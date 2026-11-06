"""Paired fixed-data test of candidate regions and independent value heads."""
import argparse
import copy
import json
from pathlib import Path
import tarfile
import time

import torch

from .benefit_footprint_model import FootprintLearner, FootprintSplitNetwork
from .benefit_rd_choice_losses import fit_choice_targets
from .benefit_rd_choice_report import stratified_report
from .benefit_rd_choice import print_strata
from .benefit_stage_one import make_learner
from .benefit_stage_one_data import read_groups, compact_group
from .budgeted_allocation_state import load_checkpoint
from .modular_experiment import json_write
from .spatial_benefit_replay import SpatialReplay


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('collect', 'train'), required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--data', help='completed new candidate-region dataset for offline training')
    p.add_argument('--comparison', help='original completed benefit_stage_one run')
    p.add_argument('--codec', help='frozen multi-scene codec training directory')
    p.add_argument('--pipeline-root')
    p.add_argument('--manifest')
    p.add_argument('--scene', default='truck')
    p.add_argument('--device', default='cpu')
    p.add_argument('--resume', action='store_true', help='resume collection or paired offline training')
    p.add_argument('--train-records', type=int, default=4096)
    p.add_argument('--validation-records', type=int, default=512)
    p.add_argument('--label-groups', type=int, default=64)
    p.add_argument('--candidate-batch', type=int, default=64)
    p.add_argument('--max-collect-windows', type=int, default=150)
    p.add_argument('--updates', type=int, default=2000)
    p.add_argument('--evaluate-every', type=int, default=250)
    p.add_argument('--evaluation-batch', type=int, default=8)
    p.add_argument('--cpu-threads', type=int, default=4)
    return p


@torch.inference_mode()
def evaluate(learner, replay, entries, batch):
    learner.network.eval()
    groups, predictions = [], []
    for start in range(0, len(entries), batch):
        records = list(read_groups(replay, entries[start:start+batch]))
        outputs = learner.predict_group_views([[v['inputs'] for v in g['views']] for g in records])
        groups.extend(compact_group(g) for g in records)
        predictions.extend(torch.stack(o).cpu().double() for o in outputs)
    return groups, predictions


def train(args):
    if not args.data:
        raise ValueError('--data required for training')
    data, out = Path(args.data).resolve(), Path(args.out).resolve()
    if out.is_relative_to(data) or data.is_relative_to(out):
        raise ValueError('training output must be separate from source dataset')
    if out.exists() and not args.resume:
        raise FileExistsError('use new training output; completed dataset can be reused')
    if not (data/'complete.json').is_file():
        raise ValueError('complete candidate-region collection first')
    metadata = json.loads((data/'metadata.json').read_text(encoding='utf8'))
    manifest = json.loads((data/'data_manifest.json').read_text(encoding='utf8'))
    initial = load_checkpoint(data/'initial.pt')
    replay_root = Path(manifest['replay_root']).resolve()
    if not replay_root.is_dir():
        raise FileNotFoundError(f'candidate-region replay missing: {replay_root}')
    replay = SpatialReplay(replay_root, max_groups=1,
        validation_groups=len(manifest['validation']), cache_mib=256)
    training, validation = manifest['training'], manifest['validation']
    if not training or not validation or {e['id'] for e in training} & {e['id'] for e in validation}:
        raise ValueError('nonempty separate training/validation required; record overlap')
    validation_sources = set()
    for g in read_groups(replay, validation):
        if not g.get('validation'):
            raise ValueError('validation record has wrong role')
        validation_sources.add(g['source'])
    def training_records():
        for g in read_groups(replay, training):
            if g.get('validation') or g['source'] in validation_sources:
                raise ValueError('training/validation source overlap')
            if any('footprints' not in v['inputs'] for v in g['views']):
                raise ValueError('missing real candidate-region observations')
            yield g
    targets = fit_choice_targets(training_records())
    config = metadata['config']
    from .cli import seed_all
    seed_all(metadata['seed'])
    saved = dict(settings=config, learner=initial['learner'])
    original = make_learner(saved, args.device)
    upgraded = FootprintLearner(None, args.device, hidden=config['hidden'], lr=config['benefit_lr'],
        dropout=config['dropout'], weight_decay=config['weight_decay'],
        decision_weight=config['decision_weight'], tie_fraction=config['tie_fraction'],
        common_weight=config['common_weight'], pair_weight=config['pair_weight'])
    upgraded.load_state_dict(copy.deepcopy(initial['learner']))
    upgraded.network = FootprintSplitNetwork.from_spatial(initial['learner']['network'], config['dropout']).to(args.device)
    upgraded.optimizer = torch.optim.AdamW(upgraded.network.parameters(), lr=config['benefit_lr'],
                                          weight_decay=config['weight_decay'])
    learners = dict(original=original, upgraded=upgraded)
    settings = dict(version='footprint_split_controls_v1', data=str(data), metadata=metadata,
        choice_targets=targets, updates_per_arm=args.updates, batch_groups=config['batch_groups'],
        sampling='same uniform shuffled cycles; original source-importance weights',
        loss='unchanged CorrectionLearner in both arms', device=args.device,
        parameter_counts={k: sum(p.numel() for p in v.network.parameters()) for k, v in learners.items()})
    if args.resume:
        previous = json.loads((out/'settings.json').read_text(encoding='utf8'))
        if previous != settings:
            raise ValueError('paired training resume settings differ')
        if (out/'complete.json').is_file() and (out/'review.tar.gz').is_file():
            return json.loads((out/'summary.json').read_text(encoding='utf8'))
        checkpoint = load_checkpoint(out/'latest.pt')
    else:
        out.mkdir(parents=True)
        (out/'metrics').mkdir()
        json_write(out/'settings.json', settings)
        checkpoint = None
    history, last_predictions, compact = [], {}, None
    started = time.perf_counter()
    def measure(update):
        nonlocal compact
        row = dict(additional_updates=update)
        for name, learner in learners.items():
            compact, predictions = evaluate(learner, replay, validation, args.evaluation_batch)
            last_predictions[name] = predictions
            report = stratified_report(compact, predictions, targets['large_gap_threshold'])
            row[name] = dict(validation=report)
            print(f'\n【{name}】更新 {update}/{args.updates}', flush=True)
            print_strata(report)
        with (out/'metrics/history.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        history.append(row)
    generator = torch.Generator().manual_seed(metadata['seed']+1)
    visits = [0]*len(training)
    order, cursor = [], 0
    completed = 0
    if checkpoint:
        for name, learner in learners.items():
            learner.load_state_dict(checkpoint['learners'][name])
        completed = checkpoint['completed']
        visits, order, cursor = checkpoint['visits'], checkpoint['order'], checkpoint['cursor']
        generator.set_state(checkpoint['generator'])
        initial_difference = checkpoint['initial_difference']
        history.extend(json.loads(line) for line in (out/'metrics/history.jsonl').read_text(encoding='utf8').splitlines()
                       if json.loads(line)['additional_updates'] <= completed)
        with (out/'metrics/history.jsonl').open('w', encoding='utf8') as stream:
            for row in history:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
    else:
        measure(0)
        initial_difference = max(float((a-b).abs().max()) for a, b in zip(*last_predictions.values()))
        if initial_difference > 1e-6:
            raise RuntimeError('paired initial predictions differ')
    def save_checkpoint(update):
        from .benefit_footprint_collect import atomic_save
        atomic_save(dict(learners={k: v.state_dict() for k, v in learners.items()},
            completed=update, visits=visits, order=order, cursor=cursor, generator=generator.get_state(),
            initial_difference=initial_difference), out/'latest.pt')
    if not checkpoint:
        save_checkpoint(0)
    for update in range(completed+1, args.updates+1):
        chosen = []
        while len(chosen) < config['batch_groups']:
            if cursor == len(order):
                order = torch.randperm(len(training), generator=generator).tolist(); cursor = 0
            take = min(config['batch_groups']-len(chosen), len(order)-cursor)
            chosen.extend(order[cursor:cursor+take]); cursor += take
        records = list(read_groups(replay, [training[i] for i in chosen]))
        for name, learner in learners.items():
            torch.manual_seed(metadata['seed']+100000+update)
            if torch.device(args.device).type == 'cuda':
                torch.cuda.manual_seed_all(metadata['seed']+100000+update)
            learner.network.train()
            losses = learner._batch_losses(records)
            weight = losses['total'].new_tensor([g.get('importance_weight', 1.) for g in records])
            if not torch.isfinite(weight).all() or (weight <= 0).any():
                raise ValueError('invalid source-importance weights')
            loss = (losses['total']*weight).mean()
            learner.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.network.parameters(), 10., error_if_nonfinite=True)
            learner.optimizer.step(); learner.updates += 1
        for i in chosen:
            visits[i] += 1
        if update % 50 == 0:
            print(f'两组进度 {update}/{args.updates}，累计 {time.perf_counter()-started:.1f}s。', flush=True)
        if update % args.evaluate_every == 0 or update == args.updates:
            measure(update)
            save_checkpoint(update)
    if completed == args.updates:
        for name, learner in learners.items():
            compact, last_predictions[name] = evaluate(learner, replay, validation, args.evaluation_batch)
    for name, learner in learners.items():
        from .benefit_footprint_collect import atomic_save
        (out/name).mkdir(exist_ok=True)
        atomic_save(dict(learner=learner.state_dict(), architecture=name, settings=settings), out/name/'final.pt')
    with (out/'metrics/final_predictions.jsonl').open('w', encoding='utf8') as stream:
        for i, (e, g) in enumerate(zip(validation, compact)):
            stream.write(json.dumps(dict(id=e['id'], source=g['source'], step=g['step'],
                truth=[v['values'].tolist() for v in g['views']], rates=g['rates'].tolist(),
                **{name: p[i].tolist() for name, p in last_predictions.items()}), allow_nan=False)+'\n')
    result = dict(status='complete', trained_arms=2, updates_per_arm=args.updates,
        initial_max_prediction_difference=initial_difference, before=history[0], after=history[-1],
        training_coverage=dict(draws=sum(visits), min_visits=min(visits), max_visits=max(visits),
                               never_visited=sum(v == 0 for v in visits)),
        elapsed_seconds=time.perf_counter()-started)
    json_write(out/'summary.json', result)
    with tarfile.open(out/'review.tmp', 'w:gz') as archive:
        for rel in ('settings.json', 'summary.json', 'metrics/history.jsonl', 'metrics/final_predictions.jsonl'):
            archive.add(out/rel, arcname=rel)
    (out/'review.tmp').replace(out/'review.tar.gz')
    json_write(out/'complete.json', dict(status='complete', updates_per_arm=args.updates))
    print(f'完成。下载 {out / "review.tar.gz"}', flush=True)
    return result


def main(args):
    if min(args.train_records, args.validation_records, args.label_groups, args.candidate_batch,
           args.max_collect_windows, args.updates, args.evaluate_every, args.evaluation_batch, args.cpu_threads) < 1:
        raise ValueError('positive experiment budgets required')
    torch.set_num_threads(args.cpu_threads)
    if args.phase == 'collect':
        if not args.comparison or not args.codec:
            raise ValueError('--comparison and --codec required for collection')
        from .benefit_footprint_collect import collect
        return collect(args)
    return train(args)


if __name__ == '__main__':
    main(parser().parse_args())
