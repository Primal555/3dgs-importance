"""One offline RD-choice arm with full uniform coverage and a saved control."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tarfile
import time

import torch

from .benefit_stage_one_data import read_groups, compact_group
from .benefit_stage_one_diagnostic import read_network, reconcile
from .benefit_stage_one_report import summarize_predictions
from .benefit_rd_choice_data import load_trial
from .benefit_rd_choice_losses import RDChoiceLearner, OBJECTIVE, fit_choice_targets
from .benefit_rd_choice_report import stratified_report
from .modular_experiment import json_write
from .spatial_benefit import pack_inputs


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='completed benefit_stage_one comparison')
    p.add_argument('--out', required=True, help='new independent output')
    p.add_argument('--replay-root')
    p.add_argument('--pipeline-root', help='relocated original pipeline metadata directory')
    p.add_argument('--batch-groups', type=int, help='actual original offline batch size; required for legacy comparisons without recorded batch metadata')
    p.add_argument('--device', default='cpu')
    p.add_argument('--evaluate-every', type=int, default=250)
    p.add_argument('--evaluation-batch', type=int, default=8)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--replay-cache-mib', type=float, default=256.)
    p.add_argument('--component-weight', type=float, default=.25)
    return p


@torch.inference_mode()
def predict(network, replay, entries, batch, device):
    network.eval()
    groups, predictions = [], []
    for start in range(0, len(entries), batch):
        records = list(read_groups(replay, entries[start:start+batch]))
        counts = [len(g['views']) for g in records]
        inputs = pack_inputs([v['inputs'] for g in records for v in g['views']], device)
        output = network(*inputs).cpu().double()
        groups.extend(compact_group(g) for g in records)
        predictions.extend(output.split(counts))
    return groups, predictions


def verified_report(groups, predictions, recorded, label):
    report = summarize_predictions(groups, predictions)
    checks = reconcile(report, recorded)
    failures = [k for k, v in checks.items() if not v['matched']]
    if failures:
        raise ValueError(f'{label} metrics mismatch: {failures[:5]}')
    return report


def print_strata(report):
    for key, title in (('all', '全部样本'), ('ordinary', '普通样本'), ('large_gap', '高差距样本')):
        row = report[key]
        if not row['records']:
            print(f'  {title}：记录=0。', flush=True)
            continue
        value = row['value_error']
        print(f'  {title}：记录={row["records"]}，三档regret={row["three_tier"]["regret_mean"]:.7g}，'
              f'命中={row["three_tier"]["best_tier_hit_fraction"]:.2%}，'
              f'档间RD MAE={row["positive_window"]["rd_mae"]:.7g}，'
              f'档间RD RMSE={row["positive_window"]["rd_rmse"]:.7g}，'
              f'逐视角RD RMSE={value["per_view_rd_rmse"]:.7g}，'
              f'逐视角MSE/结构RMSE={value["per_view_component_rmse"]}，'
              f'最大错选代价={row["selection_tail"]["max_regret"]:.7g}。', flush=True)


def main(args):
    if (min(args.evaluate_every, args.evaluation_batch, args.cpu_threads) < 1 or
            not math.isfinite(args.component_weight) or args.component_weight <= 0):
        raise ValueError('positive budgets and component weight required')
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    if out.is_relative_to(run) or run.is_relative_to(out):
        raise ValueError('source and output must be separate')
    if out.exists():
        raise FileExistsError('use a new RD-choice output directory')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu')
    torch.set_num_threads(args.cpu_threads)
    started = time.perf_counter()
    saved = load_trial(run, args.replay_root, args.pipeline_root, args.replay_cache_mib)
    replay, manifest = saved['replay'], saved['manifest']
    if out.is_relative_to(replay.root) or replay.root.is_relative_to(out):
        raise ValueError('source replay and output must be separate')
    validation_sources = set()
    for g in read_groups(replay, manifest['validation']):
        if g.get('supervision') == 'action_gap' or not g.get('validation'):
            raise ValueError('independent complete validation teachers required')
        validation_sources.add(g['source'])
    def training_records():
        for i, g in enumerate(read_groups(replay, manifest['training']), 1):
            if g.get('validation') or g['source'] in validation_sources:
                raise ValueError('training/validation source overlap')
            if i % 512 == 0:
                print(f'拟合训练温度与固定分层阈值 {i}/{len(manifest["training"])}。', flush=True)
            yield g
    targets = fit_choice_targets(training_records())
    initial, config = saved['initial'], saved['config']
    actual_batch = initial['settings'].get('batch_groups')
    if actual_batch is None:
        if args.batch_groups is None:
            raise ValueError('legacy comparison did not record actual batch size; supply the original --batch-groups')
        actual_batch = args.batch_groups
        batch_provenance = 'explicit original legacy batch size'
    else:
        if args.batch_groups is not None and args.batch_groups != actual_batch:
            raise ValueError('batch size differs from recorded original comparison')
        batch_provenance = 'initial.pt recorded actual batch size'
    if actual_batch < 1:
        raise ValueError('positive original batch size required')
    learner = RDChoiceLearner(None, device, hidden=config['hidden'], lr=config['benefit_lr'],
        dropout=config['dropout'], weight_decay=config['weight_decay'],
        decision_weight=config['decision_weight'], tie_fraction=config['tie_fraction'],
        common_weight=config['common_weight'], pair_weight=config['pair_weight'],
        training_probe_groups=config['training_probe_groups'])
    learner.load_state_dict(copy.deepcopy(initial['learner']))
    learner.optimizer = torch.optim.AdamW(learner.network.parameters(), lr=config['benefit_lr'],
                                          weight_decay=config['weight_decay'])
    learner.configure_objective(targets, args.component_weight)
    groups, initial_predictions = predict(learner.network, replay, manifest['validation'], args.evaluation_batch, device)
    source_summary = saved['summary']
    verified_report(groups, initial_predictions, source_summary['before']['original']['fixed_validation'], 'source initial')
    control, control_state, control_digest = read_network(run/'original/final.pt', device)
    if (control_state['arm'] != 'original' or
            control_state['settings']['source_checkpoint_sha256'] != source_summary['source_snapshot_sha256'] or
            control_state['additional_updates'] != source_summary['additional_updates']):
        raise ValueError('reference final identity mismatch')
    groups, control_predictions = predict(control, replay, manifest['validation'], args.evaluation_batch, device)
    verified_report(groups, control_predictions, source_summary['after']['original']['fixed_validation'], 'reference final')
    del control
    reference = stratified_report(groups, control_predictions, targets['large_gap_threshold'])
    updates = source_summary['additional_updates']
    seed = initial['settings']['seed']
    settings = dict(version=OBJECTIVE, source_run=str(run), source_initial_sha256=saved['initial_sha256'],
        source_checkpoint_sha256=source_summary['source_snapshot_sha256'], original_final_sha256=control_digest,
        evaluator_configuration_source=saved['config_source'], evaluator_configuration=config,
        updates=updates, seed=seed, batch_groups=actual_batch, batch_provenance=batch_provenance,
        sampling='uniform shuffled cycles; original source-importance weights',
        training_groups=len(manifest['training']), validation_groups=len(manifest['validation']),
        main_rd_scale=float(learner.scales['rd']), component_weight=args.component_weight,
        joint_temperature_factor=(targets['temperature']/float(learner.scales['rd']))**2,
        decision_weight=learner.decision_weight, choice_targets=targets, device=str(device),
        optimizer='fresh AdamW, original LR and weight decay', final_selection='last update only',
        manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())
    out.mkdir(parents=True)
    (out/'metrics').mkdir()
    json_write(out/'settings.json', settings)
    json_write(out/'reference.json', reference)
    print(f'单组离线更新={updates}，固定训练记录={len(manifest["training"])}，'
          f'原组末步对照已复核；每批={actual_batch}。', flush=True)
    print(f'共同RD尺度={settings["main_rd_scale"]:.7g}，联合温度={targets["temperature"]:.7g}；'
          f'逐视角分量辅助权重={args.component_weight:g}。', flush=True)
    history = []
    def measure(update, loss_values=None):
        groups, predictions = predict(learner.network, replay, manifest['validation'], args.evaluation_batch, device)
        report = stratified_report(groups, predictions, targets['large_gap_threshold'])
        probe_g, probe_p = predict(learner.network, replay, manifest['recent_training'], args.evaluation_batch, device)
        probe = summarize_predictions(probe_g, probe_p)
        row = dict(additional_updates=update, validation=report, recent_training=probe,
                   training_losses=loss_values, elapsed_seconds=time.perf_counter()-started)
        history.append(row)
        with (out/'metrics/history.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(f'\n离线更新 {update}/{updates}：', flush=True)
        print_strata(report)
        return groups, predictions
    measure(0)
    generator = torch.Generator().manual_seed(seed+1)
    order, cursor = [], 0
    visits = [0]*len(manifest['training'])
    loss_sums, loss_count = {}, 0
    for update in range(1, updates+1):
        selected = []
        while len(selected) < actual_batch:
            if cursor == len(order):
                order = torch.randperm(len(visits), generator=generator).tolist()
                cursor = 0
            take = min(actual_batch-len(selected), len(order)-cursor)
            selected.extend(order[cursor:cursor+take]); cursor += take
        records = list(read_groups(replay, [manifest['training'][i] for i in selected]))
        torch.manual_seed(seed+100000+update)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed+100000+update)
        learner.network.train()
        losses = learner._batch_losses(records)
        weights = losses['total'].new_tensor([g.get('importance_weight', 1.) for g in records])
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('positive finite source-importance weights required')
        loss = (losses['total']*weights).mean()
        learner.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learner.network.parameters(), 10., error_if_nonfinite=True)
        learner.optimizer.step()
        learner.updates += 1
        for i in selected:
            visits[i] += 1
        for key, value in losses.items():
            loss_sums[key] = loss_sums.get(key, 0.)+float((value.detach()*weights).mean())
        loss_count += 1
        if update % 50 == 0:
            print(f'进度 {update}/{updates}，累计 {time.perf_counter()-started:.1f}s。', flush=True)
        if update % args.evaluate_every == 0 or update == updates:
            final_groups, final_predictions = measure(update, {k: v/loss_count for k, v in loss_sums.items()})
            loss_sums, loss_count = {}, 0
    if updates == 0:
        final_groups, final_predictions = groups, initial_predictions
    torch.save(dict(learner=learner.state_dict(), settings=settings, diagnostic_only=True,
                    additional_updates=updates), out/'final.pt')
    result = dict(status='complete', version=OBJECTIVE, additional_updates=updates,
        source_initial_verified=True, reference_final_verified=True, renderer_calls=0, trained_arms=1,
        before=history[0], after=history[-1], reference=reference,
        training_coverage=dict(draws=sum(visits), min_visits=min(visits), max_visits=max(visits),
                               never_visited=sum(v == 0 for v in visits)),
        elapsed_seconds=time.perf_counter()-started)
    with (out/'metrics/final_predictions.jsonl').open('w', encoding='utf8') as stream:
        for entry, g, p, reference_p in zip(manifest['validation'], final_groups, final_predictions, control_predictions):
            stream.write(json.dumps(dict(id=entry['id'], source=g['source'], step=g['step'],
                rates=torch.as_tensor(g['rates']).tolist(),
                truth=[v['values'].tolist() for v in g['views']],
                original_prediction=reference_p.tolist(),
                prediction=p.tolist()), allow_nan=False)+'\n')
    json_write(out/'summary.json', result)
    with tarfile.open(out/'review.tar.gz', 'w:gz') as archive:
        for rel in ('settings.json', 'reference.json', 'summary.json', 'metrics/history.jsonl', 'metrics/final_predictions.jsonl'):
            archive.add(out/rel, arcname=rel)
    print(f'完成。下载 {out / "review.tar.gz"}', flush=True)
    return result


if __name__ == '__main__':
    main(parser().parse_args())
