"""Train RankNet/GradNorm on the original shared footprint split-head model."""
import argparse
import json
import math
from pathlib import Path
import tarfile
import time

import torch

from .benefit_footprint_collect import atomic_save
from .benefit_footprint_controls import evaluate
from .benefit_footprint_rank_gradnorm import OBJECTIVE, UniformCycles, make_learner
from .benefit_isolated_positive_controls import load_data, tensor_digest
from .benefit_isolated_positive_report import compare_choices
from .benefit_rank_gradnorm import GradNormBalancer
from .benefit_rank_gradnorm_controls import importance
from .benefit_rd_choice_report import stratified_report
from .benefit_stage_one import make_learner as make_original
from .benefit_stage_one_data import read_groups
from .budgeted_allocation_state import load_checkpoint
from .cli import seed_all
from .modular_experiment import json_write


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='completed footprint_split run')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--updates', type=int, default=2000)
    p.add_argument('--evaluate-every', type=int, default=250)
    p.add_argument('--evaluation-batch', type=int, default=8)
    p.add_argument('--training-probe', type=int, default=128)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--gradnorm-alpha', type=float, default=1.5)
    p.add_argument('--weight-lr', type=float, default=.025)
    p.add_argument('--rate-ema', type=float, default=.95)
    p.add_argument('--resume', action='store_true')
    return p


def aggregate(losses, records, device):
    weights = importance(records, device)
    return torch.stack([(losses[key]*weights).mean() for key in ('magnitude', 'ranking')])


@torch.no_grad()
def training_references(learner, replay, entries, batch):
    learner.network.eval(); total = torch.zeros(2, dtype=torch.float64)
    for start in range(0, len(entries), batch):
        records = list(read_groups(replay, entries[start:start+batch]))
        if any(g.get('validation') for g in records):
            raise ValueError('training references require training records')
        total += aggregate(learner._batch_losses(records), records, learner.device).cpu()*len(records)
        if start == 0 or start+batch >= len(entries) or (start+batch) % 512 == 0:
            print(f'训练损失基准 {min(start+batch,len(entries))}/{len(entries)}；参数未更新。', flush=True)
    return (total/len(entries)).tolist()


def branch_digests(network):
    return {name: tensor_digest(getattr(network, name).state_dict()) for name in
            ('points', 'profile_encoder', 'profile_projection', 'common_head', 'relative_head')}


def main(args):
    if min(args.updates, args.evaluate_every, args.evaluation_batch, args.training_probe, args.cpu_threads) < 1:
        raise ValueError('positive training and evaluation budgets required')
    if (not math.isfinite(args.gradnorm_alpha) or args.gradnorm_alpha < 0 or
            not math.isfinite(args.weight_lr) or args.weight_lr <= 0 or not 0 <= args.rate_ema < 1):
        raise ValueError('invalid GradNorm hyperparameters')
    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    if source.is_relative_to(out) or out.is_relative_to(source):
        raise ValueError('source/output overlap')
    if out.exists() and not args.resume:
        raise FileExistsError('new output or --resume required')
    torch.set_num_threads(args.cpu_threads)
    _, replay, training, validation, metadata, initial, previous, groups, references, hashes = load_data(source)
    if out.is_relative_to(replay.root.resolve()) or replay.root.resolve().is_relative_to(out):
        raise ValueError('output/replay overlap')
    config = metadata['config']; seed = metadata['seed']; seed_all(seed)
    temperature = previous['choice_targets']['temperature']
    threshold = previous['choice_targets']['large_gap_threshold']
    learner = make_learner(initial, config, args.device, temperature)
    initial_branches = branch_digests(learner.network)
    normalizers = tensor_digest(dict(learner.network.named_buffers()))
    sampler = UniformCycles(len(training), config['batch_groups'], seed+1)
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(seed+2))
    probe = [training[i] for i in order[:min(args.training_probe, len(training))].tolist()]
    names = ['magnitude', 'ranking']
    settings = dict(version=OBJECTIVE, source=str(source), source_hashes=hashes, source_config=config,
        initialization='data/initial.pt; original split construction order', initial_branch_digests=initial_branches,
        architecture='FootprintSplitNetwork; shared trainable point/CNN features and two trainable heads',
        updates=args.updates, reference_updates=previous['updates_per_arm'], device=args.device, seed=seed,
        evaluate_every=args.evaluate_every, evaluation_batch=args.evaluation_batch,
        training_probe_ids=[e['id'] for e in probe], tasks=names, temperature=temperature,
        temperature_fit_scope='training records only', threshold=threshold, regression_scale=float(learner.scales['rd']),
        regression_coefficients=dict(centered=1+2*config['pair_weight'],
            common=1+config['common_weight']+config['pair_weight']/2),
        rank_pairs='all six unordered q0/q1/q2/q3 RD pairs; camera-window mean',
        gradnorm_alpha=args.gradnorm_alpha, weight_lr=args.weight_lr, rate_ema=args.rate_ema,
        gradient_parameters='points.2.weight and bias; shared before both heads',
        sampling='original uniform shuffled cycles', batch_groups=config['batch_groups'],
        weighting='original source importance, weighted batch mean', selection='final update only',
        trainable_parameters=sum(p.numel() for p in learner.network.parameters() if p.requires_grad))
    stored = json.loads((out/'settings.json').read_text(encoding='utf8')) if args.resume else None
    refs = stored['initial_reference_losses'] if stored else training_references(learner, replay, training, args.evaluation_batch)
    settings['initial_reference_losses'] = refs
    balancer = GradNormBalancer(names, refs, device=args.device,
        alpha=args.gradnorm_alpha, lr=args.weight_lr, ema=args.rate_ema)
    checkpoint = None
    if args.resume:
        if stored != settings:
            raise ValueError('resume settings mismatch')
        if (out/'complete.json').is_file() and (out/'review.tar.gz').is_file():
            return json.loads((out/'summary.json').read_text(encoding='utf8'))
        checkpoint = load_checkpoint(out/'latest.pt')
    else:
        out.mkdir(parents=True); (out/'metrics').mkdir(); json_write(out/'settings.json', settings)
    baselines = {k: stratified_report(groups, p, threshold) for k, p in references.items()}
    completed = 0; history = []; last_predictions = None; last_stats = None
    if checkpoint:
        learner.load_state_dict(checkpoint['learner']); balancer.load_state_dict(checkpoint['balancer'])
        sampler.load_state_dict(checkpoint['sampler']); completed = checkpoint['completed']
        difference = checkpoint['initial_difference']; last_stats = checkpoint['last_stats']
        for name, key in (('history', 'additional_updates'), ('optimization', 'update')):
            path = out/'metrics'/f'{name}.jsonl'
            if path.is_file():
                rows = [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]
                rows = [r for r in rows if r[key] <= completed]
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf8')
                if name == 'history':
                    history = rows
    else:
        original = make_original(dict(settings=config, learner=initial['learner']), args.device)
        _, a = evaluate(original, replay, validation, args.evaluation_batch)
        _, b = evaluate(learner, replay, validation, args.evaluation_batch)
        difference = max(float((x-y).abs().max()) for x, y in zip(a, b))
        if difference > 1e-6:
            raise RuntimeError('initial effective functions differ')
        del original
    started = time.perf_counter()
    def measure(update, append=True):
        nonlocal last_predictions
        _, last_predictions = evaluate(learner, replay, validation, args.evaluation_batch)
        tg, tp = evaluate(learner, replay, probe, args.evaluation_batch)
        row = dict(additional_updates=update, validation=stratified_report(groups, last_predictions, threshold),
            training_probe=stratified_report(tg, tp, threshold), task_weights=balancer.weights().detach().cpu().tolist(),
            **{'against_'+k: compare_choices(groups, p, last_predictions) for k, p in references.items()})
        if tensor_digest(dict(learner.network.named_buffers())) != normalizers:
            raise RuntimeError('fixed normalizers changed')
        if append:
            history.append(row)
            with (out/'metrics/history.jsonl').open('a', encoding='utf8') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
        print(f'\n共享覆盖区域双头 RankNet＋GradNorm {update}/{args.updates}', flush=True)
        for scope in ('all', 'ordinary', 'large_gap'):
            r = row['validation'][scope]
            if r['three_tier'] is not None:
                print(f"  {scope}: regret={r['three_tier']['regret_mean']:.7g}, hit={r['three_tier']['best_tier_hit_fraction']:.2%}, pair_RMSE={r['positive_window']['rd_rmse']:.7g}", flush=True)
        for key in references:
            c = row['against_'+key]; ex = c['excluding_largest_improvement']
            print(f"  对比{key}末步：改善{c['beneficial_changes']}，恶化{c['harmful_changes']}；排除最大收益regret={ex['reference_regret']}->{ex['candidate_regret']}", flush=True)
        print(f"  非零变化命中率={row['against_original']['active']['candidate_best_hit_fraction']}；任务权重={row['task_weights']}", flush=True)
    def save():
        atomic_save(dict(learner=learner.state_dict(), balancer=balancer.state_dict(), sampler=sampler.state_dict(),
            completed=completed, initial_difference=difference, last_stats=last_stats), out/'latest.pt')
    if not checkpoint:
        measure(0); save()
    shared = list(learner.network.points[2].parameters())
    for update in range(completed+1, args.updates+1):
        chosen = sampler.next(); records = list(read_groups(replay, [training[i] for i in chosen]))
        torch.manual_seed(seed+100000+update)
        if torch.device(args.device).type == 'cuda':
            torch.cuda.manual_seed_all(seed+100000+update)
        learner.network.train(); learner.optimizer.zero_grad(set_to_none=True)
        tasks = aggregate(learner._batch_losses(records), records, learner.device)
        last_stats = balancer.measure(tasks, shared)
        objective = (tasks*balancer.weights().detach()).sum(); objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(learner.network.parameters(), 10., error_if_nonfinite=True)
        learner.optimizer.step(); balancer.update_weights(); learner.updates += 1; completed = update
        last_stats.update(update=update, network_loss=float(objective.detach()), network_grad_norm=float(norm))
        if update == 1 or update % 10 == 0:
            with (out/'metrics/optimization.jsonl').open('a', encoding='utf8') as f:
                f.write(json.dumps(last_stats, allow_nan=False)+'\n')
        if update % 50 == 0:
            print(f'更新 {update}/{args.updates}: loss={float(objective.detach()):.6g}, grad={float(norm):.5g}, elapsed={time.perf_counter()-started:.1f}s', flush=True)
        if update % args.evaluate_every == 0 or update == args.updates:
            measure(update); save()
    if last_predictions is None:
        measure(completed, append=False)
    changed = {k: v != initial_branches[k] for k, v in branch_digests(learner.network).items()}
    atomic_save(dict(learner=learner.state_dict(), balancer=balancer.state_dict(), settings=settings,
        architecture=OBJECTIVE), out/'final.pt')
    with (out/'metrics/final_predictions.jsonl').open('w', encoding='utf8') as f:
        for i, (entry, group) in enumerate(zip(validation, groups)):
            row = dict(id=entry['id'], source=group['source'], step=group['step'],
                truth=[v['values'].tolist() for v in group['views']], rates=torch.as_tensor(group['rates']).tolist(),
                footprint_rank_gradnorm=last_predictions[i].tolist(), **{k: p[i].tolist() for k, p in references.items()})
            f.write(json.dumps(row, allow_nan=False)+'\n')
    result = dict(status='complete', updates=completed, initial_max_prediction_difference=difference,
        branch_changed=changed, normalizers_unchanged=tensor_digest(dict(learner.network.named_buffers())) == normalizers,
        baselines=baselines, before=history[0], after=history[-1], last_optimization=last_stats,
        training_coverage=dict(draws=sum(sampler.visits), min_visits=min(sampler.visits),
            max_visits=max(sampler.visits), never_visited=sum(v == 0 for v in sampler.visits)))
    json_write(out/'summary.json', result)
    with tarfile.open(out/'review.tmp', 'w:gz') as archive:
        for rel in ('settings.json', 'summary.json', 'metrics/history.jsonl', 'metrics/optimization.jsonl', 'metrics/final_predictions.jsonl'):
            if (out/rel).is_file():
                archive.add(out/rel, arcname=rel)
    (out/'review.tmp').replace(out/'review.tar.gz')
    json_write(out/'complete.json', dict(status='complete', updates=completed, architecture=OBJECTIVE))
    print(f'完成。下载 {out/"review.tar.gz"}', flush=True)
    return result


if __name__ == '__main__':
    main(parser().parse_args())
