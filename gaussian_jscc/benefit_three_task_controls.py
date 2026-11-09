"""Fixed-label comparison of separate value tasks at equal sample budgets."""
import argparse
import json
import math
from pathlib import Path
import tarfile
import time

import torch

from .benefit_footprint_collect import atomic_save
from .benefit_footprint_controls import evaluate
from .benefit_footprint_rank_gradnorm_controls import branch_digests
from .benefit_isolated_positive_controls import load_data, tensor_digest
from .benefit_isolated_positive_report import compare_choices
from .benefit_rank_gradnorm import GradNormBalancer
from .benefit_rank_gradnorm_controls import importance, stratum
from .benefit_rd_choice_report import stratified_report
from .benefit_stage_one_data import read_groups
from .benefit_three_task import (OBJECTIVE, TASKS, TaskAccumulator, make_learner,
                                make_sampler, sampling_weights)
from .benefit_view_gap import (OBJECTIVE as VIEW_GAP, RD_DIRECTION_WEIGHT, RANK_NUMERIC_RATIO,
                               assign_with_rank_budget, gap_report)
from .budgeted_allocation_state import load_checkpoint
from .cli import seed_all
from .modular_experiment import json_write


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='completed footprint_split run with data and training')
    p.add_argument('--out', required=True)
    p.add_argument('--arm', choices=('both', 'batch8', 'batch64'), default='both')
    p.add_argument('--objective', choices=('original', 'view_gap'), default='original')
    p.add_argument('--samples', type=int, default=16000, help='record visits per arm, including repeats')
    p.add_argument('--microbatch', type=int, default=8)
    p.add_argument('--evaluate-every-samples', type=int, default=2048)
    p.add_argument('--gradient-every-samples', type=int, default=128)
    p.add_argument('--evaluation-batch', type=int, default=8)
    p.add_argument('--training-probe', type=int, default=128)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--gradnorm-alpha', type=float, default=1.5)
    p.add_argument('--weight-lr', type=float, default=.025)
    p.add_argument('--rate-ema', type=float, default=.95)
    p.add_argument('--resume', action='store_true')
    return p


@torch.no_grad()
def training_references(learner, replay, training, batch):
    learner.network.eval()
    total = torch.zeros(len(TASKS), dtype=torch.double)
    for start in range(0, len(training), batch):
        records = list(read_groups(replay, training[start:start+batch]))
        losses = learner._batch_losses(records)
        weights = importance(records, learner.device)
        total += torch.stack([(losses[k]*weights).sum() for k in TASKS]).cpu()
        if start == 0 or start+batch >= len(training) or (start+batch) % 512 == 0:
            print(f'Training references: {min(start+batch, len(training))}/{len(training)}', flush=True)
    return (total/len(training)).tolist()


def append_json(path, row):
    with path.open('a', encoding='utf8') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')


def trim_records(path, samples):
    lines = path.read_text(encoding='utf8').splitlines(keepends=True) if path.is_file() else []
    rows = []
    for i, line in enumerate(lines):
        if not line.strip(): continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines)-1 and not line.endswith(('\n', '\r')):
                break  # A partial append after the last committed checkpoint.
            raise
    rows = [r for r in rows if r['sample_visits'] <= samples]
    path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows), encoding='utf8')
    return rows


def stability(history, budget):
    rows = [r for r in history if r['sample_visits'] >= budget*.75]
    result = dict(scope='last quarter of sample budget; fixed validation set', evaluations=len(rows))
    for scope in ('all', 'ordinary', 'large_gap'):
        values = [r['validation'][scope]['three_tier']['regret_mean'] for r in rows
                  if r['validation'][scope]['three_tier'] is not None]
        result[scope] = None if not values else dict(mean=sum(values)/len(values), minimum=min(values),
            maximum=max(values), final=values[-1])
    return result


def check_resume_settings(stored, requested, resume, *, suite=False):
    if stored == requested: return
    budgets = {'sample_budget'} if suite else {'sample_budget', 'optimizer_updates'}
    same = {k: v for k, v in stored.items() if k not in budgets} == {
        k: v for k, v in requested.items() if k not in budgets}
    if not (resume and same and requested['sample_budget'] >= stored['sample_budget']):
        raise ValueError('suite settings mismatch' if suite else 'resume settings mismatch')


def run_arm(args, loaded, out, batch):
    _, replay, training, validation, metadata, initial, previous, groups, references, hashes = loaded
    config = metadata['config']; seed = metadata['seed']; seed_all(seed)
    temperature = previous['choice_targets']['temperature']; threshold = previous['choice_targets']['large_gap_threshold']
    objective = VIEW_GAP if args.objective == 'view_gap' else OBJECTIVE
    learner = make_learner(initial, config, args.device, temperature, objective=objective)
    initial_digests = branch_digests(learner.network)
    normalizers = tensor_digest(dict(learner.network.named_buffers()))
    labels = [stratum(g, threshold) for g in read_groups(replay, training)]
    sampler = make_sampler(labels, batch, seed+1, stratified=batch == 64)
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(seed+2))
    probe = [training[i] for i in order[:min(args.training_probe, len(training))].tolist()]
    settings = dict(version=objective, source=str(Path(args.source).resolve()), source_hashes=hashes,
        source_config=config, initialization='data/initial.pt; shared original footprint construction',
        initial_branch_digests=initial_digests, seed=seed, device=args.device, cpu_threads=args.cpu_threads,
        sample_budget=args.samples, effective_batch=batch, microbatch=args.microbatch,
        optimizer_updates=args.samples//batch, tasks=list(TASKS), temperature=temperature,
        rank_pairs=[[1, 2], [1, 3], [2, 3]], regression_scale=float(learner.scales['rd']),
        numeric_coefficients=dict(common=1+config['common_weight']+config['pair_weight']/2,
                                  relative=1+2*config['pair_weight']),
        threshold=threshold, strata_counts={str(s): labels.count(s) for s in (0, 1)},
        sampling='proportional stratum cycles with population/selection correction' if batch == 64 else 'uniform shuffled cycles',
        batch_quotas={str(k): v for k, v in getattr(sampler, 'quotas', {}).items()},
        weighting='original importance times sampling correction; mean over effective batch',
        gradnorm_alpha=args.gradnorm_alpha, weight_lr=args.weight_lr, rate_ema=args.rate_ema,
        gradient_parameters='points.2.weight and bias; accumulated vectors before norm',
        evaluate_every_samples=args.evaluate_every_samples, gradient_every_samples=args.gradient_every_samples,
        evaluation_batch=args.evaluation_batch, training_probe_ids=[e['id'] for e in probe],
        trainable_parameters=sum(p.numel() for p in learner.network.parameters()), selection='final update only')
    if args.objective == 'view_gap':
        settings.update(rank_loss_gain=temperature/float(learner.scales['rd']),
            relative_rd_direction_weight=RD_DIRECTION_WEIGHT, relative_component_contrast_weight=1.,
            relative_scope='per view, tier-centered RD sum and component contrast',
            rank_numeric_gradient_ratio=RANK_NUMERIC_RATIO, rank_budget_parameters='all trainable parameters')
    stored = json.loads((out/'settings.json').read_text(encoding='utf8')) if (out/'settings.json').is_file() else None
    refs = stored['initial_reference_losses'] if stored else training_references(learner, replay, training, args.evaluation_batch)
    settings['initial_reference_losses'] = refs
    if stored is not None: check_resume_settings(stored, settings, args.resume)
    if out.exists() and not args.resume: raise FileExistsError('new output or --resume required')
    balancer = GradNormBalancer(list(TASKS), refs, device=args.device,
        alpha=args.gradnorm_alpha, lr=args.weight_lr, ema=args.rate_ema)
    completed = 0; history = []; last_stats = None
    gradient_totals = dict(updates=0, clipped=0, rank_limited=0, rank_factor_sum=0., max_gradient=0.)
    if (out/'latest.pt').is_file():
        checkpoint = load_checkpoint(out/'latest.pt')
        learner.load_state_dict(checkpoint['learner']); balancer.load_state_dict(checkpoint['balancer'])
        sampler.load_state_dict(checkpoint['sampler']); completed = checkpoint['completed']
        last_stats = checkpoint['last_stats']
        if args.objective == 'view_gap': gradient_totals = checkpoint['gradient_totals']
        history = trim_records(out/'metrics/history.jsonl', completed*batch)
        trim_records(out/'metrics/optimization.jsonl', completed*batch)
    else:
        if (out/'complete.json').is_file(): raise FileNotFoundError('completed run needs latest.pt to resume')
        out.mkdir(parents=True, exist_ok=True); (out/'metrics').mkdir(exist_ok=True)
    if (out/'complete.json').is_file() and completed*batch == args.samples:
        return json.loads((out/'summary.json').read_text(encoding='utf8'))
    if completed*batch < args.samples:
        for marker in (out/'complete.json', Path(args.out)/'complete.json'):
            marker.unlink(missing_ok=True)
        if stored and stored['sample_budget'] < args.samples:
            review = Path(args.out)/'review.tar.gz'
            previous_review = review.with_name(f'review_{stored["sample_budget"]}.tar.gz')
            if review.is_file() and not previous_review.exists(): review.rename(previous_review)
            print(f'Continue samples={completed*batch}->{args.samples}; optimizer, sampler and GradNorm restored.', flush=True)
    json_write(out/'settings.json', settings)
    parameters = list(learner.network.parameters())
    shared_ids = {id(p) for p in learner.network.points[2].parameters()}
    shared_indices = [i for i, p in enumerate(parameters) if id(p) in shared_ids]
    last_predictions = None
    started = time.perf_counter()

    def measure():
        nonlocal last_predictions
        _, last_predictions = evaluate(learner, replay, validation, args.evaluation_batch)
        tg, tp = evaluate(learner, replay, probe, args.evaluation_batch)
        row = dict(sample_visits=completed*batch, optimizer_updates=completed,
            validation=stratified_report(groups, last_predictions, threshold),
            training_probe=stratified_report(tg, tp, threshold), task_weights=balancer.weights().detach().cpu().tolist(),
            **{'against_'+k: compare_choices(groups, p, last_predictions) for k, p in references.items()})
        if args.objective == 'view_gap':
            row.update(gap_quality=gap_report(groups, last_predictions, threshold),
                training_gap_quality=gap_report(tg, tp, threshold), gradient_totals=dict(gradient_totals))
        if tensor_digest(dict(learner.network.named_buffers())) != normalizers: raise RuntimeError('normalizers changed')
        history.append(row); append_json(out/'metrics/history.jsonl', row)
        print(f'\n[{out.name}] samples={completed*batch}/{args.samples}, updates={completed}, weights={row["task_weights"]}', flush=True)
        for scope in ('all', 'ordinary', 'large_gap'):
            m = row['validation'][scope]
            if m['three_tier']:
                print(f'  {scope}: n={m["records"]}, regret={m["three_tier"]["regret_mean"]:.7g}, '
                      f'hit={m["three_tier"]["best_tier_hit_fraction"]:.2%}, '
                      f'max_regret={m["selection_tail"]["max_regret"]:.7g}', flush=True)
                if args.objective == 'view_gap':
                    gap = row['gap_quality'][scope]
                    print(f'    view_gap_RMSE={gap["per_view_pair_rmse"]:.7g}, '
                          f'window_gap_RMSE={gap["window_pair_rmse"]:.7g}, '
                          f'complete_order={gap["complete_order_correct_fraction"]:.2%}, '
                          f'correct_choice_gap_RMSE={gap["correct_choice_pair_rmse"]}', flush=True)
        if args.objective == 'view_gap' and gradient_totals['updates']:
            n = gradient_totals['updates']
            print(f'  all updates: clipped={gradient_totals["clipped"]/n:.2%}, '
                  f'rank_limited={gradient_totals["rank_limited"]/n:.2%}, '
                  f'mean_rank_factor={gradient_totals["rank_factor_sum"]/n:.4f}', flush=True)
        train_regret = row['training_probe']['all']['three_tier']['regret_mean']
        print(f'  fixed_training_probe_regret={train_regret:.7g}', flush=True)
        return row

    def save():
        atomic_save(dict(learner=learner.state_dict(), balancer=balancer.state_dict(), sampler=sampler.state_dict(),
                         completed=completed, last_stats=last_stats, gradient_totals=gradient_totals), out/'latest.pt')

    if not history: measure(); save()
    for update in range(completed+1, args.samples//batch+1):
        sampled = sampler.next(); correction = sampling_weights(labels, sampler, sampled)
        records = list(read_groups(replay, [training[i] for i in sampled]))
        learner.network.train(); learner.optimizer.zero_grad(set_to_none=True)
        weights = importance(records, learner.device)*correction.to(learner.device)/batch
        accumulator = TaskAccumulator(parameters, shared_indices)
        diagnostics = update == 1 or update*batch % args.gradient_every_samples == 0 or update*batch == args.samples
        for start in range(0, batch, args.microbatch):
            end = min(start+args.microbatch, batch)
            torch.manual_seed(seed+100000+(update-1)*batch+start)
            if torch.device(args.device).type == 'cuda': torch.cuda.manual_seed_all(seed+100000+(update-1)*batch+start)
            losses = learner._batch_losses(records[start:end])
            accumulator.add(torch.stack([losses[k] for k in TASKS], -1), weights[start:end],
                            [labels[i] for i in sampled[start:end]], diagnostics=diagnostics)
        last_stats = balancer.measure_gradients(accumulator.losses, accumulator.shared_norms())
        task_weights = balancer.weights().detach()
        if args.objective == 'view_gap':
            applied_weights, budget_report = assign_with_rank_budget(accumulator, task_weights)
            last_stats.update(budget_report)
        else:
            applied_weights = task_weights
            accumulator.assign(task_weights)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
        learner.optimizer.step(); balancer.update_weights(); learner.updates += 1; completed = update
        last_stats.update(update=update, sample_visits=update*batch, network_grad_norm=float(norm),
            clipped=bool(norm > 10), network_loss=float((accumulator.losses*task_weights).sum()),
            strata=accumulator.stratum_report(applied_weights) if diagnostics else None)
        if args.objective == 'view_gap':
            gradient_totals['updates'] += 1
            gradient_totals['clipped'] += int(norm > 10)
            gradient_totals['rank_limited'] += int(budget_report['rank_gradient_limited'])
            gradient_totals['rank_factor_sum'] += budget_report['rank_gradient_factor']
            gradient_totals['max_gradient'] = max(gradient_totals['max_gradient'], float(norm))
            last_stats['applied_weighted_loss'] = float((accumulator.losses*applied_weights).sum())
        if diagnostics:
            append_json(out/'metrics/optimization.jsonl', last_stats)
            print(f'[{out.name}] samples={update*batch}/{args.samples}, update={update}, '
                  f'loss={last_stats["network_loss"]:.6g}, grad={float(norm):.5g}, elapsed={time.perf_counter()-started:.1f}s', flush=True)
            sr = last_stats['strata']
            print(f'  common/relative/rank losses={last_stats["losses"]}, weights={last_stats["weights"]}', flush=True)
            if args.objective == 'view_gap':
                print(f'  all-parameter numeric_grad={last_stats["numeric_gradient_norm"]:.5g}, '
                      f'rank_grad={last_stats["requested_rank_gradient_norm"]:.5g}'
                      f'->{last_stats["applied_rank_gradient_norm"]:.5g}, '
                      f'rank_factor={last_stats["rank_gradient_factor"]:.4f}', flush=True)
            print(f'  shared-gradient contributions: ordinary(n={sr["ordinary"]["records"]})='
                  f'{sr["ordinary"]["combined_shared_gradient_norm"]:.5g}, '
                  f'large_gap(n={sr["large_gap"]["records"]})={sr["large_gap"]["combined_shared_gradient_norm"]:.5g}, '
                  f'cosine={sr["gradient_cosine"]}, clipped={last_stats["clipped"]}', flush=True)
        if update*batch % args.evaluate_every_samples == 0 or update*batch == args.samples:
            measure(); save()
    if last_predictions is None:
        # Rebuild a report after interruption during final packaging, without updates.
        _, last_predictions = evaluate(learner, replay, validation, args.evaluation_batch)
    atomic_save(dict(learner=learner.state_dict(), balancer=balancer.state_dict(), settings=settings), out/'final.pt')
    with (out/'metrics/final_predictions.jsonl').open('w', encoding='utf8') as f:
        for i, (entry, group) in enumerate(zip(validation, groups)):
            row = dict(id=entry['id'], source=group['source'], step=group['step'],
                truth=[v['values'].tolist() for v in group['views']], rates=torch.as_tensor(group['rates']).tolist(),
                three_task=last_predictions[i].tolist(), **{k: p[i].tolist() for k, p in references.items()})
            f.write(json.dumps(row, allow_nan=False)+'\n')
    result = dict(status='complete', effective_batch=batch, sample_visits=completed*batch, optimizer_updates=completed,
        initial_branch_digests=initial_digests,
        branch_changed={k: v != initial_digests[k] for k, v in branch_digests(learner.network).items()},
        normalizers_unchanged=tensor_digest(dict(learner.network.named_buffers())) == normalizers,
        before=history[0], after=history[-1], stability=stability(history, args.samples),
        baselines={k: stratified_report(groups, p, threshold) for k, p in references.items()},
        training_coverage=dict(draws=sum(sampler.visits), min_visits=min(sampler.visits), max_visits=max(sampler.visits),
                               never_visited=sum(v == 0 for v in sampler.visits)), last_optimization=last_stats)
    if args.objective == 'view_gap': result['gradient_totals'] = gradient_totals
    json_write(out/'summary.json', result)
    json_write(out/'complete.json', dict(status='complete', sample_visits=completed*batch, optimizer_updates=completed))
    return result


def main(args):
    numbers = (args.samples, args.microbatch, args.evaluate_every_samples, args.gradient_every_samples,
               args.evaluation_batch, args.training_probe, args.cpu_threads)
    if min(numbers) < 1: raise ValueError('positive sample and execution budgets required')
    if (not math.isfinite(args.gradnorm_alpha) or args.gradnorm_alpha < 0 or
            not math.isfinite(args.weight_lr) or args.weight_lr <= 0 or not 0 <= args.rate_ema < 1):
        raise ValueError('invalid GradNorm settings')
    batches = [8, 64] if args.arm == 'both' else [int(args.arm[5:])]
    if any(args.samples % b or args.evaluate_every_samples % b or args.gradient_every_samples % b for b in batches):
        raise ValueError('sample budget and reporting intervals must be divisible by effective batch')
    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    if source.is_relative_to(out) or out.is_relative_to(source): raise ValueError('source/output overlap')
    if out.exists() and not args.resume: raise FileExistsError('new output or --resume required')
    torch.set_num_threads(args.cpu_threads)
    loaded = load_data(source)
    replay_path = loaded[1].root.resolve()
    if out.is_relative_to(replay_path) or replay_path.is_relative_to(out): raise ValueError('output/replay overlap')
    plan = dict(version=VIEW_GAP if args.objective == 'view_gap' else OBJECTIVE,
                source=str(source), batches=batches, sample_budget=args.samples)
    if (out/'plan.json').is_file():
        check_resume_settings(json.loads((out/'plan.json').read_text(encoding='utf8')), plan, args.resume, suite=True)
    out.mkdir(parents=True, exist_ok=True)
    if not (out/'plan.json').is_file(): json_write(out/'plan.json', plan)
    results = {f'split_batch{batch}': run_arm(args, loaded, out/f'split_batch{batch}', batch) for batch in batches}
    json_write(out/'plan.json', plan)
    result = dict(status='complete', sample_budget_per_arm=args.samples, arms={k: dict(
        optimizer_updates=r['optimizer_updates'], sample_visits=r['sample_visits'],
        validation=r['after']['validation'], stability=r['stability']) for k, r in results.items()})
    json_write(out/'summary.json', result)
    with tarfile.open(out/'review.tmp', 'w:gz') as archive:
        for rel in ('plan.json', 'summary.json'):
            archive.add(out/rel, arcname=rel)
        for arm in results:
            for rel in ('settings.json', 'summary.json', 'metrics/history.jsonl', 'metrics/optimization.jsonl', 'metrics/final_predictions.jsonl'):
                archive.add(out/arm/rel, arcname=f'{arm}/{rel}')
    (out/'review.tmp').replace(out/'review.tar.gz')
    json_write(out/'complete.json', dict(status='complete', arms=list(results)))
    print(f'Complete. Download {out/"review.tar.gz"}', flush=True)
    return result


if __name__ == '__main__':
    main(parser().parse_args())
