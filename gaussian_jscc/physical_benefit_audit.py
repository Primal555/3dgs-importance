"""One bounded offline audit of a completed physical-benefit pretraining run."""
import argparse
import copy
import hashlib
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .physical_benefit import PhysicalLearner
from .physical_benefit_audit_metrics import (decision_report, error_oracles,
    feature_collision_probe, codec_pairing_probe)


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_records(run, entries, validation):
    result = []
    digest = hashlib.sha256()
    root = (run/'replay').resolve()
    for i, entry in enumerate(entries):
        path = (root/entry['path'].replace('\\', '/')).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'invalid/missing replay shard: {path}')
        content = path.read_bytes()
        if len(content) != entry['size']:
            raise ValueError(f'replay shard size mismatch: {path}')
        digest.update(entry['id'].encode()); digest.update(content)
        group = torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
        if bool(group['validation']) != validation or not group['views']:
            raise ValueError('invalid replay split or empty views')
        weight = float(group.get('importance_weight', 1.))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('invalid original sampling weight')
        group['_x'] = torch.stack([v['inputs']['features'] for v in group['views']]).float()
        group['_y'] = torch.stack([v['values'] for v in group['views']]).double()
        if (group['_x'].shape[1:] != (4, 158) or group['_y'].shape != (len(group['views']), 3, 2)
                or not torch.isfinite(group['_x']).all() or not torch.isfinite(group['_y']).all()):
            raise ValueError('invalid cached physical features or labels')
        result.append(group)
        if (i+1) % 2000 == 0:
            print(f'loaded {"validation" if validation else "training"}: {i+1}/{len(entries)}', flush=True)
    if not result:
        raise ValueError('empty replay split')
    return result, digest.hexdigest()


@torch.inference_mode()
def predictions(learner, records):
    learner.network.eval()
    result = []
    # Each complete group keeps its original number of views.
    for start in range(0, len(records), 128):
        chosen = records[start:start+128]
        x = torch.cat([g['_x'] for g in chosen]).to(learner.device)
        pred = learner.network(x).double().cpu()
        offset = 0
        for group in chosen:
            count = len(group['_x'])
            result.append(pred[offset:offset+count].mean(0)); offset += count
    return torch.stack(result)


def arrays(records):
    truth = torch.stack([g['_y'].mean(0) for g in records])
    rates = torch.stack([torch.as_tensor(g['rates']).double() for g in records])
    weights = torch.tensor([g.get('importance_weight', 1.) for g in records], dtype=torch.float64)
    return truth, rates, weights


def evaluate(learner, records):
    pred = predictions(learner, records)
    truth, rates, weights = arrays(records)
    return dict(record_uniform=decision_report(pred, truth, rates, torch.ones_like(weights)),
        source_importance_weighted=decision_report(pred, truth, rates, weights))


def fixed_fit(source, groups, *, updates, batch_size, evaluate_every, seed, observer):
    """Replay an unchanged objective on a private fork, with measured coverage."""
    fork = PhysicalLearner(None, source.device, hidden=source.network.layers[0].out_features,
        dropout=source.network.dropout, decision_weight=source.decision_weight, tie_fraction=source.tie_fraction)
    fork.load_state_dict(copy.deepcopy(source.state_dict()))
    visits = [0]*len(groups)
    generator = random.Random(seed)
    order, cursor = [], 0
    norms, regression_losses, decision_losses = [], [], []
    history = [dict(additional_updates=0, metrics=observer(fork))]
    for step in range(1, updates+1):
        selected = []
        while len(selected) < batch_size:
            if cursor == len(order):
                order = list(range(len(groups))); generator.shuffle(order); cursor = 0
            take = min(batch_size-len(selected), len(order)-cursor)
            selected.extend(order[cursor:cursor+take]); cursor += take
        fork.network.train()
        losses = [fork._group_loss(groups[i]) for i in selected]
        loss = sum(item[0]*float(groups[i].get('importance_weight', 1.))
                   for i, item in zip(selected, losses))/len(selected)
        fork.optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = nn.utils.clip_grad_norm_(fork.network.parameters(), 10., error_if_nonfinite=True)
        fork.optimizer.step(); fork.updates += 1
        norms.append(float(norm))
        regression_losses.append(sum(float(item[1].detach())*float(groups[i].get('importance_weight', 1.))
                                     for i, item in zip(selected, losses))/len(selected))
        decision_losses.append(sum(float(item[2].detach())*float(groups[i].get('importance_weight', 1.))
                                   for i, item in zip(selected, losses))/len(selected))
        for i in selected:
            visits[i] += 1
        if step % evaluate_every == 0 or step == updates:
            metrics = observer(fork)
            history.append(dict(additional_updates=step, loss=float(loss.detach()),
                gradient_norm_before_clip=float(norm),
                cumulative_gradient_clip_fraction=float(np.mean(np.asarray(norms) > 10)),
                cumulative_gradient_max=max(norms),
                cumulative_weighted_regression_loss=float(np.mean(regression_losses)),
                cumulative_weighted_decision_loss=float(np.mean(decision_losses)), metrics=metrics))
            a = metrics.get('fit_pool', {}).get('source_importance_weighted', {})
            b = metrics.get('validation', {}).get('record_uniform', {})
            print(f'fixed fit {step}/{updates}: pool four-regret={a.get("four_state_regret")}, '
                  f'validation four-regret={b.get("four_state_regret")}', flush=True)
    return fork, history, visits


def dataset_report(learner, records, thresholds):
    pred = predictions(learner, records)
    truth, rates, weights = arrays(records)
    cost = truth.sum(-1)+rates
    magnitude = cost.abs().amax(-1)
    view_spread, view_sign_disagreement = [], []
    for group, mean_cost, rate in zip(records, cost, rates):
        view_cost = group['_y'].sum(-1)+rate
        view_spread.append(float((view_cost-mean_cost).square().mean().sqrt()))
        informative = mean_cost.abs() > 1e-14
        disagree = (view_cost*mean_cost < 0) & informative
        view_sign_disagreement.append(float(disagree.double().mean()))
    normalized = (torch.cat([g['_x'] for g in records])-learner.network.input_mean.cpu())/learner.network.input_scale.cpu()
    standardized_labels = torch.cat([g['_y'] for g in records])/learner.scales['value']
    report = dict(record_uniform=decision_report(pred, truth, rates, torch.ones_like(weights)),
        source_importance_weighted=decision_report(pred, truth, rates, weights),
        window_range=[min(g['step'] for g in records), max(g['step'] for g in records)],
        unique_sources=len({int(g['source']) for g in records}),
        unique_source_window_groups=len({(int(g['source']), int(g['step'])) for g in records}),
        mean_per_view_rd_spread=float(np.average(view_spread, weights=weights.numpy())),
        per_view_vs_mean_direction_disagreement=float(np.average(view_sign_disagreement, weights=weights.numpy())),
        sampling_weight_quantiles=torch.quantile(weights, torch.tensor([0., .1, .5, .9, 1.], dtype=torch.float64)).tolist(),
        target_magnitude_quantiles=torch.quantile(magnitude, torch.tensor([.5, .9, .99, 1.], dtype=torch.float64)).tolist(),
        # Full 8192-group replay has >2**24 feature values, above torch.quantile's limit.
        normalized_input_abs_p99=float(np.quantile(normalized.abs().numpy(), .99)),
        normalized_input_abs_max=float(normalized.abs().max()),
        normalized_input_fraction_above_10=float((normalized.abs() > 10).float().mean()),
        normalized_label_abs_p99=float(torch.quantile(standardized_labels.abs().flatten(), .99)),
        normalized_label_abs_max=float(standardized_labels.abs().max()),
        error_oracles=error_oracles(pred, truth, rates, weights), cost_strata={})
    bounds = [-float('inf')]+thresholds+[float('inf')]
    for i in range(len(bounds)-1):
        mask = (magnitude > bounds[i]) & (magnitude <= bounds[i+1])
        if mask.any():
            report['cost_strata'][str(i)] = dict(count=int(mask.sum()),
                metrics=decision_report(pred[mask], truth[mask], rates[mask], weights[mask]))
    true_keep, pred_keep = cost.mean(-1), (pred.sum(-1)+rates).mean(-1)
    wrong = (true_keep.abs() > 1e-14) & (true_keep*pred_keep <= 0)
    regrets = torch.where(wrong, true_keep.abs(), 0.)
    w = weights/weights.sum()
    top = torch.argsort(w*regrets, descending=True)[:10]
    report['highest_weighted_keep_regret'] = [dict(source=int(records[i]['source']), window=int(records[i]['step']),
        importance_weight=float(weights[i]), actual_tier_costs=cost[i].tolist(),
        predicted_tier_costs=(pred[i].sum(-1)+rates[i]).tolist(),
        conditional_regret=float(regrets[i]), weighted_contribution=float(w[i]*regrets[i]),
        per_view_conditional_cost=(records[i]['_y'].sum(-1).mean(-1)+rates[i].mean()).tolist())
        for i in top.tolist() if regrets[i] > 0]
    report['weighted_top10_regret_fraction'] = float((w[top]*regrets[top]).sum()/(w*regrets).sum()) if regrets.sum() else 0.
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--fit-updates', type=int, default=1000)
    p.add_argument('--fit-groups', type=int, default=128)
    p.add_argument('--batch-groups', type=int, default=4)
    p.add_argument('--evaluate-every', type=int, default=250)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    return p


def main(args):
    if (args.fit_updates < 0 or min(args.fit_groups, args.batch_groups, args.evaluate_every, args.cpu_threads) < 1):
        raise ValueError('invalid audit budget')
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    if out == run or out.is_relative_to(run) or run.is_relative_to(out) or out.exists():
        raise ValueError('use a new audit directory separate from the source run')
    torch.set_num_threads(args.cpu_threads); torch.manual_seed(args.seed)
    path = run/'checkpoints/latest/state.pt'
    digest = file_hash(path)
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state['family'] != 'physical_benefit_allocation_v1'
            or not state['progress'].get('pretrain_complete') or state['progress']['phase'] != 'pretrain'):
        raise ValueError('requires completed physical-benefit pretraining before allocation')
    settings = state['plan']['settings']
    print('Reading indexed replay; frozen evaluator and labels; CPU only.', flush=True)
    train, train_hash = load_records(run, state['replay']['training'], False)
    validation, val_hash = load_records(run, state['replay']['validation'], True)
    if {g['source'] for g in train} & {g['source'] for g in validation}:
        raise ValueError('training/validation source leakage')
    learner = PhysicalLearner(None, 'cpu', hidden=settings['hidden'], dropout=settings['dropout'],
        decision_weight=settings['decision_weight'], tie_fraction=settings['tie_fraction'], weight_decay=settings['weight_decay'])
    learner.load_state_dict(state['learner'])
    out.mkdir(parents=True)
    started = time.perf_counter()
    metadata = dict(source_checkpoint=str(path), source_sha256=digest, training_records=len(train),
        validation_records=len(validation), training_labels_sha256=train_hash, validation_labels_sha256=val_hash,
        normalization_refitted=False, probability_table_updated=False, renderer_calls_on_real_scene=0,
        historical_group_visits='not recorded; exact past per-group coverage cannot be reconstructed',
        historical_group_draws=learner.updates*settings['batch_groups'],
        historical_priority_changed_records_lower_bound=sum(
            state['replay']['priorities'][entry['id']] != 1. for entry in state['replay']['training']),
        historical_priority_note='changed priority proves at least one update; unchanged priority does not prove zero updates',
        recorded_preflight_sha256=file_hash(run/'preflight.json') if (run/'preflight.json').is_file() else None,
        source_family=state['family'], beta=settings.get('beta'),
        fit_settings=dict(updates=args.fit_updates, groups=min(args.fit_groups, len(train)),
            batch_groups=args.batch_groups, evaluate_every=args.evaluate_every),
        sampling='training: 0.5 uniform + 0.25 impact + 0.25 difficulty; validation: uniform held-out blocks',
        fit_selection='half random training records; remainder largest true absolute RD cost; diagnostic distribution')
    log_path = run/'metrics/loss.jsonl'
    if log_path.is_file():
        probes = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        norms = [float(row['grad_norm']) for row in probes if row.get('grad_norm') is not None]
        metadata['historical_gradient_samples'] = dict(count=len(norms),
            max_before_clip=max(norms) if norms else None,
            fraction_above_clip=float(np.mean(np.asarray(norms) > 10)) if norms else None,
            scope='last evaluator update in each logged window; not every historical update')
    write_json(out/'settings.json', metadata)
    collision = feature_collision_probe()
    # Actual loaded codec, deterministic synthetic blocks: no PLY/photo access.
    from .model_factory import config_from_dict, build_codec
    from .data import Geometry
    model = build_codec(config_from_dict(state['plan']['codec_config']))
    model.load_state_dict(state['state_dict']); model.eval().requires_grad_(False)
    geometry = state['scene_record']['geometry']
    model.set_scene_geometry(Geometry(geometry['lower'], geometry['span'], geometry['bits']))
    f = torch.randn(2, min(model.cfg.block_size, 32), model.cfg.attr_dim+3)*.1
    f[..., :3] = torch.rand_like(f[..., :3])
    codec = codec_pairing_probe(model, f, args.seed)
    del model
    write_json(out/'mechanism_checks.json', dict(feature_collision=collision, codec_pairing=codec))
    t, r, _ = arrays(train)
    severity = (t.sum(-1)+r).abs().amax(-1)
    thresholds = torch.quantile(severity, torch.tensor([.5, .9, .99], dtype=torch.float64)).tolist()
    before = dict(training=dataset_report(learner, train, thresholds),
                  validation=dataset_report(learner, validation, thresholds), cost_bin_thresholds=thresholds)
    write_json(out/'before.json', before)
    generator = random.Random(args.seed)
    count = min(args.fit_groups, len(train))
    selected = generator.sample(range(len(train)), (count+1)//2)
    used = set(selected)
    selected += [i for i in torch.argsort(severity, descending=True).tolist() if i not in used][:count-len(selected)]
    pool = [train[i] for i in selected]
    def observe(model):
        return dict(fit_pool=evaluate(model, pool), validation=evaluate(model, validation))
    fork, history, visits = fixed_fit(learner, pool, updates=args.fit_updates, batch_size=args.batch_groups,
        evaluate_every=args.evaluate_every, seed=args.seed, observer=observe)
    write_json(out/'fixed_fit_history.json', history)
    write_json(out/'fixed_fit_visits.json', [dict(source=int(g['source']), window=int(g['step']), visits=v)
                                             for g, v in zip(pool, visits)])
    if not all(torch.equal(v, learner.network.state_dict()[k]) for k, v in state['learner']['network'].items()):
        raise RuntimeError('source evaluator changed')
    if not all(torch.equal(fork.network.state_dict()[k], learner.network.state_dict()[k])
               for k in ('input_mean', 'input_scale', 'output_scale')):
        raise RuntimeError('fixed normalization changed')
    unchanged = file_hash(path) == digest
    if not unchanged:
        raise RuntimeError('source checkpoint changed during audit')
    summary = dict(status='complete', source_checkpoint_unchanged=unchanged, **metadata,
        feature_collision=collision, codec_pairing=codec,
        before=dict(training=before['training']['source_importance_weighted'],
                    validation=before['validation']['record_uniform']),
        fixed_fit_before=history[0], fixed_fit_after=history[-1],
        fixed_fit_visit_min=min(visits), fixed_fit_visit_max=max(visits),
        elapsed_seconds=time.perf_counter()-started,
        evidence_limits=['synthetic collision proves insufficiency, not its real-scene frequency',
            'fixed-pool fit tests learnability on its selected records, not global convergence',
            'oracle corrections use truth and are diagnostic only',
            'CPU codec pairing does not certify CUDA execution or regenerate old labels'])
    write_json(out/'summary.json', summary)
    print(json.dumps(dict(status='complete', summary=str(out/'summary.json'),
        weighted_training_four_regret=summary['before']['training']['four_state_regret'],
        validation_four_regret=summary['before']['validation']['four_state_regret'],
        collision_opposite_directions=collision['rd_a'] < 0 < collision['rd_b'],
        source_checkpoint_unchanged=unchanged), ensure_ascii=False), flush=True)
    return summary


if __name__ == '__main__':
    main(parser().parse_args())
