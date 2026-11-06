"""Paired offline loss repair with explicit validation and zero scene renders."""
import argparse
import copy
import json
import math
from pathlib import Path
import tarfile
import time

import torch

from .benefit_stage_one_data import snapshot, fixed_entries, read_groups, compact_group
from .benefit_stage_one_losses import PositivePairLearner, fit_positive_scales, OBJECTIVE
from .benefit_stage_one_report import summarize_predictions, print_report
from .modular_experiment import json_write
from .neural_correction_learning import CorrectionLearner


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='wide full-history run or its state.pt')
    p.add_argument('--out', required=True, help='new independent offline output')
    p.add_argument('--replay-root', help='relocate unchanged replay shards')
    p.add_argument('--device', default='cpu')
    p.add_argument('--updates', type=int, default=2000, help='network optimizer updates per arm')
    p.add_argument('--evaluate-every', type=int, default=250)
    p.add_argument('--train-groups', type=int, default=4096)
    p.add_argument('--probe-groups', type=int, default=128)
    p.add_argument('--batch-groups', type=int, help='default: original training batch size')
    p.add_argument('--evaluation-batch', type=int, default=8)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--replay-cache-mib', type=float, default=256.)
    p.add_argument('--q0-aux-weight', type=float, default=.25)
    p.add_argument('--seed', type=int, default=46006)
    return p


def make_learner(saved, device, repaired=False, positive_scales=None, q0_aux_weight=.25):
    s = saved['settings']
    kind = PositivePairLearner if repaired else CorrectionLearner
    learner = kind(None, device, hidden=s['hidden'], lr=s['benefit_lr'], dropout=s['dropout'],
        weight_decay=s['weight_decay'], decision_weight=s['decision_weight'], tie_fraction=s['tie_fraction'],
        common_weight=s['common_weight'], pair_weight=s['pair_weight'],
        training_probe_groups=s['training_probe_groups'])
    learner.load_state_dict(copy.deepcopy(saved['learner']))
    # Match optimizer initialization across changed and unchanged objectives.
    learner.optimizer = torch.optim.AdamW(learner.network.parameters(), lr=s['benefit_lr'],
                                          weight_decay=s['weight_decay'])
    if repaired:
        learner.configure_objective(positive_scales, q0_aux_weight)
    return learner


@torch.inference_mode()
def evaluate(learner, replay, entries, batch):
    groups, predictions = [], []
    for start in range(0, len(entries), batch):
        records = list(read_groups(replay, entries[start:start+batch]))
        output = learner.predict_group_views([[v['inputs'] for v in g['views']] for g in records])
        groups.extend(compact_group(g) for g in records)
        predictions.extend(torch.stack(p).cpu().double() for p in output)
    return summarize_predictions(groups, predictions)


def comparison(original, repaired):
    a, b = original['fixed_validation'], repaired['fixed_validation']
    fields = dict(positive_selection_regret=(a['three_tier']['regret_mean'], b['three_tier']['regret_mean']),
        positive_direction_regret=(a['positive_window']['direction_regret_mean'], b['positive_window']['direction_regret_mean']),
        positive_rd_rmse=(a['positive_window']['rd_rmse'], b['positive_window']['rd_rmse']),
        q0_common_rmse=(a['q0_auxiliary']['common_rd_rmse'], b['q0_auxiliary']['common_rd_rmse']))
    return {key: dict(original=x, repaired=y, difference=y-x,
                     relative_change=y/x-1 if x else None) for key, (x, y) in fields.items()}


def main(args):
    if (args.updates < 0 or min(args.evaluate_every, args.train_groups, args.probe_groups,
            args.evaluation_batch, args.cpu_threads) < 1 or
            (args.batch_groups is not None and args.batch_groups < 1) or
            not math.isfinite(args.q0_aux_weight) or args.q0_aux_weight < 0):
        raise ValueError('invalid offline budget or auxiliary weight')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu')
    out = Path(args.out).resolve()
    run = Path(args.run).resolve()
    if out.is_relative_to(run) or run.is_relative_to(out):
        raise ValueError('source and output must be separate')
    if out.exists():
        raise FileExistsError('use a new offline output directory')
    torch.set_num_threads(args.cpu_threads)
    started = time.perf_counter()
    print('第一阶段：固定检查点与回放；不渲染场景，不更新概率表。', flush=True)
    print('读取检查点快照；原训练可以继续，离线组只读取此刻保存的状态。', flush=True)
    saved = snapshot(run, args.replay_root, args.replay_cache_mib)
    print(f'评估网络配置来源：{saved["evaluator_settings_source"]}', flush=True)
    if (out.is_relative_to(saved['source_root']) or
            out.is_relative_to(saved['replay'].root) or saved['replay'].root.is_relative_to(out)):
        raise ValueError('offline output must be separate from source and replay')
    replay = saved['replay']
    train, validation = fixed_entries(replay, args.train_groups, args.seed)
    validation_sources = set()
    for g in read_groups(replay, validation):
        if g.get('supervision') == 'action_gap':
            raise ValueError('fixed validation requires complete four-state teacher records')
        validation_sources.add(g['source'])
    training_sources = set()
    print(f'固定训练记录={len(train)}/{len(replay.training)}，独立固定验证记录={len(validation)}'
          f'（{len(validation_sources)} 个源点）。', flush=True)
    def fitting_records():
        for i, g in enumerate(read_groups(replay, train), 1):
            if g['source'] in validation_sources or g.get('validation', False):
                raise ValueError('training/validation source overlap')
            training_sources.add(g['source'])
            if i % 512 == 0:
                print(f'拟合正档专用尺度：{i}/{len(train)} 条训练标签。', flush=True)
            yield g
    scales = fit_positive_scales(fitting_records(),
        original_rd_scale=saved['learner']['scales']['rd'],
        original_tie_fraction=saved['settings']['tie_fraction'])
    recent = train[-min(args.probe_groups, len(train)):]
    learners = dict(original=make_learner(saved, device),
        repaired=make_learner(saved, device, True, scales, args.q0_aux_weight))
    if any(not torch.equal(a, b) for a, b in zip(learners['original'].network.state_dict().values(),
                                               learners['repaired'].network.state_dict().values())):
        raise RuntimeError('paired initial weights differ')
    out.mkdir(parents=True)
    (out/'metrics').mkdir()
    settings = dict(version=OBJECTIVE, source_checkpoint=saved['source_checkpoint'],
        evaluator_settings_source=saved['evaluator_settings_source'],
        source_checkpoint_sha256=saved['source_sha256'], source_step=saved['source_step'],
        source_network_updates=saved['source_network_updates'], additional_updates_per_arm=args.updates,
        training_groups=len(train), available_training_groups=len(replay.training),
        training_sources=len(training_sources), validation_groups=len(validation),
        validation_unique_sources=len(validation_sources),
        validation_scope='independent fixed early contexts; no new on-policy holdout labels',
        recent_scope='latest records within fixed training subset; already trained',
        optimizer='fresh AdamW in both arms', learning_rate=saved['settings']['benefit_lr'],
        batch_groups=args.batch_groups or saved['settings']['batch_groups'],
        evaluator_configuration=dict(saved['settings']),
        weight_decay=saved['settings']['weight_decay'], decision_weight=saved['settings']['decision_weight'],
        original_q0_rd_scale=float(learners['original'].scales['rd']),
        positive_component_scales=scales['components'].tolist(), positive_rd_scale=float(scales['rd']),
        scale_training_types=scales['counts'], scale_floor=scales['floor'],
        training_positive_comparisons=scales['positive_comparisons'],
        original_ignored_training_comparisons=scales['original_ignored_comparisons'],
        repaired_direction_hard_ignore_band=0., q0_aux_weight=args.q0_aux_weight,
        sampling='fixed uniform replay subset; shared shuffled batches; original source-importance weights',
        normalization='source input/output normalizers unchanged', seed=args.seed,
        renderer_calls=0, new_teacher_labels=0, probability_table_updated=False,
        selection='final update only', device=str(device))
    json_write(out/'settings.json', settings)
    json_write(out/'data_manifest.json', dict(replay_root=str(replay.root), training=train,
                                             validation=validation, recent_training=recent))
    torch.save(dict(learner=saved['learner'], settings=settings), out/'initial.pt')
    print(f'每组 {args.updates} 次网络更新；两组共享样本与初始权重，AdamW 均重新初始化。', flush=True)
    print(f'旧q0尺度={settings["original_q0_rd_scale"]:.6g}；'
          f'新正档RD尺度={settings["positive_rd_scale"]:.6g}；q0辅助权重={args.q0_aux_weight:g}。', flush=True)
    print(f'固定训练标签中，旧方向忽略区覆盖 '
          f'{scales["original_ignored_comparisons"]}/{scales["positive_comparisons"]} 个正档比较。', flush=True)
    history = []
    def measure(update):
        row = dict(additional_updates=update, elapsed_seconds=time.perf_counter()-started)
        print(f'\n===== 离线网络更新 {update}/{args.updates} =====', flush=True)
        for name, learner in learners.items():
            label = '原损失组' if name == 'original' else '正档损失修复组'
            print(f'【{label}】', flush=True)
            report = dict(fixed_validation=evaluate(learner, replay, validation, args.evaluation_batch),
                          recent_training=evaluate(learner, replay, recent, args.evaluation_batch))
            for scope, value in report.items():
                print_report(scope, value)
            row[name] = report
        row['comparison'] = comparison(row['original'], row['repaired'])
        c = row['comparison']['positive_selection_regret']
        print(f'固定留出集三档选优代价：原组 {c["original"]:.6g} → 修复组 {c["repaired"]:.6g}'
              f'，差值={c["difference"]:+.6g}（负值有利）。', flush=True)
        with (out/'metrics/history.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        history.append(row)
    measure(0)
    generator = torch.Generator().manual_seed(args.seed+1)
    order, cursor = [], 0
    batch_size = args.batch_groups or saved['settings']['batch_groups']
    training_started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for update in range(1, args.updates+1):
        selected = []
        while len(selected) < batch_size:
            if cursor == len(order):
                order = torch.randperm(len(train), generator=generator).tolist()
                cursor = 0
            take = min(batch_size-len(selected), len(order)-cursor)
            selected.extend(train[i] for i in order[cursor:cursor+take])
            cursor += take
        groups = list(read_groups(replay, selected))
        for learner in learners.values():
            torch.manual_seed(args.seed+100000+update)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(args.seed+100000+update)
            learner.network.train()
            evaluated = learner._batch_losses(groups)
            weights = evaluated['total'].new_tensor([g.get('importance_weight', 1.) for g in groups])
            loss = (evaluated['total']*weights).mean()
            learner.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.network.parameters(), 10., error_if_nonfinite=True)
            learner.optimizer.step()
            learner.updates += 1
        if update % 50 == 0:
            elapsed = time.perf_counter()-training_started
            print(f'离线进度 {update}/{args.updates}：两组共 {2*update} 次更新，累计 {elapsed:.1f}s。', flush=True)
        if update % args.evaluate_every == 0 or update == args.updates:
            measure(update)
    for name, learner in learners.items():
        folder = out/name
        folder.mkdir()
        torch.save(dict(learner=learner.state_dict(), settings=settings,
            arm=name, diagnostic_only=True, additional_updates=args.updates), folder/'final.pt')
    result = dict(status='complete', additional_updates=args.updates,
        training_groups=len(train), validation_groups=len(validation), renderer_calls=0,
        probability_table_updated=False, source_snapshot_sha256=saved['source_sha256'],
        elapsed_seconds=time.perf_counter()-started, comparison=history[-1]['comparison'],
        before=history[0], after=history[-1],
        peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None,
        scope='fixed replay comparison; deployment and current-policy independent validation are not measured')
    json_write(out/'summary.json', result)
    with tarfile.open(out/'review.tar.gz', 'w:gz') as archive:
        for rel in ('settings.json', 'summary.json', 'metrics/history.jsonl'):
            archive.add(out/rel, arcname=rel)
    print(json.dumps({k: result[k] for k in ('status', 'additional_updates', 'training_groups',
        'validation_groups', 'renderer_calls', 'elapsed_seconds', 'peak_allocated_mib')}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    main(parser().parse_args())
