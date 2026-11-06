"""Read-only paired final-weight inference on the original fixed holdout shards."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
import time

import torch

from .benefit_stage_one_diagnostic_metrics import compare_record, summarize_records
from .benefit_stage_one_report import summarize_predictions
from .budgeted_allocation_state import load_checkpoint
from .modular_experiment import json_write
from .spatial_benefit import SpatialBenefitNetwork, pack_inputs
from .spatial_benefit_replay import SpatialReplay


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='completed benefit_stage_one comparison directory')
    p.add_argument('--out', required=True, help='new separate diagnostic directory')
    p.add_argument('--replay-root', help='relocated unchanged validation shards')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-groups', type=int, default=8)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--replay-cache-mib', type=float, default=256.)
    return p


def read_network(path, device):
    payload = path.read_bytes()
    state = load_checkpoint(io.BytesIO(payload))
    weights = state['learner']['network']
    net = SpatialBenefitNetwork(hidden=weights['points.0.weight'].shape[0], dropout=0.).to(device)
    net.load_state_dict(weights, strict=True)
    net.eval().requires_grad_(False)
    return net, state, hashlib.sha256(payload).hexdigest()


def reconcile(actual, expected, prefix=''):
    """Compare all report leaves; counts and metadata are exact, floats tolerant."""
    if isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys():
        return {k: v for key in actual for k, v in
                reconcile(actual[key], expected[key], f'{prefix}.{key}').items()}
    if isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        return {k: v for i, (a, b) in enumerate(zip(actual, expected)) for k, v in
                reconcile(a, b, f'{prefix}[{i}]').items()}
    if isinstance(actual, float) and isinstance(expected, (int, float)):
        matched = math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-12)
    else:
        matched = actual == expected
    return {prefix: dict(actual=actual, expected=expected, matched=matched)}


@torch.inference_mode()
def main(args):
    if min(args.batch_groups, args.cpu_threads, args.top_k) < 1:
        raise ValueError('positive batch, threads and top_k required')
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    manifest = json.loads((run/'data_manifest.json').read_text(encoding='utf8'))
    root = Path(args.replay_root or manifest['replay_root']).resolve()
    if out.is_relative_to(run) or run.is_relative_to(out) or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('diagnostic output must be separate from source and replay')
    if out.exists():
        raise FileExistsError('use a new diagnostic output directory')
    if not root.is_dir():
        raise FileNotFoundError(f'replay directory missing: {root}; supply --replay-root after relocation')
    entries = manifest['validation']
    if not entries or len({e['id'] for e in entries}) != len(entries):
        raise ValueError('nonempty unique fixed validation entries required')
    for e in entries:
        path = (root/e['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f'validation shard missing or outside replay: {path}')
    summary = json.loads((run/'summary.json').read_text(encoding='utf8'))
    if summary['status'] != 'complete':
        raise ValueError('completed paired comparison required')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu')
    torch.set_num_threads(args.cpu_threads)
    started = time.perf_counter()
    networks, hashes = {}, {}
    for arm in ('original', 'repaired'):
        net, state, digest = read_network(run/arm/'final.pt', device)
        if (state['arm'] != arm or state['settings']['source_checkpoint_sha256'] != summary['source_snapshot_sha256']
                or state['additional_updates'] != summary['additional_updates']):
            raise ValueError(f'paired checkpoint identity mismatch: {arm}')
        networks[arm], hashes[arm] = net, digest
    replay = SpatialReplay(root, max_groups=1, validation_groups=len(entries), cache_mib=args.replay_cache_mib)
    rows, groups, predictions = [], [], {k: [] for k in networks}
    print(f'固定留出记录={len(entries)}；两组末步权重；设备={device}。', flush=True)
    for start in range(0, len(entries), args.batch_groups):
        batch_entries = entries[start:start+args.batch_groups]
        batch = [replay._read(e) for e in batch_entries]
        if any(g.get('supervision') == 'action_gap' or not g.get('validation') for g in batch):
            raise ValueError('fixed independent complete validation records required')
        counts = [len(g['views']) for g in batch]
        inputs = pack_inputs([v['inputs'] for g in batch for v in g['views']], device)
        output = {name: list(net(*inputs).cpu().double().split(counts)) for name, net in networks.items()}
        for i, (entry, g) in enumerate(zip(batch_entries, batch)):
            rows.append(compare_record(entry, g, output['original'][i], output['repaired'][i]))
            groups.append(dict(rates=g['rates'], views=[dict(values=v['values']) for v in g['views']]))
            for arm in networks:
                predictions[arm].append(output[arm][i])
        if start == 0 or start+len(batch) == len(entries) or (start+len(batch)) % 64 == 0:
            print(f'逐样本对照 {start+len(batch)}/{len(entries)}。', flush=True)
    result = summarize_records(rows, args.top_k)
    reports = {arm: summarize_predictions(groups, predictions[arm]) for arm in networks}
    checks = {}
    for arm, report in reports.items():
        recorded = summary['after'][arm]['fixed_validation']
        checks.update(reconcile(report, recorded, arm))
    matched = all(c['matched'] for c in checks.values())
    result.update(status='complete' if matched else 'reconciliation_mismatch',
        reconciliation=dict(matched=matched, checks=checks), aggregate=reports,
        source_run=str(run), replay_root=str(root), checkpoint_sha256=hashes,
        validation_manifest_sha256=hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
        optimizer_updates=0, renderer_calls=0, device=str(device), elapsed_seconds=time.perf_counter()-started,
        scope='fixed early-context holdout; auxiliary q0 is not the active dense deletion branch')
    out.mkdir(parents=True)
    with (out/'records.jsonl').open('w', encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    json_write(out/'summary.json', result)
    with tarfile.open(out/'review.tar.gz', 'w:gz') as archive:
        for rel in ('summary.json', 'records.jsonl'):
            archive.add(out/rel, arcname=rel)
    pair = result['paired']
    print(f'原指标复核一致={matched}；改选={pair["changed_records"]}/{len(rows)}。', flush=True)
    print(f'有害改选={pair["harmful_switches"]}，损失合计={pair["harm_total"]:.8g}；'
          f'有益改选={pair["beneficial_switches"]}，收益合计={pair["gain_total"]:.8g}。', flush=True)
    print(f'平均选档代价变化={pair["mean_regret_change"]:+.8g}；'
          f'最严重10例占新增损失={pair["top10_harm_fraction"]:.2%}。', flush=True)
    print(f'实际改选分量均值［MSE，结构，通信］：{pair["mean_switch_components"]}', flush=True)
    print(f'下载：{out / "review.tar.gz"}', flush=True)
    if not matched:
        raise RuntimeError('fixed-holdout reconciliation mismatch; inspect saved summary before interpretation')
    return result


if __name__ == '__main__':
    main(parser().parse_args())
