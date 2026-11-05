"""使用数字文件协议重算 JSCC 侧信息；仅需 CPU 和已有指标文件。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tarfile

from .digital_link import LinkConfig, file_channel_cost


def side_information_cost(row, config: LinkConfig = LinkConfig()) -> dict:
    """三路各自封装，属性按发送复符号计数；不更改传入的历史账本。"""
    payload = row['payload_complex_symbols']
    if isinstance(payload, bool) or not isinstance(payload, int) or payload < 0:
        raise ValueError('payload_complex_symbols 必须是非负整数')
    streams = {name: file_channel_cost(row[field], config) for name, field in (
        ('coordinates', 'xyz_bytes'), ('metadata', 'metadata_bytes'), ('scales', 'scale_stream_bytes'))}
    side = sum(stream['total_channel_uses'] for stream in streams.values())
    return dict(version='digital_side_file_v1', link=asdict(config), streams=streams,
                framing='three separately framed files', payload_complex_symbols=payload,
                side_channel_uses=side, total_channel_uses=payload + side,
                side_link_simulated=False, reliability='reliable-side delivery assumed',
                rate_scope='attribute complex symbols + three digital side files; shared weights preinstalled',
                arq=False, cost_unit='complex_channel_use_per_delivery')


@contextmanager
def _reader(path):
    """读取目录或 review 包，直接读取成员，不解压。"""
    path = Path(path)
    if path.is_dir():
        def read(name):
            return json.loads((path / name).read_text(encoding='utf-8'))
        yield read
    else:
        with tarfile.open(path, 'r:*') as archive:
            def read(name):
                try:
                    member = archive.extractfile(name)
                except KeyError as error:
                    raise FileNotFoundError(f'{path}:{name}') from error
                if member is None:
                    raise FileNotFoundError(f'{path}:{name}')
                return json.load(member)
            yield read


def _allocation_file(read, name):
    for prefix in ('allocation/', ''):
        try:
            return read(prefix + name)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(name)


def _fingerprint(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_report(allocation, baselines, selected='selected') -> dict:
    """baselines 为显式的（结果目录或包，method）列表，避免混用同名工作点。"""
    with _reader(allocation) as read:
        review = read('review_summary.json')
        training = _allocation_file(read, 'training.json')
        identity = _allocation_file(read, f'evaluation/{selected}/metrics.json')['identity']
    if selected not in review['rows']:
        raise ValueError(f'缺少工作点：{selected}')
    for key in ('codec_id', 'tier_sha256'):
        if key in identity and review['rows'][selected].get(key) != identity[key]:
            raise ValueError(f'所选评估与账本 {key} 不一致')
    records, common_link, dataset = [], None, None
    for source, method in baselines:
        with _reader(source) as read:
            summary = read('summary.json')
        if summary['status'] != 'complete':
            raise ValueError(f'数字基线未完成：{source}')
        plan = summary['plan']
        config = LinkConfig(**plan['link'])
        signature = (config.k, config.n, config.bits_per_symbol)
        if common_link is not None and signature != common_link:
            raise ValueError('数字基线 link 配置不一致')
        common_link = signature
        if plan.get('arq') is not False:
            raise ValueError('当前账本要求无 ARQ 的一次交付')
        if plan['scene'] != training['spec']['name']:
            raise ValueError('scene 不一致')
        evaluation = plan['evaluation']
        if evaluation['camera_names'] != identity['views']:
            raise ValueError('camera 名单或顺序不一致')
        for key in ('resolution', 'images', 'white_background'):
            if evaluation[key] != training[key]:
                raise ValueError(f'evaluation {key} 不一致')
        expected_source = training['spec'].get('source')
        if expected_source is not None and expected_source != plan['input']['source']:
            raise ValueError('照片 source 路径不一致')
        current_dataset = plan['input']['dataset_identity']
        if dataset is not None and current_dataset != dataset:
            raise ValueError('数字基线 dataset_identity 不一致')
        dataset = current_dataset
        matches = [r for r in summary['results'] if r['method'] == method and r['snr_db'] == identity['snr']]
        if len(matches) != 1:
            raise ValueError(f'需要唯一的 method/SNR 结果：{source}, {method}')
        result = matches[0]
        trials = [r for r in summary['trials'] if r['method'] == method and r['snr_db'] == identity['snr']]
        if len(trials) != result['trials'] or not trials:
            raise ValueError('基线 trial 数量不一致')
        for trial in trials:
            uses = file_channel_cost(trial['payload_bytes'], config)['total_channel_uses']
            if trial['total_channel_uses'] != uses or result['total_channel_uses'] != uses:
                raise ValueError('基线实际成本与文件协议不一致')
        records.append(dict(result, source=str(source), summary_sha256=_fingerprint(summary),
                            input=plan['input'], link=asdict(config), link_simulated=True))
    if not records:
        raise ValueError('至少指定一组数字基线')
    config = LinkConfig(**records[0]['link'])
    rows = {}
    for label, row in review['rows'].items():
        if not isinstance(row, dict):
            continue
        cost = side_information_cost(row, config)
        rows[label] = dict(comparison_accounting=cost,
                          historical_total_channel_uses=row['total_channel_uses'],
                          historical_rd_score=row.get('rd_score'),
                          **{k: row[k] for k in ('photo_psnr', 'photo_ssim', 'photo_lpips',
                                                'source_psnr', 'source_ssim', 'source_lpips') if k in row})
    for result in records:
        result['cost_saving_vs_baseline'] = 1 - rows[selected]['comparison_accounting']['total_channel_uses'] / result['total_channel_uses']
    return dict(version='unified_side_comparison_v1', allocation_source=str(allocation),
                allocation_review_sha256=_fingerprint(review), selected=selected, snr_db=identity['snr'],
                camera_names=identity['views'], allocation=rows, baselines=records,
                historical_training_and_selection_unchanged=True,
                dataset_check='baseline dataset hashes matched; JSCC photo path and evaluation settings matched')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation', type=Path, required=True)
    parser.add_argument('--baseline', nargs=2, action='append', metavar=('RUN', 'METHOD'), required=True)
    parser.add_argument('--selected', default='selected')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    for source in [args.allocation, *(Path(pair[0]) for pair in args.baseline)]:
        if source.is_dir() and args.out.resolve() == source.resolve():
            parser.error('--out 必须与输入实验目录分开')
    report = build_report(args.allocation, args.baseline, args.selected)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    table = []
    for label, row in report['allocation'].items():
        cost = row['comparison_accounting']
        table.append(dict(method='ours_' + label, total_channel_uses=cost['total_channel_uses'],
                          payload_complex_symbols=cost['payload_complex_symbols'],
                          side_channel_uses=cost['side_channel_uses'],
                          **{k: row.get(k) for k in ('photo_psnr', 'photo_ssim', 'photo_lpips')},
                          success_rate=None, quality_conditioning='reliable_side_assumption'))
    for row in report['baselines']:
        table.append(dict(method=row['method'], total_channel_uses=row['total_channel_uses'],
                          payload_complex_symbols=None, side_channel_uses=None,
                          **{k: (row['metrics'] or {}).get(k) for k in ('photo_psnr', 'photo_ssim', 'photo_lpips')},
                          success_rate=row['success_rate'], quality_conditioning=row['quality_conditioning']))
    with (args.out/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for row in table:
        print(f"{row['method']}: {row['total_channel_uses']/1e6:.6f} M complex channel uses")
    print(f'结果：{args.out}')


if __name__ == '__main__':
    main()
