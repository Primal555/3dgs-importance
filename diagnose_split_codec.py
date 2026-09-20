"""Random-start fixed-block fitting check; not a scene/render benchmark.

No old checkpoint is used. Train on alternating sampled complete blocks and
report BOTH fitted blocks and held-out blocks, with none/10 dB AWGN and all
positive tiers plus mixed tiers. Loss and rates match the bootstrap experiment.
This small diagnostic does not automatically approve a full scene training run.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.data import read_ply, prepare, to_features, fit_feature_statistics
from gaussian_jscc.optimization import clip_codec_gradients, preserved_rng, update_stats
from gaussian_jscc.spatial_response import spatial_response_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ply', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--architecture', choices=['learned_split', 'learned_joint', 'learned_split_logcov'], default='learned_split')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--sample-blocks', type=int, default=4)
    p.add_argument('--block-size', type=int, default=256)
    p.add_argument('--hidden', type=int, default=96)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--window', type=int, default=32)
    p.add_argument('--train-channel', choices=['none', 'awgn'], default='none')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--validate-every', type=int, default=100)
    args = p.parse_args()
    if args.steps < 1 or args.sample_blocks < 4 or args.validate_every < 1:
        p.error('steps/validate-every > 0 and sample-blocks >= 4 required')
    if not 0 < args.lr < float('inf'):
        p.error('positive finite lr required')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    raw, degree = read_ply(args.ply)
    m = GaussianCodec(CodecConfig(architecture=args.architecture, sh_degree=degree,
                                 hidden=args.hidden, depth=args.depth, block_size=args.block_size,
                                 decoder_window=args.window)).to(device)
    fit_feature_statistics(raw, m)
    raw, geometry, _ = prepare(raw, m.cfg.morton_bits)
    count = len(raw) // args.block_size
    if count < args.sample_blocks:
        p.error('not enough complete blocks in source PLY')
    indices = torch.linspace(0, count-1, args.sample_blocks).long()
    source = torch.stack([raw[i*args.block_size:(i+1)*args.block_size] for i in indices]).to(device)
    features, _ = to_features(source.flatten(0, 1), geometry, m)
    features = features.reshape(args.sample_blocks, args.block_size, -1)
    fitted, heldout = features[::2], features[1::2]
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(vars(args), initialization='random weights; PLY feature statistics',
                    objective=('spatial_logcov_v1' if args.architecture == 'learned_split_logcov' else 'spatial_response_v3')+'; fine_weight=1; no clipping', snr=10,
                    parameters=sum(v.numel() for v in m.parameters()),
                    sample_indices=indices.tolist(), training_entries='even', heldout_entries='odd',
                    scope='fixed-block diagnostic only; no camera/render PSNR; no XYZ side stream')
    (out/'config.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def layout(f, label):
        if label == 'mixed':
            return (torch.arange(f.shape[1], device=device) % 3 + 1)[None].expand(f.shape[:2])
        return torch.full(f.shape[:2], int(label), device=device, dtype=torch.long)

    def loss_for(f, q, channel):
        pred = m.forward_tier_batches(f, f[..., :3], F.one_hot(q, 4).to(f), 10., channel)[0]
        return spatial_response_loss(pred.flatten(0, 1), f.flatten(0, 1), geometry, m,
                                     directions=torch.eye(3, device=device), fine_weight=1.)

    def append(name, record):
        with (out/name).open('a', encoding='utf-8') as h:
            h.write(json.dumps(record, allow_nan=False)+'\n')

    @torch.no_grad()
    def evaluate(step):
        m.eval()
        rows = []
        with preserved_rng(device):
            for name, f in [('fitted', fitted), ('heldout', heldout)]:
                for channel in ('none', 'awgn'):
                    for label in ('1', '2', '3', 'mixed'):
                        torch.manual_seed(args.seed + 100)
                        loss, stats = loss_for(f, layout(f, label), channel)
                        row = dict(step=step, split=name, channel=channel, layout=label,
                                   loss=float(loss), **stats)
                        rows.append(row)
                        append('validation.jsonl', row)
                        if label == '3':
                            print(f'{step} {name} {channel} q3: XYZ RMSE={stats["xyz_rmse_world"]:.5f}, '
                                  f'shape={stats["spatial_shape_objective"]:.5f}', flush=True)
        m.train()
        return rows

    before = evaluate(0)
    optimizer = torch.optim.Adam(m.parameters(), lr=args.lr)
    start = time.perf_counter()
    for step in range(1, args.steps+1):
        optimizer.zero_grad(set_to_none=True)
        label = ('1', '2', '3', 'mixed')[(step-1) % 4]
        loss, stats = loss_for(fitted, layout(fitted, label), args.train_channel)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite objective; optimizer not stepped')
        loss.backward()
        norm, gradient_stats = clip_codec_gradients(m, mode='none')
        old = {n: v.detach().clone() for n, v in m.named_parameters()}
        optimizer.step()
        append('loss.jsonl', dict(step=step, loss=float(loss.detach()), grad_norm=float(norm),
                                 layout=label, **stats, **gradient_stats, updates=update_stats(m, old)))
        if step % 25 == 0 or step == 1:
            print(f'{step}/{args.steps}: loss={float(loss.detach()):.5f}, grad={float(norm):.3f}', flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            after = evaluate(step)
    (out/'summary.json').write_text(json.dumps(dict(before=before, after=after,
        seconds=time.perf_counter()-start, scope=manifest['scope']), indent=2, allow_nan=False), encoding='utf-8')
    print(f'Saved diagnostic logs: {out}; no checkpoint loaded or saved.', flush=True)


if __name__ == '__main__':
    main()
