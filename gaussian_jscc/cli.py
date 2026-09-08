"""Train, transmit, receive and evaluate the Gaussian context JSCC codec."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .codec import CodecConfig, GaussianCodec
from .data import (attribute_loss, load_tiers, prepare, read_ply, to_features,
                   to_raw, write_ply)
from .transport import load_checkpoint, receive, save_checkpoint, transmit


def device_for(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device cpu for codec-only checks")
    return torch.device(name)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_tiers(n, device, drop=0.):
    # Both uniform and mixed layouts are covered with one shared codec.
    if random.random() < .5:
        q = torch.full((n,), random.randint(1, 3), dtype=torch.long, device=device)
    else:
        q = torch.randint(1, 4, (n,), device=device)
    if drop:
        q[torch.rand(n, device=device) < drop] = 0
    return q


def train(args):
    from tqdm import trange

    if args.steps < 0 or args.render_steps < 0 or args.steps + args.render_steps == 0:
        raise ValueError("request at least one training step")
    if not 0 <= args.attribute_drop < 1:
        raise ValueError("attribute-drop must be in [0,1)")
    if args.render_steps and (not args.source or not args.device.startswith("cuda")):
        raise ValueError("render training requires --source and CUDA")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    device = device_for(args.device)
    seed_all(args.seed)
    raw, degree = read_ply(args.ply)
    fixed_q = load_tiers(args.rate_map, len(raw)) if args.rate_map else None
    if fixed_q is not None and not (fixed_q > 0).any():
        raise ValueError("cannot train on an all-dropped scene")
    if args.init:
        model = load_checkpoint(args.init, device)
        if model.cfg.sh_degree != degree:
            raise ValueError("checkpoint SH degree does not match input")
    else:
        cfg = CodecConfig(sh_degree=degree, hidden=args.hidden, grid_dim=args.grid_dim,
                          depth=args.depth, planes=not args.no_planes, levels=tuple(args.levels),
                          rates=tuple(args.rates), block_size=args.block_size,
                          morton_bits=args.morton_bits)
        model = GaussianCodec(cfg).to(device)
        model.attr_mean.copy_(raw[:, 3:].mean(0).to(device))
        model.attr_std.copy_(raw[:, 3:].std(0, unbiased=False).clamp_min(.01).to(device))
    raw, geometry, fixed_q = prepare(raw, model.cfg.morton_bits, fixed_q)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    starts = list(range(0, len(raw), model.cfg.block_size))
    if fixed_q is not None:
        starts = [s for s in starts if (fixed_q[s:s + model.cfg.block_size] > 0).any()]
    cameras = None
    if args.render_steps:
        from .rendering import load_cameras
        cameras = load_cameras(args.source, args.resolution, args.white_background, args.images, "train")
    record = vars(args).copy()
    record.pop("func", None)
    (out / "training.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    progress = trange(args.steps + args.render_steps, desc="Gaussian JSCC training")
    with (out / "loss.jsonl").open("w", encoding="utf-8") as log:
        for step in progress:
            snr = random.uniform(*args.snr_range)
            optimizer.zero_grad(set_to_none=True)
            is_render = step >= args.steps
            if not is_render:
                start = random.choice(starts)
                rb = raw[start:start + model.cfg.block_size].to(device)
                qb = (fixed_q[start:start + len(rb)].to(device) if fixed_q is not None
                      else sample_tiers(len(rb), device, drop=args.attribute_drop))
                keep = qb > 0
                if not keep.any():
                    qb[0] = 1
                    keep = qb > 0
                rb, qb = rb[keep], qb[keep]
                features, unit = to_features(rb, geometry, model)
                pred, seed = model(features, unit, qb, snr, args.channel, return_seed=True)
                loss = attribute_loss(pred, features) + args.seed_position_weight * torch.nn.functional.smooth_l1_loss(
                    seed, features[:, :3])
                image_loss = None
            else:
                from .rendering import render
                from utils.loss_utils import ssim

                # Render the COMPLETE received scene. Checkpoint each spatial block
                # so its context activations are recomputed during backpropagation.
                rows, aux_losses = [], []
                total_kept = 0
                for start in starts:
                    rb = raw[start:start + model.cfg.block_size].to(device)
                    qb = (fixed_q[start:start + len(rb)].to(device) if fixed_q is not None
                          else sample_tiers(len(rb), device))
                    keep = qb > 0
                    if not keep.any():
                        continue
                    rb, qb = rb[keep], qb[keep]
                    features, unit = to_features(rb, geometry, model)
                    # Pass SNR/channel as bound defaults, not loop-captured tensors.
                    def forward(f, u, q, gamma=snr, kind=args.channel):
                        return model(f, u, q, gamma, kind, return_seed=True)
                    pred, seed = checkpoint(forward, features, unit, qb, use_reentrant=False,
                                            preserve_rng_state=True)
                    rows.append(to_raw(pred, geometry, model))
                    block_loss = attribute_loss(pred, features) + args.seed_position_weight * torch.nn.functional.smooth_l1_loss(
                        seed, features[:, :3])
                    aux_losses.append(block_loss * len(rb))
                    total_kept += len(rb)
                reconstructed = torch.cat(rows)
                camera = random.choice(cameras)
                image = render(reconstructed, camera, degree, args.white_background)
                gt = camera.original_image[:3].to(device)
                image_loss = .8 * (image - gt).abs().mean() + .2 * (1 - ssim(image, gt))
                loss = image_loss + args.attr_weight * torch.stack(aux_losses).sum() / total_kept
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss; no checkpoint saved for this step")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            values = {"step": step + 1, "phase": "render" if is_render else "attribute",
                      "loss": float(loss.detach()), "snr": snr, "grad_norm": float(norm)}
            if image_loss is not None:
                values["render_loss"] = float(image_loss.detach())
            log.write(json.dumps(values) + "\n")
            if (step + 1) % 10 == 0:
                log.flush()
                progress.set_postfix(loss=f"{values['loss']:.5f}", phase=values["phase"])
            if (step + 1) % args.save_every == 0:
                save_checkpoint(out / f"codec_{step + 1}.pt", model, step + 1, record)
        save_checkpoint(out / "codec.pt", model, args.steps + args.render_steps, record)
    print(f"Saved shared codec: {out / 'codec.pt'}")


def send(args):
    model = load_checkpoint(args.checkpoint, device_for(args.device))
    raw, degree = read_ply(args.ply)
    if degree != model.cfg.sh_degree:
        raise ValueError("SH degree mismatch")
    if args.allocation:
        from .route2 import load_mask, hard_tiers
        mask = load_mask(args.allocation, raw, model, device_for(args.device))
        q = hard_tiers(mask, args.snr)
    else:
        q = load_tiers(args.rate_map, len(raw), args.uniform_tier)
    stats = transmit(model, raw, q, args.snr, args.channel, args.seed, args.out,
                     args.metadata_code_rate, args.metadata_modulation_bits)
    print(json.dumps(stats, indent=2))


def decode(args):
    model = load_checkpoint(args.checkpoint, device_for(args.device))
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    raw = receive(model, args.packet)
    write_ply(args.out, raw, model.cfg.sh_degree)
    print(f"Decoded {len(raw)} Gaussians to {args.out}")


def evaluate(args):
    model = load_checkpoint(args.checkpoint, device_for(args.device))
    raw, degree = read_ply(args.ply)
    if degree != model.cfg.sh_degree:
        raise ValueError("SH degree mismatch")
    if args.source and not args.device.startswith("cuda"):
        raise ValueError("render evaluation requires CUDA")
    if (args.rate_map or args.allocation) and args.tiers != [1, 2, 3]:
        raise ValueError("--rate-map/--allocation uses its own tiers; omit --tiers")
    learned = None
    if args.allocation:
        from .route2 import load_mask, hard_tiers
        learned = load_mask(args.allocation, raw, model, device_for(args.device))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cameras = None
    reference = None
    if args.source:
        from .rendering import load_cameras
        cameras = load_cameras(args.source, args.resolution, args.white_background, args.images, "test")
        reference = raw.to(args.device)
    results = []
    options = [None] if args.rate_map or learned is not None else args.tiers
    for tier in options:
        for snr in args.snrs:
            q = hard_tiers(learned, snr) if learned is not None else load_tiers(args.rate_map, len(raw), tier)
            ordered, _, oq = prepare(raw, model.cfg.morton_bits, q)
            target = ordered[oq > 0]
            for trial in range(args.trials):
                label = f"{'map' if tier is None else 'tier' + str(tier)}_snr{snr:g}_trial{trial}"
                run = out / label
                stats = transmit(model, raw, q, snr, args.channel, args.seed + trial, run,
                                 args.metadata_code_rate, args.metadata_modulation_bits)
                recovered = receive(model, run)
                stats["label"] = label
                stats["position_rmse"] = (float((recovered[:, :3] - target[:, :3]).square().mean().sqrt())
                                          if len(target) else None)
                stats["attribute_mse"] = (float((recovered[:, 3:] - target[:, 3:]).square().mean())
                                          if len(target) else None)
                if args.save_ply:
                    write_ply(run / "point_cloud.ply", recovered, degree)
                if cameras:
                    from .rendering import evaluate_views
                    metrics = evaluate_views(recovered.to(args.device), reference, cameras, degree,
                                             args.white_background, run / "views" if args.save_images else None,
                                             args.lpips)
                    (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                    stats.update(metrics["mean"])
                results.append(stats)
                (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
                print(json.dumps(stats), flush=True)
    from .plots import safe_plot
    safe_plot("evaluation", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    from .route2 import add_parsers
    add_parsers(sub)
    from .plots import add_parser as add_plot_parser
    add_plot_parser(sub)
    from .benchmark import add_parser as add_benchmark_parser
    add_benchmark_parser(sub)
    p = sub.add_parser("train")
    p.set_defaults(func=train)
    p.add_argument("--ply", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--init", help="initialize from shared codec; architecture/statistics stay fixed")
    p.add_argument("--steps", type=int, default=20000, help="attribute training steps")
    p.add_argument("--render-steps", type=int, default=0, help="subsequent full-scene rendering steps")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--attr-weight", type=float, default=.1)
    p.add_argument("--snr-range", type=float, nargs=2, default=[0., 20.])
    p.add_argument("--rates", type=int, nargs=4, default=[0, 8, 16, 32])
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--grid-dim", type=int, default=16)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--levels", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--no-planes", action="store_true")
    p.add_argument("--block-size", type=int, default=4096)
    p.add_argument("--morton-bits", type=int, default=16,
                   help="sender-only spatial sorting precision; coordinates are not metadata")
    p.add_argument("--seed-position-weight", type=float, default=.2,
                   help="auxiliary loss for decoder position bootstrap")
    p.add_argument("--attribute-drop", type=float, default=.05,
                   help="random context dropout during attribute warmup; use 0 for strict codec isolation")
    p.add_argument("--rate-map")
    train_parser = p
    p = sub.add_parser("transmit")
    p.set_defaults(func=send)
    p.add_argument("--ply", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="new receiver packet directory")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--rate-map")
    group.add_argument("--allocation", help="matching route2.pt; choose hard q at this SNR")
    group.add_argument("--uniform-tier", type=int, choices=[1, 2, 3])
    p.add_argument("--snr", type=float, default=10)
    send_parser = p
    p = sub.add_parser("decode")
    p.set_defaults(func=decode)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--packet", required=True)
    p.add_argument("--out", required=True, help="new output PLY")
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("evaluate")
    p.set_defaults(func=evaluate)
    p.add_argument("--ply", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--rate-map")
    group.add_argument("--allocation", help="matching route2.pt; recompute hard q for every SNR")
    p.add_argument("--tiers", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3])
    p.add_argument("--snrs", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--save-ply", action="store_true")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--lpips", action="store_true")
    eval_parser = p
    for p in (train_parser, send_parser, eval_parser):
        p.add_argument("--device", default="cuda")
        p.add_argument("--channel", choices=["none", "awgn", "rayleigh"], default="awgn")
        p.add_argument("--seed", type=int, default=42)
    for p in (train_parser, eval_parser):
        p.add_argument("--source", help="COLMAP/Blender source images and cameras")
        p.add_argument("--resolution", type=int, default=2)
        p.add_argument("--images", default="images")
        p.add_argument("--white-background", action="store_true")
    for p in (send_parser, eval_parser):
        p.add_argument("--metadata-code-rate", type=float,
                       help="fixed FEC rate for overhead accounting; omitted = ideal AWGN capacity")
        p.add_argument("--metadata-modulation-bits", type=int, default=2)
    args = parser.parse_args()
    if getattr(args, "save_every", 1) < 1 or getattr(args, "trials", 1) < 1:
        parser.error("save-every and trials must be positive")
    if hasattr(args, "snr_range") and args.snr_range[0] > args.snr_range[1]:
        parser.error("snr-range must be increasing")
    if getattr(args, "seed_position_weight", 0) < 0:
        parser.error("seed-position-weight must be nonnegative")
    args.func(args)


if __name__ == "__main__":
    main()
