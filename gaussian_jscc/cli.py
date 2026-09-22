"""Train, transmit, receive and evaluate the learned Gaussian JSCC codec."""

import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from .data import load_tiers, prepare, read_ply, write_ply
from .transport import load_checkpoint, receive, transmit


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
    from .learned_train import add_parser as add_learned_parser
    add_learned_parser(sub)
    from .representation_train import add_parser as add_representation_parser
    add_representation_parser(sub)
    from .center_attribute_train import add_parser as add_center_attribute_parser
    add_center_attribute_parser(sub)
    from .plots import add_parser as add_plot_parser
    add_plot_parser(sub)
    from .benchmark import add_parser as add_benchmark_parser
    add_benchmark_parser(sub)
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
    for p in (send_parser, eval_parser):
        p.add_argument("--device", default="cuda")
        p.add_argument("--channel", choices=["none", "awgn", "rayleigh"], default="awgn")
        p.add_argument("--seed", type=int, default=42)
    for p in (eval_parser,):
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
    args.func(args)


if __name__ == "__main__":
    main()
