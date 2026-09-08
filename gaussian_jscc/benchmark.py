"""Isolated Gaussian JSCC codec benchmark with no pruning or learned allocation."""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from .data import prepare, read_ply, write_ply
from .transport import load_checkpoint, receive, transmit


def parameter_metrics(source, recovered):
    """Group-aware errors for matched, identically ordered Gaussian rows."""
    if source.shape != recovered.shape or source.ndim != 2 or source.shape[1] < 14:
        raise ValueError("source and recovered Gaussians must have the same trained PLY layout")
    difference = recovered - source
    span = source[:, :3].amax(0) - source[:, :3].amin(0)
    diagonal = float(span.square().sum().sqrt().clamp_min(1e-12))
    source_q = F.normalize(source[:, 7:11], dim=-1)
    recovered_q = F.normalize(recovered[:, 7:11], dim=-1)
    cosine = (source_q * recovered_q).sum(-1).abs().clamp(0, 1)
    angles = 2 * cosine.acos() * (180 / math.pi)
    result = {
        "position_rmse": float(difference[:, :3].square().mean().sqrt()),
        "position_nrmse_bbox_diagonal": float(difference[:, :3].square().mean().sqrt()) / diagonal,
        "opacity_alpha_mae": float((source[:, 3].sigmoid() - recovered[:, 3].sigmoid()).abs().mean()),
        "log_scale_rmse": float(difference[:, 4:7].square().mean().sqrt()),
        "rotation_angle_mean_deg": float(angles.mean()),
        "rotation_angle_p95_deg": float(torch.quantile(angles, .95)),
        "dc_rmse": float(difference[:, 11:14].square().mean().sqrt()),
        "all_parameter_rmse": float(difference.square().mean().sqrt()),
    }
    result["sh_rest_rmse"] = (float(difference[:, 14:].square().mean().sqrt())
                               if source.shape[1] > 14 else None)
    return result


def benchmark_codec(args):
    from .cli import device_for
    from .plots import safe_plot

    device = device_for(args.device)
    model = load_checkpoint(args.checkpoint, device)
    raw, degree = read_ply(args.ply)
    if degree != model.cfg.sh_degree:
        raise ValueError("checkpoint SH degree mismatch")
    if args.source and not args.device.startswith("cuda"):
        raise ValueError("render benchmark requires CUDA; omit --source for parameter-only testing")
    if (args.save_images or args.lpips) and not args.source:
        raise ValueError("--save-images/--lpips require --source")
    if len(set(args.channels)) != len(args.channels):
        raise ValueError("channels must be unique")
    if any(tier not in (1, 2, 3) for tier in args.tiers):
        raise ValueError("codec isolation supports only positive tiers 1, 2 and 3")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.pop("func", None)
    config.update(scope="all source Gaussians retained; no route2 allocation",
                  interpretation={"none": "codec bottleneck/reconstruction loss",
                                  "noisy": "codec bottleneck plus channel loss"})
    (out / "benchmark_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8")

    ordered, _, _ = prepare(raw, model.cfg.morton_bits)
    cameras = None
    reference = None
    if args.source:
        from .rendering import load_cameras
        cameras = load_cameras(args.source, args.resolution, args.white_background, args.images, "test")
        reference = raw.to(device)

    results = []
    total = sum((1 if kind == "none" else args.trials) * len(args.tiers) * len(args.snrs)
                for kind in args.channels)
    current = 0
    for channel_kind in args.channels:
        for tier in args.tiers:
            q = torch.full((len(raw),), tier, dtype=torch.long)
            for snr in args.snrs:
                trial_count = 1 if channel_kind == "none" else args.trials
                for trial in range(trial_count):
                    current += 1
                    label = f"{channel_kind}_tier{tier}_snr{snr:g}_trial{trial}"
                    run = out / label
                    run.mkdir()
                    if args.keep_packets:
                        packet = run / "packet"
                        temporary = None
                    else:
                        temporary = TemporaryDirectory(prefix="packet_", dir=out)
                        packet = Path(temporary.name) / "packet"
                    try:
                        stats = transmit(model, raw, q, snr, channel_kind, args.seed + trial,
                                         packet, args.metadata_code_rate,
                                         args.metadata_modulation_bits)
                        recovered = receive(model, packet)
                    finally:
                        if temporary is not None:
                            temporary.cleanup()
                    if len(recovered) != len(ordered) or not torch.isfinite(recovered).all():
                        raise RuntimeError("codec benchmark recovered an invalid Gaussian set")
                    stats.update(parameter_metrics(ordered, recovered))
                    stats.update(label=label, benchmark_series=f"{channel_kind}_tier{tier}",
                                 tier=tier, trial=trial,
                                 expected_payload_complex_symbols=len(raw) * model.cfg.rates[tier])
                    if stats["payload_complex_symbols"] != stats["expected_payload_complex_symbols"]:
                        raise RuntimeError("payload length does not match the configured uniform tier")
                    if args.save_ply:
                        write_ply(run / "point_cloud.ply", recovered, degree)
                    if cameras:
                        from .rendering import evaluate_views
                        metrics = evaluate_views(recovered.to(device), reference, cameras, degree,
                                                 args.white_background,
                                                 run / "views" if args.save_images else None,
                                                 args.lpips)
                        (run / "metrics.json").write_text(
                            json.dumps(metrics, indent=2), encoding="utf-8")
                        stats.update(metrics["mean"])
                    (run / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
                    results.append(stats)
                    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
                    print(f"[{current}/{total}] {label}: position_nrmse="
                          f"{stats['position_nrmse_bbox_diagonal']:.6g}, "
                          f"symbols/G={stats['total_uses_per_source_gaussian']:.4g}", flush=True)
    safe_plot("evaluation", out)
    print(f"Saved isolated codec benchmark to {out}")


def add_parser(sub):
    parser = sub.add_parser("benchmark-codec", help="measure codec-only loss with all Gaussians retained")
    parser.set_defaults(func=benchmark_codec)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tiers", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument("--snrs", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    parser.add_argument("--channels", nargs="+", choices=["none", "awgn", "rayleigh"],
                        default=["none", "awgn"])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", help="COLMAP/Blender scene for render-domain codec loss")
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--images", default="images")
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--save-ply", action="store_true")
    parser.add_argument("--keep-packets", action="store_true",
                        help="retain metadata.bin/received.npy for every run (large)")
    parser.add_argument("--metadata-code-rate", type=float)
    parser.add_argument("--metadata-modulation-bits", type=int, default=2)
