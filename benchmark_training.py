"""Measure full-scene codec training time/memory without updating any weights."""

import argparse
import json
from pathlib import Path
import random
import statistics
import time

import torch
from torch.nn.utils.rnn import pad_sequence

from gaussian_jscc.cli import device_for, sample_tiers, seed_all
from gaussian_jscc.data import prepare, read_ply, to_features
from gaussian_jscc.training import full_scene_step
from gaussian_jscc.transport import load_checkpoint
from gaussian_jscc.optimization import clip_codec_gradients
from gaussian_jscc.learned_objective import projection_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True, help="new timing output directory")
    parser.add_argument("--blocks-per-batch", type=int, nargs="+", default=[32])
    parser.add_argument("--backward", nargs="+", choices=["direct", "replay", "checkpoint"], default=["direct"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--snr", type=float, default=10.)
    parser.add_argument("--channel", choices=["none", "awgn", "rayleigh"], default="awgn")
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--images", default="images")
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--training-data-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-target", choices=["source", "images"], default="source")
    parser.add_argument('--attr-weight', type=float, default=1.)
    parser.add_argument('--projection-weight', type=float, default=.05)
    parser.add_argument('--clip-mode', choices=['none','global','branch'], default='none')
    parser.add_argument('--clip-norm', type=float, default=10.)
    args = parser.parse_args()
    if min(args.blocks_per_batch) < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("positive batch sizes/iterations and nonnegative warmup required")
    device = device_for(args.device)
    if device.type != "cuda":
        parser.error("full-scene render timing requires CUDA; CPU unit tests only validate gradients")
    from gaussian_jscc.rendering import load_cameras, render, RenderReference
    from utils.loss_utils import ssim

    seed_all(args.seed)
    model = load_checkpoint(args.checkpoint, device).train()
    raw, degree = read_ply(args.ply)
    if degree != model.cfg.sh_degree:
        raise ValueError("checkpoint SH degree mismatch")
    raw, geometry, _ = prepare(raw, model.cfg.morton_bits)
    source_parameters = raw.to(device)
    storage = device if args.training_data_device == "cuda" else torch.device("cpu")
    with torch.no_grad():
        blocks = [to_features(rb.to(device), geometry, model)[0].to(storage)
                  for rb in raw.split(model.cfg.block_size)]
    cameras = load_cameras(args.source, args.resolution, args.white_background, args.images, "train")
    reference = RenderReference(raw, degree, args.white_background, args.render_target)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update({"torch_version": str(torch.__version__),
                   "gpu": torch.cuda.get_device_name(device), "gaussians": len(raw),
                   "codec_block_size": model.cfg.block_size,
                   "codec_config": model.cfg.to_dict(),
                   "note": "No optimizer step. Teacher-cache generation excluded from step timing; not a pre-change baseline."})
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    results, summaries = [], []
    with (out / "timings.jsonl").open("w", encoding="utf-8") as log:
        for mode in args.backward:
            for size in args.blocks_per_batch:
                # Same camera sequence and same per-Gaussian q across configurations.
                groups = [pad_sequence(blocks[i:i + size], batch_first=True)
                          for i in range(0, len(blocks), size)]
                measured = []
                torch.cuda.empty_cache()
                for iteration in range(args.warmup + args.iterations):
                    seed_all(args.seed + iteration)
                    camera = random.choice(cameras)
                    qs = [sample_tiers(len(b), torch.device("cpu")).to(storage) for b in blocks]
                    batches = [(f, pad_sequence(qs[i * size:(i + 1) * size], batch_first=True))
                               for i, f in enumerate(groups)]
                    gt = reference.get(camera, device)

                    def distortion(scene):
                        image = render(scene, camera, degree, args.white_background)
                        return (.8 * (image - gt).abs().mean() + .2 * (1 - ssim(image, gt))
                                + args.projection_weight * projection_loss(scene, source_parameters, camera))

                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    loss, stats = full_scene_step(model, batches, geometry, args.snr, args.channel,
                                                  distortion, attr_weight=args.attr_weight, mode=mode, profile=True)
                    norm, _ = clip_codec_gradients(model, args.clip_norm, args.clip_mode)
                    torch.cuda.synchronize(device)
                    row = {"backward": mode, "blocks_per_batch": size, "iteration": iteration,
                           "warmup": iteration < args.warmup, "camera": camera.image_name,
                           "total_seconds": time.perf_counter() - started,
                           "loss": float(loss), "grad_norm": float(norm), **stats}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    results.append(row)
                    if not row["warmup"]:
                        measured.append(row)
                    print(f"{mode}, blocks={size}, step={iteration + 1}: "
                          f"{row['total_seconds']:.2f}s, "
                          f"peak={row['peak_allocated_mib']:.0f} MiB" +
                          (" (warmup)" if row["warmup"] else ""), flush=True)
                summary = {"backward": mode, "blocks_per_batch": size,
                           "median_seconds": statistics.median(r["total_seconds"] for r in measured),
                           "max_peak_allocated_mib": max(r["peak_allocated_mib"] for r in measured)}
                summaries.append(summary)
                (out / "results.json").write_text(json.dumps({"summary": summaries, "steps": results}, indent=2),
                                                  encoding="utf-8")
                del batches, groups
    print(f"Saved timings to {out / 'results.json'}; codec weights were not changed.")


if __name__ == "__main__":
    main()
