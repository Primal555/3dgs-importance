"""Joint optimization of per-Gaussian categorical masks and the shared codec."""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import pad_sequence

from .allocation import (GaussianTierMask, expected_rate,
                         scene_fingerprint)
from .codec import CodecConfig, GaussianCodec
from .data import prepare, read_ply, to_features
from .transport import load_checkpoint, model_id, save_checkpoint
from .losses import (add_arguments, config_options, configure_training, reconstruction_loss,
                     objective_stats, initialize_position_head)
from .optimization import clip_codec_gradients
from .training import joint_scene_step


def save_joint(out, suffix, model, mask, fingerprint, step, record):
    save_checkpoint(out / f"codec{suffix}.pt", model, step, record)
    torch.save({"version": 1, "count": len(mask.logits), "scene_fingerprint": fingerprint,
                "snr_conditioned": mask.snr_slopes is not None,
                "state_dict": mask.state_dict(), "codec_id": model_id(model),
                "rates": list(model.cfg.rates), "step": step, "training": record},
               out / f"route2{suffix}.pt")


def load_mask(path, raw, model, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("version") != 1 or saved["scene_fingerprint"] != scene_fingerprint(raw):
        raise ValueError("route2 checkpoint does not match this PLY and row order")
    if saved["codec_id"] != model_id(model) or tuple(saved["rates"]) != model.cfg.rates:
        raise ValueError("route2 checkpoint requires its matching codec checkpoint")
    mask = GaussianTierMask(saved["count"], snr_conditioned=saved["snr_conditioned"])
    mask.load_state_dict(saved["state_dict"])
    return mask.to(device).eval()


@torch.no_grad()
def hard_tiers(mask, snr):
    device = mask.logits.device
    return torch.cat([mask.scores(torch.arange(start, min(start + 65536, len(mask.logits)), device=device), snr)
                     .argmax(-1).cpu() for start in range(0, len(mask.logits), 65536)])


def export_mask(args):
    from .cli import device_for
    device = device_for(args.device)
    raw, _ = read_ply(args.ply)
    model = load_checkpoint(args.checkpoint, device)
    mask = load_mask(args.allocation, raw, model, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(raw), 65536):
            ids = torch.arange(start, min(start + 65536, len(raw)), device=device)
            probabilities.append(mask.probabilities(ids, args.snr).cpu())
    probabilities = torch.cat(probabilities)
    tiers = probabilities.argmax(-1).numpy().astype(np.uint8)
    np.save(out / "tiers.npy", tiers)
    np.save(out / "probabilities.npy", probabilities.numpy())
    info = {"snr_db": args.snr, "original_ply_row_order": True,
            "decision": "argmax; no hard total-budget guarantee",
            "scene_fingerprint": scene_fingerprint(raw), "codec_id": model_id(model),
            "rates": list(model.cfg.rates),
            "tier_counts": np.bincount(tiers, minlength=4).tolist(),
            "expected_payload_symbols": float(expected_rate(probabilities.double(), model.cfg.rates).sum()),
            "hard_payload_symbols": int(np.asarray(model.cfg.rates)[tiers].sum())}
    (out / "allocation.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    from .plots import safe_plot
    safe_plot("allocation", out, xyz=raw[:, :3].numpy(), rates=model.cfg.rates, snr=args.snr)


def train_joint(args):
    from tqdm import trange
    from .cli import device_for, sample_tiers, seed_all

    if args.joint_steps < 1 or args.warmup_steps < 0 or args.save_every < 1:
        raise ValueError("joint-steps/save-every must be positive; warmup-steps nonnegative")
    if args.blocks_per_batch < 1 or args.profile_every < 0:
        raise ValueError("blocks-per-batch must be positive; profile-every must be nonnegative")
    for name in ("beta", "attr_weight", "seed_position_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if min(args.tau_start, args.tau_end, args.lr, args.mask_lr) <= 0:
        raise ValueError("temperatures and learning rates must be positive")
    if args.snr_range[0] > args.snr_range[1]:
        raise ValueError("snr-range must be increasing")
    if not args.attribute_only and (not args.source or not args.device.startswith("cuda")):
        raise ValueError("route2 render optimization requires --source and CUDA; --attribute-only is diagnostic")
    seed_all(args.seed)
    device = device_for(args.device)
    raw, degree = read_ply(args.ply)
    fingerprint = scene_fingerprint(raw)
    if args.codec_init:
        model = load_checkpoint(args.codec_init, device)
        if model.cfg.sh_degree != degree:
            raise ValueError("checkpoint SH degree mismatch")
    else:
        cfg = CodecConfig(sh_degree=degree, hidden=args.hidden, grid_dim=args.grid_dim,
                          depth=args.depth, levels=tuple(args.levels), rates=tuple(args.rates),
                          block_size=args.block_size, **config_options(args))
        model = GaussianCodec(cfg).to(device)
        model.attr_mean.copy_(raw[:, 3:].mean(0).to(device))
        model.attr_std.copy_(raw[:, 3:].std(0, unbiased=False).clamp_min(.01).to(device))
    configure_training(model, args)
    prior = np.load(args.existence_prior, allow_pickle=False) if args.existence_prior else None
    mask = GaussianTierMask(len(raw), prior, args.condition_snr).to(device)
    raw, geometry, order = prepare(raw, model.cfg.morton_bits, torch.arange(len(raw)))
    initialize_position_head(model, raw, geometry)
    order = order.to(device)
    starts = list(range(0, len(raw), model.cfg.block_size))
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr},
                                  {"params": mask.parameters(), "lr": args.mask_lr}])
    cameras = None
    reference = None
    if not args.attribute_only:
        # Fail early if the differentiable inactive-mask kernel is unavailable.
        import mask_diff_gaussian_rasterization  # noqa: F401
        from .rendering import load_cameras
        cameras = load_cameras(args.source, args.resolution, args.white_background, args.images, "train")
        from .rendering import RenderReference
        reference = RenderReference(raw, degree, args.white_background, args.render_target)
    render_batches = []
    if not args.attribute_only:
        storage = device if args.training_data_device == "cuda" else torch.device("cpu")
        with torch.no_grad():
            feature_blocks = [to_features(rb.to(device), geometry, model)[0].to(storage)
                              for rb in raw.split(model.cfg.block_size)]
        id_blocks = list(order.to(storage).split(model.cfg.block_size))
        for begin in range(0, len(feature_blocks), args.blocks_per_batch):
            end = begin + args.blocks_per_batch
            render_batches.append((pad_sequence(feature_blocks[begin:end], batch_first=True),
                                   pad_sequence(id_blocks[begin:end], batch_first=True, padding_value=-1)))
        del feature_blocks, id_blocks
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    record = vars(args).copy()
    record.pop("func", None)
    record["codec_config"] = model.cfg.to_dict()
    record["seed_position_weight_effective"] = 0
    record["rate_objective"] = "expected payload symbols per source Gaussian / maximum tier length"
    record["metadata_objective"] = "2-bit full tier map assumed fixed; actual zlib bytes measured only on transmit"
    record["gradient_estimator"] = "hard Gumbel forward, biased straight-through backward"
    (out / "training.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    model.train()
    mask.train()
    total_steps = args.warmup_steps + args.joint_steps
    progress = trange(total_steps, desc="Route2 joint training")
    with (out / "loss.jsonl").open("w", encoding="utf-8") as log:
        for step in progress:
            step_started = time.perf_counter()
            backward_done = False
            snr = random.uniform(*args.snr_range)
            optimizer.zero_grad(set_to_none=True)
            tau = args.tau_start * (args.tau_end / args.tau_start) ** (
                max(0, step - args.warmup_steps) / max(1, args.joint_steps - 1))
            if step < args.warmup_steps:
                start = random.choice(starts)
                rb = raw[start:start + model.cfg.block_size].to(device)
                f, xyz = to_features(rb, geometry, model)
                q = sample_tiers(len(rb), device)
                pred, seed = model(f, xyz, q, snr, args.channel, return_seed=True)
                loss, terms = reconstruction_loss(pred, f, geometry, model, return_terms=True)
                values = {"phase": "codec_warmup",
                          **{f"{key}_loss": float(value.detach().mean()) for key, value in terms.items()}}
            elif not args.attribute_only:
                from .rendering import render
                from utils.loss_utils import ssim
                camera = random.choice(cameras)
                gt = reference.get(camera, device)

                def distortion(scene, existence):
                    image = render(scene, camera, degree, args.white_background, existence)
                    return .8 * (image - gt).abs().mean() + .2 * (1 - ssim(image, gt))

                profiled = bool(args.profile_every and (step - args.warmup_steps) % args.profile_every == 0)
                loss, values = joint_scene_step(
                    model, mask, render_batches, geometry, snr, args.channel, distortion,
                    tau, args.beta, args.attr_weight, args.render_backward, profiled)
                values.update(phase="joint_render", distortion=values["render_loss"])
                backward_done = True
            else:
                auxiliary, expected, hard_counts = [], [], []
                # Explicit CPU diagnostic proxy, never used in render training.
                selected = [random.choice(starts)]
                diagnostic = []
                for start in selected:
                    rb = raw[start:start + model.cfg.block_size].to(device)
                    f, xyz = to_features(rb, geometry, model)
                    ids = order[start:start + len(rb)]
                    scores = mask.scores(ids, snr)

                    def forward(features, coordinates, logits, gamma=snr, temperature=tau):
                        choices = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
                        pred, seed, active = model.forward_tiers(features, coordinates, choices, gamma, args.channel)
                        return pred, seed, active, choices

                    pred, seed, active, choices = checkpoint(forward, f, xyz, scores,
                                                             use_reentrant=False, preserve_rng_state=True)
                    probs = scores.softmax(-1)
                    expected.append(expected_rate(probs, model.cfg.rates).sum())
                    hard_counts.append(choices.detach().sum(0))
                    auxiliary.append(reconstruction_loss(pred, f, geometry, model, active=active, reduction="sum"))
                    diagnostic.append(F.smooth_l1_loss(pred * active[:, None], f, reduction="sum") / f.shape[1])
                count = sum(int(c.sum()) for c in hard_counts)
                payload_mean = torch.stack(expected).sum() / count
                rate_loss = payload_mean / model.cfg.rates[-1]
                aux_loss = torch.stack(auxiliary).sum() / count
                distortion = torch.stack(diagnostic).sum() / count
                loss = distortion + args.attr_weight * aux_loss + args.beta * rate_loss
                counts = torch.stack(hard_counts).sum(0)
                values = {"phase": "joint_attribute_diagnostic",
                          "distortion": float(distortion.detach()), "aux_loss": float(aux_loss.detach()),
                          "expected_symbols_per_gaussian": float(payload_mean.detach()),
                          "sampled_tier_counts": counts.cpu().tolist(), "temperature": tau}
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite route2 loss")
            if not backward_done:
                loss.backward()
            codec_norm,clip_stats = clip_codec_gradients(model,args.clip_norm,args.clip_mode)
            mask_norm = torch.nn.utils.clip_grad_norm_(mask.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            values.update(step=step + 1, snr=snr, loss=float(loss.detach()),
                          codec_grad_norm=float(codec_norm), mask_grad_norm=float(mask_norm),
                          step_seconds=time.perf_counter() - step_started)
            values.update(objective_stats(values, model, 1. if step < args.warmup_steps else args.attr_weight))
            values.update(clip_stats)
            log.write(json.dumps(values) + "\n")
            log.flush()
            progress.set_postfix(loss=f"{values['loss']:.5f}", phase=values["phase"])
            if (step + 1) % args.save_every == 0:
                save_joint(out, f"_{step + 1}", model, mask, fingerprint, step + 1, record)
        save_joint(out, "", model, mask, fingerprint, total_steps, record)
    from .plots import safe_plot
    safe_plot("training", out)
    print(f"Saved matched codec.pt and route2.pt to {out}")


def add_parsers(sub):
    p = sub.add_parser("train-route2", help="joint four-way masks and Gaussian JSCC")
    p.set_defaults(func=train_joint)
    p.add_argument("--ply", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--codec-init")
    p.add_argument("--existence-prior", help="optional [N] probabilities in original PLY order")
    p.add_argument("--condition-snr", action="store_true", help="learn per-Gaussian SNR slopes as well as four logits")
    p.add_argument("--warmup-steps", type=int, default=20000)
    p.add_argument("--joint-steps", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mask-lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=.01)
    p.add_argument("--tau-start", type=float, default=1.)
    p.add_argument("--tau-end", type=float, default=.3)
    p.add_argument("--attr-weight", type=float, default=1.)
    p.add_argument("--seed-position-weight", type=float, default=.2)
    add_arguments(p)
    p.add_argument("--snr-range", type=float, nargs=2, default=[0., 20.])
    p.add_argument("--rates", type=int, nargs=4, default=[0, 8, 16, 32])
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--grid-dim", type=int, default=16)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--levels", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--block-size", type=int, default=4096)
    p.add_argument("--blocks-per-batch", type=int, default=32)
    p.add_argument("--render-backward", choices=["replay", "checkpoint"], default="replay")
    p.add_argument("--training-data-device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--profile-every", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--source")
    p.add_argument("--resolution", type=int, default=2)
    p.add_argument("--images", default="images")
    p.add_argument("--white-background", action="store_true")
    p.add_argument("--channel", choices=["none", "awgn", "rayleigh"], default="awgn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attribute-only", action="store_true", help="diagnostic proxy; not render importance training")
    p = sub.add_parser("export-route2", help="export learned probabilities and hard q in original PLY order")
    p.set_defaults(func=export_mask)
    p.add_argument("--ply", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--allocation", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--snr", type=float, default=10.)
    p.add_argument("--device", default="cuda")
