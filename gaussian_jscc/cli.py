"""Train, transmit, receive and evaluate the Gaussian context JSCC codec."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .codec import CodecConfig, GaussianCodec
from .data import (load_tiers, prepare, read_ply, to_features,
                   to_raw, write_ply)
from .transport import load_checkpoint, receive, save_checkpoint, transmit
from .training import full_scene_step
from .losses import (add_arguments, config_options, configure_training, reconstruction_loss,
                     objective_stats, initialize_position_head, position_training_inputs)
from .optimization import all_tier_attribute_step, clip_codec_gradients, preserved_rng


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

    if args.fixed_snr is not None:
        if not math.isfinite(args.fixed_snr):
            raise ValueError('fixed-snr must be finite')
        args.snr_range = [args.fixed_snr, args.fixed_snr]
        args.position_eval_snrs = [args.fixed_snr]
    if not all(math.isfinite(x) for x in args.snr_range) or args.snr_range[0] > args.snr_range[1]:
        raise ValueError('snr-range must be finite and ordered')
    if args.geometry_only and (args.render_steps or args.rate_map or args.attribute_drop or args.tier_training != 'all'):
        raise ValueError('geometry-only requires render-steps 0, tier-training all, attribute-drop 0, no rate-map')
    if args.geometry_clean_weight is not None and (not math.isfinite(args.geometry_clean_weight) or args.geometry_clean_weight < 0):
        raise ValueError('geometry-clean-weight must be finite and nonnegative')
    if args.position_patience < 0 or args.position_min_delta < 0 or not math.isfinite(args.position_min_delta):
        raise ValueError('invalid position stopping settings')
    if args.position_patience and (not args.geometry_only or not args.position_eval_every or args.fixed_snr is None or args.channel != 'awgn'):
        raise ValueError('position-patience requires geometry-only, AWGN, fixed-snr and position evaluation')
    if args.steps < 0 or args.render_steps < 0 or args.steps + args.render_steps == 0:
        raise ValueError("request at least one training step")
    if not 0 <= args.attribute_drop < 1:
        raise ValueError("attribute-drop must be in [0,1)")
    if args.render_steps and (not args.source or not args.device.startswith("cuda")):
        raise ValueError("render training requires --source and CUDA")
    if args.blocks_per_batch < 1 or args.profile_every < 0:
        raise ValueError("blocks-per-batch must be positive; profile-every must be nonnegative")
    if args.tier_training == "all" and (args.rate_map or args.attribute_drop):
        raise ValueError("all-tier training requires --attribute-drop 0 and no --rate-map")
    if args.position_eval_every < 0 or args.position_eval_blocks < 1:
        raise ValueError("invalid fixed position evaluation settings")
    if not all(math.isfinite(v) for v in args.position_eval_snrs):
        raise ValueError("position-eval-snrs must be finite")
    if args.render_lr is None:
        args.render_lr = args.lr * .25
    if not all(math.isfinite(x) and x > 0 for x in (args.lr, args.render_lr)):
        raise ValueError("lr and render-lr must be finite and positive")
    if not math.isfinite(args.attr_weight) or args.attr_weight < 0:
        raise ValueError("attr-weight must be finite and nonnegative")
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
                          morton_bits=args.morton_bits, **config_options(args))
        model = GaussianCodec(cfg).to(device)
        model.attr_mean.copy_(raw[:, 3:].mean(0).to(device))
        model.attr_std.copy_(raw[:, 3:].std(0, unbiased=False).clamp_min(.01).to(device))
    configure_training(model, args)
    if args.geometry_clean_weight is None:
        args.geometry_clean_weight = (model.cfg.geometry_clean_weight if model.cfg.position_head=='reference_v6'
                                      else 1. if model.cfg.position_head == 'block_pilot_v5' else 0.)
    if args.geometry_only:
        if model.cfg.position_head not in ('block_relative_v4', 'block_pilot_v5', 'reference_v6'):
            raise ValueError('geometry-only requires a block geometry position head')
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('block_geometry.'))
        print('Geometry-only: all previous codec parameters frozen; train new geometric payload at fixed budgets.')
    raw, geometry, fixed_q = prepare(raw, model.cfg.morton_bits, fixed_q)
    initialize_position_head(model, raw, geometry)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    model.train()
    starts = list(range(0, len(raw), model.cfg.block_size))
    if fixed_q is not None:
        starts = [s for s in starts if (fixed_q[s:s + model.cfg.block_size] > 0).any()]
    # Frozen source attributes/statistics: normalize once, not on every step.
    # Only source inputs are cached; learned context and decoded xyz are recomputed.
    cache_device = device if args.training_data_device == "cuda" else torch.device("cpu")
    feature_blocks, tier_blocks = [], []
    with torch.no_grad():
        for start in starts:
            rb = raw[start:start + model.cfg.block_size].to(device)
            f, _ = to_features(rb, geometry, model)
            feature_blocks.append(f.to(cache_device))
            tier_blocks.append(None if fixed_q is None else
                               fixed_q[start:start + len(rb)].to(cache_device))
    # A batch never changes packet boundaries. The short final block is padded
    # with q0 slots which are excluded from context, power, losses and rendering.
    render_groups = [pad_sequence(feature_blocks[i:i + args.blocks_per_batch], batch_first=True)
                     for i in range(0, len(feature_blocks), args.blocks_per_batch)] if args.render_steps else []
    if args.init:
        print("Loaded codec weights; Adam state and step schedule start fresh (--init is not exact resume).")
    print(f"Training cache: {cache_device}; render blocks/batch: {args.blocks_per_batch}; "
          f"backward: {args.render_backward}")
    cameras = None
    reference = None
    if args.render_steps:
        from .rendering import load_cameras
        cameras = load_cameras(args.source, args.resolution, args.white_background, args.images, "train")
        from .rendering import RenderReference
        reference = RenderReference(raw, degree, args.white_background, args.render_target)
    record = vars(args).copy()
    record.pop("func", None)
    record["codec_config"] = model.cfg.to_dict()
    record["seed_position_weight_effective"] = 0  # no seed stage in geometry-first
    if args.geometry_only:
        record['training_objective'] = ('block_local_v4: local-radius SmoothL1(beta=.01) + bbox SmoothL1(beta=.001); '
                                        f'noisy objective + {args.geometry_clean_weight} * clean objective')
    if model.cfg.position_head=='reference_v6':
        record['training_objective']='v6 actual compact-group scale (floor .001), bounded Gaussian-size weight [1,4], world-axis correction; persistent clean geometry constraint'
        record['render_geometry_weight']='cfg.geometry_weight, not reduced by attr_weight'
        record['mask_gradient']='biased ST for slots/power/context; hard retained-row grouping and phase unwrap branches detached'
        if model.cfg.individual_tiers:
            record['grouping']='fixed source-row intervals; within-group retained rows compacted for phase multiplexing'
            record['tier_training_layouts']='75% mixed, 25% randomly selected uniform tier when --tier-training all'
            record['mask_gradient']='biased ST; group boundaries fixed; within-group retention and phase unwrap remain discrete'
    (out / "training.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    fixed_evaluations = []
    best_score, best_step, stale_evaluations = float('inf'), None, 0
    def fixed_position_evaluation(step):
        nonlocal best_score, best_step, stale_evaluations
        if args.position_eval_every:
            from types import SimpleNamespace
            from .gradient_diagnostics import evaluate_blocks
            options=SimpleNamespace(eval_blocks=args.position_eval_blocks,eval_snrs=args.position_eval_snrs,seed=args.seed)
            was_training=model.training
            try:
                model.eval()
                with preserved_rng(device):
                    rows=evaluate_blocks(model,feature_blocks,geometry,options,step)
            finally:
                model.train(was_training)
            fixed_evaluations.extend(rows)
            (out / "position_evaluation.json").write_text(json.dumps(fixed_evaluations,indent=2),encoding="utf-8")
            print(f"[Fixed position evaluation {step}] " + "; ".join(
                f"q{r['tier']} RMSE={r['position_rmse']:.5g}, P95={r['distance_p95']:.5g}"
                for r in rows if r['channel']=='none'),flush=True)
            if args.geometry_only and args.fixed_snr is not None and args.channel == 'awgn':
                selected = [r for r in rows if r['channel'] == 'awgn' and r['snr'] == args.fixed_snr]
                score = sum(r['unclipped_position_rmse'] for r in selected) / len(selected)
                improved = math.isfinite(score) and score < best_score - args.position_min_delta
                if improved:
                    best_score, best_step, stale_evaluations = score, step, 0
                    save_checkpoint(out / 'codec_best.pt', model, step, record)
                else:
                    stale_evaluations += 1
                print(f'[Fixed AWGN {args.fixed_snr:g} dB] mean raw XYZ RMSE={score:.6g}; best={best_score:.6g} at {best_step}', flush=True)
                (out / 'position_selection.json').write_text(json.dumps({
                    'metric': 'mean_over_tiers_unclipped_position_rmse', 'snr': args.fixed_snr,
                    'best_step': best_step, 'best_score': best_score, 'latest_step': step,
                    'latest_score': score, 'stale_evaluations': stale_evaluations,
                    'scope': 'fixed source-scene blocks; not held-out scene generalization'}, indent=2), encoding='utf-8')
                return bool(args.position_patience and stale_evaluations >= args.position_patience)
        return False
    fixed_position_evaluation(0)
    progress = trange(args.steps + args.render_steps, desc="Gaussian JSCC training")
    completed_steps = 0
    with (out / "loss.jsonl").open("w", encoding="utf-8") as log:
        for step in progress:
            step_started = time.perf_counter()
            snr = random.uniform(*args.snr_range)
            optimizer.zero_grad(set_to_none=True)
            backward_done = False
            is_render = step >= args.steps
            if step == args.steps:
                for group in optimizer.param_groups:
                    group["lr"] = args.render_lr
            if not is_render:
                index = random.randrange(len(feature_blocks))
                features = feature_blocks[index].to(device)
                if args.geometry_only:
                    from .optimization import all_tier_geometry_step
                    loss, render_values = all_tier_geometry_step(model, features, geometry, snr, args.channel,
                                                                args.geometry_clean_weight)
                    backward_done = True
                elif args.tier_training == "all":
                    loss,render_values=all_tier_attribute_step(model,features,geometry,snr,args.channel)
                    backward_done=True
                else:
                    qb = (tier_blocks[index].to(device) if fixed_q is not None
                          else sample_tiers(len(features), device, drop=args.attribute_drop))
                    keep = qb > 0
                    if not keep.any():
                        qb[0] = 1
                        keep = qb > 0
                    if not model.cfg.individual_tiers:
                        features, qb = features[keep], qb[keep]
                    unit = features[:, :3]
                    pred, seed = model(features, unit, qb, snr, args.channel, return_seed=True)
                    loss, terms = reconstruction_loss(pred, features, geometry, model, return_terms=True,
                                                      active=(qb>0).to(features) if model.cfg.individual_tiers else None,
                                                      **position_training_inputs(model,features,qb,snr))
                    render_values = {f"{key}_loss": float(value.detach().mean()) for key, value in terms.items()}
                image_loss = None
            else:
                from .rendering import render
                from utils.loss_utils import ssim

                camera = random.choice(cameras)
                gt = reference.get(camera, device)
                def distortion(scene):
                    image = render(scene, camera, degree, args.white_background)
                    return .8 * (image - gt).abs().mean() + .2 * (1 - ssim(image, gt))
                batches = []
                for group_index, f in enumerate(render_groups):
                    begin = group_index * args.blocks_per_batch
                    qs = [tier_blocks[i] if fixed_q is not None else
                          sample_tiers(len(feature_blocks[i]), cache_device)
                          for i in range(begin, min(begin + args.blocks_per_batch, len(feature_blocks)))]
                    batches.append((f, pad_sequence(qs, batch_first=True)))
                profiled = bool(args.profile_every and (step - args.steps) % args.profile_every == 0)
                loss, render_values = full_scene_step(
                    model, batches, geometry, snr, args.channel, distortion,
                    args.attr_weight, args.seed_position_weight, args.render_backward, profiled)
                image_loss = render_values["render_loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss; no checkpoint saved for this step")
            if not is_render and not backward_done:
                loss.backward()
            if args.profile_every and (step+1)%args.profile_every==0:
                from .gradient_diagnostics import update_stats, largest_gradients
                before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
                render_values['largest_parameter_gradients']=largest_gradients(model)
            else:
                before=None
            norm,clip_stats = clip_codec_gradients(model,args.clip_norm,args.clip_mode)
            optimizer.step()
            if before is not None:
                render_values['updates']=update_stats(model,before)
                del before
            values = {"step": step + 1, "phase": "geometry" if args.geometry_only else ("render" if is_render else "attribute"),
                      "loss": float(loss.detach()), "snr": snr, "grad_norm": float(norm),
                      "step_seconds": time.perf_counter() - step_started, **render_values}
            if image_loss is not None:
                values["render_loss"] = float(image_loss)
            if args.geometry_only:
                values.update(loss_profile='reference_v6' if model.cfg.position_head=='reference_v6' else 'block_local_v4',
                              geometry_contribution=values['loss'])
            else:
                values.update(objective_stats(values, model, args.attr_weight if is_render else 1.))
            values.update(clip_stats)
            values["learning_rate"] = optimizer.param_groups[0]["lr"]
            log.write(json.dumps(values) + "\n")
            if is_render or (step + 1) % 10 == 0:
                log.flush()
                progress.set_postfix(loss=f"{values['loss']:.5f}", phase=values["phase"],
                                     sec=f"{values['step_seconds']:.2f}")
            if (step + 1) % args.save_every == 0:
                save_checkpoint(out / f"codec_{step + 1}.pt", model, step + 1, record)
            completed_steps = step + 1
            if args.position_eval_every and ((step+1)%args.position_eval_every==0 or step+1 in (args.steps,args.steps+args.render_steps)):
                if fixed_position_evaluation(step+1):
                    print(f'Early stop after {completed_steps} steps; best fixed-condition checkpoint: codec_best.pt')
                    break
        save_checkpoint(out / "codec.pt", model, completed_steps, record)
    from .plots import safe_plot
    safe_plot("training", out)
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
    from .learned_train import add_parser as add_learned_parser
    add_learned_parser(sub)
    from .plots import add_parser as add_plot_parser
    add_plot_parser(sub)
    from .benchmark import add_parser as add_benchmark_parser
    add_benchmark_parser(sub)
    p = sub.add_parser("train")
    p.set_defaults(func=train)
    p.add_argument("--ply", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--init", help="initialize shared codec weights/statistics; head changes require --upgrade-position-head")
    p.add_argument("--steps", type=int, default=20000, help="attribute training steps")
    p.add_argument("--render-steps", type=int, default=0, help="subsequent full-scene rendering steps")
    p.add_argument("--blocks-per-batch", type=int, default=32,
                   help="independent spatial blocks processed together during full-scene render training")
    p.add_argument("--render-backward", choices=["replay", "checkpoint"], default="replay",
                   help="exact full-scene gradient replay (bounded codec memory) or checkpoint reference")
    p.add_argument("--training-data-device", choices=["cpu", "cuda"], default="cuda",
                   help="cache fixed normalized source features; cuda uses the selected --device")
    p.add_argument("--profile-every", type=int, default=10,
                   help="synchronize and record render phase timings/peak memory every N steps; 0 disables")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--render-lr", type=float,
                   help="render fine-tuning LR; default 0.25 * lr, without resetting Adam state")
    p.add_argument("--attr-weight", type=float, default=1.,
                   help="render-stage reconstruction multiplier; default keeps the auxiliary active at weight 1")
    p.add_argument("--snr-range", type=float, nargs=2, default=[0., 20.])
    p.add_argument('--fixed-snr', type=float, help='override training range and fixed evaluation SNR; training noise stays random')
    p.add_argument('--geometry-only', action='store_true', help='train only block-relative geometric payload; freeze all other weights')
    p.add_argument('--position-patience', type=int, default=0, help='stop after N fixed AWGN evaluations without improvement; 0 disables')
    p.add_argument('--position-min-delta', type=float, default=0., help='minimum improvement in scene-unit raw XYZ RMSE')
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
                   help="legacy compatibility only; ignored by geometry-first (no seed stage)")
    add_arguments(p)
    p.add_argument("--attribute-drop", type=float, default=.05,
                   help="random context dropout during attribute warmup; use 0 for strict codec isolation")
    p.add_argument("--tier-training",choices=("sample","all"),default="sample",
                   help="all: same block/SNR at q1/q2/q3, average gradients then one optimizer step")
    p.add_argument("--position-eval-every",type=int,default=0,help="0 disables fixed-block per-tier position evaluation")
    p.add_argument("--position-eval-blocks",type=int,default=8)
    p.add_argument("--position-eval-snrs",type=float,nargs='+',default=[0.,10.,20.])
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
