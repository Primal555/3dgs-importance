"""Explicit new-mainline entry point: python -m gaussian_jscc train-learned."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .codec import CodecConfig, GaussianCodec
from .data import read_ply, Geometry, morton_order, to_features, to_raw
from .losses import reconstruction_loss
from .learned_objective import projection_loss
from .learned_training import discrete_joint_step, decode_batches, hard_layout
from .optimization import clip_codec_gradients, preserved_rng
from .training import full_scene_step
from .transport import load_checkpoint, save_checkpoint


def add_parser(sub):
    p = sub.add_parser('train-learned', aliases=['train'], description=__doc__)
    p.set_defaults(func=train)
    for key in ('ply', 'out'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--init', help='learned_joint checkpoint ONLY; starts a fresh optimizer/schedule')
    p.add_argument('--allocation-init', help='matching route2.pt; requires --init and positive --joint-steps')
    p.add_argument('--existence-prior', help='optional .npy probabilities in ORIGINAL input PLY row order')
    p.add_argument('--source')
    p.add_argument('--device', default='cuda')
    p.add_argument('--snr', type=float, default=10.)
    p.add_argument('--channel', choices=['awgn', 'none'], default='awgn')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--render-steps', type=int, default=1000)
    p.add_argument('--joint-steps', type=int, default=0)
    p.add_argument('--render-ramp', type=int, default=200)
    p.add_argument('--block-size', type=int, default=256)
    p.add_argument('--blocks-per-batch', type=int, default=32)
    p.add_argument('--decoder-window', type=int, default=32)
    p.add_argument('--attention-heads', type=int, default=4)
    p.add_argument('--hidden', type=int, default=96)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--grid-dim', type=int, default=16)
    p.add_argument('--levels', nargs='+', type=int, default=[4, 8])
    p.add_argument('--rates', nargs=4, type=int, default=[0, 8, 16, 32])
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--mask-lr', type=float, default=1e-3)
    p.add_argument('--mask-samples', type=int, default=2)
    p.add_argument('--beta', type=float, default=.01)
    p.add_argument('--drop', type=float, default=.05)
    p.add_argument('--auxiliary-weight', type=float, default=1.)
    p.add_argument('--render-weight', type=float, default=1.)
    p.add_argument('--projection-weight', type=float, default=.05)
    p.add_argument('--xyz-loss-scale', type=float, default=.05)
    p.add_argument('--geometry-weight', type=float, default=1.)
    p.add_argument('--shape-weight', type=float, default=.25)
    p.add_argument('--scale-weight', type=float, default=1.)
    p.add_argument('--opacity-weight', type=float, default=1.)
    p.add_argument('--dc-weight', type=float, default=1.)
    p.add_argument('--sh-weight', type=float, default=.25)
    p.add_argument('--power-floor', type=float, default=.01)
    p.add_argument('--clip-mode', choices=['none', 'global', 'branch'], default='none')
    p.add_argument('--clip-norm', type=float, default=10., help='used only when clipping is enabled; not a universal threshold')
    p.add_argument('--render-backward', choices=['replay', 'checkpoint'], default='replay')
    p.add_argument('--training-data-device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--resolution', type=int, default=2)
    p.add_argument('--images', default='images')
    p.add_argument('--white-background', action='store_true')
    p.add_argument('--validate-every', type=int, default=100)
    p.add_argument('--validation-blocks', type=int, default=8)
    p.add_argument('--validation-views', type=int, default=2)
    p.add_argument('--patience', type=int, default=8, help='fixed-validation checks without improvement; zero disables stopping')
    p.add_argument('--min-delta', type=float, default=1e-4)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu-threads', type=int, default=4)


def train(args):
    from .cli import seed_all, device_for
    from .allocation import GaussianTierMask, scene_fingerprint
    from .route2 import save_joint
    device = device_for(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(args.cpu_threads)
    if min(args.steps, args.render_steps, args.joint_steps, args.patience) < 0 or args.steps+args.render_steps+args.joint_steps < 1:
        raise ValueError('invalid stage lengths')
    if min(args.validate_every, args.validation_blocks, args.validation_views, args.save_every,
           args.blocks_per_batch, args.render_ramp) < 1:
        raise ValueError('counts and ramp must be positive')
    if not 0 <= args.drop < 1 or args.mask_samples < 2:
        raise ValueError('invalid drop probability or mask-samples')
    for key in ('lr', 'mask_lr', 'clip_norm', 'xyz_loss_scale', 'power_floor'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f'{key} must be positive and finite')
    for key in ('beta', 'auxiliary_weight', 'render_weight', 'projection_weight', 'min_delta'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError(f'{key} must be nonnegative and finite')
    if not math.isfinite(args.snr):
        raise ValueError('SNR must be finite')
    needs_render = bool(args.render_steps or args.joint_steps)
    if needs_render and (not args.source or not args.device.startswith('cuda')):
        raise ValueError('render/joint stages require CUDA and --source; CPU codec checks use both stage lengths 0')
    if args.allocation_init and (not args.init or not args.joint_steps):
        raise ValueError('allocation-init requires init and joint-steps')
    if args.existence_prior and (not args.joint_steps or args.allocation_init):
        raise ValueError('existence-prior requires joint-steps and no allocation-init')
    seed_all(args.seed)
    original, degree = read_ply(args.ply)
    fingerprint = scene_fingerprint(original) if args.joint_steps else None
    if args.init:
        model = load_checkpoint(args.init, device).train()
        if model.cfg.architecture != 'learned_joint' or model.cfg.sh_degree != degree:
            raise ValueError('new learned architecture requires matching learned_joint weights; omit --init for old models')
        print('Loaded learned_joint weights/statistics. Fresh optimizer and schedule; not an exact resume.')
        print('Checkpoint architecture and loss configuration retained; CLI model-construction settings apply only without --init.')
    else:
        cfg = CodecConfig(architecture='learned_joint', loss_profile='learned_v1', sh_degree=degree,
                          hidden=args.hidden, grid_dim=args.grid_dim, depth=args.depth, levels=tuple(args.levels),
                          planes=False, rates=tuple(args.rates), block_size=args.block_size,
                          decoder_window=args.decoder_window, attention_heads=args.attention_heads,
                          power_floor=args.power_floor, xyz_loss_scale=args.xyz_loss_scale,
                          **{key+'_weight': getattr(args, key+'_weight') for key in ('geometry','shape','scale','opacity','dc','sh')})
        model = GaussianCodec(cfg).to(device)
        model.attr_mean.copy_(original[:, 3:].mean(0).to(device))
        model.attr_std.copy_(original[:, 3:].std(0, unbiased=False).clamp_min(.01).to(device))
    if args.allocation_init:
        from .route2 import load_mask
        mask = load_mask(args.allocation_init, original, model, device).train()
    else:
        prior = np.load(args.existence_prior, allow_pickle=False) if args.existence_prior else None
        mask = GaussianTierMask(len(original), existence_prior=prior).to(device) if args.joint_steps else None
    geometry = Geometry.fit(original[:, :3], model.cfg.morton_bits)
    order = torch.from_numpy(morton_order(geometry.quantize(original[:, :3]).numpy()).astype(np.int64))
    raw = original[order]
    inverse_order = torch.argsort(order)
    del original
    cache_device = device if args.training_data_device == 'cuda' else torch.device('cpu')
    blocks, ids = [], []
    with torch.no_grad():
        for start in range(0, len(raw), model.cfg.block_size):
            f, _ = to_features(raw[start:start+model.cfg.block_size].to(device), geometry, model)
            blocks.append(f.to(cache_device))
            ids.append(order[start:start+len(f)].to(cache_device))
    validation_indices = sorted(set(np.linspace(0, len(blocks)-1, min(args.validation_blocks, len(blocks)), dtype=int).tolist()))
    training_indices = [i for i in range(len(blocks)) if i not in validation_indices]
    if not training_indices:
        training_indices = list(range(len(blocks)))
    groups = [pad_sequence(blocks[i:i+args.blocks_per_batch], batch_first=True)
              for i in range(0, len(blocks), args.blocks_per_batch)]
    group_ids = [pad_sequence(ids[i:i+args.blocks_per_batch], batch_first=True, padding_value=-1)
                 for i in range(0, len(blocks), args.blocks_per_batch)]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mask_optimizer = torch.optim.Adam(mask.parameters(), lr=args.mask_lr) if mask is not None else None
    cameras = val_cameras = reference = None
    if needs_render:
        from .rendering import load_cameras, RenderReference
        cameras = load_cameras(args.source, args.resolution, args.white_background, args.images, 'train')
        if len(cameras) <= args.validation_views:
            raise ValueError('need more training cameras than validation-views')
        val_cameras, cameras = cameras[:args.validation_views], cameras[args.validation_views:]
        reference = RenderReference(raw, degree, args.white_background, 'source')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    record = {k:v for k,v in vars(args).items() if k != 'func'}
    record.update(codec_config=model.cfg.to_dict(), fixed_snr=True,
                  scope='scene-trained codec, not evidence of cross-scene generalization',
                  validation_blocks=validation_indices, validation_block_overlap=not bool(set(range(len(blocks)))-set(validation_indices)),
                  metadata='reliable counted global bbox + per-row 2bit tier syntax; shared codec weights excluded',
                  mask_gradient='REINFORCE render/projection task reward, exact expected payload rate; no ST zero-position surrogate',
                  budget='Lagrangian expected payload penalty, not a hard cap',
                  loss_design='engineering hyperparameters; no inverse tiny covariance, no learned loss weights')
    (out/'training.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'learned_joint: fixed {args.channel} {args.snr:g} dB; source block {model.cfg.block_size}; '
          f'per-Gaussian rates {model.cfg.rates}; no geometry sub-budget; clip={args.clip_mode}', flush=True)

    def task(scene, camera):
        from .rendering import render
        from utils.loss_utils import ssim
        image = render(scene, camera, degree, args.white_background)
        target = reference.get(camera, device)
        return .8*(image-target).abs().mean()+.2*(1-ssim(image, target))

    @torch.no_grad()
    def validate(step, phase):
        with preserved_rng(device):
            seed_all(args.seed+10000)
            model.eval()
            entries = []
            layouts = (1, 2, 3, None) if phase != 'joint' else ('mask',)
            for tier in layouts:
                errors, auxs = [], []
                if tier != 'mask':
                    for i in validation_indices:
                        f = blocks[i].to(device)
                        q = hard_layout(ids[i].to(device), tier, 0.)
                        pred = model(f, f[:, :3], q, args.snr, args.channel)
                        auxs.append(float(reconstruction_loss(pred, f, geometry, model)))
                        delta = (to_raw(pred, geometry, model)[:, :3]-raw[sum(len(b) for b in blocks[:i]):sum(len(b) for b in blocks[:i+1]), :3].to(device))
                        errors.append(delta.square().mean())
                entry = {'layout': 'mixed' if tier is None else tier}
                if auxs:
                    entry.update(auxiliary=sum(auxs)/len(auxs), xyz_rmse=float(torch.stack(errors).mean().sqrt()))
                if phase != 'attribute':
                    qs = [torch.where(gid >= 0, mask.scores(gid.clamp_min(0).to(device), args.snr).argmax(-1).to(gid), 0)
                          if tier == 'mask' else hard_layout(gid, tier, 0.) for gid in group_ids]
                    scene = decode_batches(model, groups, qs, args.snr, args.channel, geometry)
                    losses = [float(task(scene, camera)) for camera in val_cameras]
                    entry['render_loss'] = sum(losses)/len(losses)
                    entry['symbols_per_gaussian'] = sum(float(torch.tensor(model.cfg.rates, device=q.device)[q].sum()) for q in qs)/len(raw)
                    # Save actual held-out-view panels, including the photograph.
                    from .rendering import render
                    from PIL import Image
                    camera = val_cameras[0]
                    panel = torch.cat((camera.original_image[:3].to(device), reference.get(camera, device),
                                       render(scene, camera, degree, args.white_background)), 2).clamp(0, 1)
                    folder = out/'validation_images'/f'{step:06d}'
                    folder.mkdir(parents=True, exist_ok=True)
                    Image.fromarray((panel.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(folder/f'{entry["layout"]}.png')
                entries.append(entry)
            key = 'auxiliary' if phase == 'attribute' else 'render_loss'
            score = sum(e[key] for e in entries)/len(entries)
            if phase == 'joint':
                score += args.beta*entries[0]['symbols_per_gaussian']/model.cfg.rates[-1]
            model.train()
            row = {'step':step, 'phase':phase, 'score':score, 'score_definition':key, 'layouts':entries}
            with (out/'validation.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row)+'\n')
            return score

    step = 0
    for phase, maximum in [('attribute', args.steps), ('render', args.render_steps), ('joint', args.joint_steps)]:
        if not maximum:
            continue
        best, stale = validate(step, phase), 0
        if phase == 'joint':
            save_joint(out, '_best_joint', model, mask, fingerprint, step, record)
        else:
            save_checkpoint(out/f'codec_best_{phase}.pt', model, step, record)
        for local_step in range(1, maximum+1):
            started = time.perf_counter()
            step += 1
            optimizer.zero_grad(set_to_none=True)
            if mask_optimizer is not None:
                mask_optimizer.zero_grad(set_to_none=True)
            stats = {}
            ramp = 0. if phase == 'attribute' else min(1., local_step/args.render_ramp) if phase == 'render' else 1.
            if phase == 'attribute':
                selected = [training_indices[i] for i in torch.randint(len(training_indices), (args.blocks_per_batch,)).tolist()]
                f = pad_sequence([blocks[i] for i in selected], batch_first=True).to(device)
                gi = pad_sequence([ids[i] for i in selected], batch_first=True, padding_value=-1).to(device)
                # Uniform and mixed per-primitive layouts; never one tier per local group.
                tier = int(torch.randint(1,4,())) if torch.rand(()) < .25 else None
                q = hard_layout(gi, tier, args.drop)
                if not (q > 0).any():
                    q[gi >= 0] = 1  # codec pretraining only, never modifies policy samples
                choices = torch.nn.functional.one_hot(q, 4).to(f)
                pred, _, _ = model.forward_tier_batches(f, f[..., :3], choices, args.snr, args.channel)
                aux, terms = reconstruction_loss(pred[q>0], f[q>0], geometry, model, return_terms=True)
                loss = args.auxiliary_weight*aux
                loss.backward()
                stats = {'aux_loss':float(aux.detach()), **{k+'_loss':float(v.detach().mean()) for k,v in terms.items()}}
            else:
                camera = cameras[int(torch.randint(len(cameras), ()))]
                if phase == 'render':
                    tier = int(torch.randint(1,4,())) if torch.rand(()) < .25 else None
                    qs = [hard_layout(gi, tier, args.drop) for gi in group_ids]
                    if not any((q>0).any() for q in qs):
                        qs = [hard_layout(gi, 1) for gi in group_ids]
                    target = torch.cat([to_raw(f[q>0].to(device), geometry, model) for f,q in zip(groups, qs)])
                    task_components = {}
                    def distortion(scene):
                        image_term = task(scene,camera)
                        projection_term = projection_loss(scene,target,camera)
                        task_components.update(render_task_unweighted=float(image_term.detach()),
                                               projection_loss=float(projection_term.detach()),
                                               render_contribution=ramp*args.render_weight*float(image_term.detach()),
                                               projection_contribution=ramp*args.projection_weight*float(projection_term.detach()))
                        return ramp*(args.render_weight*image_term + args.projection_weight*projection_term)
                    loss, stats = full_scene_step(model, list(zip(groups,qs)), geometry, args.snr, args.channel,
                                                  distortion, attr_weight=args.auxiliary_weight, mode=args.render_backward)
                    stats.update(task_components)
                else:
                    def joint_distortion(scene, retained_ids):
                        target = raw[inverse_order[retained_ids.cpu()]].to(device)
                        return (args.render_weight*task(scene,camera)
                                + args.projection_weight*projection_loss(scene,target,camera))
                    loss, stats = discrete_joint_step(model, mask, groups, group_ids, geometry, args.snr, args.channel,
                                                      joint_distortion, beta=args.beta,
                                                      auxiliary_weight=args.auxiliary_weight, samples=args.mask_samples,
                                                      mode=args.render_backward)
            norm, gradient_stats = clip_codec_gradients(model, args.clip_norm, args.clip_mode)
            before = {name:p.detach().clone() for name,p in model.named_parameters()}
            if phase == 'joint':
                mask_norm = torch.stack([p.grad.norm() for p in mask.parameters() if p.grad is not None]).norm()
                if not torch.isfinite(mask_norm):
                    raise RuntimeError('nonfinite mask gradient; no mask step performed')
                stats['mask_grad_norm'] = float(mask_norm)
                mask_optimizer.step()
            optimizer.step()
            from .optimization import update_stats
            updates = update_stats(model, before)
            update_norm = math.sqrt(sum(v['update_norm']**2 for v in updates.values()))
            stats['updates'] = updates
            if 'aux_loss' in stats:
                stats['aux_contribution'] = args.auxiliary_weight*stats['aux_loss']
            for name in ('geometry','shape','scale','opacity','dc','sh'):
                if name+'_loss' in stats:
                    stats[name+'_contribution'] = args.auxiliary_weight*getattr(model.cfg,name+'_weight')*stats[name+'_loss']
            row = {'step':step, 'phase':phase, 'loss_profile':'learned_v1', 'loss':float(loss.detach()),
                   'snr':args.snr, 'render_ramp':ramp, 'grad_norm':float(norm), 'update_norm':float(update_norm),
                   'step_seconds':time.perf_counter()-started, **stats, **gradient_stats}
            with (out/'loss.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row)+'\n')
            if local_step == 1 or local_step % 10 == 0:
                print(f'{phase} {local_step}/{maximum}: loss={float(loss.detach()):.6f}, grad={float(norm):.3f}, '
                      f'update={float(update_norm):.6g}, sec={row["step_seconds"]:.2f}', flush=True)
            if step % args.save_every == 0:
                if phase == 'joint':
                    save_joint(out, f'_{step}', model, mask, fingerprint, step, record)
                else:
                    save_checkpoint(out/f'codec_{step}.pt', model, step, record)
            if local_step % args.validate_every == 0 or local_step == maximum:
                score = validate(step, phase)
                if score < best-args.min_delta:
                    best, stale = score, 0
                    if phase == 'joint':
                        save_joint(out, '_best_joint', model, mask, fingerprint, step, record)
                    else:
                        save_checkpoint(out/f'codec_best_{phase}.pt', model, step, record)
                else:
                    stale += 1
                if args.patience and stale >= args.patience and (phase != 'render' or local_step >= args.render_ramp):
                    print(f'{phase}: fixed validation stopped improving ({stale} checks); ending phase.', flush=True)
                    break
    if args.joint_steps:
        save_joint(out, '', model, mask, fingerprint, step, record)
    else:
        save_checkpoint(out/'codec.pt', model, step, record)
    print(f'Saved learned codec: {out / "codec.pt"}', flush=True)
    from .plots import safe_plot
    safe_plot('training', out)
