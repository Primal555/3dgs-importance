"""Scene-balanced shared codec; geometry and hierarchical allocation remain per scene.

Run with python -m gaussian_jscc.multiscene. Step budgets are PER SCENE,
and an update is an actual optimizer step, not a microbatch or scene sweep.
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .allocation import GaussianTierMask, scene_fingerprint
from .allocation_diagnostics import AllocationCostMeter, load_existence_prior, record_allocation
from .codec import CodecConfig, GaussianCodec
from .data import Geometry, morton_order, read_ply, to_features
from .learned_training import hard_layout, layout_schedule
from .local_response import local_response_loss
from .mask_gradient import local_mask_step
from .optimization import clip_codec_gradients, preserved_rng
from .position_delivery import PositionCostMeter
from .render_objective import MultiViewRenderTask, MaskedMultiViewRenderTask, split_cameras
from .render_validation import append_json, validate_render
from .training import full_scene_step
from .transport import load_checkpoint, model_id, save_checkpoint


def manifest(path):
    path = Path(path).resolve()
    record = json.loads(path.read_text(encoding='utf-8'))
    root = (path.parent / record.get('root', '.')).resolve()
    scenes, names, ply_paths = [], set(), set()
    for item in record['scenes']:
        s = dict(item)
        if not re.fullmatch(r'[A-Za-z0-9_-]+', s['name']) or s['name'] in names:
            raise ValueError('scene names must be unique, filesystem-safe identifiers')
        if s['role'] not in ('train', 'heldout'):
            raise ValueError('scene role must be train or heldout')
        names.add(s['name'])
        for key in ('ply', 'source'):
            s[key] = str((root / s[key]).resolve())
        if s['ply'].casefold() in ply_paths:
            raise ValueError('each PLY can appear only once; scene aliases must not cross splits')
        ply_paths.add(s['ply'].casefold())
        if s.get('prior') not in (None, 'none', 'auto', 'ply'):
            s['prior'] = str((root / s['prior']).resolve())
        scenes.append(s)
    if not scenes:
        raise ValueError('empty manifest')
    return scenes


def shared_statistics(raw_scenes):
    """Equal scene mass, float64 streaming moments; never accepts heldout data implicitly."""
    means, seconds = [], []
    for raw in raw_scenes:
        total = torch.zeros(raw.shape[1]-3, dtype=torch.float64)
        square = torch.zeros_like(total)
        for block in raw[:, 3:].split(65536):
            x = block.double()
            total += x.sum(0)
            square += x.square().sum(0)
        means.append(total / len(raw))
        seconds.append(square / len(raw))
    mean = torch.stack(means).mean(0)
    std = (torch.stack(seconds).mean(0)-mean.square()).clamp_min(0).sqrt().clamp_min(.01)
    return mean.float(), std.float()


class Scene:
    def __init__(self, spec, raw, degree, model, args, out, with_mask=True, cameras=True):
        self.spec, self.name = spec, spec['name']
        self.out = Path(out) / 'scenes' / self.name
        self.out.mkdir(parents=True, exist_ok=True)
        self.fingerprint = scene_fingerprint(raw)
        self.degree = degree
        self.geometry = Geometry.fit(raw[:, :3], model.cfg.morton_bits)
        self.order = torch.from_numpy(morton_order(self.geometry.quantize(raw[:, :3]).numpy()).astype(np.int64))
        self.prior, self.prior_info = load_existence_prior(spec.get('prior', 'auto'), spec['ply'], len(raw))
        self.raw = raw[self.order]
        device = next(model.parameters()).device
        self.blocks, self.ids = [], []
        with torch.no_grad():
            for start in range(0, len(raw), model.cfg.block_size):
                f, _ = to_features(self.raw[start:start+model.cfg.block_size].to(device), self.geometry, model)
                self.blocks.append(f.cpu())
                self.ids.append(self.order[start:start+len(f)])
        self.groups = [pad_sequence(self.blocks[i:i+args.blocks_per_batch], batch_first=True)
                       for i in range(0, len(self.blocks), args.blocks_per_batch)]
        self.group_ids = [pad_sequence(self.ids[i:i+args.blocks_per_batch], batch_first=True, padding_value=-1)
                          for i in range(0, len(self.blocks), args.blocks_per_batch)]
        self.position_meter = PositionCostMeter(model.cfg, self.geometry.normalize(self.raw[:, :3]))
        self.rate_meter = AllocationCostMeter(model.cfg, self.position_meter, len(raw), args.net_bits_per_use)
        self.mask = GaussianTierMask(len(raw), existence_prior=self.prior, tier_count=len(model.cfg.rates)) if with_mask else None
        self.optimizer = torch.optim.Adam([
            {'params': [self.mask.keep_logits], 'lr': args.keep_lr},
            {'params': [self.mask.logits], 'lr': args.mask_lr}], eps=args.mask_adam_eps) if with_mask else None
        self.cameras, self.val_cameras, self.reference = [], [], None
        self.visits = {}
        if cameras:
            from .rendering import load_cameras, RenderReference
            all_cameras = load_cameras(spec['source'], args.resolution, args.white_background, args.images, 'train')
            self.cameras, self.val_cameras, train_ids, val_ids = split_cameras(all_cameras, args.validation_views)
            if args.views_per_step > len(self.cameras):
                raise ValueError(f'{self.name}: insufficient training views')
            self.reference = RenderReference(self.raw, degree, args.white_background, 'source')
            (self.out/'scene.json').write_text(json.dumps({**spec, 'source_gaussians': len(raw),
                'fingerprint': self.fingerprint, 'geometry': self.geometry.to_dict(), 'prior': self.prior_info,
                'train_view_indices': train_ids, 'validation_view_indices': val_ids,
                'train_view_names': [str(c.image_name) for c in self.cameras],
                'validation_view_names': [str(c.image_name) for c in self.val_cameras]}, indent=2), encoding='utf-8')

    def move_mask(self, device):
        # Only the active scene's table and Adam moments occupy accelerator memory.
        if self.mask is not None:
            old = dict(self.mask.named_parameters())
            self.mask.to(device)
            new = dict(self.mask.named_parameters())
            # Handle device conversions that replace Parameter objects as well as
            # PyTorch's usual in-place data conversion, preserving Adam ownership.
            mapping = {old[k]:new[k] for k in old}
            for group in self.optimizer.param_groups:
                group['params'] = [mapping[p] for p in group['params']]
            for previous,current in mapping.items():
                if previous is not current and previous in self.optimizer.state:
                    self.optimizer.state[current] = self.optimizer.state.pop(previous)
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value) and key != 'step':
                        state[key] = value.to(device)


def save_bundle(out, label, model, scenes, step, record):
    folder = Path(out)/'checkpoints'/label
    folder.mkdir(parents=True, exist_ok=True)
    snapshot_record = dict(record, checkpoint_label=label)
    save_checkpoint(folder/'codec.pt', model, step, snapshot_record)
    identity = model_id(model)
    names = []
    for s in scenes:
        if s.mask is None:
            continue
        names.append(s.name)
        torch.save({'version': 2, 'count': len(s.raw), 'scene_fingerprint': s.fingerprint,
                    'snr_conditioned': False, 'state_dict': {k:v.detach().cpu() for k,v in s.mask.state_dict().items()},
                    'codec_id': identity, 'rates': list(model.cfg.rates), 'step': step, 'training': snapshot_record}, folder/f'{s.name}.pt')
    (folder/'bundle.json').write_text(json.dumps({'codec_id': identity, 'step': step, 'allocations': names,
        'scene_visits': {s.name:s.visits for s in scenes}, 'training': snapshot_record}, indent=2), encoding='utf-8')
    return folder


def validate(model, scenes, args, step, phase):
    results = []
    for s in scenes:
        s.move_mask(next(model.parameters()).device)
        use_mask = s.mask if phase in ('allocation', 'joint') else None
        with preserved_rng(next(model.parameters()).device):
            if use_mask is not None:
                record_allocation(s.out, step, s.mask, args.snr, model.cfg.rates, s.prior, args.seed)
            r = validate_render(model, s.groups, s.group_ids, s.raw, s.geometry, s.val_cameras,
                s.reference, args.snr, args.channel, args.validation_trials, args.seed, s.out, step, phase,
                mask=use_mask, beta=args.beta, white_background=args.white_background,
                position_net_bits_per_use=args.net_bits_per_use, position_meter=s.position_meter,
                allocation_meter=s.rate_meter if use_mask is not None else None)
        results.append({'scene': s.name, 'score': r['score']})
        s.move_mask('cpu')
    # Each scene contributes exactly once, independent of point/view counts.
    score = sum(r['score'] for r in results)/len(results)
    append_json(Path(args.out)/'validation_macro.jsonl', {'step':step, 'phase':phase, 'score':score, 'scenes':results})
    return score


def train(args):
    from .cli import device_for, seed_all
    selected = [s for s in manifest(args.manifest) if s['role'] == ('heldout' if args.adapt else 'train')]
    if args.train_scenes:
        if args.adapt or not set(args.train_scenes).issubset({s['name'] for s in selected}):
            raise ValueError('--train-scenes must select existing training scenes; not heldout adaptation')
        selected = [s for s in selected if s['name'] in args.train_scenes]
    if not selected:
        raise ValueError('no scenes selected')
    if args.adapt and args.allocation_only:
        raise ValueError('choose heldout adaptation OR training-scene allocation-only')
    if args.adapt and (not args.checkpoint or args.bootstrap_steps or args.render_steps or args.joint_steps):
        raise ValueError('adapt requires --checkpoint and zero bootstrap/render/joint steps; only allocation is fitted')
    if args.allocation_only and (not args.checkpoint or args.bootstrap_steps or args.render_steps or args.joint_steps):
        raise ValueError('allocation-only requires checkpoint and zero bootstrap/render/joint steps')
    if args.checkpoint and not (args.adapt or args.allocation_only):
        raise ValueError('shared training starts randomly; checkpoint is for frozen allocation fitting')
    counts = [args.bootstrap_steps, args.render_steps, args.allocation_steps, args.joint_steps]
    if min(counts) < 0 or sum(counts) < 1:
        raise ValueError('invalid per-scene phase budgets')
    if min(args.validate_every, args.save_every, args.blocks_per_batch, args.views_per_step,
           args.validation_views, args.validation_trials, args.local_response_views) < 1:
        raise ValueError('counts must be positive')
    if not 0 <= args.drop < 1 or not math.isfinite(args.beta) or args.beta < 0:
        raise ValueError('invalid drop or beta')
    for key in ('lr','render_lr','mask_lr','keep_lr','mask_adam_eps','net_bits_per_use'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key) <= 0:
            raise ValueError(f'invalid {key}')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'use a new output directory: {out}')
    device = device_for(args.device)
    seed_all(args.seed)
    if device.type == 'cpu':
        torch.set_num_threads(args.cpu_threads)
    loaded = [read_ply(s['ply']) for s in selected]
    degrees = {degree for _,degree in loaded}
    if len(degrees) != 1:
        raise ValueError('shared codec requires matching SH degree')
    degree = next(iter(degrees))
    if args.checkpoint:
        model = load_checkpoint(args.checkpoint, device)
        if model.cfg.sh_degree != degree or model.cfg.prefix_mode != 'progressive' or model.cfg.position_delivery != 'quantized':
            raise ValueError('adaptation requires compatible progressive reliable-coordinate codec')
        source_record = torch.load(args.checkpoint,map_location='cpu',weights_only=True).get('training',{})
        trained = source_record.get('codec_training_scenes', source_record.get('selected_scenes', []))
        if args.adapt and any(s['name'] in {t['name'] for t in trained} for s in selected):
            raise ValueError('heldout scene was used to train the supplied codec')
        # Channel/resolution/feature layout of the frozen initializer must remain
        # the same in beta controls and adaptation, not quietly revert to defaults.
        for key in ('snr','channel','resolution','white_background','images','net_bits_per_use'):
            if key in source_record:
                setattr(args,key,source_record[key])
    else:
        model = GaussianCodec(CodecConfig(architecture='learned_joint', loss_profile='learned_v1',
            sh_degree=degree, hidden=args.hidden, depth=args.depth, grid_dim=16, levels=(4,8), planes=False,
            rates=tuple(args.rates), block_size=args.block_size, decoder_window=args.decoder_window,
            attention_heads=4, prefix_mode='progressive', position_delivery='quantized', position_bits=16,
            position_compression='delta_zlib', position_compression_level=6)).to(device)
        mean, std = shared_statistics([raw for raw,_ in loaded])
        model.attr_mean.copy_(mean.to(device))
        model.attr_std.copy_(std.to(device))
    out.mkdir(parents=True)
    record = {k:v for k,v in vars(args).items()}
    provenance = torch.load(args.checkpoint,map_location='cpu',weights_only=True).get('training',{}) if args.checkpoint else {}
    record.update(selected_scenes=selected, codec_config=model.cfg.to_dict(),
        codec_training_scenes=provenance.get('codec_training_scenes',provenance.get('selected_scenes',selected)),
        normalization='checkpoint unchanged' if args.checkpoint else 'equal-scene first/second moments; training PLYs only',
        scope='heldout allocator adaptation; codec frozen' if args.adapt else 'shared training scenes only',
        step_definition='one optimizer update; budgets per scene; round robin, equal scene mass',
        source_target='original PLY rendering; photos are separate evaluation references',
        checkpoint_semantics='weights/tables, not exact optimizer resume',
        total_updates=len(selected)*sum(counts))
    (out/'training.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    scenes = [Scene(spec, raw, d, model, args, out, with_mask=bool(args.allocation_steps or args.joint_steps))
              for spec,(raw,d) in zip(selected,loaded)]
    del loaded
    if args.allocation_steps or args.joint_steps:
        from .mask_checks import check_masked_renderer
        parity = check_masked_renderer(scenes[0].raw.to(device), scenes[0].cameras[0], degree, args.white_background)
        (out/'mask_renderer_check.json').write_text(json.dumps(parity, indent=2), encoding='utf-8')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule, step = layout_schedule(model.cfg.rates), 0
    print(f'Scenes: {[s.name for s in scenes]}; per-scene budgets {counts}; total actual updates {record["total_updates"]}', flush=True)
    save_bundle(out, 'initial', model, scenes, step, record)
    for phase, budget in zip(('bootstrap','render','allocation','joint'), counts):
        if not budget:
            continue
        if phase == 'render':
            optimizer.state.clear()
        for group in optimizer.param_groups:
            group['lr'] = args.lr if phase == 'bootstrap' else args.render_lr
        best = validate(model, scenes, args, step, phase)
        save_bundle(out, 'best_'+phase, model, scenes, step, record)
        for visit in range(1, budget+1):
            for s in scenes:
                start = time.perf_counter()
                step += 1
                s.visits[phase] = visit
                model.train()
                optimizer.zero_grad(set_to_none=True)
                tier = schedule[(visit-1)%len(schedule)]
                stats = {'layout': 'mixed' if tier is None else str(tier)}
                if phase == 'bootstrap':
                    indices = torch.randint(len(s.blocks), (args.blocks_per_batch,)).tolist()
                    f = pad_sequence([s.blocks[i] for i in indices], batch_first=True).to(device)
                    ids = pad_sequence([s.ids[i] for i in indices], batch_first=True, padding_value=-1).to(device)
                    q = hard_layout(ids, tier, args.drop, len(model.cfg.rates))
                    if not (q>0).any():
                        q[ids>=0] = 1
                    choices = torch.nn.functional.one_hot(q, len(model.cfg.rates)).to(f)
                    pred,_,_ = model.forward_tier_batches(f, f[...,:3], choices, args.snr, args.channel)
                    loss, details = local_response_loss(pred[q>0], f[q>0], s.geometry, model, args.local_response_views)
                    loss.backward()
                else:
                    views = torch.randperm(len(s.cameras))[:args.views_per_step].tolist()
                    camera_batch = [s.cameras[i] for i in views]
                    if phase == 'render':
                        task = MultiViewRenderTask(camera_batch, s.reference, degree, args.white_background)
                        qs = [hard_layout(ids,tier,args.drop,len(model.cfg.rates)) for ids in s.group_ids]
                        loss, details = full_scene_step(model, list(zip(s.groups,qs)), s.geometry,
                            args.snr, args.channel, task, attr_weight=0., mode='replay')
                        details.update(task.stats)
                    else:
                        s.move_mask(device)
                        s.optimizer.zero_grad(set_to_none=True)
                        task = MaskedMultiViewRenderTask(camera_batch, s.reference, degree, args.white_background)
                        loss, details = local_mask_step(model, s.mask, s.groups, s.group_ids, s.geometry,
                            args.snr, args.channel, task, args.beta, s.rate_meter, train_codec=phase=='joint', mode='replay')
                        stats['layout'] = 'learned_mask'
                    stats['training_view_indices'] = views
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite loss; no optimizer steps performed')
                norm, _ = clip_codec_gradients(model, 10., 'none')
                if not math.isfinite(float(norm)):
                    raise RuntimeError('nonfinite codec gradient')
                if phase in ('allocation','joint'):
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in s.mask.parameters()):
                        raise RuntimeError('nonfinite allocator gradient')
                    s.optimizer.step()
                    s.optimizer.zero_grad(set_to_none=True)
                    s.move_mask('cpu')
                if phase != 'allocation':
                    optimizer.step()
                row = {'step':step, 'scene_step':visit, 'phase':phase, 'scene':s.name,
                       'loss':float(loss.detach()), 'grad_norm':float(norm), 'lr':optimizer.param_groups[0]['lr'],
                       'codec_updated':phase!='allocation', 'step_seconds':time.perf_counter()-start, **stats, **details}
                append_json(out/'loss.jsonl',row)
                append_json(s.out/'loss.jsonl',row)
                if visit==1 or visit%10==0:
                    print(f'{s.name} {phase} {visit}/{budget}: loss={row["loss"]:.6g}, q={stats["layout"]}, sec={row["step_seconds"]:.2f}',flush=True)
            if visit%args.validate_every==0 or visit==budget:
                score = validate(model, scenes, args, step, phase)
                if score < best:
                    best = score
                    save_bundle(out, 'best_'+phase, model, scenes, step, record)
                from .multiscene_plots import plot_training
                plot_training(out)
            if visit%args.save_every==0:
                save_bundle(out, 'latest', model, scenes, step, record)
        save_bundle(out, 'end_'+phase, model, scenes, step, record)
    save_bundle(out, 'final', model, scenes, step, record)
    (out/'complete.json').write_text(json.dumps({'updates':step,'scene_visits':{s.name:s.visits for s in scenes}},indent=2),encoding='utf-8')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='configs/multiscene_tandt_db.json')
    p.add_argument('--train-scenes', nargs='+', help='optional training-scene subset for single-scene controls')
    p.add_argument('--out', required=True)
    p.add_argument('--adapt', action='store_true', help='fit heldout scene tables only; never update shared codec/statistics')
    p.add_argument('--allocation-only', action='store_true', help='fit fresh training-scene tables on a frozen checkpoint; beta controls')
    p.add_argument('--checkpoint')
    p.add_argument('--device', default='cuda')
    p.add_argument('--channel', choices=['awgn','none'], default='awgn')
    p.add_argument('--snr', type=float, default=10.)
    p.add_argument('--rates', nargs='+', type=int, default=[0,4,12,24])
    for key, default in [('bootstrap-steps',5000),('render-steps',5000),('allocation-steps',1000),('joint-steps',1000),
                         ('blocks-per-batch',64),('block-size',256),('decoder-window',32),('hidden',96),('depth',2),
                         ('local-response-views',4),('views-per-step',2),('validation-views',4),('validation-trials',2),
                         ('validate-every',500),('save-every',500),('resolution',2),('seed',42),('cpu-threads',4)]:
        p.add_argument('--'+key,type=int,default=default)
    for key, default in [('lr',1e-4),('render-lr',1e-4),('mask-lr',.001),('keep-lr',.01),('mask-adam-eps',1e-15),
                         ('beta',.01),('drop',.05),('net-bits-per-use',2.)]:
        p.add_argument('--'+key,type=float,default=default)
    p.add_argument('--images',default='images')
    p.add_argument('--white-background',action='store_true')
    return p


if __name__ == '__main__':
    train(parser().parse_args())
