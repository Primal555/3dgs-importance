"""Center-only pretraining -> frozen-center attributes -> joint image refinement.

All three phases are CLEAN representation learning, not JSCC or tier training.
Phase lengths are upper bounds; explicit validation gates control progression.
"""
import json
import math
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .codec import CodecConfig, GaussianCodec
from .data import read_ply, Geometry, morton_order, fit_feature_statistics, to_features, to_scene
from .center_attribute_codec import center_loss
from .center_attribute_validation import validate, plot_run
from .learned_train import bootstrap_split
from .representation_train import norm, rng_state, restore_rng, write_state
from .render_validation import append_json
from .render_objective import MultiViewRenderTask, spaced_indices
from .transport import save_checkpoint, load_checkpoint
from .training import full_scene_step

PHASES = ('center', 'attribute', 'joint')


def add_parser(sub):
    p = sub.add_parser('train-center-attributes', description=__doc__)
    p.set_defaults(func=train)
    for name in ('ply', 'source', 'out', 'resume'):
        p.add_argument('--'+name)
    p.add_argument('--device', default='cuda')
    for name, default in dict(center_steps=5000, attribute_steps=1000, joint_steps=1000,
            min_center_steps=500, min_attribute_steps=200, min_joint_steps=200,
            transition_patience=3, stop_patience=5, hidden=96, depth=2, decoder_depth=4,
            attention_heads=4, latent_dim=64, center_latent_dim=32, block_size=256,
            blocks_per_batch=32, render_blocks_per_batch=64, views_per_step=1,
            validate_every=100, render_every=500, save_every=500, profile_every=10,
            validation_blocks=16, validation_views=4, validation_region_size=512,
            resolution=2, seed=42, cpu_threads=4).items():
        p.add_argument('--'+name.replace('_', '-'), type=int, default=default)
    for name, default in dict(center_lr=2e-4, attribute_lr=2e-4, joint_center_lr=2e-4,
            joint_attribute_lr=2e-4, center_smoothing=.001, center_max_gap_db=3.,
            attribute_min_improvement=.05, validation_relative_improvement=.005,
            clip_norm=0.).items():
        p.add_argument('--'+name.replace('_', '-'), type=float, default=default)
    p.add_argument('--render-backward', choices=['replay', 'direct'], default='replay')
    p.add_argument('--images', default='images')
    p.add_argument('--white-background', action='store_true')


def check_args(args):
    if not args.ply or not args.out:
        raise ValueError('--ply and --out are required')
    if args.center_steps < 1 or min(args.attribute_steps, args.joint_steps) < 0:
        raise ValueError('random-start center phase must have positive budget; later budgets cannot be negative')
    if args.joint_steps and not args.attribute_steps:
        raise ValueError('joint training requires attribute adaptation first')
    if (args.attribute_steps or args.joint_steps) and not args.source:
        raise ValueError('attribute/joint phases require source cameras and actual image rendering')
    positive = ('center_lr', 'attribute_lr', 'joint_center_lr', 'joint_attribute_lr', 'center_smoothing',
                'blocks_per_batch', 'render_blocks_per_batch', 'views_per_step', 'validate_every',
                'render_every', 'save_every', 'profile_every', 'validation_blocks', 'validation_views',
                'transition_patience', 'stop_patience', 'cpu_threads')
    for name in positive:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive and finite')
    for name in ('center_max_gap_db', 'clip_norm', 'min_center_steps', 'min_attribute_steps', 'min_joint_steps'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f'{name} must be nonnegative and finite')
    for name in ('attribute_min_improvement', 'validation_relative_improvement'):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) < 1:
            raise ValueError(f'{name} must be in [0,1)')


def make_batches(blocks, size):
    result = []
    for start in range(0, len(blocks), size):
        group = blocks[start:start+size]
        f = pad_sequence(group, batch_first=True)
        active = torch.arange(f.shape[1])[None] < torch.tensor([len(x) for x in group])[:, None]
        result.append((f, active.long()*3))
    return result


def optimizer_for(model, phase, args):
    groups = []
    for name, module in model.learned.named_children():
        if phase == 'joint':
            lr = args.joint_center_lr if name.startswith('center') else args.joint_attribute_lr
        else:
            lr = args.center_lr if phase == 'center' else args.attribute_lr
        parameters = [p for p in module.parameters() if p.requires_grad]
        if parameters:
            groups.append({'params': parameters, 'lr': lr, 'name': name})
    return torch.optim.Adam(groups)


def render_step(model, batches, geometry, task, mode='replay', profile=False):
    def forward(f, q):
        pred = model.reconstruct_clean(f, q > 0)[q > 0]
        # No teacher-coordinate substitution and no parameter auxiliary loss.
        return to_scene(pred, geometry, model), pred.sum()*0, {}
    return full_scene_step(model, batches, geometry, 0, 'none', task,
                           attr_weight=0, seed_weight=0, mode=mode,
                           profile=profile, batch_forward=forward)


def center_ready(validation, args):
    return validation['render'] is not None and validation['render']['center_gap_db'] <= args.center_max_gap_db


def attribute_ready(initial, best, args):
    gain = (initial-best)/max(initial, 1e-12)
    return math.isfinite(gain) and gain >= args.attribute_min_improvement, gain


def archive_tail(out, counts, step):
    archive = out/f'interrupted_tail_{time.time_ns()}'
    for name, count in counts.items():
        path = out/name
        lines = path.read_text(encoding='utf-8').splitlines() if path.exists() else []
        if len(lines) > count:
            archive.mkdir(exist_ok=True)
            shutil.copy2(path, archive/name)
            path.write_text('\n'.join(lines[:count])+'\n', encoding='utf-8')
    # Only known generated numeric step directories are moved, within this run.
    images = out/'images'
    if images.exists():
        for child in images.iterdir():
            if child.is_dir() and child.name.isdigit() and int(child.name) > step:
                (archive/'images').mkdir(parents=True, exist_ok=True)
                shutil.move(str(child), str(archive/'images'/child.name))


def train(args):
    from .cli import seed_all, device_for
    from .allocation import scene_fingerprint
    from .rendering import load_cameras, RenderReference
    resumed = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if resumed:
        if resumed.get('format') != 'center_attribute_training_v1':
            raise ValueError('not a center/attribute training state')
        args = SimpleNamespace(**resumed['arguments'])
        print('Exact resume: stored arguments, model, optimizer and RNG win.', flush=True)
    check_args(args)
    seed_all(args.seed)
    device = device_for(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(args.cpu_threads)
    out = Path(args.out).resolve()
    if not resumed and out.exists() and any(p.name not in ('console.log', 'run.pid') for p in out.iterdir()):
        raise FileExistsError('choose a new output directory or use --resume')
    original, degree = read_ply(args.ply)
    fingerprint = scene_fingerprint(original)
    cfg = CodecConfig(architecture='learned_split_logcov', context_mode='multiscale_self',
        encoder_attention='geometric_point', decoder_attention='transformer_trunk',
        sh_degree=degree, hidden=args.hidden, depth=args.depth, decoder_depth=args.decoder_depth,
        attention_heads=args.attention_heads, block_size=args.block_size,
        representation_dim=args.latent_dim, center_latent_dim=args.center_latent_dim)
    if not cfg.center_latent_dim:
        raise ValueError('this trainer requires positive center-latent-dim')
    model = GaussianCodec(cfg).to(device)
    if resumed:
        if fingerprint != resumed['fingerprint'] or cfg.to_dict() != resumed['config']:
            raise ValueError('resume scene or configuration differs')
        model.load_state_dict(resumed['model'])
    else:
        fit_feature_statistics(original, model)
    geometry = Geometry.fit(original[:, :3], cfg.morton_bits)
    order = torch.from_numpy(morton_order(geometry.quantize(original[:, :3]).numpy()).astype(np.int64))
    raw = original[order]
    del original
    with torch.no_grad():
        blocks = [to_features(chunk.to(device), geometry, model)[0].cpu() for chunk in raw.split(cfg.block_size)]
    fitted, heldout = bootstrap_split(len(raw), cfg.block_size, args.validation_blocks, args.validation_region_size)
    batches = make_batches(blocks, args.render_blocks_per_batch)
    cameras, train_cameras, reference = [], [], None
    if args.source:
        all_validation = load_cameras(args.source, args.resolution, args.white_background, args.images, 'test')
        cameras = [all_validation[i] for i in spaced_indices(len(all_validation), args.validation_views)]
        train_cameras = load_cameras(args.source, args.resolution, args.white_background, args.images, 'train')
        if not train_cameras:
            raise ValueError('no training cameras')
        if set(str(c.image_name) for c in cameras) & set(str(c.image_name) for c in train_cameras):
            raise ValueError('training and validation camera names overlap')
        reference = RenderReference(raw, degree, args.white_background, target='source')
    out.mkdir(parents=True, exist_ok=True)
    arguments = {k: v for k, v in vars(args).items() if k != 'func'}
    arguments.update(out=str(out), ply=str(Path(args.ply).resolve()))
    if args.source:
        arguments['source'] = str(Path(args.source).resolve())
    record = {'arguments': arguments, 'config': cfg.to_dict(), 'fingerprint': fingerprint,
        'geometry': geometry.to_dict(), 'fitted_blocks': fitted, 'heldout_blocks': heldout,
        'center_loss': 'mean(sqrt(||XYZ_pred-XYZ_source||_world^2 + tau^2)-tau); no axis/bbox denominator',
        'attribute_joint_loss': 'source-render image MSE only; no weighted parameter auxiliary',
        'communication': 'not implemented/trained in this clean-only experiment; latent dimensions are not channel uses',
        'position_delivery': 'learned only; source/12bit positions used exclusively in labelled validation diagnostics',
        'lr_schedule': 'constant explicit branch LRs; fresh Adam at each phase',
        'training_views': [str(c.image_name) for c in train_cameras],
        'validation_views': [str(c.image_name) for c in cameras],
        'scope': 'scene-specific; global statistics use whole source PLY; B/C train all Gaussians but not heldout cameras',
        'gates': 'engineering criteria, not proven optimal; budgets are caps, not mandatory phase lengths'}
    if not resumed:
        (out/'training.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    state = resumed['progress'] if resumed else {
        'phase': 'center', 'phase_step': 0, 'phase_index': 0, 'step': 0, 'status': 'running',
        'phase_initialized': False, 'end_reason': None,
        'best': {}, 'best_steps': {}, 'stale': 0, 'initial_attribute_mse': None,
        'last_validation': None, 'completed': {p: 0 for p in PHASES}}
    optimizer_state = resumed['optimizer'] if resumed else None
    if resumed:
        archive_tail(out, resumed['log_counts'], state['step'])
        for phase, step in state['best_steps'].items():
            shutil.copy2(out/f'codec_selected_{phase}_{step}.pt', out/f'codec_best_{phase}.pt')
        restore_rng(resumed['rng'])

    def save(optimizer, boundary=False):
        if not boundary:
            save_checkpoint(out/f'codec_{state["step"]}.pt', model, state['step'], {**record, 'phase': state['phase']})
            save_checkpoint(out/'codec_last.pt', model, state['step'], {**record, 'phase': state['phase']})
        names = ('loss.jsonl', 'validation.jsonl', 'transitions.jsonl')
        counts = {name: len((out/name).read_text(encoding='utf-8').splitlines()) if (out/name).exists() else 0 for name in names}
        write_state(out/'training_state.pt', {'format': 'center_attribute_training_v1', 'arguments': arguments,
            'config': cfg.to_dict(), 'fingerprint': fingerprint, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None, 'progress': state,
            'rng': rng_state(), 'log_counts': counts})

    def observe(images=False):
        result = validate(model, blocks, heldout, batches, raw, geometry, cameras, reference, args, state, out, images)
        phase = state['phase']
        score = (-result['render']['center_only']['source_psnr'] if result['render'] else result['centers']['world_rmse']) \
            if phase == 'center' else result['render']['full']['source_mse']
        if not math.isfinite(score):
            raise FloatingPointError('nonfinite validation score')
        previous = state['best'].get(phase, float('inf'))
        if score < previous:
            selected = out/f'codec_selected_{phase}_{state["step"]}.pt'
            save_checkpoint(selected, model, state['step'], {**record, 'phase': phase, 'selection_score': score})
            shutil.copy2(selected, out/f'codec_best_{phase}.pt')
            state['best'][phase], state['best_steps'][phase] = score, state['step']
        # Count plateau against a significant-improvement anchor, not tiny record updates.
        anchor = state.get('plateau_anchor', float('inf'))
        if score < anchor-args.validation_relative_improvement*max(abs(anchor), 1e-12) or not math.isfinite(anchor):
            state['plateau_anchor'], state['stale'] = score, 0
        else:
            state['stale'] += 1
        if phase == 'attribute' and state['initial_attribute_mse'] is None:
            state['initial_attribute_mse'] = score
        state['last_validation'] = result
        return result

    while state['phase_index'] < len(PHASES) and state['status'] == 'running':
        phase = PHASES[state['phase_index']]
        maximum = getattr(args, phase+'_steps')
        if maximum == 0:
            state['phase_index'] += 1
            continue
        state['phase'] = phase
        model.train()
        model.learned.set_phase(phase)
        optimizer = optimizer_for(model, phase, args)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            optimizer_state = None
        if not state['phase_initialized']:
            state['stale'] = 0
            state['plateau_anchor'] = float('inf')
            observe(images=state['step'] == 0)
            state['phase_initialized'] = True
            save(optimizer, boundary=state['step'] > 0)
        reason = state['end_reason'] or 'budget_cap'
        while state['phase_step'] < maximum and not state['end_reason']:
            started = time.perf_counter()
            profile = state['phase_step'] == 0 or (state['step']+1) % args.profile_every == 0
            model.zero_grad(set_to_none=True)
            if phase == 'center':
                selected = [fitted[i] for i in torch.randint(len(fitted), (args.blocks_per_batch,)).tolist()]
                group = [blocks[i] for i in selected]
                f = pad_sequence(group, batch_first=True).to(device)
                active = torch.arange(f.shape[1], device=device)[None] < torch.tensor([len(x) for x in group], device=device)[:, None]
                pred_xyz = model.learned.centers(f[..., :3], active)
                loss = center_loss(pred_xyz[active], f[..., :3][active], geometry, args.center_smoothing)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite center loss')
                loss.backward()
                stats = {}
            else:
                ids = torch.randperm(len(train_cameras))[:min(args.views_per_step, len(train_cameras))].tolist()
                task = MultiViewRenderTask([train_cameras[i] for i in ids], reference, degree, args.white_background)
                loss, stats = render_step(model, batches, geometry, task, args.render_backward, profile)
                stats.update(training_camera_ids=ids)
            modules = model.learned.module_parameters()
            gradients = {name: norm([p.grad for p in params]) for name, params in modules.items()}
            total = math.sqrt(sum(g*g for g in gradients.values()))
            if not math.isfinite(total):
                raise FloatingPointError('nonfinite gradients; optimizer not stepped')
            clip = min(1., args.clip_norm/(total+1e-6)) if args.clip_norm else 1.
            if args.clip_norm:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip_norm, error_if_nonfinite=True)
            before = {name: [p.detach().clone() for p in params] for name, params in modules.items()} if profile else {}
            optimizer.step()
            if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
                raise FloatingPointError('nonfinite parameters; recover previous training_state.pt')
            state['step'] += 1
            state['phase_step'] += 1
            state['completed'][phase] = state['phase_step']
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            row = {'step': state['step'], 'phase': phase, 'phase_step': state['phase_step'],
                'loss': float(loss.detach()), 'objective': 'world_center_distance' if phase == 'center' else 'image_mse',
                'grad_norm': total, 'module_grad_norms': gradients, 'clip_factor': clip,
                'lrs': {g['name']: g['lr'] for g in optimizer.param_groups},
                'seconds': time.perf_counter()-started, 'stats': stats}
            if profile:
                row['module_updates'] = {name: norm([p.detach()-v for p, v in zip(params, before[name])]) for name, params in modules.items()}
                print(f'{phase} {state["phase_step"]}/{maximum}: loss={row["loss"]:.6g}, grad={total:.4g}, sec={row["seconds"]:.2f}', flush=True)
            append_json(out/'loss.jsonl', row)
            images_due = state['step'] % args.render_every == 0 or state['phase_step'] == maximum
            due = state['step'] % args.validate_every == 0 or images_due
            switch = False
            if due:
                result = observe(images=images_due)
                if phase == 'center' and state['phase_step'] >= args.min_center_steps and center_ready(result, args):
                    switch, reason = True, 'center_render_gate'
                elif phase == 'attribute' and state['phase_step'] >= args.min_attribute_steps:
                    ready, _ = attribute_ready(state['initial_attribute_mse'], state['best'][phase], args)
                    if ready and state['stale'] >= args.transition_patience:
                        switch, reason = True, 'attribute_improved_then_plateaued'
                elif phase == 'joint' and state['phase_step'] >= args.min_joint_steps and state['stale'] >= args.stop_patience:
                    switch, reason = True, 'joint_validation_plateau'
                if images_due:
                    plot_run(out)
            if switch or state['phase_step'] == maximum:
                state['end_reason'] = reason
            if due or state['step'] % args.save_every == 0:
                save(optimizer)
            if switch:
                break
        # Choose by fixed validation, not the last noisy minibatch. Save exact
        # latest recovery state before rolling weights back to the selected model.
        if state['last_validation']['step'] != state['step']:
            observe(images=True)
        save(optimizer)
        selected = load_checkpoint(out/f'codec_best_{phase}.pt', device)
        model.load_state_dict(selected.state_dict())
        del selected
        save_checkpoint(out/f'codec_{phase}.pt', model, state['best_steps'][phase], {**record, 'phase': phase,
            'selected_at_step': state['best_steps'][phase]})
        passed, gain = True, None
        if phase == 'center' and args.attribute_steps:
            # The best center score was measured against the same fixed 12bit reference.
            r = state['last_validation']['render']
            passed = r is not None and r['quantized12']['source_psnr']+state['best'][phase] <= args.center_max_gap_db
        if phase == 'attribute' and args.joint_steps:
            passed, gain = attribute_ready(state['initial_attribute_mse'], state['best'][phase], args)
        append_json(out/'transitions.jsonl', {'step': state['step'], 'phase': phase,
            'reason': reason, 'passed': passed, 'selected_step': state['best_steps'][phase],
            'attribute_relative_improvement': gain})
        state['phase_index'] += 1
        state['phase_step'] = 0
        state['phase_initialized'] = False
        state['end_reason'] = None
        if not passed:
            state['status'] = 'gate_stopped'
            print(f'STOP: {phase} quality gate failed; later phases were not run.', flush=True)
        elif state['phase_index'] == len(PHASES):
            state['status'] = 'complete'
        # Boundary is recoverable with fresh optimizer in the next phase.
        save(None, boundary=True)
    if state['status'] == 'running':
        state['status'] = 'complete'
    save(None, boundary=True)
    selected_step = state['best_steps'][state['phase']]
    save_checkpoint(out/'codec.pt', model, selected_step, {**record, 'phase': state['phase'],
        'selection': 'best validation checkpoint of last executed phase', 'status': state['status']})
    summary = {**state, 'export_selected_step': selected_step, 'render_evaluated': bool(cameras), 'output': str(out)}
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plot_run(out)
    print(json.dumps({'completed': state['completed'], 'status': state['status'], 'output': str(out)}), flush=True)
