"""Train clean Gaussian representations first, then fixed-q3 JSCC adapters.

The existing spatial-response objective remains the default; teacher-axis is an
opt-in representation-only position experiment. Scene renders are validation.
"""
import json
import math
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .codec import CodecConfig, GaussianCodec
from .data import read_ply, Geometry, morton_order, to_features, fit_feature_statistics
from .learned_train import bootstrap_split
from .optimization import preserved_rng
from .render_validation import append_json
from .representation_validation import paths, objective, validate_blocks, validate_images
from .transport import save_checkpoint, load_checkpoint

PHASES = ('representation', 'adapter', 'joint')


def add_parser(sub):
    p = sub.add_parser('train-representation', description=__doc__)
    p.set_defaults(func=train)
    p.add_argument('--ply')
    p.add_argument('--out')
    choice = p.add_mutually_exclusive_group()
    choice.add_argument('--init', help='separated codec weights only; fresh optimizer/schedule')
    choice.add_argument('--resume', help='exact continuation from training_state.pt, stored arguments win')
    p.add_argument('--source', help='enable fixed-camera PSNR/SSIM and PNGs (requires CUDA rasterizer)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--representation-steps', type=int, default=5000)
    p.add_argument('--adapter-steps', type=int, default=0)
    p.add_argument('--joint-steps', type=int, default=0)
    p.add_argument('--latent-dim', type=int, default=64)
    p.add_argument('--communication-depth', type=int, default=2)
    p.add_argument('--hidden', type=int, default=96)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--decoder-depth', type=int, default=4)
    p.add_argument('--attention-heads', type=int, default=4)
    p.add_argument('--block-size', type=int, default=256)
    p.add_argument('--blocks-per-batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--clean-weight', type=float, default=1., help='explicit joint clean-path engineering coefficient')
    p.add_argument('--spatial-fine-weight', type=float, default=1.)
    p.add_argument('--position-objective', choices=['scene-scale', 'teacher-axis'], default='scene-scale')
    p.add_argument('--axis-floor-percentile', type=float, default=1., help='teacher-axis: percentile across all source axis scales, fitted once')
    p.add_argument('--axis-floor-world', type=float, help='optional explicit teacher-axis scale floor in scene units')
    p.add_argument('--local-response-views', type=int, default=4)
    p.add_argument('--channel', choices=['none', 'awgn'], default='none')
    p.add_argument('--snr', type=float, default=10.)
    p.add_argument('--power-floor', type=float, default=.01)
    p.add_argument('--clip-norm', type=float, default=0., help='0 disables clipping; otherwise explicit global threshold')
    p.add_argument('--validate-every', type=int, default=100)
    p.add_argument('--render-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--profile-every', type=int, default=10)
    p.add_argument('--validation-blocks', type=int, default=16)
    p.add_argument('--validation-views', type=int, default=4)
    p.add_argument('--validation-region-size', type=int, default=512)
    p.add_argument('--min-clean-psnr', type=float, help='required with cameras before communication training; source-render PSNR')
    p.add_argument('--max-clean-loss', type=float, help='CPU/no-camera diagnostic gate, NOT a render quality guarantee')
    p.add_argument('--max-adapter-psnr-drop', type=float, help='required with cameras before joint tuning')
    p.add_argument('--max-adapter-loss-ratio', type=float, help='CPU/no-camera gate before joint tuning')
    p.add_argument('--resolution', type=int, default=2)
    p.add_argument('--images', default='images')
    p.add_argument('--white-background', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu-threads', type=int, default=4)


def check_args(args):
    # Older representation training states keep their original objective on resume.
    for key, value in dict(position_objective='scene-scale', axis_floor_percentile=1., axis_floor_world=None).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if args.position_objective == 'teacher-axis':
        if args.adapter_steps or args.joint_steps:
            raise ValueError('teacher-axis is a representation-only experiment; disable adapter/joint steps')
        if args.spatial_fine_weight != 1.:
            raise ValueError('teacher-axis replaces the old position loss; do not adjust spatial-fine-weight')
        if not math.isfinite(args.axis_floor_percentile) or not 0 <= args.axis_floor_percentile <= 100:
            raise ValueError('axis-floor-percentile must be in [0,100]')
        if args.axis_floor_world is not None and (not math.isfinite(args.axis_floor_world) or args.axis_floor_world <= 0):
            raise ValueError('axis-floor-world must be positive and finite')
    if not args.ply or not args.out:
        raise ValueError('--ply and --out required unless --resume is supplied')
    counts = [getattr(args, p+'_steps') for p in PHASES]
    if min(counts) < 0 or sum(counts) < 1:
        raise ValueError('phase lengths must be nonnegative and at least one positive')
    if min(args.latent_dim, args.blocks_per_batch, args.validate_every, args.render_every,
           args.save_every, args.profile_every, args.validation_blocks, args.validation_views,
           args.cpu_threads, args.local_response_views) < 1:
        raise ValueError('dimensions, counts and intervals must be positive')
    for name in ('lr', 'power_floor'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive and finite')
    for name in ('clean_weight', 'spatial_fine_weight', 'clip_norm'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if args.joint_steps and args.clean_weight == 0:
        raise ValueError('joint phase must retain a positive clean reconstruction objective')
    if not math.isfinite(args.snr):
        raise ValueError('SNR must be finite')
    for name in ('min_clean_psnr', 'max_clean_loss', 'max_adapter_psnr_drop', 'max_adapter_loss_ratio'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or (name != 'min_clean_psnr' and value < 0)):
            raise ValueError(f'invalid gate {name}')
    if not args.init and not args.representation_steps:
        raise ValueError('random start requires representation training')
    if not args.init and args.joint_steps and not args.adapter_steps:
        raise ValueError('random start requires adapter warmup before joint tuning')
    if args.adapter_steps or args.joint_steps:
        gate = args.min_clean_psnr if args.source else args.max_clean_loss
        if gate is None:
            raise ValueError('communication requires an explicit --min-clean-psnr (camera) or --max-clean-loss (diagnostic) gate')
    if args.joint_steps:
        gate = args.max_adapter_psnr_drop if args.source else args.max_adapter_loss_ratio
        if gate is None:
            raise ValueError('joint tuning requires an explicit adapter quality gate')


def norm(values):
    values = [v for v in values if v is not None]
    # One host synchronization per module, not one per parameter tensor.
    return float(torch.stack([v.detach().double().square().sum() for v in values]).sum().sqrt()) if values else 0.


def phase_objective(model, features, active, geometry, args, phase, directions):
    if phase == 'representation':
        clean = model.reconstruct_clean(features, active)
        if getattr(args, 'position_objective', 'scene-scale') == 'teacher-axis':
            loss, stats, components = objective(clean[active], features[active], geometry, model, args, directions, return_components=True)
            # Profile actual /3 objective contributions, not just the combined loss.
            return loss, components, stats
        clean_loss, stats = objective(clean[active], features[active], geometry, model, args, directions)
        return clean_loss, {'clean': clean_loss}, stats
    clean, comm, y, recovered = paths(model, features, active, args.snr, args.channel)
    comm_loss, stats = objective(comm[active], features[active], geometry, model, args, directions)
    stats['latent_mse'] = float((y[active]-recovered[active]).detach().square().mean())
    if phase == 'adapter':
        # Representation decoder has requires_grad=False, but autograd through
        # its input remains enabled. NO no_grad() around the communication path.
        return comm_loss, {'communication': comm_loss}, stats
    clean_loss, _ = objective(clean[active], features[active], geometry, model, args, directions)
    terms = {'communication': comm_loss, 'weighted_clean': args.clean_weight*clean_loss}
    return sum(terms.values()), terms, stats


def rng_state():
    state = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]]}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('exact resume requires the same number of visible CUDA devices')
        torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python'])
    s = state['numpy']
    np.random.set_state((s[0], np.array(s[1], dtype=np.uint32), s[2], s[3], s[4]))


def write_state(path, state):
    temporary = Path(str(path)+'.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def recover_outputs(out, step, best_steps):
    """Archive uncheckpointed observations; never erase interrupted-run evidence."""
    archive = out/('interrupted_tail_'+str(time.time_ns()))
    for name in ('loss.jsonl', 'validation.jsonl', 'render_validation.jsonl', 'transitions.jsonl'):
        path = out/name
        if not path.exists():
            continue
        keep = []
        changed = False
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                changed = True
                continue
            if row['step'] <= step:
                keep.append(line)
            else:
                changed = True
        if changed:
            archive.mkdir(exist_ok=True)
            shutil.copy2(path, archive/name)
            path.write_text('\n'.join(keep)+'\n', encoding='utf-8')
    images = out/'images'
    if images.exists():
        for child in images.iterdir():
            if child.is_dir() and child.name.isdigit() and int(child.name) > step:
                (archive/'images').mkdir(parents=True, exist_ok=True)
                shutil.move(str(child), str(archive/'images'/child.name))
    for phase, selected in best_steps.items():
        if selected is not None:
            source = out/f'codec_selected_{phase}_{selected}.pt'
            if not source.exists():
                raise FileNotFoundError(f'Exact resume needs saved selected checkpoint: {source}')
            shutil.copy2(source, out/f'codec_best_{phase}.pt')
    if archive.exists():
        print(f'Uncheckpointed logs/images preserved in {archive}', flush=True)


def transition_gate(phase, block_result, render_result, args):
    if render_result is not None:
        clean = render_result['clean']['source_psnr']
        passed = clean >= args.min_clean_psnr
        details = {'clean_source_psnr': clean, 'minimum': args.min_clean_psnr}
        if phase == 'joint':
            drop = render_result['communication_psnr_drop']
            passed &= drop <= args.max_adapter_psnr_drop
            details.update(adapter_psnr_drop=drop, maximum_drop=args.max_adapter_psnr_drop)
    else:
        clean = block_result['clean']['loss']
        passed = clean <= args.max_clean_loss
        details = {'clean_heldout_loss': clean, 'maximum': args.max_clean_loss,
                   'warning': 'no camera/render quality evidence'}
        if phase == 'joint':
            ratio = block_result['communication']['loss']/max(clean, 1e-12)
            passed &= ratio <= args.max_adapter_loss_ratio
            details.update(adapter_loss_ratio=ratio, maximum_ratio=args.max_adapter_loss_ratio)
    return {'entering': phase, 'passed': bool(passed), **details}


def train(args):
    from .cli import seed_all, device_for
    from .allocation import scene_fingerprint
    resumed = None
    if args.resume:
        resumed = torch.load(args.resume, map_location='cpu', weights_only=True)
        if resumed.get('format') != 'representation_training_v1':
            raise ValueError('not a representation training state')
        # Explicitly use stored immutable arguments, not silent CLI overrides.
        from types import SimpleNamespace
        args = SimpleNamespace(**resumed['arguments'])
        print('Exact resume: using stored arguments, optimizer and RNG; CLI overrides are ignored.', flush=True)
    check_args(args)
    device = device_for(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(args.cpu_threads)
    if args.source and device.type != 'cuda':
        raise ValueError('scene render validation needs CUDA; omit --source for CPU structural diagnostics')
    out = Path(args.out).resolve()
    if not resumed and out.exists() and any(p.name not in ('console.log', 'run.pid') for p in out.iterdir()):
        raise FileExistsError(f'Use a new output directory, or --resume: {out}')
    seed_all(args.seed)
    original, degree = read_ply(args.ply)
    fingerprint = scene_fingerprint(original)
    axis_floor = None
    if args.position_objective == 'teacher-axis':
        from .axis_position import fit_axis_floor
        axis_floor = fit_axis_floor(original, args.axis_floor_percentile, args.axis_floor_world)
        if resumed and resumed.get('axis_floor') is not None:
            if axis_floor['world'] != resumed['axis_floor']['world']:
                raise ValueError('axis supervision floor changed on resume')
            axis_floor = resumed['axis_floor']
        args.axis_floor_world = axis_floor['world']
        print('Teacher-axis supervision: '+json.dumps(axis_floor), flush=True)
    if resumed:
        if fingerprint != resumed['fingerprint']:
            raise ValueError('resume PLY differs from original training scene')
        model = GaussianCodec(CodecConfig.from_dict(resumed['config'])).to(device)
        model.load_state_dict(resumed['model'])
    elif args.init:
        model = load_checkpoint(args.init, device)
        if not model.cfg.representation_dim or model.cfg.center_latent_dim or model.cfg.sh_degree != degree:
            raise ValueError('initializer must be a separated codec with matching SH degree')
        print('Weights/statistics initialization only: optimizer and schedule start fresh; checkpoint architecture wins.', flush=True)
    else:
        cfg = CodecConfig(architecture='learned_split_logcov', context_mode='multiscale_self',
                          encoder_attention='geometric_point', decoder_attention='transformer_trunk',
                          sh_degree=degree, hidden=args.hidden, depth=args.depth, decoder_depth=args.decoder_depth,
                          attention_heads=args.attention_heads, block_size=args.block_size,
                          representation_dim=args.latent_dim, communication_depth=args.communication_depth,
                          power_floor=args.power_floor)
        model = GaussianCodec(cfg).to(device)
        fit_feature_statistics(original, model)
    if model.cfg.rates != (0, 8, 16, 32):
        raise ValueError('this isolated experiment requires rates=(0,8,16,32) and fixed q3')
    geometry = Geometry.fit(original[:, :3], model.cfg.morton_bits)
    order = torch.from_numpy(morton_order(geometry.quantize(original[:, :3]).numpy()).astype(np.int64))
    raw = original[order]
    del original
    with torch.no_grad():
        blocks = [to_features(chunk.to(device), geometry, model)[0].cpu()
                  for chunk in raw.split(model.cfg.block_size)]
    training, heldout = bootstrap_split(len(raw), model.cfg.block_size, args.validation_blocks,
                                        args.validation_region_size)
    if set(training) & set(heldout):
        raise ValueError('strict disjoint training/heldout blocks required')
    cameras, reference = [], None
    if args.source:
        from .rendering import load_cameras, RenderReference
        from .render_objective import spaced_indices
        all_cameras = load_cameras(args.source, args.resolution, args.white_background, args.images, 'test')
        cameras = [all_cameras[i] for i in spaced_indices(len(all_cameras), args.validation_views)]
        reference = RenderReference(raw, degree, args.white_background)
    out.mkdir(parents=True, exist_ok=True)
    arguments = {k: v for k, v in vars(args).items() if k != 'func'}
    arguments.update(out=str(out), ply=str(Path(args.ply).resolve()))
    if args.source:
        arguments['source'] = str(Path(args.source).resolve())
    record = {'arguments': arguments, 'config': model.cfg.to_dict(), 'fingerprint': fingerprint,
              'training_blocks': training, 'heldout_blocks': heldout, 'geometry': geometry.to_dict(),
              'validation_views': [str(getattr(c, 'image_name', i)) for i, c in enumerate(cameras)],
              'objective': 'existing spatial_response_v3_logcov; unchanged in all phases',
              'phase_design': 'clean representation -> frozen-representation adapters -> joint with clean auxiliary',
              'scope': 'scene-fitted codec, not unseen-scene generalization; global feature stats use input PLY',
              'channel_budget': 'fixed q3=32 complex symbols per Gaussian; clean latent is NOT a transmitted rate',
              'metadata': 'existing reliable global bbox, model statistics and packet/tier syntax; no per-point XYZ side stream',
              'render_training': False, 'lr_schedule': 'constant; fresh Adam at each phase boundary',
              'gate_note': 'thresholds are explicit user/engineering criteria, not proven optimal values'}
    if axis_floor is not None:
        record.update(objective='teacher_axis_pseudo_huber_v1', axis_floor=axis_floor,
                      loss_design='(sqrt(1+sum((R_source.T * delta_xyz / clamped_source_axes)^2))-1 + logcov_shape + centered_appearance)/3',
                      old_position_terms='diagnostic only; no contribution to training objective',
                      checkpoint_comparison='raw loss not comparable to scene-scale runs; compare fixed render and world/relative errors')
    if not resumed:
        (out/'training.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    completed = resumed['completed'] if resumed else {p: 0 for p in PHASES}
    best = resumed['best'] if resumed else {p: float('inf') for p in PHASES}
    best_steps = resumed['best_steps'] if resumed else {p: None for p in PHASES}
    step = sum(completed.values())
    if resumed:
        recover_outputs(out, step, best_steps)
    model.train()

    def validate(phase, render=False):
        b = validate_blocks(model, blocks, heldout, geometry, args, step, phase, out)
        r = validate_images(model, blocks, raw, geometry, cameras, reference, args, step, phase, out) if cameras and render else None
        score = b['clean' if phase == 'representation' else 'communication']['loss']
        if score < best[phase]:
            best[phase] = score
            best_steps[phase] = step
            selected = out/f'codec_selected_{phase}_{step}.pt'
            save_checkpoint(selected, model, step, {**record, 'phase': phase, 'selection': 'heldout spatial response, not render PSNR'})
            shutil.copy2(selected, out/f'codec_best_{phase}.pt')
        return b, r

    def save(phase, optimizer):
        save_checkpoint(out/f'codec_{step}.pt', model, step, {**record, 'phase': phase})
        save_checkpoint(out/'codec.pt', model, step, {**record, 'phase': phase})
        write_state(out/'training_state.pt', {'format': 'representation_training_v1', 'arguments': arguments,
                    'config': model.cfg.to_dict(), 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'phase': phase, 'completed': completed, 'best': best, 'best_steps': best_steps,
                    'fingerprint': fingerprint, 'axis_floor': axis_floor, 'rng': rng_state()})

    if resumed:
        restore_rng(resumed['rng'])
    stopped = None
    for phase in PHASES:
        maximum = getattr(args, phase+'_steps')
        if completed[phase] >= maximum:
            continue
        model.learned.set_phase(phase)
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        if resumed and resumed['phase'] == phase:
            optimizer.load_state_dict(resumed['optimizer'])
        if completed[phase] == 0:
            b, r = validate(phase, render=True)
            if phase != 'representation':
                gate = transition_gate(phase, b, r, args)
                append_json(out/'transitions.jsonl', {'step': step, **gate})
                if not gate['passed']:
                    stopped = gate
                    save_checkpoint(out/'codec.pt', model, step, {**record, 'phase': phase, 'gate_failed': gate})
                    print(f'STOP before {phase}: quality gate not met; no communication/joint updates performed.', flush=True)
                    break
            # Save a recoverable boundary, even if the first training step fails.
            save(phase, optimizer)
        for local_step in range(completed[phase]+1, maximum+1):
            started = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            selected = [training[i] for i in torch.randint(len(training), (args.blocks_per_batch,)).tolist()]
            group = [blocks[i] for i in selected]
            f = pad_sequence(group, batch_first=True).to(device)
            lengths = torch.tensor([len(v) for v in group], device=device)
            active = torch.arange(f.shape[1], device=device)[None] < lengths[:, None]
            directions = torch.randn(args.local_response_views, 3, device=device)
            # Also clear gradients on modules frozen at the phase boundary.
            model.zero_grad(set_to_none=True)
            loss, terms, stats = phase_objective(model, f, active, geometry, args, phase, directions)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite objective in {phase}; resume last saved training_state.pt')
            profile = local_step == 1 or (step+1) % args.profile_every == 0
            modules = model.learned.module_parameters()
            diagnostics = {}
            if profile:
                trainable = [p for p in model.parameters() if p.requires_grad]
                for name, term in terms.items():
                    gradients = torch.autograd.grad(term, trainable, retain_graph=True, allow_unused=True)
                    lookup = {id(p): g for p, g in zip(trainable, gradients)}
                    diagnostics[name] = {module: norm([lookup.get(id(p)) for p in params]) for module, params in modules.items()}
            loss.backward()
            grad_norms = {name: norm([p.grad for p in params]) for name, params in modules.items()}
            total_norm = math.sqrt(sum(g*g for g in grad_norms.values()))
            if not math.isfinite(total_norm):
                raise FloatingPointError(f'nonfinite gradients in {phase}; no optimizer update performed')
            clip_factor = min(1., args.clip_norm/(total_norm+1e-6)) if args.clip_norm else 1.
            if args.clip_norm:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip_norm, error_if_nonfinite=True)
            before = {name: [p.detach().clone() for p in params] for name, params in modules.items()} if profile else {}
            optimizer.step()
            if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
                raise FloatingPointError('nonfinite parameters after update; resume last saved checkpoint')
            completed[phase] = local_step
            step += 1
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            row = {'step': step, 'phase': phase, 'phase_step': local_step, 'loss': float(loss.detach()),
                   'terms': {k: float(v.detach()) for k, v in terms.items()}, 'grad_norm': total_norm,
                   'module_grad_norms': grad_norms, 'clip_factor': clip_factor, 'lr': args.lr,
                   'seconds': time.perf_counter()-started, 'reconstruction': stats,
                   'peak_allocated_mib': torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None}
            if profile:
                row['objective_module_grad_norms'] = diagnostics
                row['module_updates'] = {name: {'norm': norm([p.detach()-v for p, v in zip(params, before[name])]),
                    'relative': norm([p.detach()-v for p, v in zip(params, before[name])])/max(norm(before[name]), 1e-12)}
                    for name, params in modules.items()}
            append_json(out/'loss.jsonl', row)
            if profile:
                print(f'{phase} {local_step}/{maximum}: loss={row["loss"]:.6g}, grad={total_norm:.4g}, sec={row["seconds"]:.2f}', flush=True)
            render_due = step % args.render_every == 0 or local_step == maximum
            if step % args.validate_every == 0 or render_due:
                validate(phase, render=render_due)
                if render_due:
                    from .representation_plots import plot_run
                    with preserved_rng(device):
                        plot_run(out)
            if step % args.save_every == 0 or local_step == maximum:
                save(phase, optimizer)
        save_checkpoint(out/f'codec_{phase}.pt', model, step, {**record, 'phase': phase})
    summary = {'completed': completed, 'step': step, 'stopped_by_gate': stopped,
               'render_evaluated': bool(cameras), 'output': str(out)}
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    from .representation_plots import plot_run
    plot_run(out)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
