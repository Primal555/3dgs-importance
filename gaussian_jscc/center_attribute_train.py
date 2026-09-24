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
from .attribute_objective import attribute_objective
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
    p.add_argument('--extend-center-steps', type=int, default=0,
                   help='explicitly fork a center checkpoint into --out and append optimizer updates')
    p.add_argument('--continuation-lr', type=float, default=2e-5)
    p.add_argument('--continuation-end-lr', type=float, default=1e-5)
    p.add_argument('--after-center-run', help='start B/C from best center checkpoint of a completed center-only run')
    p.add_argument('--later-phase-policy', choices=['gates', 'budget'], default='gates',
                   help='budget explicitly runs all requested B/C steps without quality/plateau early exits')
    p.add_argument('--device', default='cuda')
    p.add_argument('--center-decoder-kind', choices=['transformer', 'historical_light'], default='transformer')
    p.add_argument('--center-readout-norm', choices=['layernorm', 'affine'], default='layernorm',
                   help='XYZ tap readout only; affine preserves feature mean/scale, internal Pre-LN unchanged')
    p.add_argument('--center-attention-scope', choices=['block', 'self'], default='block',
                   help='center decoder only; self retains V/output projections but disables cross-token attention')
    p.add_argument('--center-step-guard', action='store_true',
                   help='joint encoder/decoder Adam direction check and same-batch step backtracking; center phase only')
    p.add_argument('--center-max-backtracks', type=int, default=8)
    p.add_argument('--center-guard-warn-after', type=int, default=50)
    p.add_argument('--center-guard-stop-after', type=int, default=200)
    p.add_argument('--center-update-policy', choices=['adam', 'soft'], default='adam')
    p.add_argument('--center-accumulation-steps', type=int, default=1,
                   help='microbatches per center optimizer update; center-steps still counts updates')
    p.add_argument('--center-loss', choices=['distance', 'mse'], default='distance',
                   help='world-space smooth Euclidean distance or coordinate MSE (sum squared XYZ / 3N)')
    p.add_argument('--center-update-window', type=int, default=100)
    p.add_argument('--center-update-warmup', type=int, default=100)
    p.add_argument('--center-update-multiplier', type=float, default=3.)
    p.add_argument('--center-lr-schedule', choices=['constant', 'late_cosine'], default='constant')
    p.add_argument('--center-lr-decay-start', type=float, default=.5)
    p.add_argument('--center-lr-end-ratio', type=float, default=.2)
    p.add_argument('--center-probe-blocks', type=int, default=0,
                   help='fixed training-block diagnostics and sampled-block audit; zero disables')
    p.add_argument('--center-drift-every', type=int, default=0,
                   help='opt-in fixed-point before/after optimizer XYZ diagnostics; zero disables')
    for name, default in dict(center_steps=5000, attribute_steps=1000, joint_steps=1000,
            min_center_steps=500, min_attribute_steps=200, min_joint_steps=200,
            transition_patience=3, stop_patience=5, hidden=96, depth=2, decoder_depth=4,
            attention_heads=4, latent_dim=64, center_latent_dim=32, block_size=256,
            blocks_per_batch=32, render_blocks_per_batch=64, views_per_step=1, attribute_views=4,
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
    p.add_argument('--center-max-world-rmse', type=float,
                   help='explicit CPU/no-camera diagnostic gate into B; NOT a render-quality substitute')
    p.add_argument('--images', default='images')
    p.add_argument('--white-background', action='store_true')


def check_args(args):
    if args.center_accumulation_steps < 1:
        raise ValueError('center-accumulation-steps must be positive')
    if args.center_step_guard and args.center_accumulation_steps != 1:
        raise ValueError('gradient accumulation requires continuous updates, not the legacy loss-rejection guard')
    if args.center_step_guard and args.center_update_policy != 'adam':
        raise ValueError('soft continuous updates cannot be combined with the legacy rejection guard')
    if not 1 <= args.center_update_warmup <= args.center_update_window:
        raise ValueError('require 1 <= update-warmup <= update-window')
    if not math.isfinite(args.center_update_multiplier) or args.center_update_multiplier < 1:
        raise ValueError('update-multiplier must be finite and >= 1')
    if not 0 <= args.center_lr_decay_start < 1 or not 0 < args.center_lr_end_ratio <= 1:
        raise ValueError('invalid center LR schedule')
    if args.center_max_backtracks < 0:
        raise ValueError('center-max-backtracks must be nonnegative')
    if not 1 <= args.center_guard_warn_after <= args.center_guard_stop_after:
        raise ValueError('require 1 <= guard-warn-after <= guard-stop-after')
    if not args.ply or not args.out:
        raise ValueError('--ply and --out are required')
    if args.center_steps < 1 or min(args.attribute_steps, args.joint_steps) < 0:
        raise ValueError('random-start center phase must have positive budget; later budgets cannot be negative')
    if args.joint_steps and not args.attribute_steps:
        raise ValueError('joint training requires attribute adaptation first')
    if args.joint_steps and not args.source:
        raise ValueError('joint phase requires source cameras and actual image rendering')
    if args.attribute_steps and not args.source and args.center_max_world_rmse is None:
        raise ValueError('without cameras, attribute phase requires explicit --center-max-world-rmse diagnostic gate')
    if args.center_max_world_rmse is not None and (not math.isfinite(args.center_max_world_rmse) or args.center_max_world_rmse < 0):
        raise ValueError('center-max-world-rmse must be nonnegative and finite')
    positive = ('center_lr', 'attribute_lr', 'joint_center_lr', 'joint_attribute_lr', 'center_smoothing',
                'blocks_per_batch', 'render_blocks_per_batch', 'views_per_step', 'validate_every',
                'render_every', 'save_every', 'profile_every', 'validation_blocks', 'validation_views',
                'transition_patience', 'stop_patience', 'cpu_threads', 'attribute_views')
    for name in positive:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive and finite')
    for name in ('center_max_gap_db', 'clip_norm', 'min_center_steps', 'min_attribute_steps', 'min_joint_steps', 'center_probe_blocks', 'center_drift_every'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f'{name} must be nonnegative and finite')
    if args.center_drift_every and not args.center_probe_blocks:
        raise ValueError('--center-drift-every requires positive --center-probe-blocks')
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
    if validation['render'] is not None:
        return validation['render']['center_gap_db'] <= args.center_max_gap_db
    return args.center_max_world_rmse is not None and validation['centers']['world_rmse'] <= args.center_max_world_rmse


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
    for path in (out/'center_drift').glob('xyz_*.pt'):
        if int(path.stem.split('_')[1]) > step:
            (archive/'center_drift').mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(archive/'center_drift'/path.name))


def train(args):
    from .cli import seed_all, device_for
    from .allocation import scene_fingerprint
    from .rendering import load_cameras, RenderReference
    later_start = bool(getattr(args, 'after_center_run', None))
    if later_start and (args.resume or getattr(args, 'extend_center_steps', 0) or not args.out):
        raise ValueError('--after-center-run requires NEW --out and cannot combine with resume/extend')
    if later_start:
        from .center_later_phases import prepare_later_phases
        resumed = prepare_later_phases(args)
    else:
        resumed = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    extending = bool(getattr(args, 'extend_center_steps', 0))
    if extending:
        if resumed is None or not args.out:
            raise ValueError('--extend-center-steps requires --resume and a NEW --out')
        from .center_continuation import prepare_continuation
        resumed = prepare_continuation(resumed, args.resume, args.out,
                                      args.extend_center_steps, args.continuation_lr,
                                      args.continuation_end_lr)
    if resumed:
        if resumed.get('config', {}).get('center_position_layout', 'absolute') != 'absolute':
            raise ValueError('explicit block-centroid checkpoints cannot resume into pointwise absolute XYZ; start a new run')
        if resumed.get('format') != 'center_attribute_training_v2':
            raise ValueError('resume requires center_attribute_training_v2 (local attribute loss); '
                             'v1 used render-MSE in B and must resume with commit fdbbc35, not silently change objectives')
        args = SimpleNamespace(**resumed['arguments'])
        if not hasattr(args, 'center_decoder_kind'):
            args.center_decoder_kind = 'transformer'
        if not hasattr(args, 'center_probe_blocks'):
            args.center_probe_blocks = 0
        if not hasattr(args, 'center_readout_norm'):
            args.center_readout_norm = 'layernorm'
        if not hasattr(args, 'center_attention_scope'):
            args.center_attention_scope = 'block'
        if not hasattr(args, 'center_drift_every'):
            args.center_drift_every = 0
        for key, default in dict(center_step_guard=False, center_max_backtracks=8,
                                 center_guard_warn_after=50, center_guard_stop_after=200,
                                 center_update_policy='adam', center_update_window=100,
                                 center_accumulation_steps=1,
                                 center_loss='distance',
                                 center_update_warmup=100, center_update_multiplier=3.,
                                 center_lr_schedule='constant', center_lr_decay_start=.5,
                                 center_lr_end_ratio=.2).items():
            if not hasattr(args, key):
                setattr(args, key, default)
        print('Start B/C from selected BEST center weights; fresh Adam; center gate explicitly bypassed.'
              if later_start else ('Center continuation: new budget/LR; model, Adam, RNG and soft history retained.'
              if extending else 'Exact resume: stored arguments, model, optimizer and RNG win.'), flush=True)
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
        representation_dim=args.latent_dim, center_latent_dim=args.center_latent_dim,
        center_decoder_kind=args.center_decoder_kind, center_readout_norm=args.center_readout_norm,
        center_attention_scope=args.center_attention_scope)
    if not cfg.center_latent_dim:
        raise ValueError('this trainer requires positive center-latent-dim')
    model = GaussianCodec(cfg).to(device)
    print(f'Center decoder: {args.center_decoder_kind}; XYZ readout: {args.center_readout_norm}; '
          f'attention scope: {args.center_attention_scope}; internal Transformer normalization unchanged.', flush=True)
    print('Position: per-point absolute XYZ; no block-centroid branch. '
          f'Joint directional step guard: {args.center_step_guard}.', flush=True)
    print(f'Center update: {args.center_update_policy}; LR schedule: {args.center_lr_schedule}.', flush=True)
    loss_description = ('mean squared error over valid points AND XYZ axes' if args.center_loss == 'mse'
                        else 'mean smooth Euclidean distance over valid points')
    print(f'Center loss: {args.center_loss}; world coordinates; {loss_description}.', flush=True)
    print(f'Center budget: {args.center_steps} optimizer updates; '
          f'{args.center_accumulation_steps} microbatches/update, '
          f'{args.blocks_per_batch} blocks/microbatch. LR is not multiplied by accumulation.', flush=True)
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
    probes = [fitted[i] for i in spaced_indices(len(fitted), min(len(fitted), args.center_probe_blocks))] if args.center_probe_blocks else []
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
        'fitted_probe_blocks': probes,
        'center_decoder_kind': args.center_decoder_kind,
        'center_readout_norm': args.center_readout_norm,
        'center_attention_scope': args.center_attention_scope,
        'center_drift_every': args.center_drift_every,
        'position_structure': 'pointwise absolute XYZ, restored from 52edc1c; no centroid/residual split',
        'center_step_guard': args.center_step_guard,
        'center_guard_mode': 'pointwise_joint_directional' if args.center_step_guard else None,
        'center_decoder_parameters': sum(p.numel() for p in model.learned.center_decoder.parameters()),
        'center_encoder_parameters': sum(p.numel() for p in model.learned.center_encoder.parameters()),
        'center_loss_kind': args.center_loss,
        'center_loss': ('sum(||XYZ_pred-XYZ_source||_world^2)/(3*N); no smoothing or bbox denominator'
                        if args.center_loss == 'mse' else
                        'mean(sqrt(||XYZ_pred-XYZ_source||_world^2 + tau^2)-tau); no axis/bbox denominator'),
        'attribute_loss': '(physical logcov matrix MSE + centered local black/white RGB response MSE)/2; '
                          'equal mean is an engineering choice; no XYZ term, no rasterization',
        'attribute_validation': 'fixed heldout blocks, fixed directions seed+73019; not scene rendering',
        'joint_loss': 'source-render image MSE only; no parameter auxiliary',
        'communication': 'not implemented/trained in this clean-only experiment; latent dimensions are not channel uses',
        'position_delivery': 'learned only; source/12bit positions used exclusively in labelled validation diagnostics',
        'lr_schedule': {'center': args.center_lr_schedule, 'base': args.center_lr,
                        'decay_start': args.center_lr_decay_start, 'end_ratio': args.center_lr_end_ratio,
                        'later_phases': 'constant explicit LRs; fresh Adam at each phase'},
        'center_update_policy': args.center_update_policy,
        'center_accumulation': {'microbatches_per_update': args.center_accumulation_steps,
            'blocks_per_microbatch': args.blocks_per_batch,
            'step_unit': 'optimizer updates, not microbatches',
            'reduction': 'mean over all valid sampled points; duplicates count as observations'},
        'training_views': [str(c.image_name) for c in train_cameras],
        'validation_views': [str(c.image_name) for c in cameras],
        'scope': 'scene-specific; global statistics use whole PLY; A/B exclude heldout blocks; C trains all Gaussians but not heldout cameras',
        'gates': ('B/C full budgets explicitly requested; quality/plateau gates bypassed'
                  if getattr(args, 'later_phase_policy', 'gates') == 'budget' else
                  'engineering criteria, not proven optimal; budgets are caps, not mandatory phase lengths')}
    if not resumed or extending or later_start:
        (out/'training.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    state = resumed['progress'] if resumed else {
        'phase': 'center', 'phase_step': 0, 'phase_index': 0, 'step': 0, 'status': 'running',
        'phase_initialized': False, 'end_reason': None,
        'best': {}, 'best_steps': {}, 'stale': 0, 'initial_attribute_loss': None,
        'last_validation': None, 'completed': {p: 0 for p in PHASES}}
    optimizer_state = resumed['optimizer'] if resumed else None
    if resumed:
        archive_tail(out, resumed['log_counts'], state['step'])
        for phase, step in state['best_steps'].items():
            shutil.copy2(out/f'codec_selected_{phase}_{step}.pt', out/f'codec_best_{phase}.pt')
        restore_rng(resumed['rng'])

    drift = None
    if args.center_drift_every:
        from .center_drift import CenterDrift
        drift = CenterDrift(model, blocks, probes, heldout, geometry, out, args.blocks_per_batch)

    def save(optimizer, boundary=False):
        if not boundary:
            save_checkpoint(out/f'codec_{state["step"]}.pt', model, state['step'], {**record, 'phase': state['phase']})
            save_checkpoint(out/'codec_last.pt', model, state['step'], {**record, 'phase': state['phase']})
        names = ('loss.jsonl', 'validation.jsonl', 'transitions.jsonl',
                 'center_drift_validation.jsonl', 'center_drift_updates.jsonl')
        counts = {name: len((out/name).read_text(encoding='utf-8').splitlines()) if (out/name).exists() else 0 for name in names}
        write_state(out/'training_state.pt', {'format': 'center_attribute_training_v2', 'arguments': arguments,
            'config': cfg.to_dict(), 'fingerprint': fingerprint, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None, 'progress': state,
            'rng': rng_state(), 'log_counts': counts})

    def observe(images=False):
        result = validate(model, blocks, heldout, batches, raw, geometry, cameras, reference, args, state, out, images, probes)
        phase = state['phase']
        if drift is not None:
            drift.observe(state['step'])
        if phase == 'center':
            score = -result['render']['center_only']['source_psnr'] if result['render'] else result['centers']['world_rmse']
        elif phase == 'attribute':
            score = result['attributes']['loss']
        else:
            score = result['render']['full']['source_mse']
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
        if phase == 'attribute' and state['initial_attribute_loss'] is None:
            state['initial_attribute_loss'] = score
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
            observe(images=True)
            state['phase_initialized'] = True
            save(optimizer, boundary=state['step'] > 0)
        reason = state['end_reason'] or 'budget_cap'
        while state['phase_step'] < maximum and not state['end_reason']:
            started = time.perf_counter()
            if phase == 'center' and args.center_lr_schedule == 'late_cosine':
                from .center_soft_update import scheduled_center_lr
                offset = getattr(args, 'center_lr_offset', 0)
                lr = scheduled_center_lr(args.center_lr, state['phase_step']+1-offset, maximum-offset,
                                         args.center_lr_decay_start, args.center_lr_end_ratio)
                for group in optimizer.param_groups:
                    group['lr'] = lr
            guard_stopped = False
            drift_due = drift is not None and phase == 'center' and (
                state['phase_step'] == 0 or (state['step']+1) % args.center_drift_every == 0)
            profile = state['phase_step'] == 0 or (state['step']+1) % args.profile_every == 0 or drift_due
            model.zero_grad(set_to_none=True)
            contributions = {}
            if phase == 'center' and args.center_accumulation_steps > 1:
                from .center_accumulation import accumulate_center_gradients
                selections = [[fitted[i] for i in torch.randint(len(fitted), (args.blocks_per_batch,)).tolist()]
                              for _ in range(args.center_accumulation_steps)]
                loss, accumulated = accumulate_center_gradients(model, blocks, selections, geometry,
                                                                 args.center_smoothing, device, args.center_loss)
                stats = {'accumulation': accumulated}
                if args.center_probe_blocks:
                    stats['sampled_blocks'] = [i for chosen in selections for i in chosen]
                    stats['sampled_microbatches'] = selections
            elif phase in ('center', 'attribute'):
                selected = [fitted[i] for i in torch.randint(len(fitted), (args.blocks_per_batch,)).tolist()]
                group = [blocks[i] for i in selected]
                f = pad_sequence(group, batch_first=True).to(device)
                active = torch.arange(f.shape[1], device=device)[None] < torch.tensor([len(x) for x in group], device=device)[:, None]
                if phase == 'center':
                    pred_xyz = model.learned.centers(f[..., :3], active)
                    loss = center_loss(pred_xyz[active], f[..., :3][active], geometry, args.center_smoothing,
                                       kind=args.center_loss)
                    stats = {}
                    stats['accumulation'] = {'microbatches': 1, 'valid_points': sum(len(x) for x in group),
                        'points_per_microbatch': [sum(len(x) for x in group)],
                        'sampled_block_count': len(selected)}
                    if args.center_probe_blocks:
                        stats['sampled_blocks'] = selected
                else:
                    pred = model.reconstruct_clean(f, active)
                    loss, stats, contributions = attribute_objective(pred[active], f[active], geometry, model, args.attribute_views)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'nonfinite {phase} loss')
                component_gradients = {}
                if profile and contributions:
                    parameters = [p for p in model.parameters() if p.requires_grad]
                    for name, term in contributions.items():
                        grads = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
                        lookup = {id(p): g for p, g in zip(parameters, grads)}
                        component_gradients[name] = {module: norm([lookup.get(id(p)) for p in params])
                            for module, params in model.learned.module_parameters().items()}
                loss.backward()
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
            xyz_before = drift.capture(keep_latents=True) if drift_due else None
            if phase == 'center' and args.center_step_guard:
                from .center_step_guard import directional_guarded_step, update_rejection_streak
                def same_batch_loss():
                    candidate = model.learned.centers(f[..., :3], active)
                    return center_loss(candidate[active], f[..., :3][active], geometry, args.center_smoothing,
                                       kind=args.center_loss)
                guard = directional_guarded_step(optimizer, same_batch_loss, loss.detach(), args.center_max_backtracks)
                guard['mode'] = 'pointwise_joint_directional'
                warned, guard_stopped = update_rejection_streak(state, guard,
                    args.center_guard_warn_after, args.center_guard_stop_after)
                guard['rejection_streak'] = state['guard_rejection_streak']
                stats['step_guard'] = guard
                if warned:
                    print(f'WARNING: {guard["rejection_streak"]} consecutive rejected center updates.', flush=True)
                if guard_stopped:
                    print('STOP: center guard stalled; saving diagnostics, not claiming convergence.', flush=True)
            elif phase == 'center' and args.center_update_policy == 'soft':
                from .center_soft_update import soft_adam_step
                history = state.setdefault('center_update_history', {})
                stats['soft_update'] = soft_adam_step(optimizer, history,
                    window=args.center_update_window, warmup=args.center_update_warmup,
                    multiplier=args.center_update_multiplier)
                counts = state.setdefault('center_update_counts', {})
                for name, info in stats['soft_update']['groups'].items():
                    c = counts.setdefault(name, {'steps': 0, 'limited': 0, 'small_scale': 0, 'zero': 0})
                    c['steps'] += 1
                    c['limited'] += info['limited']
                    c['small_scale'] += info['scale'] < .25
                    c['zero'] += info['zero_update']
                    if c['steps'] % 100 == 0 and (c['small_scale']/c['steps'] > .1 or c['zero']/c['steps'] > .1):
                        print(f'WARNING: {name} update protection may be too strong: {c}.', flush=True)
            else:
                optimizer.step()
            if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
                raise FloatingPointError('nonfinite parameters; recover previous training_state.pt')
            state['step'] += 1
            state['phase_step'] += 1
            state['completed'][phase] = state['phase_step']
            if phase == 'center':
                work = state.setdefault('center_work', {'update_attempts': 0, 'optimizer_updates': 0,
                    'microbatches': 0, 'sampled_blocks': 0, 'processed_points': 0,
                    'counted_from_global_step': state['step']})
                work['update_attempts'] += 1
                work['optimizer_updates'] += stats.get('step_guard', {}).get('accepted', True)
                work['microbatches'] += stats['accumulation']['microbatches']
                work['sampled_blocks'] += stats['accumulation']['sampled_block_count']
                work['processed_points'] += stats['accumulation']['valid_points']
                stats['cumulative_center_work'] = dict(work)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            row = {'step': state['step'], 'phase': phase, 'phase_step': state['phase_step'],
                'loss': float(loss.detach()), 'objective': {'center': 'world_center_'+args.center_loss,
                    'attribute': 'local_attribute_response', 'joint': 'image_mse'}[phase],
                'terms': {name: float(term.detach()) for name, term in contributions.items()},
                'grad_norm': total, 'module_grad_norms': gradients, 'clip_factor': clip,
                'lrs': {g['name']: g['lr'] for g in optimizer.param_groups},
                'seconds': time.perf_counter()-started, 'stats': stats}
            if profile:
                if contributions:
                    row['objective_module_grad_norms'] = component_gradients
                row['module_updates'] = {name: norm([p.detach()-v for p, v in zip(params, before[name])]) for name, params in modules.items()}
                if drift_due:
                    drift.after_step(state['step'], xyz_before, gradients, row['module_updates'], row['lrs'], clip)
                guard = stats.get('step_guard')
                suffix = (f', post_loss={guard["loss_after"]:.6g}, step_scale={guard["scale"]:g}, '
                          f'momentum_restart={guard["momentum_restarted"]}') if guard else ''
                if 'soft_update' in stats:
                    suffix = f', lr={optimizer.param_groups[0]["lr"]:.3g}' + ''.join(
                        f', {name}_scale={info["scale"]:.3g}'
                        for name, info in stats['soft_update']['groups'].items())
                if phase == 'center':
                    suffix += (f', accum={stats["accumulation"]["microbatches"]}'
                               f', points/update={stats["accumulation"]["valid_points"]}')
                print(f'{phase} {state["phase_step"]}/{maximum}: loss={row["loss"]:.6g}, grad={total:.4g}, '
                      f'sec={row["seconds"]:.2f}{suffix}', flush=True)
            append_json(out/'loss.jsonl', row)
            images_due = state['step'] % args.render_every == 0 or state['phase_step'] == maximum
            due = state['step'] % args.validate_every == 0 or images_due or guard_stopped
            switch = False
            if due:
                result = observe(images=images_due or guard_stopped)
                if phase == 'center' and state['phase_step'] >= args.min_center_steps and center_ready(result, args):
                    switch = True
                    reason = 'center_render_gate' if result['render'] else 'center_world_rmse_diagnostic_gate'
                elif phase == 'attribute' and getattr(args, 'later_phase_policy', 'gates') == 'gates' and state['phase_step'] >= args.min_attribute_steps:
                    ready, _ = attribute_ready(state['initial_attribute_loss'], state['best'][phase], args)
                    if ready and state['stale'] >= args.transition_patience:
                        switch, reason = True, 'attribute_improved_then_plateaued'
                elif phase == 'joint' and getattr(args, 'later_phase_policy', 'gates') == 'gates' and state['phase_step'] >= args.min_joint_steps and state['stale'] >= args.stop_patience:
                    switch, reason = True, 'joint_validation_plateau'
                if images_due:
                    plot_run(out)
            if guard_stopped:
                state['status'] = 'guard_stalled'
                reason = 'center_guard_stalled'
            if switch or state['phase_step'] == maximum or guard_stopped:
                state['end_reason'] = reason
            if due or state['step'] % args.save_every == 0:
                save(optimizer)
            if switch or guard_stopped:
                break
        # Choose by fixed validation, not the last noisy minibatch. Save exact
        # latest recovery state before rolling weights back to the selected model.
        if state['last_validation']['step'] != state['step']:
            observe(images=True)
        save(optimizer)
        if phase == 'center' and (drift is not None or args.center_step_guard or args.center_update_policy == 'soft'):
            # Retain matching LAST weights + Adam before best-weight selection
            # and final state saves replace the optimizer with None.
            shutil.copy2(out/'training_state.pt', out/'training_state_last_center.pt')
        selected = load_checkpoint(out/f'codec_best_{phase}.pt', device)
        model.load_state_dict(selected.state_dict())
        del selected
        save_checkpoint(out/f'codec_{phase}.pt', model, state['best_steps'][phase], {**record, 'phase': phase,
            'selected_at_step': state['best_steps'][phase]})
        passed, gain = True, None
        if phase == 'center' and args.attribute_steps:
            # The best center score was measured against the same fixed 12bit reference.
            r = state['last_validation']['render']
            passed = (r['quantized12']['source_psnr']+state['best'][phase] <= args.center_max_gap_db) if r else \
                (args.center_max_world_rmse is not None and state['best'][phase] <= args.center_max_world_rmse)
        if phase == 'attribute' and args.joint_steps and getattr(args, 'later_phase_policy', 'gates') == 'gates':
            passed, gain = attribute_ready(state['initial_attribute_loss'], state['best'][phase], args)
        if state['status'] == 'guard_stalled':
            passed = False
        append_json(out/'transitions.jsonl', {'step': state['step'], 'phase': phase,
            'reason': reason, 'passed': passed, 'selected_step': state['best_steps'][phase],
            'selection_metric': {'center': 'center_only_psnr_or_world_rmse',
                                 'attribute': 'heldout_local_attribute_loss', 'joint': 'full_render_mse'}[phase],
            'attribute_relative_improvement': gain})
        state['phase_index'] += 1
        state['phase_step'] = 0
        state['phase_initialized'] = False
        state['end_reason'] = None
        if state['status'] == 'guard_stalled':
            print('Center guard stalled; later phases were not run.', flush=True)
        elif not passed:
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
    if args.center_step_guard:
        from .center_step_guard import guard_summary
        path = out/'loss.jsonl'
        summary['center_step_guard'] = guard_summary(
            [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else [])
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plot_run(out)
    print(json.dumps({'completed': state['completed'], 'status': state['status'], 'output': str(out)}), flush=True)
