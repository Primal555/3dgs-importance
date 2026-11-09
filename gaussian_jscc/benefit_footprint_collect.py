"""Frozen paired-label collection with resumable immutable shards."""
import hashlib
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from .benefit_footprint_features import FootprintWindow, verify_footprints
from .benefit_rd_choice_data import load_trial
from .budgeted_allocation_inputs import load_inputs
from .budgeted_allocation_state import load_checkpoint
from .cli import seed_all
from .data import read_ply
from .full_history_spatial_replay import FullHistorySpatialReplay
from .model_factory import build_codec
from .modular_checkpoint import capture_rng, restore_rng
from .modular_experiment import json_write
from .optimization import preserved_rng
from .allocation_mirror import ViewWindowScheduler
from .response_value import ResponseMirrorTable
from .response_value_window import ResponseScheduler
from .render_objective import spaced_indices
from .spatial_benefit_allocation import parser as spatial_parser, preflight
from .successive_multiscene_inputs import make_scene
from .transport import model_id


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def verify_source_artifact(saved, artifact, pipeline_settings):
    """Bind the reference/split to the predictor's frozen training normalizers."""
    source = artifact['settings']
    fields = ('seed', 'beta', 'window_views', 'calibration_views', 'aligned_spatial_inputs')
    if any(k not in pipeline_settings or source.get(k) != pipeline_settings[k] for k in fields):
        raise ValueError('pretraining/online holdout or camera settings differ')
    if artifact['reference'].get('beta') != pipeline_settings['beta']:
        raise ValueError('artifact reference coefficient differs from original pipeline')
    initial = saved['initial']['learner']['network']
    pretrained = artifact['learner']['network']
    digest = hashlib.sha256()
    for key in ('input_mean', 'input_scale', 'context_mean', 'context_scale',
                'state_mean', 'state_scale', 'output_scale'):
        if not torch.equal(initial[key], pretrained[key]):
            raise ValueError(f'initializer normalizer differs from original predictor: {key}')
        digest.update(key.encode()); digest.update(initial[key].contiguous().numpy().tobytes())
    embedded = saved['initial'].get('settings', {}).get('evaluator_configuration')
    if embedded and any(embedded.get(k) != pipeline_settings[k] for k in fields):
        raise ValueError('comparison original settings differ from selected artifact')
    return dict(normalizer_sha256=digest.hexdigest(), holdout_seed=source['seed'],
                reference_sha256=hashlib.sha256(json.dumps(artifact['reference'], sort_keys=True).encode()).hexdigest())


def collect(args, *, window_factory=FootprintWindow, label_version='candidate_footprint_labels_v1', initial_bundle=None):
    out = Path(args.out).resolve()
    if out.exists() and not args.resume:
        raise FileExistsError('use a new collection output or --resume')
    saved = initial_bundle if initial_bundle is not None else load_trial(
        args.comparison, pipeline_root=args.pipeline_root, require_replay=False)
    original = saved['initial']['settings']
    pipeline = Path(args.pipeline_root) if args.pipeline_root else Path(original['source_checkpoint']).parents[3]
    payload = (pipeline/'benefit_network.pt').read_bytes()
    artifact = load_checkpoint(io.BytesIO(payload))
    source_args = artifact['settings']
    plan_payload = (pipeline/'plan.json').read_bytes()
    pipeline_settings = json.loads(plan_payload)['settings']
    source_identity = verify_source_artifact(saved, artifact, pipeline_settings)
    if not source_args.get('aligned_spatial_inputs', False):
        raise ValueError('candidate-region controls require the aligned source experiment')
    declared = dict(train_records=args.train_records, validation_records=args.validation_records,
        label_groups=args.label_groups, candidate_batch=args.candidate_batch, scene=args.scene,
        source_initial_sha256=saved['initial_sha256'], source_artifact_sha256=hashlib.sha256(payload).hexdigest(),
        source_plan_sha256=hashlib.sha256(plan_payload).hexdigest(),
        codec_run=str(Path(args.codec).resolve()), comparison=str(Path(args.comparison).resolve()))
    if label_version != 'candidate_footprint_labels_v1': declared['label_version'] = label_version
    if (out/'complete.json').is_file() and args.resume:
        existing = json.loads((out/'metadata.json').read_text(encoding='utf8'))
        if any(existing.get(k) != v for k, v in declared.items()):
            raise ValueError('completed collection settings or identities differ')
        return json.loads((out/'complete.json').read_text(encoding='utf8'))
    # Retain the original whole-block holdout: pretrained weights never saw it.
    options = spatial_parser().parse_args(['--run', args.codec, '--scene', args.scene,
        '--out', str(out), '--beta', str(source_args['beta'])])
    for key in vars(options):
        if key in source_args and key not in ('run', 'out', 'scene', 'resume', 'dry_run', 'manifest', 'device'):
            setattr(options, key, source_args[key])
    options.device = args.device
    options.manifest = args.manifest
    options.label_groups = args.label_groups
    options.candidate_batch = args.candidate_batch
    options.renderer_backend = torch.device(args.device).type
    seed_all(options.seed)
    torch.set_num_threads(args.cpu_threads)
    codec_state, cfg, spec, old, scene_options, identity = load_inputs(options)
    model = build_codec(cfg).to(args.device)
    model.load_state_dict(codec_state['state_dict'])
    model.eval().requires_grad_(False)
    model.set_training_snrs([10.])
    codec_identity = model_id(model)
    if artifact['codec_id'] != codec_identity or tuple(cfg.rates) != (0, 8, 24, 48):
        raise ValueError('frozen codec identity or rates differ from original evaluator')
    raw, degree = read_ply(spec['ply'])
    scene = make_scene(spec, raw, degree, model, scene_options, out, old=old)
    model.set_scene_geometry(scene.geometry)
    if artifact['scene_fingerprint'] != scene.fingerprint:
        raise ValueError('source scene differs from original evaluator')
    calibration = set(spaced_indices(len(scene.cameras), options.calibration_views))
    cameras = [c for i, c in enumerate(scene.cameras) if i not in calibration]
    table = ResponseMirrorTable(len(scene.raw), max_log_odds=options.max_log_odds).to(args.device)
    scheduler = ResponseScheduler(scene.group_ids, seed=options.seed)
    views = ViewWindowScheduler([str(c.image_name) for c in cameras], options.seed+9001)
    reference = artifact['reference']
    replay = FullHistorySpatialReplay(out/'replay', seed=options.seed+123,
        max_groups=args.train_records, validation_groups=args.validation_records, cache_mib=32)
    window = window_factory(model, scene, table, scheduler, reference, options)
    metadata = dict(declared, version=label_version, config=saved['config'],
        source_identity=source_identity,
        seed=original['seed'], collection_seed=options.seed, codec_id=codec_identity,
        reference=reference, scene_fingerprint=scene.fingerprint,
        window_views=options.window_views, cameras=[str(c.image_name) for c in cameras],
        feature_layout='4 states x 2 regions x 15 channels x 9 x 9',
        renderer_backend=options.renderer_backend)
    step = 0
    if args.resume:
        state = load_checkpoint(out/'latest.pt')
        if state['metadata'] != metadata:
            raise ValueError('collection resume settings or identities differ')
        replay.load_state_dict(state['replay'])
        scheduler.load_state_dict(state['scheduler']); views.load_state_dict(state['views'])
        ws = dict(state['window'], static=load_checkpoint(out/'static.pt'))
        window.load_state_dict(ws)
        step = state['step']
        restore_rng(state['rng'])
    else:
        print('检查配对编解码与局部渲染器。', flush=True)
        with preserved_rng(torch.device(args.device)):
            result = preflight(model, scene, cameras[0], options, reference)
            result.update(verify_footprints(args.device))
        out.mkdir(parents=True, exist_ok=True)
        json_write(out/'preflight.json', result)
        json_write(out/'metadata.json', metadata)
        atomic_save(saved['initial'], out/'initial.pt')
    started = time.perf_counter()
    while len(replay.training) < args.train_records or len(replay.validation) < args.validation_records:
        step += 1
        if step > args.max_collect_windows:
            raise RuntimeError('collection window limit reached; inspect counts and resume with a larger limit')
        indices = views.next(options.window_views)
        result = window.collect(step, [cameras[i] for i in indices], SimpleNamespace(scales=None), replay, True)
        if not (out/'static.pt').exists():
            atomic_save(window.static, out/'static.pt')
        state = dict(metadata=metadata, step=step, replay=replay.state_dict(),
            scheduler=scheduler.state_dict(), views=views.state_dict(),
            window={k: v for k, v in window.state_dict().items() if k != 'static'}, rng=capture_rng())
        atomic_save(state, out/'latest.pt')
        print(f'采集窗口 {step}：训练={len(replay.training)}/{args.train_records}，'
              f'验证={len(replay.validation)}/{args.validation_records}，'
              f'teacher={result["metrics"]["teacher_seconds"]:.2f}s，累计={time.perf_counter()-started:.1f}s。', flush=True)
    if model_id(model) != codec_identity:
        raise RuntimeError('frozen codec changed')
    replay.freeze_validation()
    json_write(out/'data_manifest.json', dict(replay_root=str(replay.root),
        training=replay.training[:args.train_records], validation=replay.validation[:args.validation_records]))
    result = dict(status='complete', windows=step, train_records=args.train_records,
                  validation_records=args.validation_records, frozen_codec_verified=True)
    json_write(out/'complete.json', result)
    return result
