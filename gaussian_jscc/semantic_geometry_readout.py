"""Continue a completed split128 with full latent access for geometry."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from .cli import device_for, seed_all
from .data import read_ply
from .modular_checkpoint import restore_rng
from .modular_config import ModularCodecConfig
from .modular_experiment import json_write
from .optimization import preserved_rng
from .perceptual_metrics import LPIPSEvaluator
from .refinement_views import RefinementViewCycle
from .render_validation import append_json
from .semantic_attribute_codec import FULL_GEOMETRY, SemanticAttributeCodec, fit_loss_scales
from .semantic_attribute_experiment import FAMILY, render_update, review, save_checkpoint
from .semantic_attribute_metrics import evaluate
from .successive_multiscene_inputs import make_scene


def widen(source, config):
    model = SemanticAttributeCodec(FULL_GEOMETRY, config)
    key = 'semantic_decoder.branches.geometry.input.weight'
    previous = source[key]
    expanded = previous.new_zeros((config.hidden, 128))
    expanded[:, :16] = previous[:, :16]
    expanded[:, 96:] = previous[:, 16:]
    model.load_state_dict(dict(source, **{key: expanded}), strict=True)
    model.requires_grad_(False)
    model.semantic_decoder.branches['geometry'].requires_grad_(True)
    return model


def run(source_path, out, *, steps=500, lr=3e-6, interval=100, device_name='cuda', resume=False):
    source_path, out = Path(source_path).resolve(), Path(out).resolve()
    if (out.is_relative_to(source_path.parents[1]) or source_path.is_relative_to(out)
            or (out.exists() and not resume)):
        raise ValueError('use a separate new output directory or resume the same experiment')
    parent = torch.load(source_path, map_location='cpu', weights_only=True)
    original = parent['plan']['settings']
    if (parent['family'] != FAMILY or parent['arm'] != 'split128' or
            parent['progress']['step'] != original['bootstrap_steps'] + original['render_steps']):
        raise ValueError('completed split128 checkpoint required')
    training = parent['plan']['selected_scenes']
    if steps <= 0 or steps % len(training) or interval <= 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError('positive balanced update budget, interval and LR required')
    plan = dict(parent['plan'], settings=dict(original, arms=[FULL_GEOMETRY], bootstrap_steps=0,
        render_steps=steps, render_lr=lr, validate_every=interval, save_every=interval),
        initialization='split128 final; zero added geometry columns; frozen encoder/DC/SH',
        budgets=dict(render_total=steps, updates_per_arm=steps, render_per_scene=steps//len(training)),
        selection='fixed final; evaluation only', source_checkpoint_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        source_step=parent['progress']['step'])
    folder = out/FULL_GEOMETRY
    latest = folder/'checkpoints/latest.pt'
    if resume and (not latest.exists() or json.loads((out/'plan.json').read_text()) != plan):
        raise ValueError('resume requires matching plan, parent and latest checkpoint')
    (folder/'metrics').mkdir(parents=True, exist_ok=True)
    json_write(out/'plan.json', plan)
    args = argparse.Namespace(**plan['settings'])
    args.net_bits_per_use = 2.
    device = device_for(device_name)
    torch.set_num_threads(args.cpu_threads)
    seed_all(args.seed)
    model = widen(parent['model'], ModularCodecConfig.from_dict(parent['config'])).to(device)
    parameters = list(model.semantic_decoder.branches['geometry'].parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scenes, train_raws = [], []
    records = parent['scene_records']
    for spec in training+[parent['plan']['heldout_scene']]:
        raw, degree = read_ply(spec['ply'])
        scene = make_scene(spec, raw, degree, model, args, out, old=records[spec['name']])
        scenes.append(scene)
        if spec['role'] == 'train':
            train_raws.append(raw)
    scales = fit_loss_scales(train_raws)
    train_scenes = scenes[:len(training)]
    perceptual = LPIPSEvaluator(device) if args.lpips else None
    progress = dict(step=0, cycles=parent['progress']['cycles'].copy(), phase='render')
    log = folder/'metrics/loss.jsonl'
    if resume:
        saved = torch.load(latest, map_location='cpu', weights_only=True)
        if saved['plan'] != plan or saved['arm'] != FULL_GEOMETRY:
            raise ValueError('incompatible resume checkpoint')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        progress = saved['progress']
        restore_rng(saved['rng'])
        if log.exists():
            records_kept = [line for line in log.read_text().splitlines() if json.loads(line)['step'] <= progress['step']]
            log.write_text(''.join(line+'\n' for line in records_kept), encoding='utf8')
    cycles = {scene.name: RefinementViewCycle([str(cam.image_name) for cam in scene.cameras],
        args.seed, scene.name, 2, progress['cycles']['render/'+scene.name]) for scene in train_scenes}
    json_write(folder/'architecture.json', dict(arm=FULL_GEOMETRY, dimensions=128,
        decoder_inputs=dict(geometry=128, dc=48, sh=96), trainable='semantic_decoder.branches.geometry',
        parameters=sum(param.numel() for param in model.parameters())))

    def assess(scope):
        with preserved_rng(device):
            for scene in scenes if scope == 'test' else train_scenes:
                cameras = scene.test_cameras if scope == 'test' else scene.val_cameras
                path = folder/'evaluation'/f'{progress["step"]:06d}_{scope}_{scene.name}'
                report = evaluate(model, scene, cameras, perceptual, path, interventions=scope == 'test', images=scope == 'test')
                report.update(scope=scope, arm=FULL_GEOMETRY, step=progress['step'])
                json_write(path.with_suffix('.json'), report)
                print(f'{scope} {scene.name}: PSNR={report["rows"][0]["source_psnr"]:.4f} '
                      f'shape MSE={report["attributes"]["shape"]["mse"]:.6g}', flush=True)

    if progress['step'] == 0:
        assess('validation')
    for step in range(progress['step']+1, steps+1):
        scene = train_scenes[(step-1) % len(train_scenes)]
        cycle = cycles[scene.name]
        cameras = [scene.cameras[index] for index in cycle.next(args.views_per_step)]
        optimizer.zero_grad(set_to_none=True)
        loss, stats = render_update(model, scene, cameras, scales, args)
        torch.nn.utils.clip_grad_norm_(parameters, float('inf'), error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        optimizer.step()
        progress['step'] = step
        append_json(log, dict(step=step, scene=scene.name, loss=float(loss), **stats))
        if step == 1 or step % 10 == 0:
            print(f'geometry {step}/{steps}: scene={scene.name}, loss={float(loss):.7g}', flush=True)
        if step % interval == 0 or step == steps:
            assess('validation')
            progress['cycles'].update({'render/'+name: item.state_dict() for name, item in cycles.items()})
            save_checkpoint(latest, model, optimizer, progress, records, plan)
    save_checkpoint(folder/'checkpoints/final.pt', model, optimizer, progress, records, plan)
    assess('test')
    complete = dict(status='complete', step=steps, arm=FULL_GEOMETRY)
    json_write(folder/'complete.json', complete)
    json_write(out/'complete.json', complete)
    review(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    options = parser.parse_args()
    run(options.source, options.out, steps=options.steps, lr=options.lr, interval=options.interval, resume=options.resume)
