"""Mixed q1/q2/q3 continuation from a completed sequential refinement."""
import hashlib
import json
from pathlib import Path
import sys

import torch

from . import successive_tier_refinement as engine
from .model_factory import config_from_dict
from .modular_checkpoint import validate_scene
from .modular_experiment import json_write
from .successive_joint import encoder_digest

FAMILY = 'successive_mixed_refinement_v1'
ENDPOINTS = ('initial', 'final')


def parser():
    p = engine.parser()
    p.description = __doc__
    return p


def load_source(run):
    run = Path(run).resolve()
    path = run/'checkpoints/final/codec.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    metadata = engine.read(run/'training.json')
    complete = engine.read(run/'complete.json')
    progress = saved['training']['progress']
    cfg = config_from_dict(saved['config'])
    counts = progress.get('scene_completed', {})
    names = {s['name'] for s in metadata['selected_scenes']}
    budget = metadata['settings']['steps_per_scene']
    if (metadata['version'] != engine.FAMILY or saved.get('version') != 4 or
            progress.get('training_family') != engine.FAMILY or
            not progress.get('phase_complete') or not progress.get('channel_trained') or
            progress.get('completed_stages') != [1, 2, 3] or
            progress.get('train_snrs') != [10.] or
            cfg.channel_variant != 'successive' or tuple(cfg.rates) != (0, 8, 24, 48) or
            cfg.position_delivery != 'quantized' or cfg.position_bits != 16 or
            complete.get('status') != 'complete' or complete.get('frozen_state_verified') is not True or
            complete.get('lower_tiers_unchanged') is not True or
            complete.get('heldout_optimizer_updates') != 0 or
            saved.get('step') != progress.get('step') or complete.get('step') != saved['step'] or
            complete.get('scene_completed') != counts or set(counts) != {'q1', 'q2', 'q3'} or
            any(set(c) != names or any(v != budget for v in c.values()) for c in counts.values()) or
            saved['step'] != 3*len(names)*budget):
        raise ValueError('source must be the completed three-stage 10 dB refinement endpoint')
    if json.loads(json.dumps(saved['config'])) != metadata['codec_config']:
        raise ValueError('source codec config mismatch')
    validate_scene(saved['training']['scene_record'], metadata['scene_records'])
    if set(metadata['scene_records']) != names:
        raise ValueError('source scene records mismatch')
    stats = encoder_digest(saved['state_dict'], include_encoder=False)
    if (stats != metadata['statistics_sha256'] or
            engine.base.frozen_digest(saved['state_dict'], engine.ALL_STAGES) != metadata['frozen_state_sha256']):
        raise ValueError('source frozen semantic codec/statistics mismatch')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    identity = dict(source_checkpoint=str(path), source_sha256=digest,
        source_step=saved['step'], statistics_sha256=stats,
        training_scene_history=sorted(set(metadata.get('training_scene_history', [])) | names))
    return saved, metadata, cfg, identity


def prefixes(tier):
    if tier != 1:
        raise ValueError('mixed continuation has one joint channel-training stage')
    return engine.ALL_STAGES


def configure_optimizer(model, tier, lr):
    model.requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    groups = []
    for prefix in prefixes(tier):
        params = [p for n, p in model.named_parameters() if n.startswith(prefix)]
        if not params:
            raise ValueError('missing successive channel stage: '+prefix)
        for p in params:
            p.requires_grad_(True)
        groups.append(dict(params=params, lr=lr, name=prefix.rstrip('.')))
    return torch.optim.Adam(groups)


def inputs(args):
    saved, metadata, cfg, specs, heldouts, plan = engine.inputs(args, source_loader=load_source)
    # Preserve the preceding experiment's held-out cameras and evaluation noise.
    if args.validation_views != metadata['settings']['validation_views'] or args.seed != metadata['settings']['seed']:
        raise ValueError('retain source validation-view count and seed for paired continuation')
    total = len(specs)*args.steps_per_scene
    plan.update(version=FAMILY, total_updates=total, stage_order=[1], stage_updates={'mixed': total},
        stage_scope={'mixed': list(engine.ALL_STAGES)},
        objective='source PLY RGB MSE + latent_weight * latent MSE; resampled positive mixed tiers; 10 dB AWGN',
        optimizer_initialization='new Adam; all three channel encoder/decoder stages jointly; shared across scenes',
        selection='fixed final update; validation/test never select weights',
        endpoint_evaluation='initial and final; q1/q2/q3 and fixed mixed map; paired cameras/noise',
        training_layout='independent uniform categorical q1/q2/q3 per valid point, resampled every scene update',
        view_sampling='per-scene shuffled cycles; same source train/validation/test partition; exact resume')
    plan.pop('lower_tier_checks', None)
    plan['validation_protocol'].update(added_views='none; retain the sequential-refinement partition')
    return saved, metadata, cfg, specs, heldouts, plan


class CachedMixedBatches:
    """Regenerate the same map for forward/VJP without altering channel RNG.

    Original blocks, padding and donor context are retained. Only semantic y
    is cached; all channel symbols and their normalizers use current weights.
    """
    def __init__(self, scene, targets, seed, visit):
        if visit < 1 or len(scene.groups) != len(targets):
            raise ValueError('invalid mixed scene visit or latent cache')
        self.scene, self.targets = scene, targets
        key = f'mixed:{seed}:{scene.name}:{visit}'.encode()
        self.seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63-1)

    def __len__(self):
        return len(self.targets)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        for features, ids, y in zip(self.scene.groups, self.scene.group_ids, self.targets):
            if features.shape[:2] != ids.shape or y.shape[:2] != ids.shape:
                raise ValueError('mixed cache block layout changed')
            q = torch.randint(1, 4, ids.shape, generator=generator).to(ids.device)
            q = q.masked_fill(ids < 0, 0)
            yield torch.cat((features, y), -1), q


def summarize(out, plan):
    report = engine.summarize(out, plan, endpoints=ENDPOINTS, check_lower=False)
    report.pop('lower_tiers_unchanged')
    report.pop('lower_tier_checks')
    report['selection'] = 'fixed_final'
    report['training_layout'] = plan['training_layout']
    # Report each uniform tier explicitly; mixed gains can coexist with losses
    # at a uniform tier, so both must remain visible in the review archive.
    for row in report['rows']:
        row.pop('change_from_previous_stage', None)
        if row['endpoint'] == 'final' and row['path'] == 'awgn':
            delta = row['change_from_initial']
            print(f'Final {row["scene"]}/{row["layout"]}: source PSNR={row["source_psnr"]:.4f} '
                  f'({delta["source_psnr"]:+.4f}), photo PSNR={row["photo_psnr"]:.4f} '
                  f'({delta["photo_psnr"]:+.4f})', flush=True)
    json_write(out/'review_summary.json', report)
    return report


def main(args, max_updates=None):
    return engine.main(args, max_updates=max_updates, recipe=sys.modules[__name__])


if __name__ == '__main__':
    main(parser().parse_args())
