"""Rotate mixed/uniform delivery while refining only the receiver neighbor module."""
from collections import Counter
import json
import sys

import torch
from torch.nn import functional as F

from . import successive_mixed_refinement as mixed
from . import successive_tier_refinement as engine
from .modular_experiment import json_write
from .optimization import preserved_rng

FAMILY = 'successive_neighbor_refinement_v1'
ENDPOINTS = ('initial', 'final')
TRAINABLE = ('channel_decoder.neighbor_refiner.',)
LAYOUT_CYCLE = ('mixed', '1', 'mixed', '2', 'mixed', '3')


def parser():
    p = engine.parser()
    p.description = __doc__
    p.set_defaults(scenes=['truck', 'playroom'], heldout_scenes=[])
    return p


def prefixes(tier):
    if tier != 1:
        raise ValueError('neighbor refinement has one stage')
    return TRAINABLE


def configure_optimizer(model, tier, lr):
    prefixes(tier)
    if not bool(model.channel_decoder.neighbor_enabled):
        raise ValueError('source neighbor refinement must already be enabled')
    model.requires_grad_(False)
    for p in model.parameters():
        p.grad = None
    module = model.channel_decoder.neighbor_refiner
    module.requires_grad_(True)
    return torch.optim.Adam([dict(params=list(module.parameters()), lr=lr,
                                 name='channel_decoder.neighbor_refiner')])


def layout_for_visit(visit):
    if visit < 1:
        raise ValueError('scene visits start at one')
    return LAYOUT_CYCLE[(visit-1) % len(LAYOUT_CYCLE)]


def inputs(args):
    saved, metadata, cfg, specs, heldouts, plan = mixed.inputs(args)
    if not bool(saved['state_dict']['channel_decoder.neighbor_enabled']):
        raise ValueError('completed source must include enabled neighbor refinement')
    plan.update(version=FAMILY, trainable_prefixes=list(TRAINABLE),
        stage_scope={'mixed': list(TRAINABLE)},
        frozen_state_sha256=engine.base.frozen_digest(saved['state_dict'], TRAINABLE),
        objective='source PLY RGB MSE + latent_weight * latent MSE; receiver neighbor refinement only',
        optimizer_initialization='new Adam for inherited neighbor weights; shared across selected scenes',
        training_layout='per-scene cycle: mixed, q1, mixed, q2, mixed, q3; fresh AWGN every update',
        layout_cycle=list(LAYOUT_CYCLE),
        layout_updates_per_scene=dict(Counter(layout_for_visit(i+1) for i in range(args.steps_per_scene))),
        endpoint_diagnostics='paired first two original blocks per training scene, four layouts, noiseless/AWGN; '
                             'compare sent symbols, binary16 scales and pre-neighbor latent; report pre/post latent MSE',
        cache='immutable semantic y; channel forward uses frozen stages; render VJP trains neighbor through frozen semantic decoder')
    return saved, metadata, cfg, specs, heldouts, plan


class CachedMixedBatches:
    """Visit-indexed layout replay leaves the noise RNG untouched."""
    def __init__(self, scene, targets, seed, visit):
        self.layout = layout_for_visit(visit)
        self.batches = (mixed.CachedMixedBatches(scene, targets, seed, visit) if self.layout == 'mixed'
                        else engine.base.CachedQ3Batches(scene, targets, int(self.layout)))

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


@torch.no_grad()
def endpoint_diagnostics(model, scenes, args, out, label):
    """Small paired probes complement the full-state frozen digest and full render evaluation."""
    device = next(model.parameters()).device
    modes = {m: m.training for m in model.modules()}
    rows, values = [], {}
    baseline_path = out/'metrics/neighbor_probe_initial.pt'
    baseline = torch.load(baseline_path, map_location='cpu', weights_only=True) if label != 'initial' else None
    captured = {}

    def sender_hook(module, inputs, result):
        captured['symbols'], captured['scales'] = [x.detach().cpu().clone() for x in result]

    def neighbor_hook(module, inputs):
        captured['base'] = inputs[0].detach().cpu().clone()

    hooks = [model.channel_encoder.register_forward_hook(sender_hook),
             model.channel_decoder.neighbor_refiner.register_forward_pre_hook(neighbor_hook)]
    try:
        with preserved_rng(device):
            model.eval()
            for scene in scenes:
                model.set_scene_geometry(scene.geometry)
                features = scene.groups[0][:2].to(device)
                ids = scene.group_ids[0][:2].to(device)
                for layout in engine.LAYOUTS:
                    generator = torch.Generator().manual_seed(args.seed+7001)
                    q = (torch.randint(1, 4, ids.shape, generator=generator).to(device) if layout == 'mixed'
                         else torch.full_like(ids, int(layout)))
                    q = q.masked_fill(ids < 0, 0)
                    for path in ('finite_noiseless', 'awgn'):
                        # Same noise and original complete blocks at both endpoints.
                        torch.manual_seed(args.seed+1000)
                        captured.clear()
                        y, recovered, _ = model.forward_channel_batches(features, features[..., :3],
                            F.one_hot(q, 4).to(features), 10.,
                            'none' if path == 'finite_noiseless' else 'awgn', paired_noise=True)
                        key = f'{scene.name}/{layout}/{path}'
                        values[key] = dict(captured)
                        active, target = (q > 0).cpu(), y.cpu()
                        row = dict(scene=scene.name, layout=layout, path=path, blocks=len(ids),
                            points=int(active.sum()),
                            base_latent_mse=float((captured['base'][active]-target[active]).square().mean()),
                            refined_latent_mse=float((recovered.cpu()[active]-target[active]).square().mean()))
                        row['neighbor_mse_reduction'] = row['base_latent_mse']-row['refined_latent_mse']
                        if baseline is not None:
                            row['max_absolute_change'] = {}
                            for field in ('symbols', 'scales', 'base'):
                                a, b = captured[field], baseline[key][field]
                                torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6,
                                    msg=f'frozen {field} changed in {key}')
                                row['max_absolute_change'][field] = float((a-b).abs().max())
                            row['frozen_outputs_verified'] = True
                        rows.append(row)
    finally:
        for hook in hooks:
            hook.remove()
        for module, mode in modes.items():
            module.training = mode
    if label == 'initial':
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(values, baseline_path)
    json_write(out/f'metrics/neighbor_probe_{label}.json', dict(endpoint=label,
        scope='first two original blocks per training scene; paired tensor diagnostics', rows=rows))


def summarize(out, plan):
    report = mixed.summarize(out, plan)
    report['neighbor_probes'] = {label: engine.read(out/f'metrics/neighbor_probe_{label}.json') for label in ENDPOINTS}
    counts = {}
    for line in (out/'metrics/loss.jsonl').read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        counts.setdefault(row['scene'], Counter())[row['layout']] += 1
    expected = plan['layout_updates_per_scene']
    if set(counts) != {s['name'] for s in plan['selected_scenes']} or any(dict(v) != expected for v in counts.values()):
        raise RuntimeError('completed layout update counts differ from the plan')
    report['layout_updates'] = {name: dict(c) for name, c in counts.items()}
    report['frozen_outputs_verified'] = all(r['frozen_outputs_verified'] for r in report['neighbor_probes']['final']['rows'])
    json_write(out/'review_summary.json', report)
    return report


def main(args, max_updates=None):
    return engine.main(args, max_updates=max_updates, recipe=sys.modules[__name__])


if __name__ == '__main__':
    main(parser().parse_args())
