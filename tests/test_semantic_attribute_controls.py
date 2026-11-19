"""Real semantic modules/replay/checkpoints; CPU renderer substitutes image formation only."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gaussian_jscc import semantic_attribute_experiment as entry
from gaussian_jscc.semantic_attribute_codec import (PRIVATE_ONLY, REBALANCED, SemanticAttributeCodec, make_config,
    latent_slices, log_shape, fit_loss_scales, attribute_objective)
from gaussian_jscc.semantic_attribute_metrics import evaluate, latent_diagnostics
from gaussian_jscc.data import Geometry, to_features, to_raw, write_ply
from gaussian_jscc.modular_experiment import json_write
from gaussian_jscc.multiscene import shared_statistics
from gaussian_jscc.position_delivery import delivered_positions
from gaussian_jscc.successive_multiscene_inputs import make_scene
from gaussian_jscc.perceptual_objective import PerceptualImageObjective, PerceptualMultiViewRenderTask
from gaussian_jscc.training import full_scene_step


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1); torch.manual_seed(123)


def raw_points(seed=1, count=17):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(count, 59, generator=g)*.15
    raw[:, :3] = torch.rand(count, 3, generator=g)
    raw[:, 4:7] -= 2
    raw[:, 7:11] = torch.nn.functional.normalize(raw[:, 7:11]+torch.tensor([1., 0., 0., 0.]), dim=-1)
    return raw


def config(width=128):
    cfg = make_config(width, hidden=16, block_size=8)
    cfg.context_centers = (4, 2); cfg.context_neighbors = (2, 4)
    return cfg


def model_data(arm='split128'):
    model = SemanticAttributeCodec(arm, config(int(arm[-3:])))
    raw = raw_points(count=16)
    mean, std = shared_statistics([raw])
    model.attr_mean.copy_(mean); model.attr_std.copy_(std)
    geometry = Geometry.fit(raw[:, :3], 16); model.set_scene_geometry(geometry)
    f, _ = to_features(raw, geometry, model)
    f = f.reshape(2, 8, 59); active = torch.ones(2, 8, dtype=torch.bool)
    return model, raw, geometry, f, active


@pytest.mark.parametrize('arm', entry.TRAINABLE_ARMS)
def test_interface_and_private_latent_isolation(arm):
    model, raw, geometry, f, active = model_data(arm)
    assert not any('channel' in n for n, _ in model.named_modules())
    active[1, -2:] = False
    corrupted = f.clone(); corrupted[~active] = 999
    with torch.no_grad():
        a = model(f, active); b = model(corrupted, active)
        torch.testing.assert_close(a, b)
        assert a.shape == f.shape and torch.equal(a[~active], torch.zeros_like(a[~active]))
        z = model.encode(f, active)
        assert z.shape[-1] == int(arm[-3:])
        side = delivered_positions(f[..., :3], active.long(), model.cfg)
        baseline = model.decode(z, active, side)
        if arm.startswith('split'):
            for name, unaffected in [('geometry', slice(11, 59)), ('dc', slice(3, 11)), ('sh', slice(3, 14))]:
                start, end = model.slices[name]
                changed = z.clone(); changed[..., start:end] += .7
                pred = model.decode(changed, active, side)
                torch.testing.assert_close(pred[..., unaffected], baseline[..., unaffected], atol=0, rtol=0)
    loss, terms = attribute_objective(a[active].requires_grad_(), f[active], geometry, model,
                                      torch.tensor([[1., 0., 0.]]), fit_loss_scales([raw]))
    assert torch.isfinite(loss) and set(terms) == {'opacity', 'shape', 'dc', 'sh', 'direction_rgb'}


def test_log_covariance_invariances_and_loss_identity():
    model, raw, geometry, f, active = model_data()
    flipped = raw.clone(); flipped[:, 7:11] *= -1
    torch.testing.assert_close(log_shape(raw), log_shape(flipped))
    isotropic = raw.clone(); isotropic[:, 4:7] = -2
    rotated = isotropic.clone(); rotated[:, 7:11] = torch.randn_like(rotated[:, 7:11])
    torch.testing.assert_close(log_shape(isotropic), log_shape(rotated), atol=4e-6, rtol=1e-6)
    x = f[active].clone().requires_grad_()
    loss, _ = attribute_objective(x, f[active], geometry, model,
                                  torch.tensor([[1., 0., 0.], [0., 1., 0.]]), fit_loss_scales([raw]))
    assert float(loss.detach()) == 0
    loss.backward(); assert torch.isfinite(x.grad).all()
    assert sum(b-a for a, b in latent_slices(256, True).values()) == 256


@pytest.mark.parametrize('arm, expected_slices', [
    (REBALANCED, {'geometry': (0, 32), 'dc': (32, 40), 'sh': (40, 104), 'shared': (104, 128)}),
    (PRIVATE_ONLY, {'geometry': (0, 40), 'dc': (40, 56), 'sh': (56, 128)}),
])
def test_custom_routing_and_single_arm_checkpoint(tmp_path, arm, expected_slices):
    from gaussian_jscc.progressive_attribute_channel import load_semantic_slot
    assert arm not in entry.parser().get_default('arms')
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch.object(entry, 'diagnostic_plots'), patch.object(entry, 'write_plots'):
        source, held = build_fixture(tmp_path)
        out = tmp_path/'fresh'
        entry.main(options(source, held, out, arm=arm))
    assert entry.read(out/'review_summary.json')['completed_arms'] == [arm]
    model, _ = load_semantic_slot(out/arm/'checkpoints/final.pt')
    assert model.slices == expected_slices
    diagnostic = entry.read(out/arm/'diagnostics/truck.json')
    assert set(diagnostic['latent']) == set(expected_slices)
    assert {item['latent'] for item in diagnostic['perturbations']} == set(expected_slices)
    _, _, geometry, f, active = model_data()
    model.set_scene_geometry(geometry)
    z = torch.arange(128, dtype=f.dtype).expand(*active.shape, 128)
    inputs = {}
    def capture(name):
        def hook(module, args):
            inputs[name] = args[0].clone()
        return hook
    hooks = [branch.register_forward_pre_hook(capture(name)) for name, branch in model.semantic_decoder.branches.items()]
    try:
        model.decode(z, active, f[..., :3])
    finally:
        for hook in hooks:
            hook.remove()
    for name in ('geometry', 'dc', 'sh'):
        a, b = expected_slices[name]
        expected = z[..., a:b]
        if 'shared' in expected_slices:
            c, d = expected_slices['shared']
            expected = torch.cat((expected, z[..., c:d]), -1)
        torch.testing.assert_close(inputs[name], expected, atol=0, rtol=0)


def synthetic_render(raw, camera, *args, **kwargs):
    # Shapes and attributes influence the complete image; coordinates are distinguishable.
    color = raw[:, 11:14].mean(0)+.03*raw[:, :3].mean(0)+.01*raw[:, 4:7].square().mean(0)
    color = color.sigmoid()+camera.bias
    return color[:, None, None].expand(3, 12, 12)


def cameras(source, resolution=2, white_background=False, images='images', split='test'):
    count = 6 if split == 'train' else 2
    return [SimpleNamespace(image_name=f'{split}_{i}', camera_center=torch.tensor([3.+i, 4., 5.]),
        original_image=torch.full((3, 12, 12), .5), bias=.01*i,
        FoVx=.8, FoVy=.8, image_height=12, image_width=12,
        world_view_transform=torch.eye(4), full_proj_transform=torch.eye(4)) for i in range(count)]


class Reference:
    def __init__(self, raw): self.raw = raw
    def get(self, camera, device): return synthetic_render(self.raw.to(device), camera).detach()


def test_multiview_replay_matches_direct_gradient():
    model, raw, geometry, f, active = model_data()
    other = copy.deepcopy(model)
    scale = fit_loss_scales([raw]); dirs = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    batches = [(f[i:i+1], active[i:i+1].long()) for i in range(2)]
    results = []
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render):
        for mode, m in [('replay', model), ('checkpoint', other)]:
            task = PerceptualMultiViewRenderTask(cameras('unused'), Reference(raw), 3, False,
                                                PerceptualImageObjective('mse_ssim', 1/3, 'cpu'))
            def forward(f, q):
                keep = q > 0; p = m(f, keep)
                aux, terms = attribute_objective(p[keep], f[keep], geometry, m, dirs, scale)
                return to_raw(p[keep], geometry, m), aux, terms
            loss, _ = full_scene_step(m, batches, geometry, 0, 'none', task, attr_weight=.01,
                                      mode=mode, batch_forward=forward)
            results.append(loss)
    torch.testing.assert_close(*results, atol=1e-6, rtol=1e-5)
    for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4, msg=name)


def test_shared_neighborhood_cache_preserves_values_and_gradients():
    from gaussian_jscc.pointnet_context import SpatialPointContext
    model, raw, geometry, f, active = model_data()
    z = model.encode(f, active).detach().requires_grad_()
    z_other = z.detach().clone().requires_grad_()
    side = delivered_positions(f[..., :3], active.long(), model.cfg)
    cached = model.decode(z, active, side)
    plain = []
    a, b = model.slices['shared']
    for name, branch in model.semantic_decoder.branches.items():
        c, d = model.slices[name]
        plain.append(branch(torch.cat((z_other[..., c:d], z_other[..., a:b]), -1), active, side))
    uncached = torch.cat((side, *plain), -1)
    torch.testing.assert_close(cached, uncached, atol=0, rtol=0)
    g1 = torch.autograd.grad(cached.square().sum(), z)[0]
    g2 = torch.autograd.grad(uncached.square().sum(), z_other)[0]
    torch.testing.assert_close(g1, g2, atol=0, rtol=0)
    assert not any(hasattr(m, '_allocation_geometry_cache') for m in model.modules() if isinstance(m, SpatialPointContext))


def build_fixture(root):
    source = root/'source'; source.mkdir()
    specs, records = [], {}
    model = SemanticAttributeCodec('shared128', config())
    opts = SimpleNamespace(blocks_per_batch=2, views_per_step=2, validation_views=1,
        test_views=2, resolution=2, images='images', white_background=False, net_bits_per_use=2.)
    for i, (name, role) in enumerate([('truck', 'train'), ('playroom', 'train'), ('drjohnson', 'heldout')]):
        folder = root/name; folder.mkdir()
        ply = folder/'points.ply'; raw = raw_points(i+1)
        write_ply(ply, raw, 3)
        # Compare with the exact normalized read-back source.
        raw, degree = entry.read_ply(ply)
        spec = dict(name=name, role=role, ply=str(ply), source=str(folder), prior='none')
        scene = make_scene(spec, raw, degree, model, opts, root/'preparation')
        specs.append(spec); records[name] = scene.record
    json_write(source/'training.json', dict(selected_scenes=specs[:2], scene_records=records,
        codec_config=config().to_dict(), settings=vars(opts)))
    held = root/'held.json'; json_write(held, dict(scenes=[specs[-1]]))
    return source, held


def options(source, held, out, arm='split128'):
    return entry.parser().parse_args(['--run', str(source), '--out', str(out), '--heldout-manifest', str(held),
        '--arms', arm, '--bootstrap-steps', '2', '--render-steps', '2', '--hidden', '16',
        '--views-per-step', '2', '--validation-views', '1', '--test-views', '2',
        '--blocks-per-batch', '2', '--save-every', '1', '--validate-every', '2',
        '--no-lpips', '--device', 'cpu', '--cpu-threads', '1'])


def test_end_to_end_and_exact_resume(tmp_path):
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras):
        source, held = build_fixture(tmp_path)
        full = options(source, held, tmp_path/'full')
        entry.main(full)
        resumed = options(source, held, tmp_path/'resume')
        result = entry.main(resumed, max_updates=3)
        assert result['step'] == 3
        resumed.resume = True; entry.main(resumed)
        for name in ('full', 'resume'):
            root = tmp_path/name
            assert (root/'complete.json').exists() and (root/'review.tar.gz').exists()
            summary = entry.read(root/'review_summary.json')
            assert any(r['scene'] == 'drjohnson' and r['scope'] == 'test' for r in summary['rows'])
            assert not any(r['scene'] == 'drjohnson' and r['scope'] == 'validation' for r in summary['rows'])
            diag = entry.read(root/'split128/diagnostics/truck.json')
            assert len(diag['perturbations']) == 8
            losses = [json.loads(x) for x in (root/'split128/metrics/loss.jsonl').read_text().splitlines()]
            assert [r['step'] for r in losses] == [1, 2, 3, 4]
            for r in losses: assert len(r['cameras']) == len(set(r['cameras'])) == 2
        a = torch.load(tmp_path/'full/split128/checkpoints/latest.pt', weights_only=True)
        b = torch.load(tmp_path/'resume/split128/checkpoints/latest.pt', weights_only=True)
        for key in a['model']: torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
        assert a['progress'] == b['progress']
        with pytest.raises(ValueError, match='identical experiment settings'):
            resumed.views_per_step = 1; entry.main(resumed)


def test_loss_scales_train_only_and_target_reference_separation(tmp_path):
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras):
        source, held = build_fixture(tmp_path)
        args = options(source, held, tmp_path/'run', arm='shared128')
        args.blocks_per_batch = 1  # Execution batch may differ from source metadata.
        args.test_views = 3  # A scene with fewer original test views keeps its split.
        entry.main(args, max_updates=1)
        stats = entry.read(tmp_path/'run/attribute_statistics.json')
        train = entry.read(source/'training.json')['selected_scenes']
        raws = [entry.read_ply(s['ply'])[0] for s in train]
        mean, std = shared_statistics(raws)
        assert stats['mean'] == mean.tolist() and stats['std'] == std.tolist()
        assert stats['loss_scales'] == fit_loss_scales(raws)
        assert entry.read(tmp_path/'run/plan.json')['communication_cost'] is None


def test_all_four_arms_share_data_schedule_and_emit_compact_review(tmp_path):
    import tarfile
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch.object(entry, 'diagnostic_plots'), patch.object(entry, 'write_plots'):
        source, held = build_fixture(tmp_path)
        args = options(source, held, tmp_path/'suite')
        args.arms = list(entry.ARMS)
        entry.main(args)
        schedules = []
        for arm in args.arms:
            folder = tmp_path/'suite'/arm
            losses = [json.loads(x) for x in (folder/'metrics/loss.jsonl').read_text().splitlines()]
            schedules.append([(r['scene'], r['cameras'], r['view_coverage']) for r in losses])
            assert entry.read(folder/'complete.json')['step'] == 4
        assert all(x == schedules[0] for x in schedules)
        with tarfile.open(tmp_path/'suite/review.tar.gz') as package:
            assert not any(n.endswith('.pt') for n in package.getnames())
            assert 'review_summary.json' in package.getnames()
