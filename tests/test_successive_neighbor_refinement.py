from collections import Counter
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gaussian_jscc import successive_neighbor_refinement as entry
from gaussian_jscc import successive_tier_refinement as engine
from gaussian_jscc.multisnr_training import cached_render_step
from gaussian_jscc.render_objective import MultiViewRenderTask
from test_successive_multisnr import shared, digital_runner
from test_successive_joint import cameras
from test_render_first import synthetic_render
from test_successive_q3_refinement import load_model_scene, assert_same_checkpoint
from test_successive_tier_refinement import options


def test_balanced_layouts_replay_and_padding():
    assert Counter(entry.layout_for_visit(i) for i in range(1, 151)) == {'mixed': 75, '1': 25, '2': 25, '3': 25}
    args = entry.parser().parse_args(['--run', 'source', '--out', 'out', '--digital-python', 'python'])
    assert args.scenes == ['truck', 'playroom'] and args.heldout_scenes == []
    ids = torch.arange(32).reshape(2, 16); ids[-1, -3:] = -1
    scene = SimpleNamespace(name='truck', groups=[torch.randn(2, 16, 8)], group_ids=[ids])
    targets = [torch.randn(2, 16, 48)]
    rng = torch.get_rng_state().clone()
    for visit in range(1, 13):
        batches = entry.CachedMixedBatches(scene, targets, 42, visit)
        x, q = next(iter(batches))
        assert torch.equal(q, next(iter(entry.CachedMixedBatches(scene, targets, 42, visit)))[1])
        assert torch.equal(q == 0, ids < 0)
        if batches.layout != 'mixed':
            assert (q[ids >= 0] == int(batches.layout)).all()
    assert torch.equal(torch.get_rng_state(), rng)


def test_render_gradients_train_only_neighbor_and_keep_sent_symbols(shared, tmp_path):
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras):
        args, model, scene, _ = load_model_scene(shared, tmp_path)
        optimizer = entry.configure_optimizer(model, 1, args.lr)
        targets = [b['y'] for b in engine.cache_teacher(model, scene)]
        before = {n: v.clone() for n, v in model.state_dict().items()}
        modes = {n: m.training for n, m in model.named_modules()}
        rng = torch.get_rng_state().clone()
        entry.endpoint_diagnostics(model, [scene], args, tmp_path, 'initial')
        assert torch.equal(torch.get_rng_state(), rng)
        assert modes == {n: m.training for n, m in model.named_modules()}
        for visit in (1, 2, 4, 6):
            optimizer.zero_grad(set_to_none=True)
            task = MultiViewRenderTask(scene.cameras[:1], scene.reference, scene.degree)
            # Render-only loss must pass through the frozen semantic decoder.
            cached_render_step(model, entry.CachedMixedBatches(scene, targets, 42, visit),
                               scene.geometry, 10., task, 0.)
            gradients = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
            assert gradients and all(n.startswith(entry.TRAINABLE) for n in gradients)
            assert sum(float(g.abs().sum()) for g in gradients.values()) > 0
            optimizer.step()
        assert engine.base.frozen_digest(model.state_dict(), entry.TRAINABLE) == engine.base.frozen_digest(before, entry.TRAINABLE)
        assert any(not torch.equal(v, model.state_dict()[n]) for n, v in before.items() if n.startswith(entry.TRAINABLE))
        entry.endpoint_diagnostics(model, [scene], args, tmp_path, 'final')
        rows = engine.read(tmp_path/'metrics/neighbor_probe_final.json')['rows']
        assert len(rows) == 8 and all(r['frozen_outputs_verified'] for r in rows)
        assert all(max(r['max_absolute_change'].values()) == 0 for r in rows)
        model.channel_decoder.neighbor_enabled.fill_(False)
        with pytest.raises(ValueError, match='already be enabled'):
            entry.configure_optimizer(model, 1, args.lr)


def test_neighbor_run_exact_resume_counts_and_actual_delivery(shared, tmp_path):
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('comparison_baselines.run._execute', side_effect=digital_runner):
        source = tmp_path/'sequential'
        engine.main(options(shared, source))
        full, resumed = tmp_path/'neighbor', tmp_path/'resume'

        def args(out):
            a = options(source, out)
            a.steps_per_scene = 6
            a.heldout_scenes = []
            return a

        result = entry.main(args(full))
        assert result['step'] == 12 and result['frozen_state_verified']
        assert result['actual_side_delivery_completed'] and result['selection'] == 'fixed_final'
        entry.main(args(resumed), max_updates=5)
        a = args(resumed); a.resume = True
        entry.main(a)
        one = torch.load(full/'checkpoints/final/codec.pt', weights_only=True)
        two = torch.load(resumed/'checkpoints/final/codec.pt', weights_only=True)
        assert_same_checkpoint(one, two)
        assert one['training']['progress'] == two['training']['progress']
        report = engine.read(full/'review_summary.json')
        assert report['layout_updates'] == {s: {'mixed': 3, '1': 1, '2': 1, '3': 1} for s in ('truck', 'train')}
        assert report['frozen_outputs_verified']
        assert len(report['rows']) == 2*2*2*4
        actual = engine.read(full/'actual_delivery/truck/summary.json')['trials']
        assert len(actual) == 4
        for r in actual:
            row = next(x for x in report['rows'] if x['endpoint'] == 'final' and x['scene'] == 'truck'
                       and x['path'] == 'awgn' and x['layout'] == r['method'])
            assert r['total_channel_uses'] == row['total_channel_uses']
            assert r['metrics']['source_psnr'] == pytest.approx(row['source_psnr'], abs=1e-5)
        losses = [json.loads(line) for line in (resumed/'metrics/loss.jsonl').read_text().splitlines()]
        assert len(losses) == 12
        assert (full/'review.tar.gz').exists()
