"""Real codecs on CPU: nested prefixes, VJPs, wire reconstruction and exact resume."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test_semantic_attribute_controls import (model_data, build_fixture, options, synthetic_render,
    cameras, Reference)
from gaussian_jscc import semantic_attribute_experiment as semantic_entry
from gaussian_jscc import progressive_attribute_experiment as entry
from gaussian_jscc.data import to_raw
from gaussian_jscc.position_delivery import delivered_positions
from gaussian_jscc.progressive_attribute_channel import (EIGHT_RATES, THREE_RATES, LayerConfig, LayeredAttributeCodec,
    configure_stage, frozen_digest, state_digest, load_semantic_slot)
from gaussian_jscc.progressive_attribute_packet import export_packet, receive_packet
from gaussian_jscc.progressive_attribute_evaluation import decode_cached, assess
from gaussian_jscc.semantic_attribute_codec import PRIVATE_ONLY, fit_loss_scales
from gaussian_jscc.training import full_scene_step
from gaussian_jscc.perceptual_objective import PerceptualImageObjective, PerceptualMultiViewRenderTask


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1); torch.manual_seed(34)


def fixture(arm='split128', rates=EIGHT_RATES):
    semantic, raw, geometry, f, active = model_data(arm)
    active[1, -2:] = False
    f = f.masked_fill(~active[..., None], 0)
    model = LayeredAttributeCodec(semantic, LayerConfig(rates=rates, hidden=16, neighbor_hidden=8, neighbors=3))
    model.set_scene_geometry(geometry)
    with torch.no_grad():
        y = model.semantic.encode(f, active)
    return model, geometry, f, y, active


@pytest.mark.parametrize('arm,rates', [(arm, EIGHT_RATES) for arm in
    ('shared128', 'shared256', 'split128', 'split256')]+[(PRIVATE_ONLY, THREE_RATES)])
def test_all_semantic_interfaces_and_prefix_invariance(arm, rates):
    model, geometry, f, y, active = fixture(arm, rates)
    layers = len(rates)-1
    q = (torch.arange(16).reshape(2, 8) % layers+1)*active
    with torch.no_grad():
        z, gains = model.encode(y, q, 10.)
        for j in range(1, layers+1):
            capped = q.clamp_max(j)
            short, scales = model.encode(y, capped, 10.)
            torch.testing.assert_close(short, z[..., :2*rates[j]], atol=0, rtol=0)
            torch.testing.assert_close(scales, gains[:, :j], atol=0, rtol=0)
            a = model.forward_cached(f, y, capped, 10., generator=torch.Generator().manual_seed(7))[0]
            side = delivered_positions(f[..., :3], q, model.cfg)
            # Same seed draws a full-slot noise tensor for both paths.
            from gaussian_jscc.codec import channel
            noise = channel(torch.zeros(*q.shape, 64, 2), 10., generator=torch.Generator().manual_seed(7)).flatten(-2)
            yh, _ = model.receive(z[..., :2*rates[j]]+noise[..., :2*rates[j]], gains[:, :j], capped, side, 10.)
            b = model.semantic.decode(yh, active, side)
            torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('rates', [EIGHT_RATES, THREE_RATES])
def test_stage_optimizer_freezes_lower_layers_and_semantic(rates):
    model, geometry, f, y, active = fixture(rates=rates)
    q = active.long()*2
    with torch.no_grad():
        before = model.forward_cached(f, y, q, 10., generator=torch.Generator().manual_seed(91))[0]
    optimizer = configure_stage(model, 3, 1e-3)
    fixed = frozen_digest(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith(('encoders.2.', 'decoders.2.')) for n in trainable)
    old = state_digest(model.encoders[2].state_dict())
    for _ in range(2):
        optimizer.zero_grad()
        pred, yh, _ = model.forward_cached(f, y, active.long()*3, 10.)
        (yh[active].square().mean()+pred[active][:, 3:].square().mean()).backward()
        optimizer.step()
    assert frozen_digest(model) == fixed
    assert state_digest(model.encoders[2].state_dict()) != old
    with torch.no_grad():
        after = model.forward_cached(f, y, q, 10., generator=torch.Generator().manual_seed(91))[0]
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    next_optimizer = configure_stage(model, 4, 1e-3)
    assert not next_optimizer.state
    configure_stage(model, len(rates), 1e-3)
    assert all(n.startswith('neighbor.') for n, p in model.named_parameters() if p.requires_grad)


@pytest.mark.parametrize('rates', [EIGHT_RATES, THREE_RATES])
def test_packet_receiver_and_real_serialized_cost(tmp_path, rates):
    model, geometry, f, y, active = fixture('shared256', rates)
    # Valid padding occurs only in the last original block.
    data = torch.cat((f, y), -1)
    ids = torch.arange(16).reshape(2, 8).masked_fill(~active, -1)
    scene = SimpleNamespace(geometry=geometry, groups=[f], group_ids=[ids],
                            raw=to_raw(f[active], geometry, model), name='tiny')
    layers = len(rates)-1
    q = (torch.arange(16).reshape(2, 8) % layers+1)*active
    symbols = int(torch.tensor(rates)[q].sum())
    for spatial in (False, True):
        model.neighbor_enabled.fill_(spatial)
        root = tmp_path/str(spatial)
        measured, cost = export_packet(model, scene, [data], [q], 10., 101, root)
        expected, _ = decode_cached(model, scene, [data], [q], 10., 101)
        received = receive_packet(model, root)
        torch.testing.assert_close(received, expected, atol=0, rtol=0)
        assert measured['payload_complex_symbols'] == symbols
        assert measured['metadata_bytes'] == (root/'metadata.bin').stat().st_size
        assert cost['total_channel_uses'] == symbols+cost['side_channel_uses']
        from gaussian_jscc.transport import decode_metadata
        header, wire_q = decode_metadata((root/'metadata.bin').read_bytes())
        assert header.get('tier_id_bits', 2) == layers.bit_length() and int(wire_q.max()) == layers
        assert header['config']['rates'] == list(rates)
        # The receiver rejects wrong shared weights and corrupted scale frames.
        original = (root/'scales.bin').read_bytes()
        (root/'scales.bin').write_bytes(original[:-1])
        with pytest.raises(ValueError, match='scale framing'):
            receive_packet(model, root)


def test_full_scene_replay_gradient_matches_direct():
    model, geometry, f, y, active = fixture()
    configure_stage(model, 2, 1e-4)
    other = copy.deepcopy(model)
    raw = to_raw(f[active], geometry, model)
    views = cameras('', split='train')[:2]
    scene = SimpleNamespace(geometry=geometry, degree=3, white_background=False,
                            reference=Reference(raw))
    args = entry.parser().parse_args(['--out', 'unused', '--device', 'cpu'])
    scales = fit_loss_scales([raw])
    data = torch.cat((f, y), -1)
    batches = [(data[i:i+1], active[i:i+1].long()*2) for i in range(2)]
    gradients, losses = [], []
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render):
        for net, replay in [(model, True), (other, False)]:
            torch.manual_seed(991)
            objective = PerceptualImageObjective('mse_ssim', args.ssim_weight, 'cpu')
            task = PerceptualMultiViewRenderTask(views, scene.reference, 3, False, objective)
            forward = lambda d, q: entry.batch_objective(net, scene, d, q, views, scales, args)
            if replay:
                loss, _ = full_scene_step(net, batches, geometry, 10., 'awgn', task,
                                          attr_weight=1., batch_forward=forward)
            else:
                outputs = [forward(d, q) for d, q in batches]
                n = sum(len(o[0]) for o in outputs)
                loss = task(torch.cat([o[0] for o in outputs]))+sum(o[1]*len(o[0]) for o in outputs)/n
                loss.backward()
            losses.append(loss.detach())
            gradients.append(torch.cat([p.grad.flatten() for p in net.parameters() if p.requires_grad]))
    torch.testing.assert_close(losses[0], losses[1], atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(gradients[0], gradients[1], atol=2e-6, rtol=2e-4)


def test_slot_unfilled_plan_and_reject_unfinished(tmp_path):
    args = entry.parser().parse_args(['--out', str(tmp_path/'absent'), '--dry-run'])
    plan = entry.main(args)
    assert plan['total_updates'] == 6200 and not (tmp_path/'absent').exists()
    three = entry.main(entry.parser().parse_args(['--out', str(tmp_path/'three'), '--dry-run', '--three-tier-scratch']))
    assert three['channel']['rates'] == list(THREE_RATES) and three['total_updates'] == 24750
    assert three['bootstrap_validation_steps'] == [500, 2000, 4000, 8000]
    assert three['semantic_slot']['accepted'] == [PRIVATE_ONLY]
    assert not (tmp_path/'three').exists()
    semantic, _, _, _, _ = model_data()
    path = tmp_path/'early.pt'
    torch.save(dict(family='semantic_attribute_controls_v1', plan=dict(settings=dict(bootstrap_steps=2, render_steps=2)),
                    progress=dict(step=3)), path)
    with pytest.raises(ValueError, match='completed fixed-final'):
        load_semantic_slot(path)


@pytest.mark.parametrize('three', [False, True])
def test_training_resume_across_layer_transition(tmp_path, three):
    # Real source loader, semantic weights, optimizer phases and renderer VJPs.
    # Final digital delivery is tested separately; this test avoids subprocesses.
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('comparison_baselines.run._execute'):
        source, held = build_fixture(tmp_path)
        arm = PRIVATE_ONLY if three else 'split128'
        semantic_args = options(source, held, tmp_path/'semantic', arm=arm)
        with patch.object(semantic_entry, 'diagnostic_plots'), patch.object(semantic_entry, 'write_plots'):
            semantic_entry.main(semantic_args)
        slot = tmp_path/'semantic'/arm/'checkpoints/final.pt'
        calls = []
        def assessment(model, scene, cache, args, out, **kw):
            calls.append((scene.name, kw['scope'], kw.get('actual', False), out.parent.name, kw['layers']))
            result, _ = decode_cached(model, scene, cache,
                [torch.where(ids >= 0, 1, 0) for ids in scene.group_ids], 10., 77)
            assert torch.isfinite(result).all()
            return []
        def args(name):
            return entry.parser().parse_args(['--semantic-checkpoint', str(slot), '--out', str(tmp_path/name),
                '--bootstrap-steps', '2', '--render-steps', '2', '--neighbor-steps', '0' if three else '2', '--hidden', '16',
                '--views-per-step', '2', '--validate-every', '2', '--save-every', '1', '--trials', '1',
                '--no-lpips', '--device', 'cpu', '--cpu-threads', '1', '--digital-python', sys.executable]
                + (['--three-tier-scratch'] if three else []))
        with patch.object(entry, 'assess', side_effect=assessment):
            entry.main(args('full'))
            resumed = args('resumed')
            entry.main(resumed, max_updates=5)  # Mid layer 2, after a fresh Adam transition.
            resumed.resume = True
            entry.main(resumed)
        a = torch.load(tmp_path/'full/checkpoints/latest.pt', weights_only=True)
        b = torch.load(tmp_path/'resumed/checkpoints/latest.pt', weights_only=True)
        for key in a['state']:
            torch.testing.assert_close(a['state'][key], b['state'][key], atol=0, rtol=0)
        assert a['progress'] == b['progress'] and a['progress']['step'] == (12 if three else 34)
        assert not any(name == 'drjohnson' and scope == 'validation' for name, scope, _, _, _ in calls)
        assert any(name == 'drjohnson' and actual for name, _, actual, _, _ in calls)
        if three:
            assert ('truck', 'validation', False, 'layer_02_bootstrap_00002', [2]) in calls
            assert ('truck', 'validation', False, 'layer_03', [2, 3]) in calls
        assert (tmp_path/'full/review.tar.gz').exists()


@pytest.mark.parametrize('failed,rates,layer', [(False, EIGHT_RATES, 1), (True, EIGHT_RATES, 1),
                                               (False, THREE_RATES, 3), (True, THREE_RATES, 3)])
def test_evaluation_received_files_and_failed_cost(tmp_path, failed, rates, layer):
    import hashlib
    import shutil
    from comparison_baselines.digital_link import LinkConfig, file_channel_cost
    from gaussian_jscc import progressive_attribute_evaluation as evaluation
    model, geometry, f, y, active = fixture(rates=rates)
    raw = to_raw(f[active], geometry, model)
    ids = torch.arange(16).reshape(2, 8).masked_fill(~active, -1)
    views = cameras('', split='train')[:2]
    scene = SimpleNamespace(name='tiny', geometry=geometry, groups=[f], group_ids=[ids],
        raw=raw, test_cameras=views, degree=3, white_background=False, reference=Reference(raw))
    args = entry.parser().parse_args(['--out', str(tmp_path), '--device', 'cpu',
                                     '--digital-device', 'cpu', '--trials', '1'])
    calls = []
    def execute(python, module, arguments, log):
        assert module == 'comparison_baselines.digital_link'
        params = {str(arguments[i]): arguments[i+1] for i in range(1, len(arguments), 2)}
        src, dst, report = [Path(params[k]) for k in ('--source', '--destination', '--report')]
        cfg = LinkConfig(**{k: (str(params['--'+k.replace('_', '-')]) if k == 'device'
            else int(params['--'+k.replace('_', '-')])) for k in LinkConfig.__dataclass_fields__})
        lost = failed and src.name == 'metadata.bin'
        if not lost:
            shutil.copyfile(src, dst)
        result = dict(status='failed' if lost else 'success', config=vars(cfg),
            source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), snr_db=float(params['--snr-db']),
            seed=int(params['--seed']), noiseless=False, **file_channel_cost(src.stat().st_size, cfg))
        report.write_text(json.dumps(result)); calls.append(src.name)
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch.object(evaluation, '_execute', side_effect=execute):
        rows = evaluation.assess(model, scene, [torch.cat((f, y), -1)], args,
            tmp_path/'evaluation/final/tiny', scope='test', layers=[layer], actual=True)
    assert calls == ['xyz.bin', 'metadata.bin', 'scales.bin']
    row = next(r for r in rows if r['path'] == 'awgn')
    delivered = row['actual_delivery']
    assert delivered['status'] == ('failed' if failed else 'success')
    assert (delivered['metrics'] is None) == failed
    assert delivered['total_channel_uses'] == row['accounting']['total_channel_uses']
    assert row['packet_measured']['payload_complex_symbols'] == int(active.sum())*rates[layer]
    assert row['accounting']['side_channel_uses'] == sum(v['total_channel_uses'] for v in delivered['streams'].values())
    entry.write_review(tmp_path)
    report = json.loads((tmp_path/'review_summary.json').read_text())
    assert report['actual_delivery'][0]['success_rate'] == (0 if failed else 1)
    assert not list(tmp_path.rglob('received.npy'))
    assert not list(tmp_path.rglob('xyz.bin'))
