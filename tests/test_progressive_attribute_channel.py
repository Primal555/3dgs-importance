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
from gaussian_jscc.progressive_attribute_supervision import FIELDS, WEIGHTS, attribute_errors, fit_color_scales, grouped_objective
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


def test_joint_highest_prefix_updates_every_channel_layer():
    model, _, f, y, active = fixture(PRIVATE_ONLY, THREE_RATES)
    optimizer = configure_stage(model, 'joint', 1e-3)
    frozen = frozen_digest(model)
    modules = [*model.encoders, *model.decoders]
    before = [state_digest(module.state_dict()) for module in modules]
    pred, yh, _ = model.forward_cached(f, y, active.long()*3, 10.)
    ((yh[active]-y[active]).square().mean()+pred[active][:, 3:].square().mean()).backward()
    optimizer.step()
    assert all(state_digest(module.state_dict()) != digest for module, digest in zip(modules, before))
    assert frozen_digest(model) == frozen
    assert not model.neighbor_enabled


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


@pytest.mark.parametrize('grouped', [False, True])
def test_full_scene_replay_gradient_matches_direct(grouped):
    model, geometry, f, y, active = fixture(PRIVATE_ONLY, THREE_RATES) if grouped else fixture()
    configure_stage(model, 2, 1e-4)
    other = copy.deepcopy(model)
    raw = to_raw(f[active], geometry, model)
    views = cameras('', split='train')[:2]
    scene = SimpleNamespace(geometry=geometry, degree=3, white_background=False,
                            reference=Reference(raw))
    args = entry.parser().parse_args(['--out', 'unused', '--device', 'cpu'])
    scales = fit_loss_scales([raw])
    if grouped:
        scales.update(fit_color_scales([raw]))
    data = torch.cat((f, y), -1)
    batches = [(data[i:i+1], active[i:i+1].long()*2) for i in range(2)]
    gradients, losses = [], []
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render):
        for net, replay in [(model, True), (other, False)]:
            torch.manual_seed(991)
            objective = PerceptualImageObjective('mse_ssim', args.ssim_weight, 'cpu')
            task = PerceptualMultiViewRenderTask(views, scene.reference, 3, False, objective)
            forward = lambda d, q: entry.batch_objective(net, scene, d, q, views, scales, args,
                                                          grouped_tier=2 if grouped else None)
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


def test_local_response_matches_clean_decoder_and_updates_only_active_channel():
    model, geometry, f, y, active = fixture(PRIVATE_ONLY, THREE_RATES)
    ids = torch.arange(active.numel()).reshape_as(active).masked_fill(~active, -1)
    scene = SimpleNamespace(geometry=geometry, groups=[f], group_ids=[ids])
    cache = entry.cache_scene(model, scene, local_response=True)
    torch.testing.assert_close(cache['batches'][0], torch.cat((f, y), -1))
    configure_stage(model, 2, 1e-4)
    q = 2*active.long()
    args = entry.parser().parse_args(['--out', 'unused', '--local-response-weight', '0.1'])
    scales = fit_loss_scales([to_raw(f[active], geometry, model)])
    cameras_used = cameras('', split='train')[:2]
    centers = torch.stack([c.camera_center for c in cameras_used])
    probe = {}
    torch.manual_seed(55)
    _, _, terms = entry.batch_objective(model, scene, cache['training_batches'][0], q,
        cameras_used, scales, args, response_centers=centers, response_seed=17, gradient_probe=probe)
    assert probe['response_latent_grad_norm'] > 0 and probe['baseline_latent_grad_norm'] > 0
    with torch.no_grad():
        clean = model.semantic.decode(y, active, delivered_positions(f[..., :3], q, model.cfg))
        torch.manual_seed(55)
        pred, _, _ = model.forward_cached(f, y, q, args.snr)
        views, _ = entry.color_directions(clean[active][:, :3], geometry, centers, 8, 17)
        expected, _ = entry.local_response_loss(pred[active], clean[active], geometry, model, directions=views)
        torch.testing.assert_close(terms['local_response'], expected)
    terms['local_response'].backward()
    for module in (model.encoders[1], model.decoders[1]):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


@pytest.mark.parametrize('supervise_latent', [False, True])
def test_grouped_supervision_decouples_rgb_and_controls_latent_loss(supervise_latent):
    model, geometry, f, y, active = fixture(PRIVATE_ONLY, THREE_RATES)
    source = to_raw(f[active], geometry, model)
    candidate = source.clone()
    candidate[:, 11:14] += .1
    candidate[:, 7:11] *= -1
    directions = torch.nn.functional.normalize(torch.randn(len(source), 4, 3), dim=-1)
    errors = attribute_errors(candidate, source, directions)
    assert all((errors[key] == 0).all() for key in ('shape', 'opacity', 'high_order_rgb'))
    assert (errors['base_rgb'] > 0).all()
    candidate[:, 11:14] = source[:, 11:14]
    candidate[:, 14:] += .1
    errors = attribute_errors(candidate, source, directions)
    assert (errors['base_rgb'] == 0).all() and (errors['high_order_rgb'] > 0).all()

    args = entry.parser().parse_args(['--out', 'unused', '--three-tier-scratch', '--grouped-attribute-supervision'])
    args.grouped_latent_supervision = supervise_latent
    scales = dict(fit_loss_scales([source]), **fit_color_scales([source]))
    tiers = torch.arange(len(source)) % 3 + 1
    loss, _ = grouped_objective(candidate, source, directions, scales, tiers)
    separate = sum(grouped_objective(candidate[tiers == k], source[tiers == k], directions[tiers == k], scales, k)[0]
                   * (tiers == k).float().mean() for k in WEIGHTS)
    torch.testing.assert_close(loss, separate)
    views = cameras('', split='train')[:2]
    scene = SimpleNamespace(geometry=geometry)
    configure_stage(model, 1, 1e-4)
    old = frozen_digest(model)
    data, tiers = torch.cat((f, y), dim=-1), active.long()
    torch.manual_seed(7)
    _, loss, terms = entry.batch_objective(model, scene, data, tiers, views, scales, args, grouped_tier=1)
    assert terms['latent'].requires_grad == supervise_latent
    attrs = sum(WEIGHTS[1][key] * terms[key] for key in FIELDS)
    torch.testing.assert_close(loss, terms['latent'] + .01*attrs if supervise_latent else attrs)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoders[0].parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoders[0].parameters())
    assert frozen_digest(model) == old
    assert all(p.grad is None for p in model.semantic.parameters())


def test_slot_unfilled_plan_and_reject_unfinished(tmp_path):
    args = entry.parser().parse_args(['--out', str(tmp_path/'absent'), '--dry-run'])
    plan = entry.main(args)
    assert plan['total_updates'] == 6200 and not (tmp_path/'absent').exists()
    three = entry.main(entry.parser().parse_args(['--out', str(tmp_path/'three'), '--dry-run', '--three-tier-scratch']))
    assert three['channel']['rates'] == list(THREE_RATES) and three['total_updates'] == 24750
    assert three['bootstrap_validation_steps'] == [500, 2000, 4000, 8000]
    assert three['semantic_slot']['accepted'] == [PRIVATE_ONLY]
    assert not (tmp_path/'three').exists()
    joint = entry.main(entry.parser().parse_args(['--out', str(tmp_path/'joint'), '--dry-run', '--three-tier-joint']))
    assert joint['channel'] == three['channel'] and joint['total_updates'] == 8250
    assert joint['supervision_per_tier'] == {'bootstrap': 8000, 'render': 250}
    assert joint['trainable_prefixes'] == [8, 24, 48]
    assert joint['bootstrap_validation_steps'] == three['bootstrap_validation_steps']
    assert not (tmp_path/'joint').exists()
    grouped_flags = ['--out', str(tmp_path/'grouped'), '--dry-run', '--three-tier-scratch', '--grouped-attribute-supervision']
    grouped = entry.main(entry.parser().parse_args(grouped_flags))
    assert all(grouped[key] == three[key] for key in ('channel', 'cost_protocol'))
    assert grouped['settings'] == dict(three['settings'], render_steps=500, latent_weight=0., attribute_weight=1.)
    assert grouped['settings']['validate_every'] == 250
    assert [p['steps'] for p in grouped['schedule']] == [8000, 500] * 3
    assert grouped['total_updates'] == 25500
    gated = entry.main(entry.parser().parse_args(grouped_flags + ['--q1-content-gate']))
    assert gated['schedule'] == grouped['schedule'][:2] and gated['total_updates'] == 8500
    assert gated['settings'] == grouped['settings'] and gated['q1_content_gate']['symbols_per_gaussian'] == 8
    average_500 = entry.main(entry.parser().parse_args([
        '--out', str(tmp_path/'average_500'), '--dry-run', '--three-tier-scratch',
        '--render-steps', '500', '--record-symbol-diagnostics']))
    assert average_500['total_updates'] == 25500 and average_500['record_symbol_diagnostics']
    assert average_500['settings'] == dict(three['settings'], render_steps=500)
    assert average_500['objective'] == three['objective']
    explicit = entry.main(entry.parser().parse_args(grouped_flags + ['--render-steps', '250']))
    assert explicit['schedule'] == three['schedule'] and explicit['total_updates'] == 24750
    assert grouped['attribute_supervision']['latent_weight'] == 0.
    assert grouped['attribute_supervision']['attribute_weight']['render'] == 1.
    assert grouped['attribute_supervision']['weights_by_complex_symbols'] == {
        str(symbols): WEIGHTS[i] for i, symbols in enumerate((8, 24, 48), 1)}
    equal = entry.main(entry.parser().parse_args(grouped_flags + ['--equal-attribute-weights']))
    assert equal['settings'] == grouped['settings'] and equal['schedule'] == grouped['schedule']
    assert equal['attribute_supervision']['weights_by_complex_symbols'] == {
        str(symbols): dict.fromkeys(FIELDS, 0.25) for symbols in (8, 24, 48)}
    combined = entry.main(entry.parser().parse_args(grouped_flags + ['--grouped-latent-supervision']))
    assert combined['schedule'] == grouped['schedule']
    assert combined['settings'] == dict(grouped['settings'], latent_weight=1., attribute_weight=.01)
    assert combined['attribute_supervision'] == dict(grouped['attribute_supervision'],
        mode='grouped_attributes_latent_v2', latent_weight=1.,
        attribute_weight=dict(bootstrap=.01, render=.01, neighbor=.01))
    with pytest.raises(ValueError, match='requires --grouped-attribute-supervision'):
        entry.main(entry.parser().parse_args(['--out', str(tmp_path/'invalid'), '--dry-run',
            '--three-tier-scratch', '--grouped-latent-supervision']))
    semantic, _, _, _, _ = model_data()
    path = tmp_path/'early.pt'
    torch.save(dict(family='semantic_attribute_controls_v1', plan=dict(settings=dict(bootstrap_steps=2, render_steps=2)),
                    progress=dict(step=3)), path)
    with pytest.raises(ValueError, match='completed fixed-final'):
        load_semantic_slot(path)


@pytest.mark.parametrize('mode', ['eight', 'three-tier-scratch', 'three-tier-joint',
                                  'grouped-attribute-supervision', 'equal-attribute-weights',
                                  'grouped-latent-supervision', 'q1-content-gate', 'local-response'])
def test_training_resume_across_layer_transition(tmp_path, mode):
    # Real source loader, semantic weights, optimizer phases and renderer VJPs.
    # Final digital delivery is tested separately; this test avoids subprocesses.
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('comparison_baselines.run._execute'):
        source, held = build_fixture(tmp_path)
        three = mode != 'eight'
        joint = mode == 'three-tier-joint'
        gated = mode == 'q1-content-gate'
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
        def args(name, profile=mode):
            flags = [] if profile == 'eight' else ['--'+profile]
            if profile == 'local-response':
                flags = ['--three-tier-scratch', '--attribute-weight', '0.02', '--local-response-weight', '0.1']
            if profile in ('q1-content-gate', 'equal-attribute-weights', 'grouped-latent-supervision'):
                flags.append('--grouped-attribute-supervision')
            if '--grouped-attribute-supervision' in flags:
                flags.append('--three-tier-scratch')
            return entry.parser().parse_args(['--semantic-checkpoint', str(slot), '--out', str(tmp_path/name),
                '--bootstrap-steps', '2', '--render-steps', '2', '--neighbor-steps', '0' if three else '2', '--hidden', '16',
                '--views-per-step', '2', '--validate-every', '2', '--save-every', '1', '--trials', '1',
                '--no-lpips', '--device', 'cpu', '--cpu-threads', '1', '--digital-python', sys.executable]
                + flags)
        with patch.object(entry, 'assess', side_effect=assessment):
            entry.main(args('full'))
            resumed = args('resumed')
            entry.main(resumed, max_updates=3 if joint or gated else 5)
            resumed.resume = True
            if mode in ('equal-attribute-weights', 'grouped-latent-supervision'):
                wrong = args('resumed', 'grouped-attribute-supervision')
                wrong.resume = True
                with pytest.raises(ValueError, match='same semantic checkpoint and experiment settings'):
                    entry.main(wrong)
            entry.main(resumed)
        a = torch.load(tmp_path/'full/checkpoints/latest.pt', weights_only=True)
        b = torch.load(tmp_path/'resumed/checkpoints/latest.pt', weights_only=True)
        for key in a['state']:
            torch.testing.assert_close(a['state'][key], b['state'][key], atol=0, rtol=0)
        expected = {'eight': 34, 'three-tier-scratch': 12, 'three-tier-joint': 4,
                    'grouped-attribute-supervision': 12, 'equal-attribute-weights': 12,
                    'grouped-latent-supervision': 12, 'q1-content-gate': 4, 'local-response': 12}
        assert a['progress'] == b['progress'] and a['progress']['step'] == expected[mode]
        if mode == 'local-response':
            rows = [json.loads(line) for line in (tmp_path/'full/metrics/loss.jsonl').read_text().splitlines()]
            for row in rows:
                auxiliary = row['latent_loss']+.02*row['attributes_loss']+.1*row['local_response_loss']
                assert row['loss'] == pytest.approx(auxiliary+row.get('render_loss', 0.), rel=1e-5)
                assert row['response_latent_grad_norm'] > 0 and row['baseline_latent_grad_norm'] > 0
        assert not any(name == 'drjohnson' and scope == 'validation' for name, scope, _, _, _ in calls)
        assert any(name == 'drjohnson' and actual for name, _, actual, _, _ in calls)
        if joint:
            assert ('truck', 'validation', False, 'joint_bootstrap_00002', [1, 2, 3]) in calls
            assert ('truck', 'validation', False, 'joint_render', [1, 2, 3]) in calls
            def views(run):
                path = tmp_path/run/'metrics/loss.jsonl'
                result = {}
                for row in map(json.loads, path.read_text().splitlines()):
                    tier_metrics = row.get('tier_metrics', {str(row['layer']): row})
                    result.update({(row['phase'], row['phase_step'], tier): item['views']
                                   for tier, item in tier_metrics.items()})
                return result
            with patch.object(entry, 'assess', side_effect=assessment):
                entry.main(args('frozen', 'three-tier-scratch'))
            assert views('full') == views('frozen')
            summary = json.loads((tmp_path/'full/review_summary.json').read_text())
            assert summary['training_mode'] == 'joint_prefix_mean'
        elif gated:
            assert all(layers == [1] for _, _, _, _, layers in calls)
            assert a['state']['encoders.0.gate.readout.weight'].abs().sum() > 0
            assert json.loads((tmp_path/'full/complete.json').read_text())['rates'] == [0, 8]
            with patch.object(entry, 'assess', side_effect=assessment):
                entry.main(args('ungated', 'grouped-attribute-supervision'), max_updates=1)
            # The identity gate preserves the first q1 update's data, noise and loss.
            gated_first, ungated_first = (json.loads((tmp_path/run/'metrics/loss.jsonl').read_text().splitlines()[0])
                                          for run in ('full', 'ungated'))
            assert gated_first['views'] == ungated_first['views']
            assert gated_first['loss'] == ungated_first['loss']
        elif three:
            assert ('truck', 'validation', False, 'layer_02_bootstrap_00002', [2]) in calls
            assert ('truck', 'validation', False, 'layer_03', [2, 3]) in calls
        if a['plan'].get('attribute_supervision'):
            rows = [json.loads(line) for line in (tmp_path/'full/metrics/loss.jsonl').read_text().splitlines()]
            for row in rows:
                weights = dict.fromkeys(FIELDS, 0.25) if mode == 'equal-attribute-weights' else WEIGHTS[row['layer']]
                attrs = sum(weights[k] * row[k+'_loss'] for k in FIELDS)
                auxiliary = row['latent_loss'] + .01*attrs if mode == 'grouped-latent-supervision' else attrs
                assert row['loss'] == pytest.approx(auxiliary + row.get('render_loss', 0.), rel=1e-6)
                if row['phase'] == 'render':
                    assert row['aux_loss'] == pytest.approx(auxiliary, rel=1e-6)
            assert json.loads((tmp_path/'full/complete.json').read_text())['attribute_supervision'] == (
                a['plan']['attribute_supervision']['mode'])
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
                                     '--digital-device', 'cpu', '--trials', '1', '--record-symbol-diagnostics']
                                     + (['--three-tier-scratch', '--grouped-attribute-supervision'] if rates == THREE_RATES else []))
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
    assert row['attribute_errors']['points'] == int(active.sum())
    assert set(row['attribute_errors']['mse']) == set(FIELDS)
    assert 'symbols' not in row['diagnostics']
    clean_row = next(r for r in rows if r['path'] == 'finite_noiseless')
    stats = clean_row['diagnostics']['symbols']
    assert stats['points'] == int(active.sum())
    assert len(stats['mean_power_by_symbol']) == rates[layer]
    assert sum(stats['mean_power_by_symbol'])/rates[layer] == pytest.approx(
        clean_row['diagnostics']['transmitted_power'], rel=1e-6)
    assert sum(stats['centered_power_by_symbol']) == pytest.approx(sum(stats['real_covariance_eigenvalues']))
    assert 1 <= stats['real_covariance_effective_rank'] <= min(rates[layer]*2, stats['points']-1) + 1e-8
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
