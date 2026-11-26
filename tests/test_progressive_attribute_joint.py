"""Sequential semantic adaptation, frozen channel prefixes and paired photo endpoints."""
import json
import shutil
import sys
from argparse import Namespace
from unittest.mock import patch

import pytest
import torch

from comparison_baselines.digital_link import file_channel_cost
from gaussian_jscc import progressive_attribute_joint as joint
from gaussian_jscc.progressive_attribute_channel import LayerConfig, LayeredAttributeCodec, THREE_RATES
from gaussian_jscc.progressive_attribute_supervision import assess_attributes, summarize_attribute_recovery
from gaussian_jscc.semantic_attribute_codec import PRIVATE_ONLY
from test_semantic_attribute_controls import build_fixture, cameras, model_data, synthetic_render


def copy_sides(sender, receiver, reports, config, snr, seed, runner):
    receiver.mkdir()
    uses = 0
    for file in ('metadata.bin', 'xyz.bin', 'scales.bin'):
        shutil.copyfile(sender/file, receiver/file)
        uses += file_channel_cost((sender/file).stat().st_size, config)['total_channel_uses']
    return dict(status='success', streams={}, side_channel_uses=uses)


@pytest.fixture
def joint_source(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(101)
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('gaussian_jscc.progressive_attribute_evaluation.transmit_sides', side_effect=copy_sides):
        run, held = build_fixture(tmp_path)
        inputs = json.loads((run/'training.json').read_text())
        scenes = inputs['selected_scenes']+json.loads(held.read_text())['scenes']
        semantic, *_ = model_data(PRIVATE_ONLY)
        model = LayeredAttributeCodec(semantic, LayerConfig(rates=THREE_RATES, hidden=16, neighbor_hidden=8))
        settings = dict(snr=10., seed=42, trials=1, lpips=False, digital_device='cpu', cpu_threads=1,
                        views_per_step=2, digital_python=sys.executable)
        plan = dict(semantic_slot=dict(arm=PRIVATE_ONLY, config=semantic.cfg.to_dict()),
                    selected_scenes=scenes, scene_records=inputs['scene_records'], total_updates=12,
                    channel=model.layer_config.to_dict(), settings=settings, cost_protocol='digital_side_file_v1')
        identity = model.identity()
        (run/'checkpoints').mkdir()
        scales = dict.fromkeys(('shape', 'opacity', 'dc', 'sh', 'direction_rgb'), 1.)
        torch.save(dict(family=joint.THREE_FAMILY, plan=plan, state=model.state_dict(),
                        progress=dict(step=12), loss_scales=scales),
                   run/'checkpoints/final.pt')
        joint.json_write(run/'complete.json', dict(step=12, status='complete', rates=list(THREE_RATES), codec_id=identity))
        for spec in scenes:
            record = plan['scene_records'][spec['name']]
            scene = joint.make_scene(spec, *joint.read_ply(spec['ply']), model, Namespace(**inputs['settings']),
                                     run, old=record)
            cached = joint.cache_scene(model, scene)['batches']
            evaluation = Namespace(**settings, grouped_attribute_supervision=False)
            out = run/'evaluation/final'/scene.name
            original_rows = joint.assess(model, scene, cached, evaluation, out, scope='test',
                                         layers=joint.LAYOUTS, actual=True)
            if scene.name == 'truck':
                with patch('gaussian_jscc.progressive_attribute_evaluation.transmit_sides',
                           side_effect=AssertionError('a completed delivery was repeated')):
                    updated = joint.assess(model, scene, cached, evaluation, out, scope='test',
                                           layers=joint.LAYOUTS, actual=True, detailed_attributes=True)
                for before, after in zip(original_rows, updated):
                    assert 'details' in after.pop('attribute_errors')
                    assert after == before

        yield run, model


def test_tier_semantic_adaptation_freezes_other_layers_and_resumes(tmp_path, joint_source):
    run, model = joint_source

    def args(name):
        return joint.parser().parse_args(['--run', str(run), '--out', str(tmp_path/name),
            '--steps-per-scene', '6', '--save-every', '2', '--validate-every', '1', '--device', 'cpu'])

    joint.main(args('full'))
    part = args('resumed')
    original = joint.update
    for stop in (3, 4):
        def interrupt(net, scene, tier, visit, views, config):
            if visit == stop:
                raise InterruptedError
            return original(net, scene, tier, visit, views, config)
        with patch.object(joint, 'update', side_effect=interrupt), pytest.raises(InterruptedError):
            joint.main(part)
        part.resume = True
    joint.main(part)

    checkpoints = [torch.load(tmp_path/name/'checkpoints/final.pt', weights_only=True)
                   for name in ('full', 'resumed')]
    a, b = checkpoints
    assert a['progress'] == b['progress'] and a['progress']['step'] == 12
    for name, value in a['state'].items():
        torch.testing.assert_close(b['state'][name], value, atol=0, rtol=0)
    output = part.out
    previous = model.state_dict()
    for tier in joint.TIERS:
        stage = torch.load(output/f'checkpoints/q{tier}_end.pt', weights_only=True)
        state = stage['state']
        changed = {k for k in state if not torch.equal(previous[k], state[k])}
        prefixes = ('semantic.semantic_encoder.', 'semantic.semantic_decoder.',
                    f'encoders.{tier-1}.', f'decoders.{tier-1}.')
        assert all(any(k.startswith(prefix) for k in changed) for prefix in prefixes)
        assert all(k.startswith(prefixes) for k in changed)
        assert stage['progress']['tier'] == tier
        assert stage['optimizer']['state'] and all(v['step'] == 4 for v in stage['optimizer']['state'].values())
        evaluated = json.loads((output/f'evaluation/q{tier}_end/truck/summary.json').read_text())
        assert {r['layer'] for r in evaluated['rows']} == set(joint.TIERS)
        attributes = json.loads((output/f'evaluation/q{tier}_end/truck/attribute_recovery.json').read_text())
        assert set(attributes['tiers']) == {'1', '2', '3'}
        assert attributes['awgn_basis'] == 'reliable side files'
        assert not (output/f'evaluation/q{tier}_end/drjohnson').exists()
        previous = state
    history = [json.loads(s) for s in (output/'metrics/loss.jsonl').read_text().splitlines()]
    for scene in ('truck', 'playroom'):
        assert [r['layout'] for r in history if r['scene'] == scene] == [1, 1, 2, 2, 3, 3]
    assert not (output/'evaluation/monitor_000002/drjohnson').exists()
    clean = [json.loads((output/'evaluation'/label/'truck/clean.json').read_text())
             for label in ('monitor_000002', 'q1_end')]
    assert clean[0]['photo_psnr'] != clean[1]['photo_psnr']
    comparison = json.loads((output/'photo_comparison.json').read_text())
    assert set(comparison) == {'truck', 'playroom', 'drjohnson'}
    attributes = json.loads((output/'evaluation/final/drjohnson/attribute_recovery.json').read_text())
    assert attributes['awgn_basis'] == 'successful actual deliveries'
    assert attributes['tiers']['3']['awgn_trials_used'] == 1
    assert comparison['truck']['clean']['gain_db'] != 0.
    row = comparison['drjohnson']['layouts']['3']
    assert row['paired_successes'] == 1 and row['awgn']['gain_db'] != 0.
    assert row['noise_penalty_db']['reduction_db'] == pytest.approx(
        row['awgn']['gain_db']-row['finite_noiseless']['gain_db'])


def test_rotating_adaptation_keeps_encoder_fixed_and_resumes_adam(tmp_path, joint_source):
    run, source = joint_source
    def args(name):
        return joint.parser().parse_args(['--run', str(run), '--out', str(tmp_path/name),
            '--rotate-tiers', '--steps-per-scene', '8', '--save-every', '1',
            '--validate-every', '4', '--device', 'cpu'])

    expected_tiers = {1: (1,), 2: (2,), 3: (3,), 'mixed': (1, 2, 3)}
    previous = source.state_dict()
    save = joint.save
    def check_step(path, model, optimizer, progress, plan, scales):
        nonlocal previous
        state = model.state_dict()
        step = progress['step']
        if path.stem == 'latest' and step:
            layout = joint.LAYOUTS[((step-1)//2) % 4]
            prefixes = ['semantic.semantic_decoder.']
            for tier in expected_tiers[layout]:
                prefixes.extend((f'encoders.{tier-1}.', f'decoders.{tier-1}.'))
            changed = [key for key in state if not torch.equal(state[key], previous[key])]
            assert all(key.startswith(tuple(prefixes)) for key in changed)
            assert all(any(key.startswith(prefix) for key in changed) for prefix in prefixes)
            assert all(not p.requires_grad for p in model.semantic.semantic_encoder.parameters())
            assert not model.semantic.semantic_encoder.training
        previous = {name: value.clone() for name, value in state.items()}
        save(path, model, optimizer, progress, plan, scales)

    full = args('rotating')
    with patch.object(joint, 'save', side_effect=check_step):
        joint.main(full)
    half = args('resumed_rotation')
    original = joint.update
    def interrupt(net, scene, tier, visit, views, config):
        if tier == 'mixed' and visit == 8:
            raise InterruptedError
        return original(net, scene, tier, visit, views, config)
    with patch.object(joint, 'update', side_effect=interrupt), pytest.raises(InterruptedError):
        joint.main(half)
    half.resume = True
    joint.main(half)

    a, b = [torch.load(x.out/'checkpoints/final.pt', weights_only=True) for x in (full, half)]
    assert a['progress'] == b['progress'] and a['progress']['step'] == 16
    torch.testing.assert_close(a['state'], b['state'], atol=0, rtol=0)
    torch.testing.assert_close(a['optimizer'], b['optimizer'], atol=0, rtol=0)
    optimized = [name for name, _ in source.named_parameters()
                 if name.startswith(('semantic.semantic_decoder.', 'encoders.', 'decoders.'))]
    for index, state in a['optimizer']['state'].items():
        name = optimized[index]
        assert state['step'] == (16 if name.startswith('semantic.') else 8)
    history = [json.loads(line) for line in (half.out/'metrics/loss.jsonl').read_text().splitlines()]
    for scene in ('truck', 'playroom'):
        assert [r['layout'] for r in history if r['scene'] == scene] == list(joint.LAYOUTS)*2
    final = json.loads((half.out/'photo_comparison.json').read_text())
    assert final['drjohnson']['layouts']['mixed']['paired_successes'] == 1
    initial = json.loads((half.out/'evaluation/monitor_000000/truck/summary.json').read_text())
    assert {row['layer'] for row in initial['rows']} == set(joint.LAYOUTS)


def test_finite_noiseless_rotation_adds_visits_and_resumes_with_original_teacher(tmp_path, joint_source):
    run, source = joint_source

    def args(name):
        return joint.parser().parse_args(['--run', str(run), '--out', str(tmp_path/name),
            '--rotate-tiers', '--steps-per-scene', '20', '--finite-noiseless-every', '5',
            '--attribute-weight', '0.02', '--semantic-anchor-weight', '0.02', '--save-every', '1',
            '--validate-every', '24', '--device', 'cpu'])

    full, resumed = args('finite'), args('finite_resumed')
    update = joint.update
    teacher_state = source.semantic.state_dict()

    def check_update(net, scene, layout, visit, cameras, config, **options):
        teacher = options['teacher']
        if options['path'] == 'finite_noiseless':
            assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
            torch.testing.assert_close(teacher.state_dict(), teacher_state, atol=0, rtol=0)
        else:
            assert teacher is None
        with patch.object(net, 'transmit', wraps=net.transmit) as transmit:
            result = update(net, scene, layout, visit, cameras, config, **options)
        assert transmit.call_args_list
        assert all(call.kwargs['noiseless'] == (teacher is not None) for call in transmit.call_args_list)
        return result

    with patch.object(joint, 'update', side_effect=check_update):
        joint.main(full)

    def interrupt(net, scene, layout, visit, cameras, config, **options):
        if options['teacher'] is not None and layout == 2:
            raise InterruptedError
        return check_update(net, scene, layout, visit, cameras, config, **options)

    with patch.object(joint, 'update', side_effect=interrupt), pytest.raises(InterruptedError):
        joint.main(resumed)
    resumed.resume = True
    with patch.object(joint, 'update', side_effect=check_update):
        joint.main(resumed)

    a, b = [torch.load(x.out/'checkpoints/final.pt', weights_only=True) for x in (full, resumed)]
    assert a['progress'] == b['progress'] and a['progress']['step'] == 48
    assert a['plan']['awgn_steps_per_scene'] == 20
    assert a['plan']['finite_noiseless_steps_per_scene'] == 4
    torch.testing.assert_close(a['state'], b['state'], atol=0, rtol=0)
    torch.testing.assert_close(a['optimizer'], b['optimizer'], atol=0, rtol=0)
    history = [json.loads(row) for row in (resumed.out/'metrics/loss.jsonl').read_text().splitlines()]
    for scene in ('truck', 'playroom'):
        rows = [r for r in history if r['scene'] == scene]
        expected = [('awgn', layout) for layout in list(joint.LAYOUTS)*5]
        for index, layout in reversed(list(enumerate(joint.LAYOUTS, 1))):
            expected.insert(index*5, ('finite_noiseless', layout))
        assert [(r['path'], r['layout']) for r in rows] == expected
        assert a['progress']['cycles'][scene]['updates'] == 20
        assert a['progress']['cycles'][f'{scene}:finite_noiseless']['updates'] == 4
    for row in history:
        expected = row['latent_loss'] + 0.02*row['attributes_loss']
        if row['path'] == 'finite_noiseless':
            assert row['semantic_anchor_loss'] > 0
            expected += 0.02*row['semantic_anchor_loss']
        else:
            assert 'semantic_anchor_loss' not in row
        assert row['aux_loss'] == pytest.approx(expected, abs=1e-7)


def test_photo_gain_pairs_successful_trials_and_preserves_clean_change():
    def endpoint(clean, values):
        rows = {}
        for tier in joint.LAYOUTS:
            rows[tier, 'finite_noiseless', 0] = {'photo_psnr': 30.}
            for trial, value in enumerate(values):
                rows[tier, 'awgn', trial] = {'actual_delivery': dict(
                    status='failed' if value is None else 'success',
                    metrics=None if value is None else {'photo_psnr': value})}
        return {'photo_psnr': clean}, rows
    initial = endpoint(31., [27., 24., None])
    final = endpoint(30.5, [28., None, 29.])
    result = joint.photo_comparison(initial, final, 3)
    assert result['clean']['gain_db'] == -.5
    row = result['layouts']['3']
    assert row['paired_successes'] == 1 and row['attempted'] == 3
    assert row['awgn']['gain_db'] == row['noise_penalty_db']['reduction_db'] == 1.
    assert joint.photo_comparison(initial, endpoint(31., [None]*3), 3)['layouts']['1']['awgn'] is None


def test_attribute_details_separate_sh_degrees_and_condition_on_delivery():
    from types import SimpleNamespace
    source = torch.zeros(4, 59)
    source[:, 7] = 1.
    recovered = source.clone()
    recovered[:, 4:7] = 0.5
    recovered[:, 7] = -1.
    recovered[:, 14:].reshape(-1, 3, 15)[:, :, 3:8] = 2.
    views = [SimpleNamespace(camera_center=torch.tensor([0., 0., 1.]))]
    def measure(pred):
        return assess_attributes(pred, source, views, 'cpu', detailed=True)
    clean, noisy = measure(source), measure(recovered)
    details = noisy['details']
    assert details['sorted_log_scale_rmse'] == 0.5 and details['log_volume_bias'] == 1.5
    assert details['rotation_angle_mean_deg'] == 0.
    assert [details[f'sh_degree_{i}_coefficient_mse'] for i in (1, 2, 3)] == [0., 4., 0.]
    rows = [dict(layer=1, path='finite_noiseless', attribute_errors=clean),
            dict(layer=1, path='awgn', attribute_errors=noisy, actual_delivery={'status': 'success'}),
            dict(layer=1, path='awgn', attribute_errors=clean, actual_delivery={'status': 'failed'})]
    row = summarize_attribute_recovery(clean, rows, actual=True)['tiers']['1']
    assert row['awgn_trials_used'] == 1 and row['awgn_trials_attempted'] == 2
    assert row['awgn_minus_finite']['sh_degree_2_coefficient_mse'] == 4.
    assert summarize_attribute_recovery(clean, rows, False)['tiers']['1']['awgn_mean']['sh_degree_2_coefficient_mse'] == 2.
    rows[1]['actual_delivery']['status'] = 'failed'
    assert summarize_attribute_recovery(clean, rows, True)['tiers']['1']['awgn_mean'] is None
