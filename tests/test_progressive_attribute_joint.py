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
from gaussian_jscc.semantic_attribute_codec import PRIVATE_ONLY
from test_semantic_attribute_controls import build_fixture, cameras, model_data, synthetic_render


def copy_sides(sender, receiver, reports, config, snr, seed, runner):
    receiver.mkdir()
    uses = 0
    for file in ('metadata.bin', 'xyz.bin', 'scales.bin'):
        shutil.copyfile(sender/file, receiver/file)
        uses += file_channel_cost((sender/file).stat().st_size, config)['total_channel_uses']
    return dict(status='success', streams={}, side_channel_uses=uses)


def test_tier_semantic_adaptation_freezes_other_layers_and_resumes(tmp_path):
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
        torch.save(dict(family=joint.THREE_FAMILY, plan=plan, state=model.state_dict(), progress=dict(step=12)),
                   run/'checkpoints/final.pt')
        joint.json_write(run/'complete.json', dict(step=12, status='complete', rates=list(THREE_RATES), codec_id=identity))
        for spec in scenes:
            record = plan['scene_records'][spec['name']]
            scene = joint.make_scene(spec, *joint.read_ply(spec['ply']), model, Namespace(**inputs['settings']),
                                     run, old=record)
            joint.assess(model, scene, joint.cache_scene(model, scene)['batches'],
                Namespace(**settings, grouped_attribute_supervision=False), run/'evaluation/final'/scene.name,
                scope='test', layers=joint.LAYOUTS, actual=True)

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
    assert comparison['truck']['clean']['gain_db'] != 0.
    row = comparison['drjohnson']['layouts']['3']
    assert row['paired_successes'] == 1 and row['awgn']['gain_db'] != 0.
    assert row['noise_penalty_db']['reduction_db'] == pytest.approx(
        row['awgn']['gain_db']-row['finite_noiseless']['gain_db'])


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
