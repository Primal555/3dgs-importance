"""Exercise paired training, resume, and the mean-gate packet path on CPU."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import pytest
import torch

from comparison_baselines.digital_link import LinkConfig, file_channel_cost
from gaussian_jscc import progressive_attribute_experiment as channel
from gaussian_jscc import progressive_attribute_evaluation as evaluation
from gaussian_jscc import q1_content_gate_experiment as entry
from gaussian_jscc import semantic_attribute_experiment as semantic
from gaussian_jscc.progressive_attribute_supervision import WEIGHTS
from gaussian_jscc.semantic_attribute_codec import PRIVATE_ONLY
from test_semantic_attribute_controls import build_fixture, cameras, options, synthetic_render


def digital_fixture(python, module, arguments, log):
    assert module == 'comparison_baselines.digital_link'
    assert arguments[0] == 'transmit'
    params = dict(zip(arguments[1::2], arguments[2::2]))
    source, destination, report = [Path(params[k]) for k in ('--source', '--destination', '--report')]
    cfg = LinkConfig(device='cpu')
    snr = float(params['--snr-db'])
    success = snr != 4. or source.name != 'metadata.bin'
    if success:
        shutil.copyfile(source, destination)
    report.write_text(json.dumps(dict(status='success' if success else 'failed', config=vars(cfg),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), snr_db=snr,
        seed=int(params['--seed']), noiseless=False, **file_channel_cost(source.stat().st_size, cfg))))


def test_q1_paired_controls_and_resume(tmp_path):
    common = ['--views-per-step', '2', '--device', 'cpu', '--digital-device', 'cpu', '--cpu-threads', '1',
              '--bootstrap-steps', '2', '--render-steps', '2', '--save-every', '1', '--trials', '1',
              '--no-lpips', '--digital-python', sys.executable]
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
         patch('comparison_baselines.run._execute'), patch.object(entry, '_execute'), \
         patch.object(evaluation, '_execute', side_effect=digital_fixture):
        source, held = build_fixture(tmp_path)
        with patch.object(semantic, 'diagnostic_plots'), patch.object(semantic, 'write_plots'):
            semantic.main(options(source, held, tmp_path/'semantic', arm=PRIVATE_ONLY))
        slot = tmp_path/'semantic'/PRIVATE_ONLY/'checkpoints/final.pt'
        with patch.object(channel, 'assess', return_value=[]):
            channel.main(channel.parser().parse_args(['--out', str(tmp_path/'channel'), '--semantic-checkpoint',
                str(slot), '--three-tier-scratch', '--grouped-attribute-supervision', '--hidden', '16', *common]),
                max_updates=5)
        layer = tmp_path/'channel/checkpoints/layer_01.pt'
        args = ['--source', str(layer), *common]
        full = tmp_path/'full'
        resumed = tmp_path/'resumed'
        entry.main(entry.parser().parse_args(['--out', str(full), *args]))
        run_step = entry.train_update

        def interrupt(*values):
            if values[3]['phase'] == 'render' and values[4] == 1:
                raise RuntimeError('test interruption')
            return run_step(*values)

        with patch.object(entry, 'train_update', side_effect=interrupt), pytest.raises(RuntimeError, match='test interruption'):
            entry.main(entry.parser().parse_args(['--out', str(resumed), *args]))
        entry.main(entry.parser().parse_args(['--out', str(resumed), '--resume', *args]))

    initialization = json.loads((full/'execution_check.json').read_text())
    assert initialization['gate_parameters'] == 8416
    assert not any(initialization['initial_reconstruction_max_error'].values())
    logs = []
    for arm in entry.ARMS:
        a, b = [torch.load(root/arm/'checkpoints/final.pt', weights_only=True)['state'] for root in (full, resumed)]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        if arm == 'content_gate':
            assert a['encoders.0.gate.readout.weight'].abs().sum() > 0
        log = [json.loads(line) for line in (full/arm/'metrics/loss.jsonl').read_text().splitlines()]
        logs.append([{k: row[k] for k in ('step', 'scene', 'snr_db', 'noise_seed', 'views')} for row in log])
        for row in log:
            attrs = sum(weight*row[k+'_loss'] for k, weight in WEIGHTS[1].items())
            assert row['loss'] == pytest.approx(attrs+row.get('render_loss', 0.), rel=1e-6)
    assert logs[0] == logs[1]
    report = json.loads((full/'review_summary.json').read_text())
    for result in report['results']:
        baseline, gated, mean = [result['arms'][arm] for arm in (*entry.ARMS, 'mean_gate')]
        assert baseline['channel_uses'] == gated['channel_uses'] == mean['channel_uses']
        assert mean['actual_trials'] == 0
        if result['path'] == 'awgn':
            expected = int(result['snr_db'] != 4.)
            assert baseline['actual_success_rate'] == gated['actual_success_rate'] == expected
            assert (gated['actual_success_conditioned_quality'] is not None) == bool(expected)
    assert (resumed/'review.tar.gz').is_file()
