import copy
import importlib
import json
import tarfile

import pytest


def module():
    return importlib.import_module('comparison_baselines.side_information_accounting')


@pytest.mark.parametrize('size,words,padding', [(0,1,460), (460,1,0), (461,2,507), (968,2,0)])
def test_file_framing_boundaries(size, words, padding):
    from comparison_baselines.digital_link import LinkConfig, file_channel_cost
    result = file_channel_cost(size, LinkConfig())
    assert result['codewords'] == words
    assert result['padding_bytes'] == padding
    assert result['crc_bytes'] == 4 * words
    assert result['total_channel_uses'] == 1536 * words
    assert result['information_bits'] == words * 4096


@pytest.mark.parametrize('size', [-1, 1.5, True, 2**64])
def test_reject_invalid_file_lengths(size):
    from comparison_baselines.digital_link import LinkConfig, file_channel_cost
    with pytest.raises(ValueError):
        file_channel_cost(size, LinkConfig())


def test_three_streams_are_framed_separately_and_input_is_unchanged():
    row = dict(payload_complex_symbols=17080584, xyz_bytes=2989974,
               metadata_bytes=981, scale_stream_bytes=11476, total_channel_uses=29090308)
    before = copy.deepcopy(row)
    cost = module().side_information_cost(row)
    assert row == before
    assert cost['streams']['coordinates']['total_channel_uses'] == 9040896
    assert cost['streams']['metadata']['total_channel_uses'] == 4608
    assert cost['streams']['scales']['total_channel_uses'] == 35328
    assert cost['total_channel_uses'] == 26161416
    assert cost['side_link_simulated'] is False
    from comparison_baselines.digital_link import file_channel_cost
    combined = file_channel_cost(row['xyz_bytes'] + row['metadata_bytes'] + row['scale_stream_bytes'])
    assert cost['side_channel_uses'] == combined['total_channel_uses'] + 1536


def test_selected_cost_and_nondefault_modulation():
    from comparison_baselines.digital_link import LinkConfig
    row = dict(payload_complex_symbols=16219664, xyz_bytes=2643045,
               metadata_bytes=171498, scale_stream_bytes=17038)
    cost = module().side_information_cost(row)
    assert cost['total_channel_uses'] == 24782864
    qpsk = module().side_information_cost(row, LinkConfig(bits_per_symbol=2))
    assert qpsk['side_channel_uses'] == cost['side_channel_uses'] * 2
    assert qpsk['payload_complex_symbols'] == row['payload_complex_symbols']


def test_missing_scale_stream_is_not_silently_omitted():
    with pytest.raises(KeyError, match='scale_stream_bytes'):
        module().side_information_cost(dict(payload_complex_symbols=0, xyz_bytes=1, metadata_bytes=1))


def test_compare_report_keeps_historical_rd_and_actual_baseline_failures(tmp_path):
    views = ['a', 'b']
    allocation = tmp_path/'allocation'
    allocation.mkdir()
    row = dict(payload_complex_symbols=16219664, xyz_bytes=2643045, metadata_bytes=171498,
               scale_stream_bytes=17038, total_channel_uses=27545988, rd_score=1.229,
               photo_psnr=24.2647, photo_ssim=.857, photo_lpips=.179)
    (allocation/'review_summary.json').write_text(json.dumps(dict(rows=dict(selected=row))))
    (allocation/'training.json').write_text(json.dumps(dict(spec=dict(name='truck'),
        resolution=2, images='images', white_background=False, test_view_names=views)))
    p = allocation/'evaluation/selected'
    p.mkdir(parents=True)
    (p/'metrics.json').write_text(json.dumps(dict(identity=dict(snr=10, views=views))))
    from comparison_baselines.digital_link import LinkConfig, file_channel_cost
    from dataclasses import asdict
    baseline = tmp_path/'baseline'
    baseline.mkdir()
    uses = file_channel_cost(155722836, LinkConfig())['total_channel_uses']
    summary = dict(status='complete', plan=dict(scene='truck', arq=False,
        link=asdict(LinkConfig()), input=dict(dataset_identity='same'),
        evaluation=dict(resolution=2, images='images', white_background=False, camera_names=views)),
        results=[dict(method='maskgaussian', snr_db=10, trials=4, successes=3,
                      success_rate=.75, metrics=dict(photo_psnr=26.31), total_channel_uses=uses)],
        trials=[dict(method='maskgaussian', snr_db=10, payload_bytes=155722836,
                     total_channel_uses=uses, status=s) for s in ['success']*3+['failed']])
    (baseline/'summary.json').write_text(json.dumps(summary))
    report = module().build_report(allocation, [(baseline, 'maskgaussian')])
    ours = report['allocation']['selected']
    assert ours['comparison_accounting']['total_channel_uses'] == 24782864
    assert ours['historical_total_channel_uses'] == 27545988
    assert ours['historical_rd_score'] == 1.229
    assert report['baselines'][0]['success_rate'] == .75
    assert report['baselines'][0]['total_channel_uses'] == uses
    assert report['baselines'][0]['cost_saving_vs_baseline'] == pytest.approx(1-24782864/uses)
    assert json.loads((allocation/'review_summary.json').read_text())['rows']['selected'] == row
    archive = tmp_path/'review.tar.gz'
    with tarfile.open(archive, 'w:gz') as output:
        for path in allocation.rglob('*.json'):
            output.add(path, arcname=path.relative_to(allocation).as_posix())
    archived = module().build_report(archive, [(baseline, 'maskgaussian')])
    assert archived['allocation'] == report['allocation']
    summary['trials'][0]['total_channel_uses'] += 1
    (baseline/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='成本'):
        module().build_report(allocation, [(baseline, 'maskgaussian')])
    summary['trials'][0]['total_channel_uses'] -= 1
    summary['plan']['evaluation']['camera_names'] = ['wrong']
    (baseline/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='camera'):
        module().build_report(allocation, [(baseline, 'maskgaussian')])


def test_cli_rejects_output_overwriting_source(tmp_path):
    with pytest.raises(SystemExit) as error:
        module().main(['--allocation', str(tmp_path), '--baseline', str(tmp_path), 'maskgaussian',
                       '--out', str(tmp_path)])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []
