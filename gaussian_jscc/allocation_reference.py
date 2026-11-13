"""Reuse a completed allocation's objective scales when changing calibration views."""
import copy
import hashlib
import json
from pathlib import Path

from comparison_baselines.side_information_accounting import side_information_cost
from .allocation_distortion import calibrate


def load_reference_anchor(args, scene_record, codec_config, source):
    root = Path(args.objective_reference_run).resolve()
    data = (root/'training.json').read_bytes()
    training = json.loads(data)
    complete = json.loads((root/'complete.json').read_text(encoding='utf8'))
    if complete.get('status') != 'complete' or not complete.get('frozen_codec_verified'):
        raise ValueError('objective reference requires a completed frozen-codec allocation')
    if args.cost_protocol != 'digital_side_file_v1' or not args.structured_evidence:
        raise ValueError('objective reference requires structured digital-file RD')
    if training['codec_config'] != json.loads(json.dumps(codec_config)):
        raise ValueError('objective reference codec configuration changed')
    for key in ('source_sha256', 'statistics_sha256'):
        if training['source'].get(key) != source.get(key):
            raise ValueError('objective reference source changed: '+key)
    for key, value in scene_record.items():
        if training.get(key) != json.loads(json.dumps(value)):
            raise ValueError('objective reference scene changed: '+key)
    for key in ('beta', 'snr', 'seed', 'trials', 'distortion', 'structure_share', 'lambda_s'):
        if training['settings'][key] != getattr(args, key):
            raise ValueError('objective reference setting changed: '+key)
    ref = copy.deepcopy(training['objective_reference'])
    if ref.get('cost_protocol') != 'digital_side_file_v1':
        raise ValueError('objective reference uses a different cost protocol')
    expected = calibrate(ref['mse_ref'], ref['structure_ref'], mode=args.distortion,
                         share=args.structure_share, lambda_s=args.lambda_s)
    if any(ref.get(key) != value for key, value in expected.items()):
        raise ValueError('objective reference distortion scales are inconsistent')
    cost = side_information_cost(ref['q3_cost'])
    if (ref['c_ref'] != cost['total_channel_uses'] or
            ref['q3_cost']['total_channel_uses'] != ref['c_ref'] or
            ref['beta'] != args.beta or
            any(complete.get(key) != ref[key] for key in ('d_ref', 'c_ref', 'lambda_s', 'beta'))):
        raise ValueError('objective reference RD ledger is inconsistent')
    ref.pop('side_proxy', None)
    return dict(run=str(root), training_sha256=hashlib.sha256(data).hexdigest(),
                codec_id=complete['codec_id'], reference=ref)


def anchored_reference(anchor, full_cost):
    ref = copy.deepcopy(anchor['reference'])
    for key in ('payload_complex_symbols', 'xyz_bytes', 'metadata_bytes',
                'scale_stream_bytes', 'total_channel_uses', 'cost_protocol'):
        if full_cost[key] != ref['q3_cost'][key]:
            raise ValueError('objective reference full-q3 accounting changed: '+key)
    ref['q3_cost'] = copy.deepcopy(full_cost)
    ref['normalization_source'] = dict(run=anchor['run'], training_sha256=anchor['training_sha256'],
        view_names=list(ref['calibration_views']), scope='fixed source-allocation scales')
    return ref
