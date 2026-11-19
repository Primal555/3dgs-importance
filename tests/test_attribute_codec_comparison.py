"""Check fixed point identity, SH layout, view isolation and report resumption."""
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from comparison_baselines import attribute_codec_comparison as entry
from comparison_baselines.attribute_codec_worker import fcgs_roundtrip
from gaussian_jscc import semantic_attribute_experiment as training
from gaussian_jscc import semantic_attribute_metrics as semantic_metrics
from gaussian_jscc.semantic_attribute_codec import PRIVATE_ONLY
from test_semantic_attribute_controls import build_fixture, cameras, options, raw_points, synthetic_render


def test_fcgs_coding_layout_and_inverse_permutation(monkeypatch, tmp_path):
    # FCGS expects SH interleaved by coefficient. Project PLY uses color-major.
    raw = raw_points()
    raw[1, :3] = raw[0, :3]
    raw[:, 14:] = torch.arange(45) + 100 * torch.arange(len(raw))[:, None]
    expected = raw[:, 14:].reshape(-1, 3, 15).transpose(1, 2).flatten(1)

    class Codec:
        def __init__(self, **kwargs):
            pass

        def cuda(self):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state, strict):
            pass

        def compress(self, xyz, features, **kwargs):
            torch.testing.assert_close(features[:, 3:48], expected)
            assert upstream.sorted_orig_voxels(xyz, features)[0] is xyz
            assert upstream.compress_gaussian_params()[-1]['file_size']['total'] == 0
            torch.manual_seed(1)
            order = torch.randperm(len(xyz))
            self.encoded = features[order]
            return xyz[order], self.encoded

        def decomprss(self, **kwargs):
            xyz, = upstream.decompress_gaussian_params()
            torch.manual_seed(1)
            return xyz[torch.randperm(len(xyz))], self.encoded

    upstream = SimpleNamespace(FCGS=Codec, sorted_orig_voxels=None,
                               compress_gaussian_params=None, decompress_gaussian_params=None)
    monkeypatch.setitem(sys.modules, 'model', SimpleNamespace(FCGS_model=upstream))
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda x: x)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: {})
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'checkpoints' / 'checkpoint_0.0001.pkl').write_bytes(b'fixture')
    restored, _ = fcgs_roundtrip(tmp_path, raw.numpy(), tmp_path, 1e-4)
    np.testing.assert_array_equal(restored, raw.numpy())
    restored[0, 0] += 1
    with pytest.raises(ValueError, match='preserve shared XYZ'):
        list(entry.variants(raw, torch.from_numpy(restored)))


def test_comparison_uses_one_reference_and_resumes(tmp_path):
    torch.set_num_threads(1)
    semantic = tmp_path / 'private128'
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
            patch('gaussian_jscc.rendering.load_cameras', side_effect=cameras), \
            patch.object(training, 'write_plots'), patch.object(training, 'diagnostic_plots'):
        source, held = build_fixture(tmp_path)
        training.main(options(source, held, semantic, arm=PRIVATE_ONLY))

    repo = tmp_path / 'repo'
    (repo / '.git').mkdir(parents=True)
    out = tmp_path / 'comparison'
    args = entry.parser().parse_args([
        '--run', str(semantic), '--out', str(out), '--scenes', 'truck', 'drjohnson',
        '--fcgs-repo', str(repo), '--c3dgs-repo', str(repo),
        '--fcgs-python', sys.executable, '--c3dgs-python', sys.executable,
        '--image-views', '0', '--no-lpips',
    ])
    worker_calls = []
    semantic_xyz = {}
    seen_semantic = []

    def roundtrip(_, module, arguments, log):
        assert module == 'comparison_baselines.attribute_codec_worker'
        method = arguments[0]
        work = Path(arguments[arguments.index('--out') + 1])
        rows = np.load(work / 'input.npy')
        np.testing.assert_array_equal(rows[:, :3], semantic_xyz[log.parent.parent.name])
        if method == 'c3dgs':
            calibration = entry.read(work / 'cameras.json')
            name = log.parent.parent.name
            recorded = entry.read(semantic / 'scene_records.json')[name]
            used = {row['image_name'] for row in calibration}
            assert used and used.isdisjoint(recorded['test_view_names'] + recorded['validation_view_names'])
        rows[:, 4] += .1
        rows[:, 11] += .05
        np.save(work / 'decoded.npy', rows)
        (work / 'payload.json').write_text(json.dumps({'attribute_bytes': 1234}))
        assert (rows[:, 7:11] != 0).any()
        worker_calls.append(method)

    original_to_raw = semantic_metrics.to_raw

    def roundoff_world_xyz(*args):
        rows = original_to_raw(*args)
        rows[:, :3] += 1.52587890625e-05
        return rows

    def decode_with_shared_grid(model, scene, *, received_unit):
        canonical = scene.geometry.denormalize(received_unit)
        with patch.object(semantic_metrics, 'to_raw', side_effect=roundoff_world_xyz), \
                patch.object(model, 'decode', wraps=model.decode) as decode:
            rows = semantic_metrics.decode_scene(model, scene, received_unit=received_unit)
        receiver_xyz = torch.cat([side[active] for _, active, side in (c.args for c in decode.call_args_list)])
        assert torch.equal(receiver_xyz, received_unit)
        assert torch.equal(rows[:, :3], canonical)
        semantic_xyz[scene.name] = canonical.numpy()
        seen_semantic.append(scene.name)
        return rows

    with patch.object(entry, 'load_cameras', side_effect=cameras), \
            patch.object(entry, 'device_for', return_value=torch.device('cpu')), \
            patch.object(entry, 'decode_scene', side_effect=decode_with_shared_grid), \
            patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), \
            patch.object(entry, 'render', side_effect=synthetic_render), patch.object(entry, '_execute', side_effect=roundtrip):
        entry.main(args)
        args.resume = True
        entry.main(args)
    assert worker_calls == ['fcgs', 'c3dgs'] * 2
    assert seen_semantic == ['truck', 'drjohnson']
    for scene in ('truck', 'drjohnson'):
        records = entry.read(out / 'scenes' / scene / 'semantic' / 'metrics.json')
        assert records['payload'] is None
        assert records['xyz_input'] == 'shared_16bit_grid'
        assert set(records['variants']) == {'all_attributes', 'geometry_opacity_only', 'color_only'}
    with tarfile.open(out / 'review.tar.gz') as tar:
        assert 'review_summary.json' in tar.getnames()
        assert not any(n.endswith(('.npy', '.npz')) for n in tar.getnames())
