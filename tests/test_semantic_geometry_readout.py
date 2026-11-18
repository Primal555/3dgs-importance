from unittest.mock import patch

import torch

from gaussian_jscc import semantic_attribute_experiment as original
from gaussian_jscc.progressive_attribute_channel import load_semantic_slot
from gaussian_jscc.semantic_attribute_codec import FULL_GEOMETRY
from gaussian_jscc.semantic_geometry_readout import run, widen
from test_semantic_attribute_controls import build_fixture, options, model_data, synthetic_render, cameras


def test_migration_and_gradient_access():
    torch.set_num_threads(1)
    old, _, geometry, features, active = model_data()
    new = widen(old.state_dict(), old.cfg)
    new.set_scene_geometry(geometry)
    torch.testing.assert_close(new(features, active), old(features, active), atol=2e-6, rtol=1e-5)
    assert all(name.startswith('semantic_decoder.branches.geometry.') for name, param in new.named_parameters() if param.requires_grad)
    linear = new.semantic_decoder.branches['geometry'].input
    assert not linear.weight[:, 16:96].count_nonzero()
    new(features, active)[..., 3:11].square().mean().backward()
    assert linear.weight.grad[:, 16:96].abs().sum() > 0


def test_refinement_and_channel_handoff(tmp_path):
    torch.set_num_threads(1)
    with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render), patch(
            'gaussian_jscc.rendering.load_cameras', side_effect=cameras), patch.object(original, 'diagnostic_plots'):
        source, held = build_fixture(tmp_path)
        original.main(options(source, held, tmp_path/'source_run'))
        path = tmp_path/'source_run/split128/checkpoints/final.pt'
        before = torch.load(path, weights_only=True)['model']
        out = tmp_path/'full_geometry'
        run(path, out, steps=2, interval=1, device_name='cpu')
        network, saved = load_semantic_slot(out/FULL_GEOMETRY/'checkpoints/final.pt')
        assert saved['progress']['step'] == 2 and (out/'review.tar.gz').is_file()
        for key, value in network.state_dict().items():
            if not key.startswith('semantic_decoder.branches.geometry.'):
                assert torch.equal(value, before[key])
        assert network.semantic_decoder.branches['geometry'].input.weight[:, 16:96].count_nonzero()
