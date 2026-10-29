import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gaussian_jscc.pixel_benefit_window import PixelWindow, decode_group, decode_groups


class Codec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(rates=(0, 1, 2, 3))
        self.calls = []

    def semantic_encoder(self, features, xyz, active):
        return active.float().sum(1, keepdim=True).unsqueeze(-1).expand(-1, features.shape[1], 1)

    def forward_tier_batches(self, features, xyz, choices, snr, kind, paired_noise=False,
                             _semantic_y=None, _channel_noise=None):
        q = choices.argmax(-1)
        if _channel_noise is None:
            _channel_noise = torch.randn((*q.shape, 6)) * (10 ** (-snr / 10) / 2) ** 0.5
        self.calls.append((q.clone(), _channel_noise.clone()))
        return (torch.cat((_semantic_y, q[..., None], _channel_noise[..., :1]), -1),)


class WindowTests(unittest.TestCase):
    def test_group_reencodes_mask_and_preserves_original_noise_row(self):
        model = Codec()
        features = torch.zeros(2, 2, 3)
        q = torch.tensor([[1, 1], [0, 1]])
        noise = torch.arange(24).reshape(2, 2, 6).float()
        y = model.semantic_encoder(features, features, q > 0)
        actual = model.forward_tier_batches(features, features, torch.nn.functional.one_hot(q, 4).float(),
                                             10, 'awgn', _semantic_y=y, _channel_noise=noise)[0]
        outputs, tiers = decode_group(model, features, q, 1, 0, actual, y, noise, 10)
        self.assertEqual(tiers[:, 0].tolist(), [0, 1, 2, 3])
        self.assertEqual(outputs[:, 1, 0].tolist(), [1, 2, 2, 2])
        self.assertTrue(all(torch.equal(call[1], noise[1:2]) for call in model.calls[1:]))
        self.assertEqual(len(model.calls), 4)

    def test_batched_groups_preserve_repeated_rows_and_noise(self):
        model = Codec()
        f = torch.zeros(2, 3, 3)
        q = torch.tensor([[1, 0, 2], [0, 1, 3]])
        noise = torch.arange(36).reshape(2, 3, 6).float()
        y = model.semantic_encoder(f, f, q > 0)
        actual = model.forward_tier_batches(f, f, torch.nn.functional.one_hot(q, 4).float(),
                                             10, 'awgn', _semantic_y=y, _channel_noise=noise)[0]
        rows, columns = torch.tensor([0, 0, 1]), torch.tensor([0, 1, 2])
        outputs, tiers = decode_groups(model, f, q, rows, columns, actual, y, noise, 10)
        self.assertEqual(outputs.shape, (3, 4, 3, 3))
        self.assertEqual(tiers[1, :, 1].tolist(), [0, 1, 2, 3])
        self.assertEqual(outputs[1, :, 0, 0].tolist(), [2, 3, 3, 3])
        self.assertEqual(len(model.calls), 4)
        self.assertTrue(all(torch.equal(call[1], noise[rows]) for call in model.calls[1:]))

    def test_pretrain_uses_full_multiview_groups_and_keeps_holdout_separate(self):
        from gaussian_jscc.pixel_benefit_replay import PixelReplay
        import tempfile
        model = Codec()
        model.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        model.eval()
        ids = torch.tensor([[0, 1], [2, 3]])
        features = torch.zeros(2, 2, 3)
        scene = SimpleNamespace(groups=[features], group_ids=[ids], geometry=None, degree=0,
                                white_background=False, reference=SimpleNamespace(get=lambda c, d: torch.zeros(3, 2, 2)))
        table = SimpleNamespace(logits=torch.zeros(4, 3), sample=lambda ids, snr: (torch.ones_like(ids),))
        scheduler = SimpleNamespace(validation=torch.tensor([True, False, False, False]),
                                    generator=torch.Generator().manual_seed(3),
                                    record=lambda selected, step: None)
        args = SimpleNamespace(snr=10, beta=0, label_groups=4)
        reference = {'d_ref': 1, 'lambda_s': 0, 'c_ref': 1,
                     'side_proxy': {'xyz_per_retained': 0, 'bits_per_use': 2}}
        camera_scene = SimpleNamespace(image=torch.zeros(3, 2, 2), projected=torch.ones(4, 12),
                                       project=lambda raw, existence: torch.zeros(len(raw), 12))
        def fake_group(*a, **kw):
            return {'inputs': {'valid': torch.ones(1, 1, 16, 16)}, 'density': torch.zeros(1, 3, 2, 16, 16),
                    'height': 2, 'width': 2, 'd_ref': 1, 'feature_seconds': 1., 'teacher_seconds': 2.}
        with tempfile.TemporaryDirectory() as root, \
             patch('gaussian_jscc.pixel_benefit_window.capture', return_value=camera_scene), \
             patch('gaussian_jscc.pixel_benefit_window.build_group', side_effect=fake_group), \
             patch('gaussian_jscc.pixel_benefit_window.raw_batches', side_effect=lambda p, g, m: p):
            replay = PixelReplay(root, 1, 20, 3)
            window = PixelWindow(model, scene, table, scheduler, reference, args)
            result = window.collect(1, [object(), object()], SimpleNamespace(scales=None), replay, True)
            groups = list(replay.training_records()) + list(replay.validation_records())
            self.assertEqual(len(groups), 4)
            self.assertTrue(all(len(group['views']) == 2 for group in groups))
            self.assertTrue(all(torch.equal(group['rates'], torch.zeros(3, dtype=torch.float64)) for group in groups))
            self.assertEqual(result['metrics']['teacher_seconds'], 2 * 2 * len({g['source'] for g in groups}))
            self.assertTrue(all(group['source'] == 0 for group in replay.validation_records()))
            self.assertTrue(all(group['source'] != 0 for group in replay.training_records()))
            self.assertEqual(result['q'].tolist(), [1, 1, 1, 1])
            learner = SimpleNamespace(scales=(1, 1), predict=lambda *args: torch.full((3, 2), 2.))
            window.collect(2, [object(), object()], learner, replay, True)
            self.assertEqual(float(window.difficulty[0]), 1.)
            self.assertTrue(any(float(value) > 1 for value in window.difficulty[1:]))
            replay.freeze_validation()
            scheduler.response_rounds = lambda step: [[torch.tensor([0, 1])]]
            before = len(list(replay.validation_records()))
            allocation = window.collect(3, [object(), object()], learner, replay, False)
            self.assertEqual(allocation['tier_ids'].tolist(), [0, 3])
            self.assertTrue(torch.equal(allocation['tier_costs'], torch.full((2, 3), 4., dtype=torch.float64)))
            self.assertEqual(len(list(replay.validation_records())), before)

    def test_real_codec_cpu_tiles_and_mirror_update(self):
        import tempfile
        from test_successive_allocation import small_successive
        from gaussian_jscc.pixel_benefit_replay import PixelReplay
        from gaussian_jscc.response_value import ResponseMirrorTable
        from gaussian_jscc.response_value_window import ResponseScheduler
        torch.set_num_threads(1)
        torch.manual_seed(21)
        _, geometry, features, model = small_successive()
        model.eval().requires_grad_(False)
        ids = [torch.arange(16).reshape(2, 8)]
        table = ResponseMirrorTable(16)
        scheduler = ResponseScheduler(ids, rounds=1)
        transform = torch.eye(4)
        transform[3, 2] = 3
        camera = SimpleNamespace(image_width=16, image_height=16, FoVx=1.2, FoVy=1.2,
                                 world_view_transform=transform, full_proj_transform=torch.eye(4),
                                 camera_center=torch.zeros(3))
        scene = SimpleNamespace(groups=[features], group_ids=ids, geometry=geometry, degree=1,
                                white_background=False,
                                reference=SimpleNamespace(get=lambda c, d: torch.zeros(3, 16, 16, device=d)))
        reference = {'d_ref': 1., 'lambda_s': .1, 'c_ref': 100.,
                     'side_proxy': {'xyz_per_retained': 1., 'bits_per_use': 2.}}
        args = SimpleNamespace(snr=10., beta=.1, label_groups=2, candidate_batch=4)
        learner = SimpleNamespace(scales=(1, 1), predict=lambda inputs, h, w, d: torch.zeros(3, 2))
        with tempfile.TemporaryDirectory() as root:
            replay = PixelReplay(root, 1, 20, 2)
            window = PixelWindow(model, scene, table, scheduler, reference, args)
            result = window.collect(1, [camera], learner, replay, False)
            table.update(result['keep_costs'], result['tier_ids'], result['tier_costs'], .01, .01, 16)
            groups = list(replay.training_records()) + list(replay.validation_records())
            self.assertGreater(len(groups), 0)
            self.assertTrue(all(torch.isfinite(g['rates']).all() for g in groups))
            self.assertTrue(all(g['views'][0]['density'].shape[1:] == (3, 2, 16, 16) for g in groups))
            self.assertTrue(all(torch.isfinite(g['views'][0]['density']).all() for g in groups))


if __name__ == '__main__':
    unittest.main()
