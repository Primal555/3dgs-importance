import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch

from gaussian_jscc.entropy_training import cache_latents, train_density


class EntropyTrainingTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def options(self):
        return SimpleNamespace(groups=[2, 2], hidden=12, mixtures=3, bin_width=.25,
            validation_fraction=.25, seed=42, batch_size=9, epochs=3, lr=.001,
            min_lr=.0001, grad_clip=5., device='cpu')

    def test_source_order_and_padding_exclusion(self):
        class Encoder(torch.nn.Module):
            def forward(self, f, xyz, active): return f*active[..., None]
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.anchor = torch.nn.Parameter(torch.tensor(1.))
                self.cfg = SimpleNamespace(latent_dim=4); self.semantic_encoder = Encoder()
        model = Model().eval().requires_grad_(False)
        ids = torch.tensor([[3, 1, -1], [0, 2, 4]])
        raw = torch.arange(20.).reshape(5, 4)
        f = raw[ids.clamp_min(0)]
        scene = SimpleNamespace(raw=raw, groups=[f], group_ids=[ids])
        y, blocks = cache_latents(model, scene)
        torch.testing.assert_close(y, raw)
        self.assertEqual(blocks.tolist(), [1, 0, 1, 0, 1])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_full_visits_and_exact_epoch_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            y = torch.randn(60, 4, generator=torch.Generator().manual_seed(3))
            blocks = torch.arange(12).repeat_interleave(5)
            full, full_log = train_density(y, blocks, self.options(), root/'full', {'id': 'test'})
            train_density(y, blocks, self.options(), root/'resume', {'id': 'test'}, max_epochs=1)
            resumed, log = train_density(y, blocks, self.options(), root/'resume', {'id': 'test'}, resume=True)
            for k, value in full.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[k], rtol=0, atol=0)
            self.assertEqual(log[-1]['epoch'], 3)
            self.assertEqual(log[-1]['minimum_training_visits'], 3)
            self.assertEqual(log[-1]['training_draws'], log[-1]['train_rows']*3)
            for key in ('train_bits_per_gaussian', 'validation_bits_per_gaussian', 'updates'):
                self.assertEqual(log[-1][key], full_log[-1][key])

    def test_cached_latent_matches_actual_successive_path_for_every_positive_map(self):
        from test_successive_allocation import small_successive
        raw,geometry,f,model=small_successive()
        model.eval().requires_grad_(False)
        ids=torch.tensor([5,2,7,0,8,1,4,3,6,9,10,11,15,14,13,12]).reshape(2,8)
        scene=SimpleNamespace(raw=raw,groups=[f],group_ids=[ids])
        y,_=cache_latents(model,scene)
        with torch.no_grad():
            for q in (torch.ones_like(ids),torch.full_like(ids,3),torch.arange(16).reshape(2,8)%3+1):
                choices=torch.nn.functional.one_hot(q,4).to(f)
                actual,_,_=model.forward_channel_batches(f,f[...,:3],choices,10.,'none')
                torch.testing.assert_close(y[ids],actual,rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_resume_rejects_changed_latent(self):
        with tempfile.TemporaryDirectory() as directory:
            y=torch.randn(60,4);blocks=torch.arange(12).repeat_interleave(5)
            train_density(y,blocks,self.options(),Path(directory),{'id':'test'},max_epochs=1)
            y[0,0]+=1
            with self.assertRaisesRegex(ValueError,'data/settings'):
                train_density(y,blocks,self.options(),Path(directory),{'id':'test'},resume=True)

    def test_first_epoch_interruption_resumes_from_initial_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);y=torch.randn(60,4);blocks=torch.arange(12).repeat_interleave(5)
            full,_=train_density(y,blocks,self.options(),root/'full',{'id':'test'})
            with patch('torch.optim.Adam.step',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    train_density(y,blocks,self.options(),root/'resume',{'id':'test'})
            recovered,_=train_density(y,blocks,self.options(),root/'resume',{'id':'test'},resume=True)
            for k,v in full.state_dict().items():
                torch.testing.assert_close(v,recovered.state_dict()[k],rtol=0,atol=0)


if __name__ == '__main__': unittest.main()
