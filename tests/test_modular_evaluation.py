import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from test_modular_training import setup
from test_render_first import synthetic_render,Reference
import test_render_first as render_fixture
from gaussian_jscc.data import to_raw


def scene_fixture(model,g,f):
    cams=render_fixture.RenderFirstTests().cameras()
    return SimpleNamespace(groups=[f],group_ids=[torch.arange(16).reshape(2,8)],raw=to_raw(f.flatten(0,1),g,model),
                           geometry=g,degree=1,val_cameras=cams,reference=Reference(),white_background=False,channel_trained=False)


class ModularEvaluationTests(unittest.TestCase):
    def test_validation_is_rng_and_parameter_neutral(self):
        from gaussian_jscc.modular_evaluation import evaluate_paths
        m,g,f,q=setup(); scene=scene_fixture(m,g,f)
        before={n:p.clone() for n,p in m.named_parameters()}; rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            rows=evaluate_paths(m,scene,step=0,phase='initial',snrs=[10],trials=2,seed=42,save_images=True,out=Path(t))
            self.assertEqual(len(list((Path(t)/'images/initial').glob('*.png'))),10)
        self.assertTrue(m.training); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for n,p in m.named_parameters(): torch.testing.assert_close(p,before[n],rtol=0,atol=0)
        clean=next(r for r in rows if r['path']=='clean')
        self.assertIsNone(clean['payload_complex_symbols']); self.assertIsNone(clean['total_channel_uses'])
        finite=[r for r in rows if r['path']=='finite_noiseless']
        self.assertEqual(clean['latent_mse'],0.)
        for row in rows:
            self.assertAlmostEqual(row['latent_nmse'],row['latent_mse']/max(row['latent_target_power'],1e-12))
        self.assertEqual(len(finite),4); self.assertFalse(finite[0]['channel_trained'])
        self.assertEqual([r['payload_complex_symbols'] for r in finite[:3]],[64,192,384])
        self.assertGreater(finite[0]['xyz_bytes'],0); self.assertGreater(finite[0]['metadata_bytes'],0)

    def test_current_clean_remeasured_after_channel_render(self):
        from gaussian_jscc.modular_evaluation import evaluate_paths
        m,g,f,q=setup(); scene=scene_fixture(m,g,f)
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            a=evaluate_paths(m,scene,step=0,phase='semantic_render',snrs=[10],trials=1,seed=42,save_images=True,out=Path(t))[0]
            with torch.no_grad(): m.semantic_decoder.heads['dc'].bias.add_(5)
            b=evaluate_paths(m,scene,step=1,phase='channel_render',snrs=[10],trials=1,seed=42,save_images=True,out=Path(t))[0]
            self.assertNotEqual(a['source_psnr'],b['source_psnr'])
            self.assertEqual(len(list((Path(t)/'images').rglob('*.png'))),0)

    def test_final_all_views_two_trials(self):
        from gaussian_jscc.modular_evaluation import evaluate_paths
        m,g,f,q=setup(); scene=scene_fixture(m,g,f); scene.val_cameras=scene.val_cameras*2
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            rows=evaluate_paths(m,scene,step=8,phase='test',snrs=[0,5,10,15,20],trials=2,seed=42,save_images=False,out=Path(t))
        awgn=[r for r in rows if r['path']=='awgn']
        self.assertEqual(len(awgn),20)
        self.assertTrue(all(r['views']==4 and r['trials']==2 for r in awgn))
        self.assertTrue(all(r['actual_snr']==r['tx_condition_snr'] for r in awgn))
        self.assertTrue(all('source_psnr_std_trials' in r and 'latent_mse' in r for r in awgn))

if __name__=='__main__': unittest.main()
