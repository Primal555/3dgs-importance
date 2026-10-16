import tempfile
import contextlib
import io
import json
import statistics
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
    def test_ssim_and_lpips_cover_all_paths_references_and_trial_aggregation(self):
        from gaussian_jscc.modular_evaluation import evaluate_paths
        from test_perceptual_metrics import backend
        m,g,f,q=setup(); scene=scene_fixture(m,g,f)
        before={n:p.clone() for n,p in m.named_parameters()}; rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as t,patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            base=evaluate_paths(m,scene,step=0,phase='initial',snrs=[10],trials=2,seed=42,save_images=False,out=Path(t))
            with patch('gaussian_jscc.perceptual_metrics._load_lpips_model',side_effect=backend), contextlib.redirect_stdout(io.StringIO()) as stream:
                measured=evaluate_paths(m,scene,step=0,phase='final',snrs=[10],trials=2,seed=42,
                                        save_images=False,out=Path(t),lpips_metrics=True)
            self.assertIn('SSIM=',stream.getvalue()); self.assertIn('LPIPS=',stream.getvalue())
            saved=[json.loads(line) for line in (Path(t)/'metrics/validation.jsonl').read_text().splitlines()]
            self.assertEqual(len(saved),18)
        self.assertTrue(torch.equal(rng,torch.get_rng_state())); self.assertTrue(m.training)
        for n,p in m.named_parameters(): torch.testing.assert_close(p,before[n],rtol=0,atol=0)
        for a,b in zip(base,measured):
            self.assertEqual(a['source_psnr'],b['source_psnr'])
            self.assertEqual(a['source_ssim'],b['source_ssim'])
            self.assertEqual(a['latent_mse'],b['latent_mse'])
            self.assertNotIn('source_lpips',a)
            self.assertEqual(b['image_metric_protocol']['lpips']['network'],'vgg')
            for ref in ('source','photo'):
                key=ref+'_lpips'
                self.assertTrue(all(key in x for x in b['per_view_trial']))
                self.assertAlmostEqual(b[key],statistics.mean(x[key] for x in b['per_view_trial']))
                trial_means=[statistics.mean(x[key] for x in b['per_view_trial'] if x['trial']==trial)
                             for trial in range(b['trials'])]
                self.assertAlmostEqual(b[key+'_std_trials'],statistics.pstdev(trial_means))

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
