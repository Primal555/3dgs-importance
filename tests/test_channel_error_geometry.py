"""Joint error directions and decoder response sensitivity have exact fixtures."""
import unittest
import torch


class ErrorGeometryTests(unittest.TestCase):
    def test_covariance_distinguishes_random_cross_dim_error_from_point_bias(self):
        from gaussian_jscc.latent_error_diagnostics import error_moments
        y=torch.zeros(2,2,dtype=torch.double)
        noise=torch.tensor([[[1.,2.],[2.,4.]],[[-1.,-2.],[-2.,-4.]]],dtype=torch.double)
        noise[...,0]+=1
        r=error_moments(y,y,noise)
        joint=r['joint_error']
        torch.testing.assert_close(torch.tensor(joint['noise_covariance']),torch.tensor([[2.5,5.],[5.,10.]]))
        torch.testing.assert_close(torch.tensor(joint['point_mean_error_second_moment']),torch.tensor([[1.,0.],[0.,0.]]))
        torch.testing.assert_close(torch.tensor(joint['error_second_moment']),torch.tensor([[3.5,5.],[5.,10.]]))
        self.assertAlmostEqual(joint['decomposition_max_abs_residual'],0.,places=12)
        self.assertAlmostEqual(r['summary']['noisy_mse'],6.75)
        self.assertAlmostEqual(joint['noise_correlation'][0][1],1.)

    def test_finite_difference_gram_and_risk_include_cross_dimension_terms(self):
        from gaussian_jscc.channel_error_geometry import finite_difference_gram,direction_risks
        def response(v):return torch.stack((2*v[0]+v[1],3*v[1]))
        point=torch.tensor([.3,-.2],dtype=torch.double)
        gram=finite_difference_gram(response,point,1e-3)
        torch.testing.assert_close(gram,torch.tensor([[2.,1.],[1.,5.]],dtype=torch.double),rtol=1e-10,atol=1e-10)
        moment=torch.tensor([[1.,-1.],[-1.,1.]],dtype=torch.double)
        report=direction_risks(moment,gram)
        self.assertAlmostEqual(report['total_linearized_risk'],5.,places=9)
        self.assertAlmostEqual(sum(r['risk'] for r in report['directions']),5.,places=9)
        self.assertAlmostEqual(sum(report['diagonal_only_contributions']),7.,places=9)
        torch.testing.assert_close(point,torch.tensor([.3,-.2],dtype=torch.double))

    def test_geometry_helpers_reject_nonfinite_or_incompatible_inputs(self):
        from gaussian_jscc.channel_error_geometry import finite_difference_gram,direction_risks
        for step in (0.,float('nan')):
            with self.assertRaises(ValueError): finite_difference_gram(lambda x:x,torch.ones(2),step)
        with self.assertRaises(ValueError):direction_risks(torch.eye(2),torch.eye(3))
        with self.assertRaises(ValueError):direction_risks(torch.eye(2),torch.full((2,2),float('nan')))

    def test_real_decoder_probe_keeps_original_blocks_rng_and_weights(self):
        import tempfile
        from pathlib import Path
        from gaussian_jscc.channel_error_geometry import diagnose_geometry
        from gaussian_jscc.modular_codec import ModularGaussianCodec
        from test_channel_spatial_denoising import config
        from test_modular_training import setup
        from test_modular_evaluation import scene_fixture
        torch.set_num_threads(1);torch.manual_seed(4)
        _,g,f,_=setup();model=ModularGaussianCodec(config(sh_degree=1,latent_dim=2))
        scene=scene_fixture(model,g,f)
        original={n:v.clone() for n,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
        sizes=[];hook=model.semantic_decoder.register_forward_pre_hook(lambda m,a:sizes.append(a[0].shape[1]))
        try:
            with tempfile.TemporaryDirectory() as t:
                report=diagnose_geometry(model,scene,Path(t),blocks=1,trials=2,color_views=2,relative_step=.01,seed=42)
        finally:hook.remove()
        self.assertTrue(model.training);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(set(sizes),{8})
        self.assertEqual(len(report['anchors']),1)
        anchor=report['anchors'][0]
        self.assertEqual({r['layout'] for r in anchor['layouts']},{'1','2','3','mixed'})
        self.assertTrue(anchor['gram_halfstep_relative_error']>=0)
        average=sum(torch.tensor(v) for v in anchor['gram_rgb'].values())/3
        torch.testing.assert_close(average,torch.tensor(anchor['gram']),rtol=1e-5,atol=1e-10)
        for row in anchor['layouts']:
            risk=row['directions']
            self.assertAlmostEqual(sum(v['risk'] for v in risk['directions']),risk['total_linearized_risk'],places=8)
            self.assertGreaterEqual(row['linearized_noiseless_risk'],0)
            self.assertEqual(set(row['linearized_risk_rgb']),{'r','g','b'})
        for n,v in original.items():torch.testing.assert_close(v,model.state_dict()[n],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
