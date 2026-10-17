import unittest
import torch
from gaussian_jscc.data import Geometry


class XYZPayloadCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        from gaussian_jscc.xyz_payload_config import XYZPayloadConfig
        from gaussian_jscc.xyz_payload_codec import XYZPayloadCodec
        torch.manual_seed(42)
        cfg = XYZPayloadConfig(sh_degree=1,hidden=16,depth=1,grid_dim=4,levels=(2,),
            block_size=8,decoder_window=4,channel_hidden=24,context_centers=(4,2),
            context_neighbors=(3,4),rates=(0,2,4,8))
        model = XYZPayloadCodec(cfg)
        model.set_scene_geometry(Geometry([1,2,3],[10,2,4],16))
        return model

    def features(self):
        torch.manual_seed(7)
        f = torch.randn(1,5,23)*.1
        f[...,:3] = torch.tensor([[[.5,.5,.5],[1,1,1],[0,0,0],[.2,.3,.7],[.9,.8,.1]]])
        return f, torch.ones(1,5,dtype=torch.bool)

    def test_config_rejects_side_coordinates_and_inconsistent_dimensions(self):
        from gaussian_jscc.xyz_payload_config import XYZPayloadConfig
        for settings in ({'position_delivery':'quantized'},{'latent_dim':48},
                         {'channel_denoiser_mode':'spatial'},{'channel_neighbor_mode':'plain'}):
            with self.assertRaises(ValueError): XYZPayloadConfig(**settings)
        self.assertEqual(XYZPayloadConfig.from_dict(XYZPayloadConfig().to_dict()),XYZPayloadConfig())

    def test_payload_is_three_metric_coordinates_plus_attributes(self):
        model=self.model(); f,active=self.features()
        y=model.encode_representation(f,f[...,:3],active)
        self.assertEqual(tuple(y.shape),(1,5,51))
        torch.testing.assert_close(y[0,:3,:3],torch.tensor([[0.,0,0],[1,.2,.4],[-1,-.2,-.4]]))
        pred=model.decode_representation(y,active.long())
        torch.testing.assert_close(pred[...,:3],f[...,:3],atol=2e-7,rtol=0)
        changed=y.detach().clone(); changed[0,0,0]=.4
        pred2=model.decode_representation(changed,active.long())
        self.assertAlmostEqual(float(pred2[0,0,0].detach()),.7,places=6)
        with self.assertRaisesRegex(ValueError,'side|external'):
            model.decode_representation(y,active.long(),f[...,:3])

    def test_padding_cannot_leak_xyz_and_all_inactive_is_finite(self):
        model=self.model(); f,active=self.features(); active[0,-1]=False
        dirty=f.clone(); dirty[~active]=float('nan')
        got=model.forward_clean_batches(dirty,dirty[...,:3],active)
        self.assertTrue(torch.isfinite(got).all()); self.assertEqual(int(got[~active].count_nonzero()),0)
        torch.testing.assert_close(got[active],model.forward_clean_batches(f,f[...,:3],active)[active])
        out=model.forward_clean_batches(dirty,dirty[...,:3],torch.zeros_like(active))
        self.assertEqual(int(out.count_nonzero()),0)

    def test_attribute_condition_keeps_gradient_into_received_xyz(self):
        model=self.model(); f,active=self.features()
        model.semantic_decoder.attributes.xyz_embedding[-1].weight.data.fill_(.1)
        y=model.encode_representation(f,f[...,:3],active).detach().requires_grad_()
        pred=model.decode_representation(y,active.long())
        pred[...,3:].square().sum().backward()
        self.assertGreater(float(y.grad[...,:3].norm()),0)

    def test_lower_prefix_cannot_read_future_symbols(self):
        model=self.model(); f,active=self.features()
        z=model.encode_full(f[0],f[0,:,:3],active[0],10.)[None]
        q=torch.ones_like(active,dtype=torch.long)
        from gaussian_jscc.codec import prefix_mask
        mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(z)
        modified=z.clone(); modified[~mask]=1000
        a=model.channel_decoder(z,mask,q,10.)
        b=model.channel_decoder(modified,mask,q,10.)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        pred=model.forward_tier_batches(f,f[...,:3],torch.nn.functional.one_hot(q,4).float(),10.,'none')[0]
        self.assertTrue(torch.isfinite(pred).all())
        self.assertEqual(tuple(pred.shape),tuple(f.shape))


if __name__=='__main__': unittest.main()
