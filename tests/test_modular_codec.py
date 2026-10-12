import inspect
import unittest
import torch


class ModularEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def modules(self):
        try:
            from gaussian_jscc.modular_config import ModularCodecConfig
            from gaussian_jscc.modular_core import SemanticEncoder
        except ImportError as exc:
            self.fail(f'modular semantic interface missing: {exc}')
        return ModularCodecConfig, SemanticEncoder

    def test_config_defaults_and_invalid_dimensions(self):
        Config, Encoder = self.modules()
        cfg = Config()
        self.assertEqual((cfg.latent_dim,cfg.hidden,cfg.depth,cfg.decoder_window,cfg.attention_heads),
                         (48,96,2,32,4))
        self.assertEqual(cfg.rates,(0,4,12,24))
        self.assertEqual(cfg.decoder_geometry,'xyz')
        self.assertEqual(Config.from_dict(cfg.to_dict()),cfg)
        for kwargs in ({'latent_dim':0},{'hidden':17},{'position_bits':12},
                       {'position_delivery':'learned'},{'architecture':'learned_joint'}):
            with self.assertRaises(ValueError): Config(**kwargs)

    def test_semantic_encoder_has_no_channel_conditions(self):
        Config, Encoder = self.modules()
        cfg=Config(sh_degree=1,hidden=16,grid_dim=4,levels=(2,),decoder_window=4,block_size=8,latent_dim=7)
        model=Encoder(cfg)
        self.assertEqual(list(inspect.signature(model.forward).parameters),['features','xyz','active'])
        f=torch.randn(2,13,3+cfg.attr_dim); f[...,:3]=torch.rand(2,13,3)
        active=torch.ones(2,13,dtype=torch.bool); active[:,9:]=False
        a=model(f,f[...,:3],active); b=model(f,f[...,:3],active)
        self.assertEqual(a.shape,(2,13,7)); torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(int(a[~active].count_nonzero()),0)
        changed=f.clone(); changed[~active]=1e4
        torch.testing.assert_close(a,model(changed,changed[...,:3],active),rtol=0,atol=0)
        self.assertFalse(torch.allclose(a[active].square().mean(-1), torch.ones_like(a[active].mean(-1))))
        a.square().sum().backward(); self.assertGreater(float(model.readout.weight.grad.norm()),0)


if __name__=='__main__': unittest.main()
