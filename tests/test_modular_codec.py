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


class ModularPathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        from gaussian_jscc.modular_config import ModularCodecConfig
        from gaussian_jscc.modular_codec import ModularGaussianCodec
        self.cfg = ModularCodecConfig(sh_degree=1, hidden=16, grid_dim=4, levels=(2,), decoder_window=4)
        self.model = ModularGaussianCodec(self.cfg)
        self.f = torch.randn(2,13,3+self.cfg.attr_dim)
        self.f[...,:3] = torch.rand(2,13,3)
        self.q = torch.full((2,13),3,dtype=torch.long)
        self.q[:,4:8] = 0
        self.q[:,-1] = 0

    def test_clean_bypasses_channel_but_noiseless_executes_it(self):
        calls=[]
        hooks=[getattr(self.model,n).register_forward_hook(lambda m,a,o: calls.append(type(m).__name__))
               for n in ('channel_encoder','channel_decoder')]
        self.model.forward_clean_batches(self.f,self.f[...,:3],self.q>0)
        self.assertEqual(calls,[])
        self.model.forward_tier_batches(self.f,self.f[...,:3],torch.nn.functional.one_hot(self.q,4).float(),10,'none')
        self.assertEqual(calls,['ChannelEncoder','ChannelDecoder'])
        for h in hooks: h.remove()

    def test_progressive_prefix_and_power(self):
        from gaussian_jscc.codec import unpack
        f=self.f[0]; active=self.q[0]>0
        full=self.model.encode_full(f,f[:,:3],active,10)
        for tier in (1,2,3):
            q=active.long()*tier
            sent=unpack(self.model.encode(f,f[:,:3],q,10),q,self.cfg.rates)
            mask=torch.arange(48)[None]<2*self.cfg.rates[tier]
            torch.testing.assert_close(sent,full*mask)
        for a,b in zip(self.cfg.rates[:-1],self.cfg.rates[1:]):
            self.assertLessEqual(float(full[:,2*a:2*b].detach().square().sum(-1).max()/(b-a)),1.00001)
        for p in self.model.channel_encoder.parameters(): p.data.zero_()
        self.assertTrue(torch.isfinite(self.model.encode_full(f,f[:,:3],active,10)).all())

    def test_xyz_auxiliary_is_delivered_and_detached(self):
        from gaussian_jscc.position_delivery import delivered_positions
        side=delivered_positions(self.f[...,:3],self.q,self.cfg).requires_grad_()
        with torch.no_grad(): self.model.semantic_decoder.xyz_embedding[-1].weight.normal_(0,.1)
        y=torch.randn(2,13,self.cfg.latent_dim,requires_grad=True)
        a=self.model.semantic_decoder(y,self.q>0,side)
        b=self.model.semantic_decoder(y,self.q>0,side+.1)
        self.assertFalse(torch.allclose(a,b))
        a.square().sum().backward(); self.assertIsNone(side.grad)
        self.assertNotIn('xyz',self.model.semantic_decoder.heads)
        clean=self.model.forward_clean_batches(self.f,self.f[...,:3],self.q>0)
        noisy=self.model.forward_tier_batches(self.f,self.f[...,:3],torch.nn.functional.one_hot(self.q,4).float(),10,'none')[0]
        torch.testing.assert_close(clean[...,:3],side.detach()); torch.testing.assert_close(noisy[...,:3],side.detach())

    def test_q0_padding_tail_and_missing_side_information(self):
        from gaussian_jscc.codec import prefix_mask
        active=self.q>0; m=prefix_mask(self.q.flatten(),self.cfg.rates).reshape(2,13,48)
        z=torch.randn(2,13,48)
        a=self.model.channel_decoder(z,m,self.q,10)
        b=self.model.channel_decoder(z.masked_fill(~m,1e5),m,self.q,10)
        torch.testing.assert_close(a,b)
        for side in (None,torch.zeros(2,12,3),torch.full((2,13,3),float('nan'))):
            with self.assertRaises(ValueError): self.model.semantic_decoder(a,active,side)
        f=self.f.clone(); f[~active]=1e5
        clean=self.model.forward_clean_batches(self.f,self.f[...,:3],active)
        torch.testing.assert_close(clean,self.model.forward_clean_batches(f,f[...,:3],active))
        zero=self.model.forward_clean_batches(self.f,self.f[...,:3],torch.zeros_like(active))
        self.assertTrue(torch.isfinite(zero).all()); self.assertEqual(int(zero.count_nonzero()),0)

if __name__=='__main__': unittest.main()
