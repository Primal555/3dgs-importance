import copy
import unittest
import torch
from test_gaussian_jscc import fixture
from gaussian_jscc.data import prepare,to_features
from gaussian_jscc.modular_config import ModularCodecConfig
from gaussian_jscc.modular_codec import ModularGaussianCodec


def setup():
    torch.set_num_threads(1)
    raw,_=fixture(16,1); raw,g,_=prepare(raw,16)
    model=ModularGaussianCodec(ModularCodecConfig(sh_degree=1,hidden=16,grid_dim=4,levels=(2,),block_size=8,decoder_window=4))
    f,_=to_features(raw,g,model)
    return model,g,f.reshape(2,8,-1),torch.full((2,8),2,dtype=torch.long)


class ModularTrainingTests(unittest.TestCase):
    def test_phase_trainable_sets_and_frozen_snapshots(self):
        from gaussian_jscc.modular_training import configure_phase,bootstrap_step,finish_update,PHASE_MODULES
        for phase in PHASE_MODULES:
            m,g,f,q=setup(); opt=configure_phase(m,phase,{n:1e-3 for n in PHASE_MODULES[phase]})
            self.assertEqual({n.split('.')[0] for n,p in m.named_parameters() if p.requires_grad},set(PHASE_MODULES[phase]))
            before={n:p.detach().clone() for n,p in m.named_parameters()}
            bootstrap_step(m,f,q,g,10,'awgn',phase)
            stats=finish_update(m,opt,before)
            for n,p in m.named_parameters():
                if n.split('.')[0] not in PHASE_MODULES[phase]: torch.testing.assert_close(p,before[n],rtol=0,atol=0)
            self.assertEqual(set(stats['updates']),{'semantic_encoder','channel_encoder','channel_decoder','semantic_decoder'})

    def test_frozen_decoder_keeps_channel_gradient(self):
        from gaussian_jscc.modular_training import configure_phase,bootstrap_step
        m,g,f,q=setup(); configure_phase(m,'channel_bootstrap',{})
        bootstrap_step(m,f,q,g,10,'none','channel_bootstrap')
        for name in ('channel_encoder','channel_decoder'):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in getattr(m,name).parameters() if p.grad is not None),0)
        self.assertTrue(all(p.grad is None for p in m.semantic_decoder.parameters()))

    def test_replay_matches_direct_all_phases(self):
        from gaussian_jscc.modular_training import configure_phase,render_step
        for phase in ('semantic_render','channel_render'):
            m,g,f,q=setup(); other=copy.deepcopy(m)
            for net in (m,other): configure_phase(net,phase,{})
            values=[]
            for mode,net in [('replay',m),('checkpoint',other)]:
                torch.manual_seed(84)
                values.append(render_step(net,[(f[:1],q[:1]),(f[1:],q[1:])],g,10,'awgn',lambda r:r.square().mean(),phase,mode=mode)[0])
            torch.testing.assert_close(values[0],values[1])
            for (n,p),(_,b) in zip(m.named_parameters(),other.named_parameters()):
                if p.grad is not None: torch.testing.assert_close(p.grad,b.grad,atol=1e-6,rtol=1e-4,msg=n)

    def test_channel_bootstrap_optimizes_latent_mse_and_rejects_nonfinite_update(self):
        from gaussian_jscc.modular_training import configure_phase,bootstrap_step,finish_update,predict
        m,g,f,q=setup(); opt=configure_phase(m,'channel_bootstrap',{})
        values={}
        hooks=[m.semantic_encoder.register_forward_hook(lambda mod,a,o:values.update(y=o.detach())),
               m.channel_decoder.register_forward_hook(lambda mod,a,o:values.update(y_hat=o.detach()))]
        torch.manual_seed(9); predict(m,f,q,10,'awgn','channel_bootstrap')
        for hook in hooks: hook.remove()
        expected=(values['y_hat']-values['y']).square().mean()
        torch.manual_seed(9); actual,diag=bootstrap_step(m,f,q,g,10,'awgn','channel_bootstrap')
        torch.testing.assert_close(actual,expected.detach())
        self.assertEqual(diag['objective'],'latent_mse')
        self.assertIn('latent_mse',diag)
        before={n:p.detach().clone() for n,p in m.named_parameters()}
        next(p for p in m.parameters() if p.grad is not None).grad.fill_(float('nan'))
        with self.assertRaises(RuntimeError): finish_update(m,opt,before)
        for n,p in m.named_parameters(): torch.testing.assert_close(p,before[n])

    def test_bootstrap_does_not_depend_on_semantic_decoder_or_attribute_targets(self):
        from gaussian_jscc.modular_training import configure_phase,bootstrap_step
        m,g,f,q=setup(); configure_phase(m,'channel_bootstrap',{})
        def unusable_decoder(module,args,output):
            raise AssertionError('latent-only bootstrap must not run semantic decoding')
        hook=m.semantic_decoder.register_forward_hook(unusable_decoder)
        try:
            loss,stats=bootstrap_step(m,f,q,g,10,'none','channel_bootstrap')
        finally:
            hook.remove()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(stats['latent_mse'],0)
        self.assertTrue(all(p.grad is None for p in m.semantic_encoder.parameters()))

    def test_joint_latent_gradient_survives_zero_render_gradient_and_padding(self):
        from gaussian_jscc.modular_training import configure_phase,render_step
        m,g,f,q=setup(); other=copy.deepcopy(m)
        q[0]=torch.tensor([1,2,3,1,2,3,1,2]); q[1]=torch.tensor([3,0,0,0,0,0,0,0])
        for net in (m,other): configure_phase(net,'channel_render',{})
        values={}
        hooks=[other.semantic_encoder.register_forward_hook(lambda mod,a,o:values.update(y=o)),
               other.channel_decoder.register_forward_hook(lambda mod,a,o:values.update(y_hat=o))]
        from gaussian_jscc.modular_training import predict
        predict(other,f,q,10,'none','channel_render')
        expected=(values['y_hat'][q>0]-values['y'][q>0].detach()).square().mean()
        weight=.25
        (weight*expected).backward()
        for hook in hooks: hook.remove()
        actual,stats=render_step(m,[(f[:1],q[:1]),(f[1:],q[1:])],g,10,'none',lambda raw:raw.sum()*0,'channel_render',channel_latent_weight=weight)
        torch.testing.assert_close(actual,weight*expected.detach())
        self.assertAlmostEqual(stats['latent_mse'],float(expected.detach()),places=6)
        self.assertAlmostEqual(stats['weighted_latent_loss'],weight*float(expected.detach()),places=6)
        self.assertEqual(stats['render_loss'],0.)
        self.assertEqual(set(stats['latent_mse_by_tier']),{'1','2','3'})
        for (name,p),(_,ref) in zip(m.named_parameters(),other.named_parameters()):
            if name.startswith(('channel_encoder.','channel_decoder.')):
                self.assertIsNotNone(p.grad)
                torch.testing.assert_close(p.grad,ref.grad,atol=1e-6,rtol=1e-4,msg=name)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.channel_decoder.parameters()),0)

    def test_latent_target_is_stop_gradient_and_padding_is_excluded(self):
        from gaussian_jscc.modular_training import latent_reconstruction_loss
        y=torch.tensor([[[1.,2.],[100.,100.]]],requires_grad=True)
        yh=torch.tensor([[[3.,1.],[-100.,-100.]]],requires_grad=True)
        q=torch.tensor([[2,0]])
        loss,_=latent_reconstruction_loss(y,yh,q)
        self.assertEqual(float(loss.detach()),2.5)
        loss.backward()
        self.assertIsNone(y.grad)
        torch.testing.assert_close(yh.grad,torch.tensor([[[2.,-1.],[0.,0.]]]))

if __name__=='__main__': unittest.main()
