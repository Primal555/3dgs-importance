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

    def test_loss_and_latent_diagnostics_are_separate(self):
        from gaussian_jscc.modular_training import configure_phase,bootstrap_step,finish_update,predict
        from gaussian_jscc.local_response import local_response_loss
        m,g,f,q=setup(); opt=configure_phase(m,'channel_bootstrap',{})
        directions=torch.eye(3)
        torch.manual_seed(9); pred=predict(m,f,q,10,'awgn','channel_bootstrap')
        expected,_=local_response_loss(pred.flatten(0,1),f.flatten(0,1),g,m,directions=directions)
        torch.manual_seed(9); actual,diag=bootstrap_step(m,f,q,g,10,'awgn','channel_bootstrap',directions)
        torch.testing.assert_close(actual,expected.detach())
        self.assertIn('latent_mse',diag)
        before={n:p.detach().clone() for n,p in m.named_parameters()}
        next(p for p in m.parameters() if p.grad is not None).grad.fill_(float('nan'))
        with self.assertRaises(RuntimeError): finish_update(m,opt,before)
        for n,p in m.named_parameters(): torch.testing.assert_close(p,before[n])

if __name__=='__main__': unittest.main()
