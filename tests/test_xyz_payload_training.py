import tempfile
import unittest
from pathlib import Path
import torch
import test_xyz_payload_codec as codec_fixture


class XYZPayloadTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_group_loss_ignores_padding_and_is_not_flat_dimension_mean(self):
        from gaussian_jscc.xyz_payload_training import payload_loss
        y=torch.zeros(1,3,51); yh=torch.ones_like(y,requires_grad=True)
        y[0,2]=float('nan'); q=torch.tensor([[1,3,0]])
        loss,stats=payload_loss(y,yh,q)
        self.assertEqual(float(loss.detach()),2.)
        self.assertEqual(float(stats['latent_mse']),1.)
        loss.backward(); self.assertEqual(int(yh.grad[0,2].count_nonzero()),0)
        self.assertAlmostEqual(float(yh.grad[0,0,0]),1/3,places=6)
        self.assertAlmostEqual(float(yh.grad[0,0,3]),1/48,places=6)
        for w in (0,-1,float('nan')):
            with self.assertRaises(ValueError): payload_loss(y,yh,q,w)

    def test_scopes_and_real_channel_learning_keep_semantics_frozen(self):
        from gaussian_jscc.xyz_payload_training import configure_phase,bootstrap_step,finish_update
        helper=codec_fixture.XYZPayloadCodecTests(); model=helper.model(); f,active=helper.features()
        q=active.long()*3
        optimizer=configure_phase(model,'channel_bootstrap',{})
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        loss,stats=bootstrap_step(model,f,q,model_geometry(),10.,'none','channel_bootstrap')
        finish_update(model,optimizer,before)
        self.assertTrue(torch.isfinite(loss)); self.assertGreater(stats['payload_xyz_mse'],0)
        self.assertTrue(any(not torch.equal(p,before[n]) for n,p in model.named_parameters() if n.startswith('channel_')))
        for n,p in model.named_parameters():
            if n.startswith('semantic_'): torch.testing.assert_close(p,before[n],rtol=0,atol=0)
        configure_phase(model,'channel_render',{})
        self.assertTrue(all(not p.requires_grad for p in model.semantic_encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.semantic_decoder.parameters()))

    def test_replay_render_sends_center_gradient_through_channel(self):
        from gaussian_jscc.xyz_payload_training import configure_phase,render_step
        helper=codec_fixture.XYZPayloadCodecTests(); model=helper.model(); f,active=helper.features(); q=active.long()*3
        configure_phase(model,'channel_render',{})
        def objective(raw): return raw[:,:3].square().mean()+raw[:,3:].square().mean()*.001
        loss,stats=render_step(model,[(f,q)],model_geometry(),10.,'none',objective,'channel_render')
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(stats['scene_gradient_norms']['xyz'],0)
        self.assertGreater(sum(float(p.grad.norm()) for p in model.channel_decoder.parameters() if p.grad is not None),0)
        self.assertAlmostEqual(float(loss),stats['render_loss']+stats['weighted_latent_loss'],places=4)

    def test_checkpoint_restores_optimizer_rng_and_rejects_legacy(self):
        from gaussian_jscc.xyz_payload_training import configure_phase,bootstrap_step,finish_update
        from gaussian_jscc.xyz_payload_checkpoint import save_checkpoint,restore_checkpoint
        from gaussian_jscc.modular_checkpoint import restore_rng
        helper=codec_fixture.XYZPayloadCodecTests(); model=helper.model(); f,a=helper.features()
        optimizer=configure_phase(model,'channel_bootstrap',{})
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        bootstrap_step(model,f,a.long()*3,model_geometry(),10.,'none','channel_bootstrap')
        finish_update(model,optimizer,before)
        progress={'phase':'channel_bootstrap','module_lrs':{},'step':1,'phase_step':1}
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'model.pt'; save_checkpoint(path,model,optimizer,progress,{'geometry':model_geometry().to_dict()})
            expected=torch.rand(3)
            restored,opt,state,rng,scene=restore_checkpoint(path,'cpu')
            restore_rng(rng); torch.testing.assert_close(torch.rand(3),expected,rtol=0,atol=0)
            for x,y in zip(model.parameters(),restored.parameters()): torch.testing.assert_close(x,y,rtol=0,atol=0)
            self.assertEqual(len(opt.state),len(optimizer.state)); self.assertEqual(state['step'],1)
            torch.save({'version':4},path)
            with self.assertRaisesRegex(ValueError,'XYZ payload'): restore_checkpoint(path,'cpu')


def model_geometry():
    from gaussian_jscc.data import Geometry
    return Geometry([1,2,3],[10,2,4],16)


if __name__=='__main__': unittest.main()
