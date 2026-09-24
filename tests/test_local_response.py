"""Real local-response/channel gradients on CPU; scene renderer mocked only in CLI test."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from gaussian_jscc.local_response import local_response_loss
from gaussian_jscc.data import to_features, write_ply
from test_learned_joint import setup
from test_progressive_prefix import progressive
from test_render_first import synthetic_render
import test_render_first as render_fixture


class LocalResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        torch.manual_seed(12)
        return progressive(16)

    def test_equal_response_zero_and_no_teacher_gradient(self):
        _,g,f,m=self.fixture()
        pred=f.clone().requires_grad_()
        target=f.clone().requires_grad_()
        loss,_=local_response_loss(pred,target,g,m)
        self.assertLess(float(loss.detach()),1e-12)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(float(pred.grad[:,:3].abs().sum()),0.)

    def test_quaternion_sign_and_translation_invariance(self):
        raw,g,f,m=self.fixture()
        changed=raw.clone()
        changed[:,7:11]*=-1
        pred,_=to_features(changed,g,m)
        pred[:,:3]+=100
        loss,_=local_response_loss(pred,f,g,m)
        self.assertLess(float(loss),1e-12)

    def test_each_attribute_head_and_encoder_receive_gradients(self):
        _,g,f,m=self.fixture()
        q=torch.arange(len(f))%3+1
        pred=m(f,f[:,:3],q,10,'awgn')
        loss,_=local_response_loss(pred,f,g,m)
        loss.backward()
        for name in ('opacity','scale','rotation','dc'):
            grad=m.learned.heads[name].weight.grad
            self.assertTrue(torch.isfinite(grad).all(),name)
            self.assertGreater(float(grad.norm()),0,name)
        self.assertIsNone(m.learned.heads['xyz'].weight.grad)
        self.assertGreater(float(m.learned.symbol_head.weight.grad.norm()),0)
        self.assertGreater(float(m.learned.embeddings[1].weight.grad.norm()),0)

    def test_high_order_sh_and_extreme_anisotropy_finite(self):
        from test_gaussian_jscc import fixture
        from gaussian_jscc.codec import CodecConfig, GaussianCodec
        from gaussian_jscc.data import prepare
        raw,_=fixture(8,degree=3)
        raw,g,_=prepare(raw,16)
        m=GaussianCodec(CodecConfig(architecture='learned_joint',loss_profile='learned_v1',
                                   position_delivery='quantized',prefix_mode='progressive',sh_degree=3,hidden=16,depth=1,
                                   grid_dim=4,levels=(2,),decoder_window=4))
        m.attr_mean.copy_(raw[:,3:].mean(0))
        m.attr_std.copy_(raw[:,3:].std(0,unbiased=False).clamp_min(.01))
        f,_=to_features(raw,g,m)
        pred=m(f,f[:,:3],torch.ones(len(f),dtype=torch.long),10,'awgn')
        loss,_=local_response_loss(pred,f,g,m,views=8)
        loss.backward()
        grad=m.learned.heads['sh'].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()),0)
        raw[:,4:7]=torch.tensor([-20.,-20.,10.])
        extreme,_=to_features(raw,g,m)
        extreme=extreme.detach().requires_grad_()
        loss,_=local_response_loss(extreme,f,g,m)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_black_white_disambiguate_opacity_color(self):
        raw,g,f,m=self.fixture()
        from utils.sh_utils import RGB2SH
        raw[:,3]=torch.logit(torch.full((len(raw),),.4))
        raw[:,11:14]=RGB2SH(torch.full((len(raw),3),.5))
        target,_=to_features(raw,g,m)
        raw[:,3]=torch.logit(torch.full((len(raw),),.8))
        raw[:,11:14]=RGB2SH(torch.full((len(raw),3),.25))
        pred,_=to_features(raw,g,m)
        _,stats=local_response_loss(pred,target,g,m)
        self.assertLess(stats['local_black_mse'],1e-12)
        self.assertGreater(stats['local_white_mse'],1e-3)

    def test_batched_matches_individual_equal_weight(self):
        _,g,f,m=self.fixture()
        pred=f.clone(); pred[:,4:7]+=.4
        directions=torch.tensor([[1.,2.,3.],[-2.,1.,-1.]])
        batch=local_response_loss(pred,f,g,m,directions=directions)[0]
        separate=torch.stack([local_response_loss(p[None],t[None],g,m,directions=directions)[0] for p,t in zip(pred,f)]).mean()
        torch.testing.assert_close(batch,separate)

    def test_learned_xyz_rejected(self):
        _,g,f,m=setup(8)
        with self.assertRaisesRegex(ValueError,'requires float32 or quantized'):
            local_response_loss(f,f,g,m)

    def test_real_codec_short_optimization(self):
        _,g,f,m=self.fixture()
        optimizer=torch.optim.Adam(m.parameters(),lr=1e-3)
        q=torch.arange(len(f))%3+1
        directions=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]])
        def evaluate():
            torch.manual_seed(901)
            return local_response_loss(m(f,f[:,:3],q,10,'awgn'),f,g,m,directions=directions)[0]
        initial=float(evaluate().detach())
        for _ in range(50):
            optimizer.zero_grad()
            loss=evaluate()
            loss.backward()
            optimizer.step()
        final=float(evaluate().detach())
        self.assertLess(final,initial*.9)

    def test_two_stage_cli_random_explicit_xyz(self):
        from gaussian_jscc.cli import main
        from gaussian_jscc.transport import load_checkpoint
        raw,_,_,_=self.fixture()
        adam_step = torch.optim.Adam.step
        observed_adam_steps = []
        def tracked_step(optimizer, *args, **kwargs):
            result = adam_step(optimizer, *args, **kwargs)
            observed_adam_steps.append(max(float(s['step']) for s in optimizer.state.values() if 'step' in s))
            return result
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_ply(root/'input.ply',raw,0)
            argv=['gaussian_jscc','train-learned','--ply',str(root/'input.ply'),'--out',str(root/'run'),
                  '--source','mock','--device','cuda','--bootstrap-steps','2','--render-steps','2',
                  '--bootstrap-objective','local-response','--position-delivery','quantized',
                  '--prefix-mode','progressive','--position-bits','16','--position-compression','delta_zlib',
                  '--lr','0.0001','--render-lr','0.0001',
                  '--validation-views','1','--validation-trials','1','--validate-every','1',
                  '--hidden','16','--depth','1','--grid-dim','4','--levels','2',
                  '--block-size','8','--blocks-per-batch','2','--decoder-window','4']
            with patch('sys.argv',argv),patch('torch.optim.Adam.step',tracked_step), \
                 patch('gaussian_jscc.cli.device_for',return_value=torch.device('cpu')), \
                 patch('gaussian_jscc.rendering.load_cameras',return_value=render_fixture.RenderFirstTests().cameras()*2), \
                 patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
                 patch('gaussian_jscc.learned_train.load_checkpoint',side_effect=AssertionError('random must not load')):
                main()
            rows=[json.loads(s) for s in (root/'run'/'loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in rows],['bootstrap','bootstrap','render','render'])
            self.assertTrue(all(r['grad_norm']>0 for r in rows))
            self.assertTrue(all(r['aux_loss']==0 for r in rows[2:]))
            self.assertEqual([r['lr'] for r in rows],[1e-4]*4)
            self.assertTrue(all(r['render_backward']=='replay' for r in rows[2:]))
            self.assertEqual([r['phase_step'] for r in rows],[1,2,1,2])
            self.assertEqual(observed_adam_steps,[1,2,1,2])
            # Attribute pretraining does not invoke the full-scene render task.
            self.assertTrue(all('render_loss' not in r for r in rows[:2]))
            self.assertTrue(all(r['objective']=='local-response' for r in rows[:2]))
            self.assertTrue((root/'run'/'codec_end_bootstrap.pt').exists())
            self.assertTrue((root/'run'/'codec_best_render.pt').exists())
            self.assertTrue((root/'run'/'codec_best_bootstrap.pt').exists())
            self.assertTrue((root/'run'/'charts'/'bootstrap_validation.png').exists())
            self.assertTrue((root/'run'/'charts'/'prefix_gains.png').exists())
            self.assertEqual(load_checkpoint(root/'run'/'codec.pt','cpu').cfg.position_delivery,'quantized')
            validations=[json.loads(s) for s in (root/'run'/'bootstrap_validation.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in validations],[0,1,2])
            scene_checks=[json.loads(s) for s in (root/'run'/'validation.jsonl').read_text().splitlines()]
            self.assertTrue(any(r['phase']=='bootstrap' and r['step']==1 for r in scene_checks))
            self.assertTrue(all(r['paired_prefix_noise'] for r in scene_checks))


if __name__=='__main__':
    unittest.main()
