import copy
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from test_gaussian_jscc import fixture
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.data import prepare, to_features, to_raw, write_ply
from gaussian_jscc.transport import save_checkpoint, load_checkpoint, transmit, receive, model_id
from gaussian_jscc.training import full_scene_step
from gaussian_jscc.learned_training import policy_objective, discrete_joint_step
from gaussian_jscc.learned_objective import projection_loss
from gaussian_jscc.losses import reconstruction_loss
from gaussian_jscc.optimization import clip_codec_gradients
from gaussian_jscc.allocation import GaussianTierMask


def setup(n=17):
    raw, _ = fixture(n, degree=0)
    cfg = CodecConfig(architecture='learned_joint', loss_profile='learned_v1', sh_degree=0,
                      hidden=16, depth=2, grid_dim=4, levels=(2,), planes=False,
                      block_size=8, decoder_window=4, rates=(0,2,4,6))
    model = GaussianCodec(cfg)
    model.attr_mean.copy_(raw[:,3:].mean(0))
    model.attr_std.copy_(raw[:,3:].std(0,unbiased=False).clamp_min(.01))
    raw,g,_ = prepare(raw,16)
    f,_ = to_features(raw,g,model)
    return raw,g,f,model


class LearnedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_no_hand_geometry_or_split_budget(self):
        _,_,_,m = setup()
        self.assertFalse(hasattr(m,'block_geometry'))
        self.assertFalse(hasattr(m,'geometry_slots'))
        self.assertEqual(m.cfg.geometry_rates, ())
        self.assertTrue(m.cfg.individual_tiers)
        with self.assertRaises(ValueError):
            CodecConfig(architecture='learned_joint',geometry_rates=(0,1,2,3))

    def test_packed_batched_noise_and_gradient_match(self):
        _,g,f,m = setup(8)
        q = torch.tensor([1,0,2,3,1,0,2,3])
        torch.manual_seed(8)
        actual = m(f,f[:,:3],q,10,'awgn')
        torch.manual_seed(8)
        dense,_,_ = m.forward_tier_batches(f[None],f[None,:,:3],F.one_hot(q,4).float()[None],10,'awgn')
        torch.testing.assert_close(actual,dense[0],rtol=2e-5,atol=1e-6)
        reconstruction_loss(actual[q>0],f[q>0],g,m).backward()
        for name in ('xyz','opacity','scale','rotation','dc'):
            grad = m.learned.heads[name].weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.norm()),0)
        self.assertGreater(float(m.learned.symbol_head.weight.grad.norm()),0)

    def test_reject_fake_st_gradients(self):
        _,_,f,m = setup(8)
        choices = F.one_hot(torch.ones(1,8,dtype=torch.long),4).float().requires_grad_()
        with self.assertRaisesRegex(ValueError,'score-function'):
            m.forward_tier_batches(f[None],f[None,:,:3],choices,10)

    def test_power_budget_empty_windows_and_singletons(self):
        for n in (1,7,8,17):
            _,_,f,m = setup(n)
            for q in (torch.zeros(n,dtype=torch.long),torch.arange(n)%4):
                z = m.encode(f,f[:,:3],q,10)
                self.assertEqual(len(z),int(torch.tensor(m.cfg.rates)[q].sum()))
                if len(z):
                    self.assertLessEqual(float(z.detach().square().sum(-1).mean()),1.00001)
                result = m.decode(z,q,10)
                self.assertTrue(torch.isfinite(result).all())
                self.assertEqual(float(result[q==0].detach().abs().sum()),0.)
                result.sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_new_receiver_has_only_packet_and_weights(self):
        raw,_,_,m = setup(17)
        q = torch.arange(len(raw))%4
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_checkpoint(root/'codec.pt',m,0)
            fresh = load_checkpoint(root/'codec.pt','cpu')
            self.assertEqual(model_id(m),model_id(fresh))
            stats = transmit(m,raw,q,10,'none',42,root/'packet')
            a = receive(m,root/'packet')
            b = receive(fresh,root/'packet')
            torch.testing.assert_close(a,b)
            self.assertEqual(len(b),int((q>0).sum()))
            self.assertEqual(stats['handcrafted_coordinate_symbols'],0)
            self.assertEqual(stats['payload_complex_symbols'],int(torch.tensor(m.cfg.rates)[q].sum()))

    def test_v4_packet_identity_survives_cleanup(self):
        _,_,_,m = setup(8)
        # Explicit pre-cleanup learned-v4 schema, not reconstructed from today's
        # serialization. Removing inactive modes must not break current packets.
        saved_config = dict(sh_degree=0, hidden=16, grid_dim=4, levels=[2], planes=False,
                            depth=2, rates=[0,2,4,6], block_size=8, morton_bits=16,
                            architecture='learned_joint', geometry_rates=[], geometry_weight=1.,
                            shape_weight=.25, opacity_weight=1., dc_weight=1., sh_weight=.25,
                            geometry_floor=.0001, loss_profile='learned_v1', scale_weight=1.,
                            position_head='learned_affine', individual_tiers=True,
                            decoder_window=4, attention_heads=4, power_floor=.01, xyz_loss_scale=.05)
        digest=hashlib.sha256(json.dumps(saved_config,sort_keys=True).encode())
        for name,value in sorted(m.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.assertEqual(model_id(m),digest.hexdigest())
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'v4.pt'
            torch.save({'version':4,'config':saved_config,'state_dict':m.state_dict()},path)
            loaded=load_checkpoint(path,'cpu')
            self.assertEqual(model_id(loaded),digest.hexdigest())

    def test_xyz_responds_to_received_symbols_and_not_clamped(self):
        _,g,f,m = setup(8)
        q = torch.ones(8,dtype=torch.long)
        z = m.encode(f,f[:,:3],q,10).detach().requires_grad_()
        m.decode(z,q,10)[:,:3].sum().backward()
        self.assertGreater(float(z.grad.norm()),0)
        pred = f.detach().clone()
        pred[:,0] = 1.5
        pred.requires_grad_()
        to_raw(pred,g,m)[:,0].sum().backward()
        self.assertTrue((pred.grad[:,0] > 0).all())
        self.assertTrue((to_raw(pred,g,m)[:,0] > g.lower[0]+g.span[0]).all())

    def test_drop_inputs_do_not_leak_into_other_outputs(self):
        _,_,f,m = setup(8)
        q = torch.tensor([1,0,2,3,1,0,2,3])
        original = m(f,f[:,:3],q,10,'none')
        altered = f.clone()
        altered[q==0] = 100
        changed = m(altered,altered[:,:3],q,10,'none')
        torch.testing.assert_close(original[q>0],changed[q>0])

    def test_replay_equals_checkpoint(self):
        _,g,f,m = setup(8)
        batches = [(f[None],torch.tensor([[1,0,2,3,1,0,2,3]]))]
        results = []
        for mode in ('checkpoint','replay'):
            model = copy.deepcopy(m)
            torch.manual_seed(9)
            loss,_ = full_scene_step(model,batches,g,10,'awgn',lambda raw:raw[:,:3].square().mean(),
                                     attr_weight=1.,mode=mode)
            results.append((loss,[p.grad for p in model.parameters()]))
        torch.testing.assert_close(results[0][0],results[1][0])
        for a,b in zip(results[0][1],results[1][1]):
            if a is not None:
                torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA for local attention and device RNG replay')
    def test_cuda_replay(self):
        _,g,f,m = setup(8)
        f,m=f.cuda(),m.cuda()
        batches=[(f[None],torch.tensor([[1,0,2,3,1,0,2,3]],device='cuda'))]
        results=[]
        for mode in ('checkpoint','replay'):
            model=copy.deepcopy(m)
            torch.cuda.manual_seed_all(123)
            loss,_=full_scene_step(model,batches,g,10,'awgn',lambda raw:raw[:,:3].square().mean(),
                                   attr_weight=1.,mode=mode)
            results.append((loss,[p.grad for p in model.parameters()]))
        torch.testing.assert_close(results[0][0],results[1][0])
        for a,b in zip(results[0][1],results[1][1]):
            if a is not None:
                torch.testing.assert_close(a,b,rtol=2e-3,atol=2e-4)

    def test_score_function_expectation_matches_exact_gradient(self):
        logits = torch.tensor([.1,.2,-.3,.4],requires_grad=True)
        costs = torch.tensor([3.,1.,.7,.5])
        p = logits.softmax(-1)
        exact = torch.autograd.grad((p*costs).sum(),logits,retain_graph=True)[0]
        expected = torch.zeros_like(logits)
        for a,b in itertools.product(range(4),repeat=2):
            loss = policy_objective([logits.log_softmax(-1)[a],logits.log_softmax(-1)[b]],[costs[a],costs[b]])
            grad = torch.autograd.grad(loss,logits,retain_graph=True)[0]
            expected += (p[a]*p[b]).detach()*grad
        torch.testing.assert_close(exact,expected)

    def test_discrete_joint_omits_zero_rows_and_backpropagates(self):
        _,g,f,m = setup(8)
        mask = GaussianTierMask(8,existence_prior=torch.full((8,),.6))
        visited = []
        def task(raw,ids):
            visited.append((len(raw),len(ids)))
            # CPU differentiable task stub, explicitly penalizes omission.
            return raw[:,:3].square().sum()/8 + (8-len(raw))*.5
        loss,stats = discrete_joint_step(m,mask,[f[None]],[torch.arange(8)[None]],g,10,'none',task)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(a==b for a,b in visited))
        self.assertGreater(float(mask.logits.grad.norm()),0)
        self.assertTrue(torch.isfinite(mask.logits.grad).all())
        self.assertEqual(stats['mask_samples'],2)

    def test_no_clipping_means_no_gradient_change(self):
        _,_,f,m = setup(8)
        m(f,f[:,:3],torch.ones(8,dtype=torch.long),10,'none').sum().backward()
        before = [p.grad.clone() if p.grad is not None else None for p in m.parameters()]
        _,stats = clip_codec_gradients(m,1.,'none')
        for old,p in zip(before,m.parameters()):
            if old is not None:
                torch.testing.assert_close(old,p.grad)
        self.assertTrue(all(v['clip_factor']==1 for v in stats['gradient_groups'].values()))

    def test_projection_negative_depth_finite(self):
        camera = SimpleNamespace(world_view_transform=torch.eye(4),FoVx=1.,FoVy=1.)
        src = torch.tensor([[.1,.2,3.]+[0.]*11])
        pred = src.clone(); pred[:,2] = -1.; pred.requires_grad_()
        loss = projection_loss(pred,src,camera)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertLess(float(pred.grad[0,2]),0)

    def test_cli_and_old_checkpoint_rejection(self):
        from gaussian_jscc.cli import main
        raw,_,_,_ = setup(24)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_ply(root/'input.ply',raw,0)
            argv = ['gaussian_jscc','train-learned','--ply',str(root/'input.ply'),'--out',str(root/'run'),
                    '--device','cpu','--steps','2','--render-steps','0','--block-size','8',
                    '--hidden','16','--decoder-window','4','--grid-dim','4','--levels','2',
                    '--depth','1','--blocks-per-batch','2','--validation-blocks','1','--save-every','1']
            with patch('sys.argv',argv),patch('gaussian_jscc.plots.safe_plot'):
                main()
            model = load_checkpoint(root/'run'/'codec.pt','cpu')
            self.assertEqual(model.cfg.architecture,'learned_joint')
            log = [json.loads(x) for x in (root/'run'/'loss.jsonl').read_text().splitlines()]
            self.assertEqual(len(log),2)
            self.assertEqual(log[0]['clip_mode'],'none')
            self.assertGreater(log[0]['update_norm'],0)
            # No historical implementation is kept solely to create a fixture.
            torch.save({'version':3,'config':{'architecture':'geometry_first'}},root/'old.pt')
            argv[argv.index('--out')+1] = str(root/'reject')
            with patch('sys.argv',argv+['--init',str(root/'old.pt')]):
                with self.assertRaisesRegex(ValueError,'learned_joint'):
                    main()

    def test_complete_stage_control_flow_with_mock_renderer(self):
        # Real codec/replay/policy gradients; ONLY camera/rasterizer boundary is
        # mocked to run on CPU. This is not a CUDA rasterizer quality test.
        from gaussian_jscc.cli import main
        raw,_,_,_ = setup(24)
        camera = SimpleNamespace(original_image=torch.full((3,8,8),.5),
                                 world_view_transform=torch.eye(4),FoVx=1.,FoVy=1.)
        def renderer(scene,*args):
            if not len(scene):
                return torch.zeros(3,8,8)
            return (scene[:,:3].sum(0)/24).sigmoid()[:,None,None].expand(3,8,8)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_ply(root/'input.ply',raw,0)
            argv=['gaussian_jscc','train-learned','--ply',str(root/'input.ply'),'--out',str(root/'run'),
                  '--device','cuda','--source','mock','--steps','1','--render-steps','2','--joint-steps','2',
                  '--views-per-step','1','--validation-trials','1','--block-size','8','--hidden','16','--decoder-window','4',
                  '--depth','1','--grid-dim','4','--levels','2','--blocks-per-batch','2',
                  '--validation-blocks','1','--validation-views','1','--validate-every','1','--save-every','1']
            with patch('sys.argv',argv),patch('gaussian_jscc.cli.device_for',return_value=torch.device('cpu')), \
                 patch('gaussian_jscc.rendering.load_cameras',return_value=[camera,camera]), \
                 patch('gaussian_jscc.rendering.render',side_effect=renderer),patch('gaussian_jscc.plots.safe_plot'):
                main()
            rows=[json.loads(x) for x in (root/'run'/'loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in rows],['bootstrap','render','render','joint','joint'])
            self.assertTrue(all(r['aux_loss']==0 for r in rows[1:]))
            self.assertTrue(all('projection_loss' not in r for r in rows))
            self.assertAlmostEqual(rows[1]['loss'],rows[1]['image_mse'],places=7)
            self.assertTrue((root/'run'/'route2.pt').exists())
            self.assertTrue(list((root/'run'/'validation_images').rglob('*.png')))
            from gaussian_jscc.route2 import load_mask
            model=load_checkpoint(root/'run'/'codec.pt','cpu')
            allocation=load_mask(root/'run'/'route2.pt',raw,model,'cpu')
            self.assertEqual(len(allocation.logits),len(raw))


if __name__ == '__main__':
    unittest.main()
