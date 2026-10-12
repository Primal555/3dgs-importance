"""Receiver geometry experiment: real codec/channel; synthetic CPU rendering only."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.data import prepare, to_features, to_raw, write_ply
from gaussian_jscc.position_delivery import delivered_positions
from gaussian_jscc.transport import load_checkpoint, model_id, receive, save_checkpoint, transmit
from test_gaussian_jscc import fixture
from test_render_first import synthetic_render, Reference
import test_render_first as render_fixture


class XYZContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def pair(self):
        raw, _ = fixture(12, degree=1)
        raw, geometry, _ = prepare(raw, 16)
        options = dict(sh_degree=1, hidden=16, depth=2, grid_dim=4, levels=(2,),
                       block_size=8, decoder_window=4, rates=(0,4,12,24),
                       prefix_mode='progressive', position_delivery='quantized', position_bits=16)
        models = []
        for mode in ('none','xyz'):
            torch.manual_seed(51)
            try:
                model = GaussianCodec(CodecConfig(**options, decoder_geometry=mode))
            except TypeError as exc:
                self.fail(f'missing opt-in receiver XYZ context: {exc}')
            model.attr_mean.copy_(raw[:,3:].mean(0))
            model.attr_std.copy_(raw[:,3:].std(0,unbiased=False).clamp_min(.01))
            models.append(model)
        features,_ = to_features(raw,geometry,models[0])
        return raw, geometry, features, *models

    def open_geometry(self, model):
        with torch.no_grad():
            torch.manual_seed(103)
            for name, parameter in model.named_parameters():
                if ('xyz_embedding.2.' in name or 'xyz_bias.2.' in name):
                    parameter.normal_(std=.04)

    def test_initial_common_weights_and_outputs_match_without_changing_payload(self):
        _,_,f,base,xyz = self.pair()
        for name,value in base.state_dict().items():
            self.assertTrue(torch.equal(value,xyz.state_dict()[name]),name)
        q = torch.tensor([1,2,3,0,1,2,3,1,2,3,0,1])
        torch.testing.assert_close(base.encode(f,f[:,:3],q,10),xyz.encode(f,f[:,:3],q,10),rtol=0,atol=0)
        torch.testing.assert_close(base(f,f[:,:3],q,10,'none'),xyz(f,f[:,:3],q,10,'none'),rtol=1e-5,atol=1e-6)
        self.assertNotIn('decoder_geometry',base.cfg.to_dict())
        self.assertEqual(xyz.cfg.to_dict()['decoder_geometry'],'xyz')

    def test_received_coordinates_affect_attributes_not_coordinate_output(self):
        _,_,f,_,model = self.pair()
        self.open_geometry(model)
        q = torch.full((len(f),),2)
        symbols = model.encode(f,f[:,:3],q,10)
        side = delivered_positions(f[:,:3],q,model.cfg)
        a = model.decode(symbols,q,10,delivered_xyz=side)
        changed = side.clone(); changed[2] = torch.tensor([.1,.8,.3])
        b = model.decode(symbols,q,10,delivered_xyz=changed)
        self.assertGreater(float((a[:,3:]-b[:,3:]).detach().abs().max()),1e-5)
        torch.testing.assert_close(b[:,:3],changed,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'XYZ'):
            model.decode(symbols,q,10)

    def test_padding_and_unreceived_suffix_cannot_contaminate_active_points(self):
        _,_,f,_,model = self.pair()
        self.open_geometry(model)
        q = torch.tensor([[1,2,3,0,1,2,3,0,0,0,0,0]])
        received = torch.randn(1,len(f),48)
        side = delivered_positions(f[None,:,:3],q,model.cfg)
        a = model.learned.decode(received,q,10,delivered_xyz=side)
        from gaussian_jscc.codec import prefix_mask
        altered = received.clone()
        absent = ~prefix_mask(q.flatten(),model.cfg.rates).reshape_as(received)
        altered[absent] = 10000
        other = side.clone(); other[q==0] = 10000
        b = model.learned.decode(altered,q,10,delivered_xyz=other)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        zeros = model.learned.decode(received,torch.zeros_like(q),10,delivered_xyz=side)
        self.assertTrue(torch.isfinite(zeros).all())
        self.assertEqual(int(torch.count_nonzero(zeros)),0)

    def test_real_forward_uses_quantized_side_not_clean_xyz_and_backpropagates(self):
        _,_,f,_,model = self.pair()
        q = torch.full((1,len(f)),3)
        recorded = []
        handle = model.learned.xyz_embedding.register_forward_pre_hook(lambda _,args: recorded.append(args[0].detach().clone()))
        output,_,_ = model.forward_tier_batches(f[None],f[None,:,:3],F.one_hot(q,4).float(),10,'none')
        handle.remove()
        side = delivered_positions(f[None,:,:3],q,model.cfg)
        torch.testing.assert_close(recorded[0],side*2-1,rtol=0,atol=0)
        (output[...,3:]-f[None,:,3:]).square().mean().backward()
        grads = {n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
        self.assertGreater(grads['learned.xyz_embedding.2.weight'],0)
        self.assertGreater(grads['learned.dec_blocks.0.xyz_bias.2.weight'],0)
        self.assertIsNone(model.learned.heads['xyz'].weight.grad)

    def test_checkpoint_actual_packet_and_replay_gradients_preserve_geometry(self):
        raw,geometry,f,_,model = self.pair()
        self.open_geometry(model)
        q = torch.tensor([1,2,3,0,1,2,3,1,2,3,0,1])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_checkpoint(root/'codec.pt',model,0)
            loaded = load_checkpoint(root/'codec.pt','cpu')
            self.assertEqual(model_id(model),model_id(loaded))
            stats = transmit(model,raw,q,10,'none',42,root/'packet')
            self.assertEqual(stats['payload_complex_symbols'],124)
            recovered = receive(loaded,root/'packet')
            expected = torch.cat([to_raw(model(f[i:i+8],f[i:i+8,:3],q[i:i+8],10,'none')[q[i:i+8]>0],geometry,model)
                                  for i in range(0,len(f),8)])
            torch.testing.assert_close(recovered,expected)
        from gaussian_jscc.training import full_scene_step
        from gaussian_jscc.render_objective import MultiViewRenderTask
        results = []
        with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            for mode in ('replay','checkpoint'):
                trial = copy.deepcopy(model)
                torch.manual_seed(25)
                loss,_ = full_scene_step(trial,[(f[None],q[None])],geometry,10,'awgn',
                    MultiViewRenderTask(render_fixture.RenderFirstTests().cameras(),Reference(),1),attr_weight=0,mode=mode)
                results.append((loss,{n:p.grad for n,p in trial.named_parameters()}))
        torch.testing.assert_close(results[0][0],results[1][0])
        for name,grad in results[0][1].items():
            if grad is not None:
                torch.testing.assert_close(grad,results[1][1][name],atol=2e-6,rtol=2e-4,msg=name)

    def test_unsupported_position_and_allocator_modes_fail_before_training(self):
        try:
            with self.assertRaises(ValueError):
                CodecConfig(decoder_geometry='xyz',position_delivery='learned')
        except TypeError as exc:
            self.fail(f'missing geometry configuration validation: {exc}')
        _,_,f,_,model = self.pair()
        with self.assertRaisesRegex(ValueError,'geometry'):
            model.forward_st_prefix_batches(f[None],f[None,:,:3],torch.ones(1,len(f),dtype=torch.long),
                                           torch.ones(1,len(f),4)/4,10)
        from gaussian_jscc.multiscene import parser,train
        options = parser().parse_args(['--out','unused','--decoder-geometry','xyz'])
        with self.assertRaisesRegex(ValueError,'geometry'):
            train(options)


class XYZExperimentTests(unittest.TestCase):
    def test_complete_random_truck_pair_produces_comparisons_and_compact_outputs(self):
        try:
            controls = importlib.import_module('gaussian_jscc.xyz_context_controls')
        except ModuleNotFoundError:
            self.fail('missing isolated XYZ context experiment')
        from gaussian_jscc import multiscene
        from gaussian_jscc.multiscene_experiments import evaluate,parser as evaluation_parser
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw,_ = fixture(12,degree=1)
            write_ply(root/'truck.ply',raw,1)
            (root/'images').mkdir()
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'scenes':[
                {'name':'truck','role':'train','ply':'truck.ply','source':'images'},
                {'name':'train','role':'train','ply':'not_loaded.ply','source':'not_loaded'}]}))
            source = root/'source'
            options = multiscene.parser().parse_args(['--manifest',str(manifest),'--out',str(source),
                '--train-scenes','truck','--device','cpu','--bootstrap-steps','1','--render-steps','1',
                '--allocation-steps','0','--joint-steps','0','--hidden','16','--depth','2','--block-size','8',
                '--decoder-window','4','--blocks-per-batch','2','--validation-views','1','--validation-trials','1',
                '--views-per-step','1','--output-profile','compact'])
            args = controls.parser().parse_args(['--run',str(source),'--out',str(root/'controls'),
                '--device','cpu','--bootstrap-steps','4','--render-steps','4','--validate-every','4',
                '--test-views','1','--trials','1','--image-views','1'])
            def dispatch(command,check):
                module = command[command.index('-m')+1]
                argv = command[command.index('-m')+2:]
                if module=='gaussian_jscc.multiscene':
                    multiscene.train(multiscene.parser().parse_args(argv))
                elif module=='gaussian_jscc.multiscene_experiments':
                    evaluate(evaluation_parser().parse_args(argv))
                else:
                    self.fail(module)
            with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
                 patch('gaussian_jscc.rendering.load_cameras',return_value=render_fixture.RenderFirstTests().cameras()), \
                 patch('gaussian_jscc.xyz_context_controls.subprocess.run',side_effect=dispatch):
                multiscene.train(options)
                controls.main(args)
            out = root/'controls'
            complete = json.loads((out/'complete.json').read_text())
            self.assertTrue(complete['paired_initializer_verified'])
            self.assertFalse(complete['allocator_trained'])
            rows = json.loads((out/'comparison.json').read_text())
            self.assertEqual(len(rows),6)
            for mode in ('none','xyz'):
                dest = out/mode
                losses = [json.loads(line) for line in (dest/'loss.jsonl').read_text().splitlines()]
                self.assertEqual(len(losses),8)
                self.assertEqual({r['layout'] for r in losses},{'1','2','3','mixed'})
                self.assertTrue(all(r['retained_gaussians']==len(raw) for r in losses if r['phase']=='render'))
                self.assertEqual({p.name for p in (dest/'scenes/truck/validation_images').iterdir()},{'000000','000008'})
            self.assertTrue((out/'charts/truck_xyz_context_loss.png').exists())
            self.assertTrue((out/'charts/truck_xyz_context_ssim.png').exists())
            self.assertTrue((out/'runtime_comparison.json').exists())
            bash = Path('C:/softwares/Git/bin/bash.exe')
            if os.name=='nt' and bash.exists():
                env = dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHON_BIN=sys.executable.replace('\\','/'))
                command = [str(bash),'scripts/test_xyz_context_truck.sh',str(source),str(root/'dry'),'--dry-run']
                result = subprocess.run(command,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('decoder_geometry_xyz:',result.stdout)
                self.assertFalse((root/'dry').exists())


if __name__=='__main__':
    unittest.main()
