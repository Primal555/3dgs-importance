"""Real codec/Adam tests; only the CUDA rasterizer is replaced on CPU."""
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

import test_residual_cooperation as fixtures
from test_render_first import synthetic_render
from gaussian_jscc.residual_cooperation import probe_batch
from gaussian_jscc.transport import model_id, load_checkpoint


class FreezeControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def module(self):
        return importlib.import_module('gaussian_jscc.residual_freeze_controls')

    def fixture(self):
        return fixtures.CooperationTests().model_scene()

    def prediction(self, model, scene, tier):
        with torch.no_grad():
            torch.manual_seed(97)
            return probe_batch(model, scene.groups[0], torch.full((1, 8), tier),
                               scene.geometry, 10., 'awgn').clone()

    def update(self, updater, model, scene, tier):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(101)
        prediction = probe_batch(model, scene.groups[0], torch.full((1, 8), tier),
                                 scene.geometry, 10., 'awgn')
        prediction[:, 3:].square().mean().backward()
        return updater.step()

    def test_stage_masks_literal_real_slots_and_tier_rows(self):
        m = self.module()
        model, _ = self.fixture()
        updater = m.StageUpdater(model, 2, 'frozen', 1e-5)
        rows = updater.masks['learned.symbol_head.weight'].any(dim=1).nonzero().flatten().tolist()
        self.assertEqual(rows, list(range(8, 24)))
        columns = updater.masks['learned.dec_in.weight'].any(dim=0).nonzero().flatten().tolist()
        self.assertEqual(columns, list(range(8, 24))+list(range(56, 72)))
        self.assertEqual(updater.masks['learned.tier.weight'].any(dim=1).tolist(), [False, False, True, False])
        self.assertFalse(model.learned.dec_in.bias.requires_grad)
        self.assertFalse(model.learned.heads['dc'].weight.requires_grad)
        self.assertTrue(model.learned.refinements[0].heads['dc'].weight.requires_grad)
        self.assertFalse(model.learned.refinements[1].heads['dc'].weight.requires_grad)

    def test_q2_q3_updates_preserve_every_lower_prefix_and_inactive_parameter(self):
        m = self.module()
        for tier in (2, 3):
            with self.subTest(tier=tier):
                model, scene = self.fixture()
                updater = m.StageUpdater(model, tier, 'frozen', 1e-4)
                before = {q:self.prediction(model, scene, q) for q in range(1, tier+1)}
                for _ in range(3):
                    result = self.update(updater, model, scene, tier)
                    self.assertGreater(result['update_norm'], 0)
                    self.assertEqual(updater.audit()['forbidden_max_abs'], 0.)
                for q in range(1, tier):
                    torch.testing.assert_close(self.prediction(model, scene, q), before[q], rtol=0, atol=0)
                self.assertGreater(float((self.prediction(model,scene,tier)-before[tier]).norm()), 0.)
                changed = model.learned.symbol_head.weight.detach()-updater.baseline['learned.symbol_head.weight']
                start, end = ((8,24) if tier==2 else (24,48))
                self.assertGreater(float(changed[start:end].norm()),0.)
                self.assertEqual(int(torch.count_nonzero(changed[:start])),0)
                self.assertEqual(int(torch.count_nonzero(changed[end:])),0)

    def test_q1_updates_shared_path_but_not_future_layers(self):
        m = self.module()
        model, scene = self.fixture()
        updater = m.StageUpdater(model, 1, 'frozen', 1e-4)
        self.update(updater,model,scene,1)
        self.assertEqual(updater.audit()['forbidden_max_abs'],0.)
        change = model.learned.fuse[0].weight.detach()-updater.baseline['learned.fuse.0.weight']
        self.assertGreater(float(change.norm()),0.)

    def test_audit_fails_if_frozen_shared_parameter_changes(self):
        m = self.module()
        model, _ = self.fixture()
        updater = m.StageUpdater(model,2,'frozen',1e-5)
        with torch.no_grad():
            model.learned.dec_in.bias.add_(.1)
        with self.assertRaisesRegex(RuntimeError,'frozen'):
            updater.audit()

    def test_audit_detects_signed_zero_byte_change(self):
        m = self.module()
        model, _ = self.fixture()
        with torch.no_grad():
            model.learned.dec_in.bias[0] = 0.
        updater = m.StageUpdater(model,2,'frozen',1e-5)
        with torch.no_grad():
            model.learned.dec_in.bias[0] = -0.
        self.assertTrue(torch.equal(model.learned.dec_in.bias,updater.baseline['learned.dec_in.bias']))
        with self.assertRaisesRegex(RuntimeError,'frozen'):
            updater.audit()

    def test_replay_can_skip_unused_cpu_gradient_snapshot(self):
        from gaussian_jscc.residual_cooperation import render_gradient
        model, scene = self.fixture()
        with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
             patch('gaussian_jscc.residual_cooperation.gradient_snapshot',side_effect=AssertionError('unneeded transfer')):
            loss, snapshot, _ = render_gradient(model,scene,scene.cameras,3,10.,'awgn',42,False,
                                               collect_snapshot=False)
        self.assertEqual(snapshot,{})
        self.assertGreater(loss,0.)
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_paired_training_checkpoints_logs_charts_and_rng_restored(self):
        m = self.module()
        model, scene = self.fixture()
        identity = model_id(model)
        flags = {n:p.requires_grad for n,p in model.named_parameters()}
        state = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temp, patch('gaussian_jscc.rendering.render',side_effect=synthetic_render):
            out = Path(temp)/'controls'
            summary = m.run_controls(model,scene,out,steps=(1,1,1),validate_every=1,
                views_per_step=1,heldout_views=1,validation_trials=1,image_views=1)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            self.assertEqual(summary['updates_per_arm'],3)
            for arm in ('joint','frozen'):
                loss = [json.loads(x) for x in (out/arm/'loss.jsonl').read_text().splitlines()]
                self.assertEqual([r['tier'] for r in loss],[1,2,3])
                self.assertTrue(all(r['update_norm']>0 for r in loss))
                validation = [json.loads(x) for x in (out/arm/'validation.jsonl').read_text().splitlines()]
                self.assertEqual(len(validation),12)
                self.assertEqual({r['tier'] for r in validation},{1,2,3})
                self.assertEqual(model_id(load_checkpoint(out/arm/'checkpoints/q3/codec.pt','cpu')),
                                 summary['arms'][arm]['final_model_id'])
            a = [json.loads(x) for x in (out/'joint/loss.jsonl').read_text().splitlines()]
            b = [json.loads(x) for x in (out/'frozen/loss.jsonl').read_text().splitlines()]
            self.assertEqual([(r['noise_seed'],r['camera_names']) for r in a],
                             [(r['noise_seed'],r['camera_names']) for r in b])
            audits = [json.loads(x) for x in (out/'frozen/freeze_audits.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['forbidden_max_abs']==0 for r in audits))
            self.assertTrue(all(v['prediction_max_abs']==0 and v['symbol_max_abs']==0
                                for r in audits for v in r['lower_prefixes']))
            self.assertEqual(len(list((out/'charts').glob('*.png'))),2)
            self.assertEqual(len(list((out/'images').rglob('*.png'))),9)
            self.assertEqual(summary['arms']['joint']['initial_model_id'],identity)
            self.assertEqual(summary['arms']['frozen']['initial_model_id'],identity)
            with self.assertRaises(FileExistsError):
                m.run_controls(model,scene,out,steps=(1,1,1),views_per_step=1)
        self.assertEqual(model_id(model),identity)
        self.assertEqual({n:p.requires_grad for n,p in model.named_parameters()},flags)

    def test_cli_real_source_identity_checkpoints_and_launch_script(self):
        m = self.module()
        from gaussian_jscc.data import write_ply
        from gaussian_jscc import multiscene
        import test_residual_prefix as codec_fixtures
        import test_render_first as render_fixtures
        raw, _, _, _, _ = codec_fixtures.ResidualPrefixTests().pair()
        cameras = render_fixtures.RenderFirstTests().cameras()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_ply(root/'truck.ply',raw,1)
            (root/'images').mkdir()
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'scenes':[{'name':'truck','role':'train','ply':'truck.ply','source':'images'}]}))
            source = root/'source'
            opts = multiscene.parser().parse_args(['--manifest',str(manifest),'--out',str(source),
                '--device','cpu','--decoder-mode','residual','--bootstrap-steps','1','--render-steps','1',
                '--allocation-steps','0','--joint-steps','0','--hidden','16','--depth','1','--block-size','8',
                '--decoder-window','4','--blocks-per-batch','2','--validation-views','1','--validation-trials','1',
                '--views-per-step','1','--output-profile','compact'])
            out = root/'controls'
            arguments = m.parser().parse_args(['--run',str(source),'--out',str(out),'--device','cpu',
                '--stage-steps','1','1','1','--validate-every','1','--views-per-step','1',
                '--heldout-views','1','--validation-trials','1'])
            with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
                 patch('gaussian_jscc.rendering.load_cameras',return_value=cameras):
                multiscene.train(opts)
                checkpoint = source/'checkpoints/final/codec.pt'
                original = checkpoint.read_bytes()
                m.main(arguments)
                self.assertEqual(checkpoint.read_bytes(),original)
            complete = json.loads((out/'complete.json').read_text())
            self.assertTrue(complete['source_checkpoint_unchanged'])
            self.assertEqual(complete['updates_per_arm'],3)
            self.assertEqual(len(list(out.rglob('codec.pt'))),6)
            self.assertEqual(len(list(out.rglob('optimizer.pt'))),6)
            arguments.out = str(source/'bad')
            with self.assertRaisesRegex(ValueError,'outside'):
                m.main(arguments)
            bash = Path('C:/softwares/Git/bin/bash.exe')
            if os.name=='nt' and bash.exists():
                env = dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHON_BIN=sys.executable.replace('\\','/'))
                env.pop('STAGE_STEPS',None)
                command = [str(bash),'scripts/test_residual_freeze_controls.sh',str(source),str(root/'dry'),'--dry-run']
                result = subprocess.run(command,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('no historical moments',result.stdout)
                self.assertEqual(json.loads(result.stdout)['stage_steps'],[1000,1000,1000])
                self.assertFalse((root/'dry').exists())
                direct = subprocess.run([sys.executable,'-m','gaussian_jscc.residual_freeze_controls',
                    '--run',str(source),'--out',str(root/'dry'),'--dry-run'],capture_output=True,text=True)
                self.assertEqual(direct.returncode,0,direct.stderr)
                self.assertEqual(json.loads(direct.stdout)['stage_steps'],[1000,1000,1000])
                env['STAGE_STEPS'] = '2 3 4'
                overridden = subprocess.run(command,env=env,capture_output=True,text=True)
                self.assertEqual(overridden.returncode,0,overridden.stderr)
                self.assertEqual(json.loads(overridden.stdout)['stage_steps'],[2,3,4])
                env.pop('CUDA_VISIBLE_DEVICES')
                refused = subprocess.run(command,env=env,capture_output=True,text=True)
                self.assertNotEqual(refused.returncode,0)

    def test_phase_adam_is_fresh_even_after_training_shared_tensor(self):
        m = self.module()
        model, scene = self.fixture()
        first = m.StageUpdater(model,1,'frozen',1e-4)
        self.update(first,model,scene,1)
        self.assertTrue(first.optimizer.state)
        second = m.StageUpdater(model,2,'frozen',1e-4)
        self.assertFalse(second.optimizer.state)
        self.update(second,model,scene,2)
        moments = second.optimizer.state[model.learned.symbol_head.weight]['exp_avg']
        self.assertEqual(int(torch.count_nonzero(moments[:8])),0)
        self.assertEqual(int(torch.count_nonzero(moments[24:])),0)

    def test_zero_and_nonfinite_allowed_gradients_stop_instead_of_counting_updates(self):
        m = self.module()
        model, _ = self.fixture()
        updater = m.StageUpdater(model,2,'frozen',1e-5)
        with self.assertRaisesRegex(RuntimeError,'zero allowed gradient'):
            updater.step()
        model.learned.symbol_head.weight.grad = torch.full_like(model.learned.symbol_head.weight,float('nan'))
        with self.assertRaisesRegex(RuntimeError,'nonfinite gradient'):
            updater.step()

    def test_failure_restores_model_flags_mode_and_rng(self):
        m = self.module()
        model, scene = self.fixture()
        model.train()
        identity = model_id(model)
        state = torch.get_rng_state().clone()
        flags = {n:p.requires_grad for n,p in model.named_parameters()}
        with tempfile.TemporaryDirectory() as temp, patch('gaussian_jscc.rendering.render',side_effect=RuntimeError('renderer failed')):
            with self.assertRaisesRegex(RuntimeError,'renderer failed'):
                m.run_controls(model,scene,Path(temp)/'controls',steps=(1,1,1),
                    views_per_step=1,heldout_views=1,make_plots=False,image_views=0)
        self.assertTrue(model.training)
        self.assertEqual(model_id(model),identity)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual({n:p.requires_grad for n,p in model.named_parameters()},flags)


if __name__ == '__main__':
    unittest.main()
