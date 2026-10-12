"""Cooperation probes use real codec/autograd; only CUDA rendering is substituted."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gaussian_jscc.transport import model_id, save_checkpoint
import test_residual_prefix as residual_fixtures
import test_render_first as render_fixtures
from test_render_first import synthetic_render, Reference


class CooperationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def module(self):
        try:
            return importlib.import_module('gaussian_jscc.residual_cooperation')
        except ModuleNotFoundError:
            self.fail('missing independent residual cooperation diagnostic')

    def model_scene(self):
        helper = residual_fixtures.ResidualPrefixTests()
        raw, geometry, features, _, model = helper.pair()
        helper.open_corrections(model)
        cameras = render_fixtures.RenderFirstTests().cameras()
        scene = SimpleNamespace(groups=[features[None]], group_ids=[torch.arange(len(features))[None]],
            geometry=geometry, degree=1, reference=Reference(), cameras=[cameras[0]], val_cameras=[cameras[1]])
        return model, scene

    def test_group_cosines_use_matching_parameters_and_undefined_zero_norm(self):
        m = self.module()
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.learned = torch.nn.Module()
                self.learned.dec_in = torch.nn.Linear(2, 1, bias=False)
                self.learned.heads = torch.nn.ModuleDict({'dc': torch.nn.Linear(1, 1, bias=False)})
        model = Tiny()
        model.learned.dec_in.weight.grad = torch.tensor([[1., 0.]])
        model.learned.heads['dc'].weight.grad = torch.tensor([[0.]])
        first = m.gradient_snapshot(model)
        model.learned.dec_in.weight.grad = torch.tensor([[-1., 0.]])
        second = m.gradient_snapshot(model)
        rows = m.gradient_relations({1: first, 2: second})
        decoder = next(r for r in rows if r['group'] == 'shared_decoder')
        self.assertEqual(decoder['cosine'], -1.)
        self.assertEqual(decoder['dot'], -1.)
        self.assertEqual(decoder['parameter_count'], 2)
        empty = next(r for r in rows if r['group'] == 'base_attributes')
        self.assertIsNone(empty['cosine'])
        self.assertEqual(empty['status'], 'zero_norm')

    def test_suffix_shuffle_preserves_prefix_noise_padding_and_signal_multiset(self):
        m = self.module()
        signal = torch.arange(2*4*48).reshape(2, 4, 48).float()
        valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
        signal = signal * valid[..., None]
        noise = torch.arange(signal.numel()).reshape_as(signal).float()/10000 * valid[..., None]
        received = signal + noise
        state = torch.get_rng_state().clone()
        shuffled = m.ablate_suffix(signal, received, valid, 'shuffle_suffix', seed=9)
        torch.testing.assert_close(shuffled[..., :24], received[..., :24], atol=0, rtol=0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(int(torch.count_nonzero(shuffled[~valid])), 0)
        for block in range(2):
            values = (shuffled-noise)[block, valid[block], 24:]
            want = signal[block, valid[block], 24:]
            torch.testing.assert_close(values.sort(dim=0).values, want.sort(dim=0).values)
        self.assertGreater(float((shuffled-received).norm()), 0.)
        blocked = m.ablate_suffix(signal, received, valid, 'zero_suffix', seed=9)
        self.assertEqual(int(torch.count_nonzero(blocked[..., 24:])), 0)

    def test_paired_codec_prefixes_unchanged_by_q3_ablation_and_xyz_delivered(self):
        m = self.module()
        model, scene = self.model_scene()
        samples = []
        for tier, variant in ((1, 'normal'), (2, 'normal'), (3, 'no_delta3'),
                              (3, 'zero_suffix'), (3, 'shuffle_suffix')):
            torch.manual_seed(53)
            raw, details = m.probe_batch(model, scene.groups[0], torch.full((1, 8), tier),
                scene.geometry, 10, 'awgn', variant=variant, shuffle_seed=81, return_stages=True)
            samples.append((raw, details))
        torch.testing.assert_close(samples[1][0], samples[2][0], atol=0, rtol=0)
        for _, stages in samples[2:]:
            torch.testing.assert_close(stages[0], samples[0][1][0], atol=0, rtol=0)
            torch.testing.assert_close(stages[1], samples[1][1][1], atol=0, rtol=0)
        for raw, _ in samples[1:]:
            torch.testing.assert_close(raw[:, :3], samples[0][0][:, :3], atol=0, rtol=0)

    def test_full_scene_gradients_match_direct_autograd_and_rng_is_preserved(self):
        m = self.module()
        model, scene = self.model_scene()
        other = copy.deepcopy(model)
        identity = model_id(model)
        state = torch.get_rng_state().clone()
        with patch('gaussian_jscc.rendering.render', side_effect=synthetic_render):
            _, snapshot, _ = m.render_gradient(model, scene, scene.cameras, tier=3,
                snr=10, kind='awgn', seed=52, white_background=False)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            torch.manual_seed(52)
            q = torch.full((1, 8), 3)
            raw = m.probe_batch(other, scene.groups[0], q, scene.geometry, 10, 'awgn')
            from gaussian_jscc.render_objective import MultiViewRenderTask
            MultiViewRenderTask(scene.cameras, scene.reference, 1)(raw).backward()
        for name, parameter in other.named_parameters():
            if parameter.requires_grad:
                actual = snapshot[name]['gradient']
                expected = torch.zeros_like(parameter).flatten() if parameter.grad is None else parameter.grad.flatten()
                torch.testing.assert_close(actual, expected.cpu(), atol=2e-6, rtol=2e-4)
        self.assertEqual(model_id(model), identity)

    def test_update_probes_are_independent_restore_weights_and_create_no_checkpoints(self):
        m = self.module()
        model, scene = self.model_scene()
        identity = model_id(model)
        state = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temp, \
             patch('gaussian_jscc.rendering.render', side_effect=synthetic_render):
            out = Path(temp)
            m.run_probes(model, scene, out, probes=1, views_per_probe=1, heldout_views=1,
                         snr=10, kind='awgn', seed=62, lrs=[1e-4, 1e-5], white_background=False,
                         make_plots=False, image_views=0)
            summary = json.loads((out/'summary.json').read_text())
            self.assertTrue(summary['source_model_unchanged'])
            self.assertEqual(summary['completed_probes'], 1)
            self.assertEqual(summary['optimizer'], 'fresh Adam for EACH independent one-step probe; no historical moments')
            updates = [json.loads(x) for x in (out/'updates.jsonl').read_text().splitlines()]
            self.assertEqual(len(updates), 36)  # 3 update tiers * 2 lrs * 2 splits * 3 evaluated tiers
            self.assertTrue(all(r['start_model_id'] == identity for r in updates))
            self.assertTrue(all(r['update_norm'] > 0 for r in updates))
            self.assertFalse(list(out.rglob('*.pt')))
            ablations = [json.loads(x) for x in (out/'ablations.jsonl').read_text().splitlines()]
            self.assertEqual({r['variant'] for r in ablations}, {'normal', 'no_delta3', 'zero_suffix', 'shuffle_suffix'})
            self.assertTrue(all(r['group'] != 'shared_encoder' for r in summary['gradient_summary']
                if r['group'].startswith('refinement')))
        self.assertEqual(model_id(model), identity)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_cli_refuses_direct_checkpoint_or_source_output_and_dry_run_is_read_only(self):
        m = self.module()
        model, _ = self.model_scene()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root/'pair'/'residual'
            (run/'checkpoints'/'final').mkdir(parents=True)
            (run/'scenes'/'truck').mkdir(parents=True)
            (run/'training.json').write_text(json.dumps({'decoder_mode':'residual','manifest':'missing.json'}))
            (run/'scenes'/'truck'/'scene.json').write_text('{}')
            save_checkpoint(run/'checkpoints'/'final'/'codec.pt',model,3)
            args = m.parser().parse_args(['--run',str(run.parent),'--out',str(root/'out'),'--dry-run'])
            m.main(args)
            self.assertFalse((root/'out').exists())
            save_checkpoint(root/'other_experiment.pt',model,3)
            args.checkpoint = str(root/'other_experiment.pt')
            with self.assertRaisesRegex(ValueError, 'selected source experiment'):
                m.main(args)
            args.checkpoint = str(run/'checkpoints'/'final'/'codec.pt')
            m.main(args)
            args.out = str(run/'diagnostic')
            with self.assertRaisesRegex(ValueError, 'outside'):
                m.main(args)

    def test_failure_during_modified_weight_evaluation_restores_model_and_rng(self):
        m = self.module()
        model, scene = self.model_scene()
        identity = model_id(model)
        state = torch.get_rng_state().clone()
        def fail_after_update(*args, **kwargs):
            if model_id(model) != identity:
                raise RuntimeError('controlled renderer failure after update')
            return synthetic_render(*args, **kwargs)
        with tempfile.TemporaryDirectory() as temp, \
             patch('gaussian_jscc.rendering.render',side_effect=fail_after_update):
            with self.assertRaisesRegex(RuntimeError, 'controlled renderer failure'):
                m.run_probes(model,scene,Path(temp),probes=1,views_per_probe=1,heldout_views=1,
                    lrs=[1e-4],make_plots=False,image_views=0)
        self.assertEqual(model_id(model),identity)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_cli_integrates_real_checkpoint_scene_identity_outputs_plots_and_shell(self):
        m = self.module()
        from gaussian_jscc.data import write_ply
        from gaussian_jscc import multiscene
        model, _ = self.model_scene()
        raw, _, _, direct, _ = residual_fixtures.ResidualPrefixTests().pair()
        cameras = render_fixtures.RenderFirstTests().cameras()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_ply(root/'truck.ply',raw,1)
            (root/'images').mkdir()
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps({'scenes':[{'name':'truck','role':'train','ply':'truck.ply','source':'images'}]}))
            source=root/'source'
            opts=multiscene.parser().parse_args(['--manifest',str(manifest),'--out',str(source),
                '--device','cpu','--decoder-mode','residual','--bootstrap-steps','1','--render-steps','1',
                '--allocation-steps','0','--joint-steps','0','--hidden','16','--depth','1','--block-size','8',
                '--decoder-window','4','--blocks-per-batch','2','--validation-views','1','--validation-trials','1',
                '--views-per-step','1','--output-profile','compact'])
            arguments=m.parser().parse_args(['--run',str(source),'--out',str(root/'diagnostic'),
                '--device','cpu','--probes','1','--views-per-probe','1','--heldout-views','1','--lrs','1e-4'])
            with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
                 patch('gaussian_jscc.rendering.load_cameras',return_value=cameras):
                multiscene.train(opts)
                checkpoint=source/'checkpoints/final/codec.pt'
                original=checkpoint.read_bytes()
                m.main(arguments)
                self.assertEqual(checkpoint.read_bytes(),original)
                save_checkpoint(source/'checkpoints'/'direct.pt',direct,3)
                arguments.checkpoint=str(source/'checkpoints'/'direct.pt')
                arguments.out=str(root/'rejected')
                with self.assertRaisesRegex(ValueError,'residual attributes'):
                    m.main(arguments)
            out=root/'diagnostic'
            complete=json.loads((out/'complete.json').read_text())
            self.assertTrue(complete['source_checkpoint_unchanged'])
            self.assertEqual(len(list((out/'charts').glob('*.png'))),3)
            self.assertEqual(len(list((out/'images').glob('*.png'))),4)
            self.assertFalse(list(out.rglob('*.pt')))
            bash=Path('C:/softwares/Git/bin/bash.exe')
            if os.name=='nt' and bash.exists():
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHON_BIN=sys.executable.replace('\\','/'))
                command=[str(bash),'scripts/test_residual_cooperation.sh',str(source),str(root/'dry'),'--dry-run']
                result=subprocess.run(command,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('no historical moments',result.stdout)
                self.assertFalse((root/'dry').exists())
                env.pop('CUDA_VISIBLE_DEVICES')
                refused=subprocess.run(command,env=env,capture_output=True,text=True)
                self.assertNotEqual(refused.returncode,0)


if __name__ == '__main__':
    unittest.main()
