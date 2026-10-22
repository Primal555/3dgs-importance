"""Full-weight joint fork, frozen encoder, replay gradients and exact resume."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def cameras(*args,**kwargs):
    from test_render_first import RenderFirstTests
    result=RenderFirstTests().cameras()
    for i,camera in enumerate(result):
        camera.camera_center=torch.tensor([5.+i,6.,7.])
        if args[-1]=='test':camera.image_name='test_'+str(i)
    return result


def source_run(root, *, context_variant='baseline'):
    from test_modular_experiment import source_fixture
    from gaussian_jscc.modular_config import ModularCodecConfig
    from gaussian_jscc.model_factory import build_codec
    from gaussian_jscc.modular_experiment import inputs,make_scene,json_write
    from gaussian_jscc.data import read_ply
    from gaussian_jscc.multiscene import shared_statistics
    from gaussian_jscc.modular_checkpoint import capture_rng,save_training_checkpoint
    from gaussian_jscc.successive_training import configure_optimizer
    original=source_fixture(root);parent=root/'parent'
    from argparse import Namespace
    metadata,manifest,spec,old=inputs(Namespace(run=str(original),manifest=None))
    context=dict(context_variant=context_variant,context_centers=(4,2),context_neighbors=(2,4)) \
        if context_variant!='baseline' else {}
    model=build_codec(ModularCodecConfig(channel_variant='successive',channel_prefix_mode='full_prefix',
        channel_power_mode='block',channel_neighbor_mode='plain',latent_dim=8,channel_hidden=16,
        hidden=16,sh_degree=1,block_size=8,depth=2,grid_dim=4,levels=(2,),attention_heads=4,
        decoder_window=4,rates=(0,8,24,48),**context))
    raw,degree=read_ply(spec['ply']);mean,std=shared_statistics([raw])
    model.attr_mean.copy_(mean);model.attr_std.copy_(std)
    scene=make_scene(spec,raw,degree,model,metadata,old,parent)
    f=scene.groups[0];q=(scene.group_ids[0]>=0).long()*3
    rates=dict(main=1e-4,neighbor=1e-4,cooperative=1e-5)
    for phase in ('main','neighbor','cooperative'):
        optimizer=configure_optimizer(model,phase,rates);model.train()
        y,recovered,_=model.forward_channel_batches(f,f[...,:3],torch.nn.functional.one_hot(q,4).to(f),10.)
        (recovered[q>0]-y.detach()[q>0]).square().mean().backward();optimizer.step()
    progress=dict(step=3,phase='channel_bootstrap',successive_phase='cooperative',
        completed=dict(main=1,neighbor=1,cooperative=1),module_lrs=rates,freeze_semantic=True,
        train_snrs=[10.],training_objective=dict(version='successive_channel_v1'))
    save_training_checkpoint(parent/'checkpoints/final/codec.pt',model,optimizer,progress,capture_rng(),scene.record)
    json_write(parent/'training.json',dict(manifest=str(manifest),source_run=str(original),scene=scene.record,
        codec_config=model.cfg.to_dict(),selected_scenes=[spec],net_bits_per_use=scene.net_bits_per_use,
        phase_budgets=dict(main=1,neighbor=1,cooperative=1),successive_channel=dict(version='successive_channel_v1')))
    json_write(parent/'complete.json',dict(status='complete',step=3,completed=progress['completed']))
    return parent


class SuccessiveJointTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(27)

    def options(self,source,out):
        from gaussian_jscc.successive_joint import parser
        return parser().parse_args(['--run',str(source),'--out',str(out),'--steps','4',
            '--device','cpu','--validate-every','2','--save-every','1',
            '--test-views','2','--render-trials','1','--no-lpips'])

    def test_replay_matches_checkpoint_gradients_with_quantized_scale_ste(self):
        from gaussian_jscc.successive_joint import joint_step
        from gaussian_jscc.modular_checkpoint import restore_training_checkpoint
        from gaussian_jscc.modular_training import configure_phase,MODULES
        from gaussian_jscc.modular_snr_controls import control_inputs
        from gaussian_jscc.modular_experiment import make_scene
        from gaussian_jscc.data import read_ply
        from gaussian_jscc.render_objective import MultiViewRenderTask
        from test_render_first import synthetic_render
        with tempfile.TemporaryDirectory() as folder, \
             patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
             patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
            root=Path(folder);parent=source_run(root)
            model,_,_,_,_=restore_training_checkpoint(parent/'checkpoints/final/codec.pt','cpu')
            configure_phase(model,'channel_render',{name:1e-5 for name in MODULES})
            metadata=json.loads((parent/'training.json').read_text())
            original,_,spec,old,_=control_inputs(metadata)
            raw,degree=read_ply(spec['ply']);scene=make_scene(spec,raw,degree,model,original,old,root/'parity')
            batches=[(f,torch.where(ids>=0,torch.arange(ids.numel()).reshape_as(ids)%3+1,0))
                     for f,ids in zip(scene.groups,scene.group_ids)]
            gradients=[];losses=[]
            for mode in ('replay','checkpoint'):
                model.zero_grad(set_to_none=True);torch.manual_seed(91)
                task=MultiViewRenderTask(scene.cameras,scene.reference,degree,scene.white_background)
                loss,_=joint_step(model,batches,scene.geometry,task,1.,mode=mode)
                losses.append(loss);gradients.append({n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None})
            torch.testing.assert_close(losses[0],losses[1],rtol=1e-6,atol=1e-8)
            self.assertEqual(set(gradients[0]),set(gradients[1]))
            for name in gradients[0]:
                torch.testing.assert_close(gradients[0][name],gradients[1][name],rtol=1e-5,atol=1e-7)

    def test_joint_fork_preserves_all_weights_freezes_encoder_and_resumes_exactly(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.successive_joint'))
        from gaussian_jscc import successive_joint as entry
        from gaussian_jscc.modular_checkpoint import restore_training_checkpoint
        from test_render_first import synthetic_render
        with tempfile.TemporaryDirectory() as folder, \
             patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
             patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
            root=Path(folder);parent=source_run(root);source_path=parent/'checkpoints/final/codec.pt'
            original=source_path.read_bytes();source=torch.load(source_path,weights_only=True)
            result=entry.main(self.options(parent,root/'full'))
            self.assertEqual(result['status'],'complete');self.assertEqual(result['step'],4)
            initial=torch.load(root/'full/checkpoints/initial/codec.pt',weights_only=True)
            self.assertFalse(initial['training']['optimizer']['state'])
            for name,value in source['state_dict'].items():
                torch.testing.assert_close(initial['state_dict'][name],value,rtol=0,atol=0)
            final=torch.load(root/'full/checkpoints/final/codec.pt',weights_only=True)
            for name,value in source['state_dict'].items():
                if name.startswith('semantic_encoder.') or name in ('attr_mean','attr_std'):
                    torch.testing.assert_close(final['state_dict'][name],value,rtol=0,atol=0)
            rows=[json.loads(s) for s in (root/'full/metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['layout'] for r in rows],['1','2','3','mixed'])
            for row in rows:
                self.assertAlmostEqual(row['loss'],row['render_loss']+row['weighted_latent_loss'],places=7)
                self.assertEqual(row['updates']['semantic_encoder']['update_norm'],0.)
                for name in ('channel_encoder','channel_decoder','semantic_decoder'):
                    self.assertGreater(row['updates'][name]['update_norm'],0.)
                    self.assertEqual(row['module_lrs'][name],1e-5)
            model,opt,progress,_,_=restore_training_checkpoint(root/'full/checkpoints/final/codec.pt','cpu')
            self.assertEqual({g['name'] for g in opt.param_groups},{'channel_encoder','channel_decoder','semantic_decoder'})
            self.assertTrue(all(not p.requires_grad for p in model.semantic_encoder.parameters()))
            self.assertTrue(bool(model.channel_decoder.neighbor_enabled))
            args=self.options(parent,root/'resumed');entry.main(args,max_updates=2)
            args.resume=True;entry.main(args)
            resumed=torch.load(root/'resumed/checkpoints/final/codec.pt',weights_only=True)
            for name,value in final['state_dict'].items():
                torch.testing.assert_close(value,resumed['state_dict'][name],rtol=0,atol=0)
            self.assertEqual(final['training']['progress'],resumed['training']['progress'])
            for key,state in final['training']['optimizer']['state'].items():
                for name,value in state.items():
                    torch.testing.assert_close(value,resumed['training']['optimizer']['state'][key][name],rtol=0,atol=0)
            self.assertEqual(source_path.read_bytes(),original)
            summary=json.loads((root/'full/comparison/summary.json').read_text())
            self.assertEqual(len(summary['rows']),18)
            self.assertIn('diagnostic',summary['clean'])
            metrics=[json.loads(s) for s in (root/'full/evaluation/final/metrics/validation.jsonl').read_text().splitlines()]
            for row in metrics:
                if row['path']!='clean':
                    self.assertGreater(row['scale_stream_bytes'],0)
                    self.assertEqual(row['total_channel_uses'],row['payload_complex_symbols']+
                        row['coordinate_channel_uses']+row['metadata_channel_uses']+row['scale_channel_uses'])

    def test_source_validation_dry_run_and_real_shell_launcher(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.successive_joint'))
        from gaussian_jscc import successive_joint as entry
        from test_render_first import synthetic_render
        with tempfile.TemporaryDirectory() as folder, \
             patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
             patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
            root=Path(folder);parent=source_run(root)
            args=self.options(parent,root/'dry');args.dry_run=True
            plan=entry.main(args);self.assertFalse(Path(args.out).exists())
            self.assertEqual(plan['trainable_modules'],['channel_encoder','channel_decoder','semantic_decoder'])
            self.assertEqual(plan['optimizer_initialization'],'new Adam; source moments not loaded')
            args.out=str(parent/'nested')
            with self.assertRaises(ValueError):entry.build_plan(args)
            args.out=str(root/'bad');args.lr=float('nan')
            with self.assertRaises(ValueError):entry.build_plan(args)
            args.lr=1e-5
            marker=parent/'complete.json';completion=json.loads(marker.read_text())
            marker.write_text(json.dumps(dict(completion,status='interrupted')))
            with self.assertRaises(ValueError):entry.build_plan(args)
            marker.write_text(json.dumps(completion))
            path=parent/'checkpoints/final/codec.pt';saved=torch.load(path,weights_only=True)
            saved['state_dict']['channel_decoder.neighbor_enabled'].fill_(False);torch.save(saved,path)
            with self.assertRaises(ValueError):entry.build_plan(args)
            saved['state_dict']['channel_decoder.neighbor_enabled'].fill_(True);torch.save(saved,path)
            bash=Path('C:/softwares/Git/bin/bash.exe')
            if bash.exists():
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHON_BIN=Path(sys.executable).as_posix(),PYTHONUTF8='1')
                command=[str(bash),'scripts/test_successive_joint_truck.sh',parent.as_posix(),
                         (root/'shell_dry').as_posix(),'--dry-run','--device','cpu','--no-lpips']
                launch=subprocess.run(command,env=env,capture_output=True,text=True,encoding='utf8',timeout=30)
                self.assertEqual(launch.returncode,0,launch.stdout+launch.stderr)
                self.assertFalse((root/'shell_dry').exists())
                env.pop('CUDA_VISIBLE_DEVICES');launch=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30)
                self.assertNotEqual(launch.returncode,0)


if __name__=='__main__':unittest.main()
