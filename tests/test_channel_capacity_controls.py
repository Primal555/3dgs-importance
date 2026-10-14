import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from gaussian_jscc.modular_config import ModularCodecConfig
from gaussian_jscc.modular_codec import ModularGaussianCodec
from gaussian_jscc.modular_checkpoint import initialize_channel_run
from gaussian_jscc.modular_training import configure_phase
import test_modular_experiment as experiment_fixture
import test_render_first as render_fixture
from test_render_first import synthetic_render


class ChannelCapacityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_optional_variant_preserves_baseline_identity_and_prefix_contract(self):
        base=ModularCodecConfig()
        self.assertNotIn('channel_variant',base.to_dict())
        # Ignored baseline channel width must not break config/checkpoint resume.
        ignored=ModularCodecConfig(channel_hidden=32)
        self.assertEqual(ModularCodecConfig.from_dict(ignored.to_dict()),ignored)
        cfg=ModularCodecConfig(hidden=16,grid_dim=4,levels=(2,),decoder_window=4,
                               channel_variant='residual',channel_hidden=32)
        self.assertEqual(ModularCodecConfig.from_dict(cfg.to_dict()),cfg)
        model=ModularGaussianCodec(cfg)
        y=torch.randn(2,7,48); q=torch.tensor([[1,2,3,0,1,2,3]]*2)
        z=model.channel_encoder(y,q>0,10)
        from gaussian_jscc.codec import prefix_mask
        mask=prefix_mask(q.flatten(),cfg.rates).reshape_as(z)
        a=model.channel_decoder(z,mask,q,10)
        b=model.channel_decoder(z.masked_fill(~mask,1e5),mask,q,10)
        torch.testing.assert_close(a,b)
        self.assertEqual(int(a[q==0].count_nonzero()),0)
        for start,end in ((0,4),(4,12),(12,24)):
            self.assertLessEqual(float(z[...,2*start:2*end].detach().square().sum(-1).max()/(end-start)),1.00001)
        a.square().mean().backward()
        self.assertGreater(float(model.channel_encoder.blocks[-1].output.weight.grad.norm()),0)
        self.assertGreater(float(model.channel_decoder.blocks[-1].output.weight.grad.norm()),0)
        for kwargs in ({'channel_variant':'unknown'},{'channel_hidden':0}):
            with self.assertRaises(ValueError): ModularCodecConfig(**kwargs)

    def test_frozen_semantic_render_still_backpropagates_through_decoder(self):
        cfg=ModularCodecConfig(hidden=16,grid_dim=4,levels=(2,),decoder_window=4)
        model=ModularGaussianCodec(cfg)
        opt=configure_phase(model,'channel_render',{},freeze_semantic=True)
        self.assertEqual([g['name'] for g in opt.param_groups],['channel_encoder','channel_decoder'])
        f=torch.randn(1,8,3+cfg.attr_dim); f[...,:3]=torch.rand(1,8,3)
        q=torch.full((1,8),3)
        pred=model.forward_tier_batches(f,f[...,:3],torch.nn.functional.one_hot(q,4).float(),10,'awgn')[0]
        pred[...,11:].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.semantic_decoder.parameters()))
        self.assertGreater(float(model.channel_decoder.input.weight.grad.norm()),0)

    def test_real_training_freezes_semantics_and_resumes_exactly(self):
        from gaussian_jscc.modular_experiment import main
        from gaussian_jscc.transport import load_checkpoint
        with tempfile.TemporaryDirectory() as t, patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras',return_value=render_fixture.RenderFirstTests().cameras()):
            root=Path(t); source=experiment_fixture.source_fixture(root); helper=experiment_fixture.ModularExperimentTests()
            main(helper.args(source,root/'teacher'),max_updates=4)
            teacher=root/'teacher/checkpoints/end_semantic_render/codec.pt'
            # Finish boundary snapshot if interrupted immediately after last update.
            if not teacher.exists():
                opts=helper.args(source,root/'teacher'); opts.resume=str(root/'teacher/checkpoints/latest/codec.pt')
                main(opts,max_updates=1)
            reference=load_checkpoint(teacher,'cpu')
            for folder in ('baseline','full','resume'):
                opts=helper.args(source,root/folder); opts.init_semantic=str(teacher)
                opts.channel_variant='baseline' if folder=='baseline' else 'residual'
                opts.channel_hidden=192 if folder=='baseline' else 32; opts.freeze_semantic=True
                main(opts,max_updates=3 if folder=='resume' else None)
                if folder=='resume':
                    opts.init_semantic=None; opts.resume=str(root/folder/'checkpoints/latest/codec.pt')
                    main(opts)
                trained=load_checkpoint(root/folder/'checkpoints/final/codec.pt','cpu')
                for name,value in reference.state_dict().items():
                    if name.startswith(('semantic_encoder.','semantic_decoder.')) or name in ('attr_mean','attr_std'):
                        torch.testing.assert_close(value,trained.state_dict()[name],rtol=0,atol=0)
            a=load_checkpoint(root/'full/checkpoints/final/codec.pt','cpu')
            b=load_checkpoint(root/'resume/checkpoints/final/codec.pt','cpu')
            for name,value in a.state_dict().items(): torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
            base=torch.load(root/'baseline/checkpoints/final/codec.pt',weights_only=True)
            full=torch.load(root/'full/checkpoints/final/codec.pt',weights_only=True)
            torch.testing.assert_close(base['training']['rng']['torch'],full['training']['rng']['torch'],rtol=0,atol=0)
            rowsets=[]
            for folder in ('baseline','full'):
                rows=[json.loads(l) for l in (root/folder/'metrics/loss.jsonl').read_text().splitlines()]
                rowsets.append([(r['layout'],r['snr'],r['training_view_indices']) for r in rows])
            self.assertEqual(rowsets[0],rowsets[1])
            from gaussian_jscc.channel_capacity_controls import summarize
            summary=summarize(root,['baseline','full'],teacher)
            self.assertTrue(all(r['semantic_weights_unchanged'] for r in summary['semantic_audit'].values()))
            self.assertTrue((root/'comparison/channel_curves.png').exists())
            full['state_dict']['semantic_decoder.input.bias'][0]+=1
            torch.save(full,root/'full/checkpoints/latest/codec.pt')
            with self.assertRaisesRegex(RuntimeError,'semantic weights'):
                summarize(root,['baseline','full'],teacher)
            with self.assertRaisesRegex(ValueError,'config'):
                initialize_channel_run(teacher,'cpu',cfg=ModularCodecConfig(hidden=32))

    def test_control_plan_has_same_teacher_and_all_tiers(self):
        from gaussian_jscc.channel_capacity_controls import parser,build_plan
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); source=experiment_fixture.source_fixture(root)
            checkpoint=root/'teacher.pt'; checkpoint.write_bytes(b'plan does not load weights')
            args=parser().parse_args(['--run',str(source),'--out',str(root/'out'),'--init-semantic',str(checkpoint)])
            plan=build_plan(args)
            self.assertEqual([j['group'] for j in plan['jobs']],['baseline','residual'])
            for job in plan['jobs']:
                self.assertIn('--freeze-semantic',job['command'])
                self.assertIn(str(checkpoint.resolve()),job['command'])
            self.assertEqual(plan['rates'],[0,4,12,24])
            args.blocks_per_batch=1
            with self.assertRaisesRegex(ValueError,'teacher batch'):
                build_plan(args)


if __name__=='__main__': unittest.main()
