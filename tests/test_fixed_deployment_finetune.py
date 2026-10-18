"""Tiny CPU training checks; no real scene rendering or full regression run."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from gaussian_jscc.modular_allocation import main, parser, save_state, configure_allocation_phase


class FixedDeploymentTests(unittest.TestCase):
    def test_finetune_requires_skipping_new_allocation_training(self):
        args = parser().parse_args(['--run','missing','--out','unused','--fixed-deployment'])
        with self.assertRaisesRegex(ValueError,'allocation-steps 0'): main(args)

    def source(self, root):
        from test_modular_allocation import small_model
        from test_modular_experiment import source_fixture
        from gaussian_jscc.allocation import GaussianTierMask
        from gaussian_jscc.modular_experiment import inputs, make_scene
        from gaussian_jscc.data import read_ply
        from gaussian_jscc.transport import model_id
        source = source_fixture(root)
        _, _, _, model = small_model()
        opts = parser().parse_args(['--run', str(source), '--out', str(root/'setup')])
        config, manifest, spec, old = inputs(opts)
        raw, degree = read_ply(spec['ply'])
        scene = make_scene(spec, raw, degree, model, config, old, root/'setup')
        run = root/'beta005'
        mask = GaussianTierMask(16)
        q = torch.arange(16) % 4
        with torch.no_grad():
            mask.keep_logits.copy_(torch.nn.functional.one_hot((q > 0).long(), 2)*40.)
            mask.logits.copy_(torch.nn.functional.one_hot((q-1).clamp_min(0), 3)*40.)
        identity = model_id(model)
        save_state(run, 'final', model, mask, torch.optim.Adam(mask.parameters()),
            configure_allocation_phase(model, 'joint', 1e-5),
            dict(step=2000, phase='joint', completed=dict(allocation=1000, joint=1000)),
            scene, dict(snr=10.), identity)
        (run/'deployment/final').mkdir(parents=True)
        np.save(run/'deployment/final/tiers.npy', q.numpy().astype(np.uint8))
        (run/'deployment/final/allocation.json').write_text(json.dumps(dict(
            source_gaussians=16, hard_tier_counts=[4,4,4,4], rates=[0,8,24,48], snr=10., step=2000)))
        (run/'complete.json').write_text(json.dumps(dict(final_codec_id=identity)))
        (run/'training.json').write_text(json.dumps(dict(config, codec_config=model.cfg.to_dict(),
            scene=scene.record, selected_scenes=[spec], beta=.005, manifest=str(manifest))))
        return run, q

    def args(self, run, out, mode='--fixed-deployment'):
        return parser().parse_args(['--run', str(run), '--out', str(out), mode, '--device','cpu',
            '--allocation-steps','0','--joint-steps','4','--beta','.005','--prefix-anchor-every','2',
            '--validate-every','2','--save-every','2','--trials','1','--test-views','1','--image-views','1','--no-lpips'])

    def test_fixed_map_updates_all_modules_without_allocator_and_exactly_resumes(self):
        from test_render_first import RenderFirstTests, synthetic_render
        from gaussian_jscc.transport import load_checkpoint
        def render(raw, camera, degree, white=False, existence=None, colors=None):
            if colors is not None:
                return colors.mean(0)[:,None,None].expand(3,8,8)
            return synthetic_render(raw,camera)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, \
             patch('gaussian_jscc.rendering.load_cameras',return_value=RenderFirstTests().cameras()), \
             patch('gaussian_jscc.rendering.render',side_effect=render):
            root = Path(directory)
            run, q = self.source(root)
            with patch('gaussian_jscc.modular_allocation.local_mask_step', side_effect=AssertionError('no mask training')):
                main(self.args(run,root/'full'))
                main(self.args(run,root/'resume'),max_updates=2)
                args = self.args(run,root/'resume'); args.resume=True; main(args)
            a = load_checkpoint(root/'full/checkpoints/final/codec.pt','cpu')
            b = load_checkpoint(root/'resume/checkpoints/final/codec.pt','cpu')
            for name,value in a.state_dict().items(): torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
            state = torch.load(root/'full/checkpoints/final/state.pt',weights_only=True)
            self.assertIsNone(state['mask']); self.assertIsNone(state['mask_optimizer'])
            torch.testing.assert_close(state['fixed_tiers'],q)
            rows = [json.loads(line) for line in (root/'full/metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows],[1,2,3,4])
            self.assertEqual([r['prefix_anchor_applied'] for r in rows],[False,True,False,True])
            for row in rows:
                self.assertEqual(row['sampled_tier_counts'],[4,4,4,4])
                self.assertEqual(row['rate_loss'],0.)
                self.assertEqual(row['aux_loss'],0.)
                self.assertTrue(all(v['update_norm']>0 for v in row['updates'].values()))
            np.testing.assert_array_equal(np.load(root/'full/deployment/final/tiers.npy'),q.numpy())
            self.assertFalse((root/'full/deployment/allocation_latest/probabilities.npy').exists())
            report = [json.loads(line) for line in (root/'full/metrics/test.jsonl').read_text().splitlines()]
            self.assertEqual({r['layout'] for r in report},{'1','2','3','deployment'})
            self.assertEqual(next(r for r in report if r['layout']=='deployment')['tier_counts'],[4,4,4,4])
            # Even a changed map with identical counts must be rejected on resume.
            np.save(run/'deployment/final/tiers.npy',q.flip(0).numpy().astype(np.uint8))
            changed = self.args(run,root/'resume'); changed.resume=True
            with self.assertRaisesRegex(ValueError,'map changed'): main(changed)
            np.save(run/'deployment/final/tiers.npy',q.numpy().astype(np.uint8))
            # Optional continuing-joint arm starts from the learned table, not a new prior.
            main(self.args(run,root/'control','--continue-allocation'),max_updates=1)
            initial = torch.load(root/'control/checkpoints/initial/state.pt',weights_only=True)
            source = torch.load(run/'checkpoints/final/allocation.pt',weights_only=True)
            for name,value in initial['mask'].items(): torch.testing.assert_close(value,source['state_dict'][name])


if __name__ == '__main__': unittest.main()
