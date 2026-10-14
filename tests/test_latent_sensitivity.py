"""Decoder audit: matched relative noise, immutable weights, no channel calls."""
import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_gaussian_jscc import fixture
from gaussian_jscc.data import prepare,write_ply,to_features
from gaussian_jscc.modular_config import ModularCodecConfig
from gaussian_jscc.modular_codec import ModularGaussianCodec
from gaussian_jscc.allocation import scene_fingerprint
from gaussian_jscc.modular_checkpoint import capture_rng,save_training_checkpoint


class LatentSensitivityTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.latent_sensitivity'), 'decoder sensitivity audit missing')
        return importlib.import_module('gaussian_jscc.latent_sensitivity')

    def model(self,variant='baseline'):
        torch.set_num_threads(1)
        return ModularGaussianCodec(ModularCodecConfig(sh_degree=1,hidden=16,grid_dim=4,levels=(2,),
            block_size=8,decoder_window=4,context_variant=variant,context_centers=(4,2),context_neighbors=(4,8)))

    def test_noise_has_exact_pointwise_relative_rms_and_preserves_rng(self):
        audit=self.module(); y=torch.tensor([[1.,2.,3.],[.1,.2,.3],[0.,0.,0.]])
        rng=torch.get_rng_state().clone(); a=audit.relative_noise(y,.2,42); b=audit.relative_noise(y,.2,42)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(a.square().mean(-1),y.square().mean(-1)*.04)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        torch.testing.assert_close(audit.relative_noise(y*7,.2,42),a*7)
        sparse=audit.relative_noise(y,.2,42,point=1)
        torch.testing.assert_close(sparse[[0,2]],torch.zeros(2,3)); torch.testing.assert_close(sparse[1],a[1])
        with self.assertRaises(ValueError): audit.relative_noise(y,-.1,42)

    def test_unclipped_rgb_and_common_source_response_match_physical_fixture(self):
        audit=self.module(); m=self.model(); raw,_=fixture(8,1); ordered,g,_=prepare(raw,16)
        f,_=to_features(ordered,g,m); dirs=torch.tensor([[1.,0,0],[-1.,0,0]])
        shifted=f.clone(); shifted[:,11]+=1
        a=audit.color_values(f,m,dirs); b=audit.color_values(shifted,m,dirs)
        self.assertAlmostEqual(float((a-b).square().mean()),.02652582384864923,places=6)
        self.assertEqual(audit.source_grid_response(f,f,g,m,dirs).shape,(8,2,25,2,3))

    def test_actual_models_frozen_noiseless_zero_control_and_spread(self):
        audit=self.module(); raw,_=fixture(17,1); raw,g,_=prepare(raw,16)
        for variant in ('baseline','both'):
            m=self.model(variant); before={n:p.clone() for n,p in m.state_dict().items()}
            with patch.object(m.channel_encoder,'forward',side_effect=AssertionError('no channel encoding')), \
                 patch.object(m.channel_decoder,'forward',side_effect=AssertionError('no channel decoding')):
                rows,spread=audit.audit_model(m,raw[:8],g,block_index=0,levels=[0.,.1],trials=2,directions=4,seed=42,sparse_level=.2)
            self.assertTrue(m.training)
            for n,p in m.state_dict().items(): torch.testing.assert_close(p,before[n],rtol=0,atol=0)
            self.assertEqual(len(rows),4); self.assertEqual(len(spread),2)
            zero=[r for r in rows if r['level']==0]
            self.assertTrue(all(r['color_mse']==0 and r['response_mse']==0 and r['input_nmse']==0 for r in zero))
            self.assertTrue(all(abs(r['input_nmse']-.01)<1e-6 for r in rows if r['level']==.1))
            self.assertTrue(all(torch.isfinite(r['_color_errors']).all() for r in rows))

    def test_cli_verifies_exact_ply_and_emits_only_compact_diagnostics(self):
        audit=self.module(); raw,_=fixture(17,1)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); source=root/'source.ply'; write_ply(source,raw,1)
            from gaussian_jscc.data import read_ply
            raw,_=read_ply(source); ordered,g,_=prepare(raw,16)
            scene={'fingerprint':scene_fingerprint(raw),'source_gaussians':17,'sh_degree':1,'geometry':g.to_dict()}
            suite=root/'suite'
            for variant in ('baseline','both'):
                m=self.model(variant)
                folder=suite/variant; folder.mkdir(parents=True)
                (folder/'training.json').write_text(json.dumps({'scene':scene,'selected_scenes':[{'name':'truck','ply':str(source)}]}))
                save_training_checkpoint(folder/'checkpoints/end_semantic_render/codec.pt',m,None,
                    {'step':3000},capture_rng(),scene)
            args=audit.parser().parse_args(['--run',str(suite),'--out',str(root/'audit'),'--blocks','2',
                '--levels','0','.1','--trials','1','--directions','4'])
            audit.main(args)
            summary=json.loads((root/'audit/summary.json').read_text())
            self.assertTrue(summary['weights_unchanged']); self.assertFalse(summary['channel_invoked'])
            self.assertEqual(len(summary['curves']),4)
            self.assertTrue((root/'audit/sensitivity.png').exists())
            self.assertTrue((root/'audit/spread.csv').exists())
            self.assertFalse(list((root/'audit').rglob('*.pt')))
            args.out=str(root/'bad'); args.ply=str(root/'wrong.ply'); write_ply(Path(args.ply),raw[:8],1)
            with self.assertRaisesRegex(ValueError,'PLY'): audit.main(args)
            self.assertFalse(Path(args.out).exists())


if __name__=='__main__': unittest.main()
