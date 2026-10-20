"""Audit contracts: paired source points, genuine stage traces and immutable weights."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch


class ChannelAuditTests(unittest.TestCase):
    def audit(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_jscc.channel_audit'),
                             'read-only channel audit entry is missing')
        from gaussian_jscc import channel_audit
        return channel_audit

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)

    def model(self):
        from gaussian_jscc.modular_codec import ModularGaussianCodec
        from gaussian_jscc.modular_config import ModularCodecConfig
        cfg=ModularCodecConfig(sh_degree=1,hidden=16,grid_dim=4,levels=(2,),block_size=8,
            decoder_window=4,latent_dim=4,channel_variant='progressive',channel_hidden=16,
            context_variant='both',context_centers=(4,2),context_neighbors=(3,3),context_radii=(2.,4.),
            channel_prefix_mode='full_prefix',rates=(0,2,4,6),channel_denoiser_mode='pointwise',
            channel_denoiser_hidden=8,channel_denoiser_neighbors=3,channel_neighbor_mode='plain',
            channel_neighbor_hidden=8,channel_neighbor_neighbors=3)
        model=ModularGaussianCodec(cfg)
        from gaussian_jscc.data import Geometry
        model.set_scene_geometry(Geometry([0,0,0],[1,2,3],16))
        for module in (*model.channel_decoder.denoisers,model.channel_decoder.neighbor_refiner):
            torch.nn.init.normal_(module.readout.weight,std=.03)
        return model

    def test_power_reports_radial_conditioning_not_false_amplitude_loss(self):
        a=self.audit()
        raw=torch.tensor([[1.,1.,2.,2.],[1.,1.,2.,2.]])
        # E=2,8 per complex use. epsilon=2 => normalized powers .5,.8.
        signal=raw/torch.tensor([2.,2.,10**.5,10**.5])
        rows=a.power_report(raw,signal,(0,1,2),2.)
        self.assertAlmostEqual(rows[0]['normalized_power']['mean'],.5)
        self.assertAlmostEqual(rows[1]['radial_log_gain']['mean'],.2)
        self.assertAlmostEqual(rows[0]['radial_log_gain']['mean'],.5)

    def test_tail_fraction_uses_point_mean_error_energy(self):
        a=self.audit()
        e=torch.tensor([[[0.],[0.],[0.],[4.]],[[0.],[0.],[0.],[4.]]])
        r=a.error_tail(e,tail_fraction=.25)
        self.assertAlmostEqual(r['top_fraction_error_share'],1.)
        self.assertEqual(r['worst_point_indices'],[3])
        with self.assertRaises(ValueError):a.error_tail(e,tail_fraction=0)

    def test_physical_response_is_quaternion_sign_invariant_and_separates_color(self):
        a=self.audit()
        attrs=torch.zeros(2,20);attrs[:,4]=1;attrs[:,1:4]=torch.tensor([-1.,-2.,-3.])
        directions=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
        base=a.attribute_response(attrs,1,directions)
        changed=attrs.clone();changed[:,4]*=-1
        same=a.response_metrics(a.attribute_response(changed,1,directions),base)
        self.assertEqual(same['log_covariance_mse'],0.)
        self.assertEqual(same['relative_covariance_mse'],0.)
        changed[:,8]=1
        color=a.response_metrics(a.attribute_response(changed,1,directions),base)
        self.assertGreater(color['r_mse'],0.)
        self.assertEqual(color['g_mse'],0.)
        self.assertEqual(color['opacity_mse'],0.)

    def test_pointwise_attention_collapse_is_checked_on_actual_module(self):
        a=self.audit();model=self.model();module=model.channel_decoder.denoisers[0]
        previous=torch.randn(1,8,4);signal=torch.randn(1,8,12)
        active=torch.tensor([[True,True,False,True,True,True,True,True]])
        mask=torch.ones_like(signal);xyz=torch.rand(1,8,3)
        args=(previous,signal,mask,xyz,active,10.)
        with torch.no_grad():output=module(*args)
        error=a.pointwise_collapse_error(module,args,output)
        self.assertLess(error,1e-6)

    def test_trace_matches_official_mixed_forward_and_removes_hooks(self):
        a=self.audit();model=self.model();decoder=model.channel_decoder
        q=torch.tensor([[1,2,3,0,3,2,1,0]])
        rx=torch.randn(1,8,12);xyz=torch.rand(1,8,3)
        from gaussian_jscc.codec import prefix_mask
        mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(rx)
        expected=decoder(rx,mask,q,10.,delivered_xyz=xyz)
        trace=a.trace_channel(decoder,rx,mask,q,10.,xyz)
        torch.testing.assert_close(trace['final']['value'],expected,rtol=0,atol=0)
        self.assertEqual(int(trace['stage2']['active'].sum()),4)
        torch.testing.assert_close(trace['stage1_denoiser']['before']+trace['stage1_denoiser']['correction'],
                                   trace['stage1_denoiser']['value'],rtol=0,atol=0)
        torch.testing.assert_close(trace['neighbor']['value'],expected,rtol=0,atol=0)
        self.assertTrue(all(not m._forward_hooks for m in decoder.modules()))

    def test_physical_noise_snr_does_not_override_trained_conditioning_support(self):
        a=self.audit();model=self.model();model.set_training_snrs([10.])
        from gaussian_jscc.data import Geometry
        from test_gaussian_jscc import fixture
        raw,_=fixture(8,1);geometry=Geometry.fit(raw[:,:3],16)
        blocks=a.select_blocks(raw,geometry,8,1,42)
        conditions=[]
        hooks=[model.channel_encoder.register_forward_pre_hook(lambda m,args:conditions.append(float(args[2]))),
               model.channel_decoder.register_forward_pre_hook(lambda m,args:conditions.append(float(args[3])))]
        try:
            with tempfile.TemporaryDirectory() as t:
                result=a.audit_model(model,blocks,geometry,Path(t),trials=2,snr=0.,response_blocks=1,directions=2)
        finally:
            for hook in hooks:hook.remove()
        self.assertEqual(set(conditions),{10.})
        self.assertEqual(result['snr'],0.)
        self.assertEqual(result['tx_condition_snr'],10.)

    def test_small_real_audit_is_paired_reproducible_and_read_only(self):
        a=self.audit();model=self.model()
        from gaussian_jscc.data import Geometry
        from test_gaussian_jscc import fixture
        raw,degree=fixture(19,1);geometry=Geometry.fit(raw[:,:3],16)
        blocks=a.select_blocks(raw,geometry,8,2,42)
        identity={k:v.clone() for k,v in model.state_dict().items()}
        rng=torch.get_rng_state().clone();seen=[]
        hook=model.semantic_decoder.register_forward_pre_hook(lambda m,args:seen.append(args[0].shape[1]))
        try:
            with tempfile.TemporaryDirectory() as t:
                opts=dict(trials=2,snr=10.,seed=42,response_blocks=1,directions=2,probe_nmse=.02)
                first=a.audit_model(model,blocks,geometry,Path(t)/'first',**opts)
                second=a.audit_model(model,blocks,geometry,Path(t)/'second',**opts)
                self.assertEqual(first['layouts'],second['layouts'])
                self.assertEqual(first['sample_sha256'],second['sample_sha256'])
                self.assertEqual(first['optimizer_updates'],0)
                self.assertTrue((Path(t)/'first/stage_metrics.csv').is_file())
                self.assertTrue((Path(t)/'first/latent_dimensions.csv').is_file())
                import csv
                with (Path(t)/'first/stage_metrics.csv').open(newline='') as handle:
                    stage=next(row for row in csv.DictReader(handle) if row['layout']=='2' and
                               row['path']=='awgn' and row['node']=='stage2')
                self.assertEqual(int(stage['latent_point_observations']),22)  # 8+3 points, two trials
                self.assertEqual(int(stage['response_point_observations']),16)  # first complete 8-point block only
                for row in first['layouts']:
                    self.assertAlmostEqual(row['moments']['summary']['decomposition_residual'],0.,places=9)
                    self.assertAlmostEqual(row['moments']['summary']['noiseless_shift_variance_identity_residual'],0.,places=9)
                    self.assertIn('log_covariance_mse',row['semantic_response']['noisy_vs_clean'])
                    self.assertIn('actual_direction_equal_nmse',row['sensitivity'])
                    self.assertIn('isotropic_equal_nmse',row['sensitivity'])
        finally:hook.remove()
        self.assertEqual(set(seen),{len(b['raw']) for b in blocks[:1]})
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in identity.items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_sampling_rejects_bad_budgets_and_keeps_original_morton_blocks(self):
        a=self.audit()
        from test_gaussian_jscc import fixture
        from gaussian_jscc.data import Geometry,prepare
        raw,_=fixture(19,1);geometry=Geometry.fit(raw[:,:3],16)
        selected=a.select_blocks(raw,geometry,8,99,42)
        ordered,_,_=prepare(raw,16)
        self.assertEqual(sorted(len(b['raw']) for b in selected),[3,8,8])
        for b in selected:
            torch.testing.assert_close(b['raw'],ordered[b['block_index']*8:(b['block_index']+1)*8])
            torch.testing.assert_close(b['raw'],raw[b['source_ids']])
        with self.assertRaises(ValueError):a.select_blocks(raw,geometry,8,0,42)

    def test_plan_refuses_source_output_overlap_and_mismatched_scene(self):
        a=self.audit();model=self.model()
        from gaussian_jscc.transport import save_checkpoint
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for name in ('new','old'):
                run=root/name;(run/'checkpoints/final').mkdir(parents=True)
                (run/'training.json').write_text(json.dumps(dict(codec_config=model.cfg.to_dict(),
                    scene={'fingerprint':'same'},manifest='unused.json')),encoding='utf-8')
                save_checkpoint(run/'checkpoints/final/codec.pt',model,6,dict(scene_record={'fingerprint':'same'}))
            opts=a.parser().parse_args(['--run',str(root/'new'),'--reference-run',str(root/'old'),
                                       '--out',str(root/'audit'),'--device','cpu','--dry-run'])
            plan=a.main(opts)
            self.assertEqual(plan['optimizer_updates'],0)
            self.assertFalse((root/'audit').exists())
            opts.out=str(root/'new/audit')
            with self.assertRaisesRegex(ValueError,'separate'):a.build_plan(opts)
            opts.out=str(root/'audit')
            meta=json.loads((root/'old/training.json').read_text());meta['scene']['fingerprint']='different'
            (root/'old/training.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'mismatch'):a.build_plan(opts)

    def test_end_to_end_two_different_teachers_share_sources_and_generate_comparison(self):
        a=self.audit()
        from gaussian_jscc.data import write_ply,read_ply,Geometry
        from gaussian_jscc.allocation import scene_fingerprint
        from gaussian_jscc.transport import save_checkpoint
        from test_gaussian_jscc import fixture
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);raw,_=fixture(19,1);degree=1
            write_ply(root/'source.ply',raw,degree);raw,degree=read_ply(root/'source.ply')
            geometry=Geometry.fit(raw[:,:3],16)
            record=dict(fingerprint=scene_fingerprint(raw),source_gaussians=19,geometry=geometry.to_dict())
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps(dict(scenes=[dict(name='truck',role='train',ply='source.ply',source='.')])) )
            hashes=[]
            for name in ('new','old'):
                model=self.model()
                # The semantic feature bases genuinely differ; do not exchange them.
                if name=='old':
                    with torch.no_grad():model.semantic_encoder.readout.bias.add_(.7)
                run=root/name;(run/'checkpoints/final').mkdir(parents=True)
                checkpoint=run/'checkpoints/final/codec.pt'
                save_checkpoint(checkpoint,model,6,dict(scene_record=record))
                hashes.append(checkpoint.read_bytes())
                (run/'training.json').write_text(json.dumps(dict(codec_config=model.cfg.to_dict(),
                    scene=record,manifest=str(manifest))),encoding='utf-8')
            opts=a.parser().parse_args(['--run',str(root/'new'),'--reference-run',str(root/'old'),
                '--out',str(root/'audit'),'--device','cpu','--blocks','2','--response-blocks','1',
                '--trials','2','--directions','2'])
            # Observe actual transmitter/receiver tensors without replacing the
            # channel or its noise generator. Different teachers must still see
            # paired noise in every full slot, not just equal random seeds.
            from unittest.mock import patch
            from gaussian_jscc.transport import load_checkpoint
            observed={};hooks=[]
            def observe(path,device):
                model=load_checkpoint(path,device);row=dict(tx=[],rx=[])
                observed[Path(path).parents[2].name]=row
                hooks.append(model.channel_encoder.register_forward_hook(lambda m,args,out:row['tx'].append(out.detach().clone())))
                hooks.append(model.channel_decoder.register_forward_pre_hook(lambda m,args:row['rx'].append(args[0].detach().clone())))
                return model
            try:
                with patch('gaussian_jscc.channel_audit.load_checkpoint',side_effect=observe):result=a.main(opts)
            finally:
                for hook in hooks:hook.remove()
            for index,(new_rx,old_rx) in enumerate(zip(observed['new']['rx'],observed['old']['rx'])):
                block=(index//3)%2  # noiseless + two noise trials, original two-block ordering
                new_noise=new_rx-observed['new']['tx'][block]
                old_noise=old_rx-observed['old']['tx'][block]
                torch.testing.assert_close(new_noise,old_noise,atol=3e-7,rtol=2e-6)
            self.assertEqual(len(result['rows']),8)
            self.assertTrue((root/'audit/comparison.png').is_file())
            self.assertTrue((root/'audit/complete.json').is_file())
            reports=[json.loads((root/'audit'/name/'audit.json').read_text()) for name in ('current','reference')]
            self.assertEqual(reports[0]['source_ids'],reports[1]['source_ids'])
            self.assertNotEqual(reports[0]['layouts'][0]['moments']['summary']['target_power'],
                                reports[1]['layouts'][0]['moments']['summary']['target_power'])
            for name,original in zip(('new','old'),hashes):
                self.assertEqual((root/name/'checkpoints/final/codec.pt').read_bytes(),original)
            with self.assertRaises(FileExistsError):a.main(opts)


if __name__=='__main__':unittest.main()
