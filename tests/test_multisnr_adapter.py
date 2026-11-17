from collections import Counter
from dataclasses import replace
import json
import sys
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from gaussian_jscc import successive_multisnr_adapter as entry
from gaussian_jscc.multisnr_adapter_training import train_step, visit_schedule, batches_for
from gaussian_jscc.multisnr_adaptation import noiseless_tier
from gaussian_jscc.multisnr_training import cached_render_step
from gaussian_jscc.model_factory import build_codec, config_from_dict
from gaussian_jscc.optimization import preserved_rng
from gaussian_jscc.render_objective import MultiViewRenderTask
from gaussian_jscc.snr_adapter import adapter_parameter, configure_optimizer
from test_successive_multisnr import shared, digital_runner
from test_successive_q3_refinement import load_model_scene, assert_same_checkpoint
from test_successive_joint import cameras
from test_render_first import synthetic_render


def options(source, out):
    return entry.parser().parse_args(['--run',str(source),'--out',str(out),'--device','cpu',
        '--digital-python',sys.executable,'--digital-device','cpu','--scenes','truck','playroom',
        '--heldout-scenes','drjohnson','--steps-per-scene','4','--save-every','1','--validate-every','2',
        '--views-per-step','1','--validation-views','1','--trials','1','--monitor-trials','1',
        '--eval-snrs','5','10','15','--noiseless-every','1','--cpu-threads','1','--no-lpips'])


def adapted(source):
    cfg=replace(source.cfg,channel_snr_adapter_hidden=8)
    model=build_codec(cfg)
    missing, unexpected=model.load_state_dict(source.state_dict(),strict=False)
    assert not unexpected and missing and all(adapter_parameter(k) for k in missing)
    model.set_training_snrs([5.,15.])
    return model


def test_per_scene_schedules_cover_noiseless_tiers_and_noise_changes():
    counts={s:Counter() for s in range(9)}
    for visit in range(1,1001):
        for scene in range(9):
            q=noiseless_tier(visit,4)
            if q: counts[scene][q]+=1
    assert all(c=={1:84,2:83,3:83} for c in counts.values())
    rng=torch.get_rng_state().clone()
    for scene in ('truck','playroom'):
        records=[visit_schedule(42,scene,i) for i in range(1,101)]
        assert Counter(r['layout'] for r in records)=={q:25 for q in ('1','2','3','mixed')}
        assert len({r['noise_seed'] for r in records})==100
        for r in records:
            assert all(5+i*10/3<=v<5+(i+1)*10/3 for i,v in enumerate(r['snrs']))
        assert records[79]==visit_schedule(42,scene,80)
    assert torch.equal(rng,torch.get_rng_state())
    assert visit_schedule(42,'truck',1)!=visit_schedule(42,'playroom',1)


def test_zero_init_and_trained_ten_db_packet_invariance(shared,tmp_path):
    with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
        _,source,scene,_=load_model_scene(shared,tmp_path)
        assert 'channel_snr_adapter_hidden' not in source.cfg.to_dict()
        model=adapted(source)
        assert entry.initialization_probe(model,source,scene,42)['passed']
        optimizer=configure_optimizer(model,1e-3)
        targets=[b['y'] for b in entry.cache_teacher(model,scene)]
        args=options(shared,tmp_path/'unused')
        original=entry.frozen_digest(model.state_dict())
        for visit in range(1,5):
            optimizer.zero_grad(set_to_none=True)
            train_step(model,scene,targets,scene.cameras[:1],args,visit)
            assert all(p.grad is None for n,p in model.named_parameters() if not adapter_parameter(n))
            assert sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if adapter_parameter(n) and p.grad is not None)>0
            optimizer.step()
        assert entry.frozen_digest(model.state_dict())==original==entry.frozen_digest(source.state_dict())
        # A rebuilt receiver must retain the adapters and wire-scale boundary.
        rebuilt=build_codec(config_from_dict(model.cfg.to_dict()))
        rebuilt.load_state_dict(model.state_dict())
        rebuilt.set_scene_geometry(scene.geometry)
        model.eval(); source.eval()
        f=scene.groups[0]; y=targets[0]; active=scene.group_ids[0]>=0
        from gaussian_jscc.position_delivery import delivered_positions
        for tier in (1,2,3):
            q=active.long()*tier
            a=source.encode_channel_packet(y,q,10.)
            b=model.encode_channel_packet(y,q,10.)
            assert a['scales']==b['scales']
            torch.testing.assert_close(a['payload'],b['payload'],atol=0,rtol=0)
            xyz=delivered_positions(f[...,:3],q,model.cfg)
            x=source.decode_channel_packet(a['payload'],q,10.,a['scales'],xyz)
            z=rebuilt.decode_channel_packet(b['payload'],q,10.,b['scales'],xyz)
            torch.testing.assert_close(x,z,atol=1e-7,rtol=1e-6)
        q=active.long()*3
        # Adapters have acquired a nonzero response outside the anchor.
        a=source.encode_channel_packet(y,q,10.)['payload']
        b=model.encode_channel_packet(y,q,6.)['payload']
        assert not torch.allclose(a,b,atol=1e-7,rtol=1e-6)


def test_three_snr_gradient_is_mean_and_uses_paired_noise(shared,tmp_path):
    with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
        _,source,scene,_=load_model_scene(shared,tmp_path)
        model=adapted(source); optimizer=configure_optimizer(model,1e-5)
        targets=[b['y'] for b in entry.cache_teacher(model,scene)]
        args=options(shared,tmp_path/'unused'); args.noiseless_weight=0
        schedule=visit_schedule(args.seed,scene.name,1)
        batches=batches_for(scene,targets,schedule['layout'],args.seed,1)
        grads=[]; noises=[]
        from gaussian_jscc.codec import channel
        def observe(symbols,snr,kind,*a,**kw):
            result=channel(symbols,snr,kind,*a,**kw)
            noises.append((snr,(result-symbols)/(10**(-snr/20)/2**.5)))
            return result
        for snr in schedule['snrs']:
            optimizer.zero_grad(set_to_none=True)
            with preserved_rng(torch.device('cpu')):
                torch.manual_seed(schedule['noise_seed'])
                cached_render_step(model,batches,scene.geometry,snr,
                    MultiViewRenderTask(scene.cameras[:1],scene.reference,scene.degree),args.latent_weight)
            grads.append({n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None})
        optimizer.zero_grad(set_to_none=True)
        with patch('gaussian_jscc.successive_codec.channel',side_effect=observe):
            stats=train_step(model,scene,targets,scene.cameras[:1],args,1)
        for n,p in model.named_parameters():
            if p.grad is not None:
                torch.testing.assert_close(p.grad,sum(g[n] for g in grads)/3,atol=1e-7,rtol=2e-5)
        firsts=[next(noise for s,noise in noises if s==snr) for snr in schedule['snrs']]
        for v in firsts[1:]:torch.testing.assert_close(v,firsts[0],atol=2e-6,rtol=2e-5)
        assert stats['full_scene_recoveries']==3


def test_full_resume_evaluation_and_actual_delivery(shared,tmp_path):
    with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('comparison_baselines.run._execute',side_effect=digital_runner), \
         patch('gaussian_jscc.multisnr_plots.write_figures'):
        full,resumed=tmp_path/'full',tmp_path/'resumed'
        result=entry.main(options(shared,full))
        assert result['step']==8 and result['ten_db_and_clean_preserved']
        assert result['noiseless_counts']=={s:{'1':2,'2':1,'3':1} for s in ('truck','playroom')}
        entry.main(options(shared,resumed),max_updates=3)
        args=options(shared,resumed); args.resume=True
        entry.main(args)
        a=torch.load(full/'checkpoints/final/codec.pt',weights_only=True)
        b=torch.load(resumed/'checkpoints/final/codec.pt',weights_only=True)
        assert_same_checkpoint(a,b)
        from gaussian_jscc.modular_checkpoint import restore_training_checkpoint
        restored,opt,progress,_,_=restore_training_checkpoint(full/'checkpoints/final/codec.pt','cpu')
        assert progress['snr_adapter_only']
        assert all(p.requires_grad==adapter_parameter(n) for n,p in restored.named_parameters())
        report=entry.read(full/'review_summary.json')
        assert report['ten_db_and_clean_preserved']
        rows=entry.read(full/'evaluation/final/truck/summary.json')['rows']
        actual=entry.read(full/'actual_delivery/truck/summary.json')['trials']
        for r in actual:
            expected=next(x for x in rows if x['path']=='awgn' and x['layout']==r['method'] and x['snr_db']==r['snr_db'])
            assert r['total_channel_uses']==expected['total_channel_uses']
            assert r['metrics']['source_psnr']==pytest.approx(expected['source_psnr'],abs=1e-5)
        assert (full/'review.tar.gz').exists()
        args=options(shared,full); args.resume=True
        assert entry.main(args)['status']=='complete'
