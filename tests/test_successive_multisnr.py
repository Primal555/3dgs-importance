"""Matched SNR, frozen scopes, cached gradients, exact resume and byte accounting."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import pytest
import torch

from comparison_baselines.digital_link import LinkConfig, file_channel_cost
from gaussian_jscc import successive_multisnr as entry
from gaussian_jscc.multisnr_training import schedule, optimizer_for, cached_render_step
from test_render_first import synthetic_render
from test_successive_joint import cameras
import test_successive_multiscene as legacy


@pytest.fixture(scope='module')
def shared(tmp_path_factory):
    root = tmp_path_factory.mktemp('shared_multisnr')
    torch.set_num_threads(1)
    with contextlib.redirect_stdout(io.StringIO()), \
         patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
        from gaussian_jscc import successive_multiscene
        parent, manifest = legacy.source_and_manifest(root)
        source = root/'shared'
        args = legacy.SuccessiveMultisceneTests().options(parent,manifest,source)
        args.steps_per_scene = 1
        successive_multiscene.main(args)
    return source


def options(source,out):
    return entry.parser().parse_args(['--run',str(source),'--out',str(out),'--device','cpu',
        '--digital-python',sys.executable,'--digital-device','cpu','--scenes','truck',
        '--heldout-scenes','drjohnson','--channel-steps-per-scene','4','--joint-steps-per-scene','2',
        '--eval-snrs','5','10','15','--save-every','1','--trials','1','--cpu-threads','1',
        '--no-lpips','--color-stat-start','0','--color-start','0'])


def digital_runner(python,module,arguments,log,*,fail=False):
    assert module=='comparison_baselines.digital_link'
    if arguments[0]=='self-test': return
    params = {str(arguments[i]):arguments[i+1] for i in range(1,len(arguments),2)}
    src,dest,report = [Path(params[k]) for k in ('--source','--destination','--report')]
    cfg = LinkConfig(**{k:(str(params['--'+k.replace('_','-')]) if k=='device'
                          else int(params['--'+k.replace('_','-')])) for k in LinkConfig.__dataclass_fields__})
    failed = fail and src.name=='metadata.bin'
    if not failed: shutil.copyfile(src,dest)
    result = dict(status='failed' if failed else 'success',config=vars(cfg),
        source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),snr_db=float(params['--snr-db']),
        seed=int(params['--seed']),noiseless=False,**file_channel_cost(src.stat().st_size,cfg))
    report.write_text(json.dumps(result))


def test_schedule_balances_every_snr_stratum_and_layout_without_rng_state():
    torch.manual_seed(13)
    before = torch.get_rng_state().clone()
    for cycle in range(4):
        values = [schedule(i+1+cycle*20,42,5.,15.) for i in range(20)]
        assert {(b,q) for _,q,b in values} == {(b,q) for b in range(5) for q in ('1','2','3','mixed')}
        assert all(5+2*b <= s < 7+2*b for s,_,b in values)
        assert len({s for s,_,_ in values})==20
    assert torch.equal(before,torch.get_rng_state())
    assert schedule(17,42,5,15)==schedule(17,42,5,15)
    assert schedule(17,42,5,15)!=schedule(17,43,5,15)


def test_full_run_and_resume_keep_frozen_encoder_and_exact_final_weights(shared,tmp_path):
    with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('comparison_baselines.run._execute',side_effect=digital_runner):
        full = tmp_path/'full'; resumed=tmp_path/'resumed'
        args=options(shared,full); result=entry.main(args)
        assert result['step']==6 and result['actual_side_delivery_completed']
        args=options(shared,resumed); entry.main(args,max_updates=4)
        frozen=torch.load(resumed/'checkpoints/latest/codec.pt',weights_only=True)
        original=torch.load(shared/'checkpoints/final/codec.pt',weights_only=True)
        for n,value in original['state_dict'].items():
            if n.startswith(('semantic_encoder.','semantic_decoder.')) or n in ('attr_mean','attr_std'):
                torch.testing.assert_close(value,frozen['state_dict'][n],rtol=0,atol=0)
        args.resume=True; entry.main(args,max_updates=1); entry.main(args)
        a=torch.load(full/'checkpoints/final/codec.pt',weights_only=True)
        b=torch.load(resumed/'checkpoints/final/codec.pt',weights_only=True)
        for n,value in a['state_dict'].items(): torch.testing.assert_close(value,b['state_dict'][n],rtol=0,atol=0)
        for n,value in original['state_dict'].items():
            if n.startswith('semantic_encoder.') or n in ('attr_mean','attr_std'):
                torch.testing.assert_close(value,a['state_dict'][n],rtol=0,atol=0)
        for prefix in ('channel_encoder.','channel_decoder.','semantic_decoder.'):
            assert any(not torch.equal(v,a['state_dict'][n]) for n,v in original['state_dict'].items() if n.startswith(prefix))
        progress=a['training']['progress']
        assert progress['completed']=={'channel_bootstrap':4,'channel_render':2}
        # Transition discarded the channel-only Adam steps.
        assert {int(s['step']) for s in a['training']['optimizer']['state'].values()}=={2}
        initial=json.loads((full/'evaluation/initial/truck/summary.json').read_text())['rows']
        final=json.loads((full/'evaluation/final/truck/summary.json').read_text())['rows']
        assert all(r['condition_snr_db']==10 for r in initial)
        assert all(r['condition_snr_db']==r['snr_db'] for r in final)
        for r in final:
            if r['path']=='clean': continue
            accounting=r['comparison_accounting']
            assert accounting['version']=='digital_side_file_v1'
            assert r['total_channel_uses']==accounting['payload_complex_symbols']+accounting['side_channel_uses']
        actual=json.loads((full/'actual_delivery/truck/summary.json').read_text())
        assert all(r['success_rate']==1 for r in actual['results'])
        for r in actual['trials']:
            match=next(x for x in final if x['path']=='awgn' and x['layout']==r['method'] and x['snr_db']==r['snr_db'])
            assert r['total_channel_uses']==match['total_channel_uses']
            assert r['metrics']['source_psnr']==pytest.approx(match['source_psnr'],abs=1e-5)
        assert not list(full.rglob('received.npy')) and not list(full.rglob('xyz.bin'))
        assert (full/'review.tar.gz').exists()
        # Completed runs are idempotent; no new gradient or digital link updates.
        args=options(shared,full);args.resume=True
        with patch('comparison_baselines.run._execute',side_effect=AssertionError('unexpected rerun')):
            entry.main(args)


def test_cached_joint_has_same_gradients_as_frozen_encoder_forward(shared,tmp_path):
    from gaussian_jscc.model_factory import build_codec
    from gaussian_jscc.channel_normalization_metrics import cache_teacher
    from gaussian_jscc.modular_training import render_step
    from gaussian_jscc.render_objective import MultiViewRenderTask
    with patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
        args=options(shared,tmp_path/'unused')
        saved,metadata,cfg,specs,_,_=entry.inputs(args)
        model=build_codec(cfg);model.load_state_dict(saved['state_dict']);model.set_training_snrs([5,15])
        raw,degree=entry.read_ply(specs[0]['ply'])
        scene=entry.make_scene(specs[0],raw,degree,model,args,tmp_path,old=metadata['scene_records']['truck'])
        cache=cache_teacher(model,scene)
        optimizer,_=optimizer_for(model,'channel_render',args)
        qs=[(ids>=0).long()*2 for ids in scene.group_ids]
        model.train();model.semantic_encoder.eval()
        task=MultiViewRenderTask(scene.cameras[:1],scene.reference,degree,False)
        original=model.forward_channel_batches
        def paired(*a,**kw): kw['paired_noise']=True; return original(*a,**kw)
        torch.manual_seed(9)
        with patch.object(model,'forward_channel_batches',side_effect=paired):
            loss,_=render_step(model,list(zip(scene.groups,qs)),scene.geometry,7.25,'awgn',task,'channel_render')
        grad={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        optimizer.zero_grad(set_to_none=True);torch.manual_seed(9)
        packed=[torch.cat((f,c['y']),-1) for f,c in zip(scene.groups,cache)]
        with patch.object(model.semantic_encoder,'forward',side_effect=AssertionError('cache missed')):
            cached,_=cached_render_step(model,list(zip(packed,qs)),scene.geometry,7.25,task,1.)
        torch.testing.assert_close(loss,cached,atol=1e-7,rtol=1e-6)
        for n,p in model.named_parameters():
            if n in grad: torch.testing.assert_close(grad[n],p.grad,atol=2e-6,rtol=2e-5)


def test_failed_side_delivery_charged_without_render_and_is_resumable(shared,tmp_path):
    from gaussian_jscc.model_factory import build_codec
    from gaussian_jscc.multisnr_evaluation import actual_delivery
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
        args=options(shared,tmp_path/'unused');args.eval_snrs=[10.]
        saved,metadata,cfg,specs,_,_=entry.inputs(args)
        model=build_codec(cfg);model.load_state_dict(saved['state_dict']);model.set_training_snrs([5,15])
        raw,degree=entry.read_ply(specs[0]['ply'])
        scene=entry.make_scene(specs[0],raw,degree,model,args,tmp_path,old=metadata['scene_records']['truck'])
        def runner(*a,**kw): return digital_runner(*a,**kw,fail=True)
        with patch('comparison_baselines.run._execute',side_effect=runner) as execute, \
             patch('gaussian_jscc.multisnr_evaluation.measure',side_effect=AssertionError('failed trial rendered')):
            report=actual_delivery(model,scene,args,tmp_path)
            assert execute.call_count==3
            assert report['results'][0]['success_rate']==0
            assert report['trials'][0]['metrics'] is None
            assert report['trials'][0]['total_channel_uses']>0
            actual_delivery(model,scene,args,tmp_path)
            assert execute.call_count==3


def test_snr_reaches_sender_receiver_and_noiseless_constraint(shared,tmp_path):
    from gaussian_jscc.model_factory import build_codec
    from gaussian_jscc.channel_normalization_metrics import cache_teacher
    from gaussian_jscc.successive_training import loss_for_batch
    from gaussian_jscc.codec import channel
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
        args=options(shared,tmp_path/'unused')
        saved,metadata,cfg,specs,_,_=entry.inputs(args)
        model=build_codec(cfg);model.load_state_dict(saved['state_dict']);model.set_training_snrs([5,15])
        raw,degree=entry.read_ply(specs[0]['ply'])
        scene=entry.make_scene(specs[0],raw,degree,model,args,tmp_path,old=metadata['scene_records']['truck'])
        batch=cache_teacher(model,scene)[0]
        optimizer_for(model,'channel_bootstrap',args)
        model.train();model.semantic_encoder.eval();model.semantic_decoder.eval()
        condition=[]
        h=model.channel_decoder.register_forward_pre_hook(lambda m,a:condition.append(a[3]))
        try:
            with patch('gaussian_jscc.successive_training.channel',wraps=channel) as noise:
                loss,_=loss_for_batch(model,model.semantic_decoder,batch['y'],batch['active'].long()*3,
                    batch['xyz'],scene.geometry,torch.stack([c.camera_center for c in scene.cameras]),
                    args,dict(successive_phase='cooperative',step=3,color_state={}),snr=6.5)
                loss.backward()
                assert noise.call_args.args[1]==6.5
            assert condition==[6.5,6.5]
            assert any('snr' in n and p.grad is not None and p.grad.abs().sum()>0
                       for n,p in model.channel_encoder.named_parameters())
        finally: h.remove()
