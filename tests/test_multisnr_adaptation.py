"""Source targets, paired prefix gradients, resume and matched MCS accounting."""
import contextlib
import io
import json
from unittest.mock import patch

import pytest
import torch

from comparison_baselines.digital_link import file_channel_cost
from comparison_baselines.side_link_policy import side_link_config
from gaussian_jscc import multisnr_adaptation as adaptation
from gaussian_jscc import successive_multisnr as entry
from gaussian_jscc.channel_normalization_metrics import cache_teacher
from gaussian_jscc.model_factory import build_codec
from gaussian_jscc.multisnr_training import optimizer_for
from test_successive_multisnr import shared, options, digital_runner
from test_successive_joint import cameras
from test_render_first import synthetic_render


def settings(source,out):
    args=options(source,out)
    extra=adaptation.parser().parse_args(['--run',str(source),'--out',str(out),'--digital-python',args.digital_python])
    for name in ('source_aligned','render_every','anchor_every','anchor_weight','side_link_policy',
                 'monitor_scenes','monitor_every','monitor_early','monitor_trials'):
        setattr(args,name,getattr(extra,name))
    args.channel_lr=1e-5;args.monitor_scenes=['truck'];args.monitor_every=2
    args.monitor_early=[1];args.monitor_trials=1;args.anchor_every=2;args.render_every=2
    return args


def setup(source,tmp_path):
    args=settings(source,tmp_path/'unused')
    saved,metadata,cfg,specs,_,_=entry.inputs(args)
    model=build_codec(cfg);model.load_state_dict(saved['state_dict']);model.set_training_snrs([5,15])
    raw,degree=entry.read_ply(specs[0]['ply'])
    scene=entry.make_scene(specs[0],raw,degree,model,args,tmp_path,old=metadata['scene_records']['truck'])
    cache=cache_teacher(model,scene)
    return args,model,scene,cache


def test_link_policy_and_frame_padding():
    low=side_link_config(5,'snr_qpsk_half_v1','cpu')
    high=side_link_config(10,'snr_qpsk_half_v1','cpu')
    assert (low.k,low.n,low.bits_per_symbol)==(4096,8192,2)
    assert (high.k,high.n,high.bits_per_symbol)==(4096,6144,4)
    for size in (0,460,461,508,1000001):
        a,b=file_channel_cost(size,low),file_channel_cost(size,high)
        assert a['codewords']==b['codewords']
        assert a['total_channel_uses']==a['codewords']*4096
        assert b['total_channel_uses']==b['codewords']*1536
    assert side_link_config(7.5,'snr_qpsk_half_v1')==side_link_config(5,'snr_qpsk_half_v1')
    assert side_link_config(5).bits_per_symbol==4


def test_source_targets_and_all_prefixes_receive_gradients(shared,tmp_path):
    with patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras):
        args,model,scene,cache=setup(shared,tmp_path)
        optimizer_for(model,'channel_bootstrap',args)
        model.train();model.semantic_encoder.eval();model.semantic_decoder.eval()
        batch=cache[0];f=scene.groups[0];q=batch['active'].long()*3
        noise=torch.randn(*q.shape,2*model.cfg.rates[-1])*.1
        main,final=adaptation.paired_decode(model,f,batch['y'],q,6.5,noise,with_main=True)
        ordinary=adaptation.paired_decode(model,f,batch['y'],q,6.5,noise)
        for a,b in zip(final,ordinary): torch.testing.assert_close(a,b,atol=0,rtol=0)
        from gaussian_jscc.local_response import local_response_loss
        targets=[]
        def observe(pred,target,*a,**kw):
            targets.append(target.detach().clone());return local_response_loss(pred,target,*a,**kw)
        with patch('gaussian_jscc.multisnr_adaptation.local_response_loss',side_effect=observe), \
             patch.object(model.semantic_encoder,'forward',side_effect=AssertionError('cache missed')):
            loss,stats=adaptation.source_batch_loss(model,f,batch,scene.geometry,
                torch.stack([c.camera_center for c in scene.cameras]),args,
                dict(step=0,color_state={}),6.5,'3')
            loss.backward()
        assert len(targets)==6
        for target in targets: torch.testing.assert_close(target,f[batch['active']],atol=0,rtol=0)
        for module in (model.channel_encoder,model.channel_decoder):
            for stage in module.stages:
                assert sum(float(p.grad.abs().sum()) for p in stage.parameters() if p.grad is not None)>0
        assert all(p.grad is None for p in model.semantic_encoder.parameters())
        assert all(p.grad is None for p in model.semantic_decoder.parameters())
        assert len(stats['prefixes'])==3


def test_v2_resume_and_quality_cost_records(shared,tmp_path):
    with contextlib.redirect_stdout(io.StringIO()), \
         patch('gaussian_jscc.rendering.render',side_effect=synthetic_render), \
         patch('gaussian_jscc.rendering.load_cameras',side_effect=cameras), \
         patch('comparison_baselines.run._execute',side_effect=digital_runner), \
         patch('gaussian_jscc.multisnr_plots.write_figures'):
        full=tmp_path/'full';split=tmp_path/'split'
        entry.main(settings(shared,full))
        args=settings(shared,split);entry.main(args,max_updates=3)
        args.resume=True;entry.main(args)
        a=torch.load(full/'checkpoints/final/codec.pt',weights_only=True)
        b=torch.load(split/'checkpoints/final/codec.pt',weights_only=True)
        source=torch.load(shared/'checkpoints/final/codec.pt',weights_only=True)
        for name,value in a['state_dict'].items():
            torch.testing.assert_close(value,b['state_dict'][name],atol=0,rtol=0)
            if name.startswith('semantic_encoder.') or name in ('attr_mean','attr_std'):
                torch.testing.assert_close(value,source['state_dict'][name],atol=0,rtol=0)
        rows=json.loads((full/'evaluation/final/truck/summary.json').read_text())['rows']
        actual=json.loads((full/'actual_delivery/truck/summary.json').read_text())['trials']
        for row in actual:
            ideal=next(x for x in rows if x['path']=='awgn' and x['layout']==row['method'] and x['snr_db']==row['snr_db'])
            assert ideal['comparison_accounting']['link']==row['comparison_accounting']['link']
            assert ideal['total_channel_uses']==row['total_channel_uses']
            assert ideal['source_psnr']==pytest.approx(row['metrics']['source_psnr'],abs=1e-5)
        monitor=json.loads((full/'evaluation/monitor_000006/truck/summary.json').read_text())['rows']
        assert all(r['scope']=='validation' for r in monitor)
        assert {v['view'] for r in monitor for v in r['views']}.isdisjoint(
            {v['view'] for r in rows for v in r['views']})
        loss_rows=[json.loads(x) for x in (full/'metrics/loss.jsonl').read_text().splitlines()]
        assert loss_rows[0]['render_update']
        assert any(r.get('anchor_weight',0)>0 for r in loss_rows)
        # Final replayed evaluations reuse existing files on repeated invocation.
        args=settings(shared,full);args.resume=True
        with patch('comparison_baselines.run._execute',side_effect=AssertionError('unexpected digital rerun')):
            entry.main(args)


def test_plotter_preserves_scope_and_reports_success_cost(tmp_path):
    from gaussian_jscc.modular_experiment import json_write
    from gaussian_jscc.multisnr_plots import write_figures
    for label,scope in (('initial','test'),('final','test'),('monitor_000000','validation'),('monitor_000009','validation')):
        rows=[dict(path='awgn',scene='truck',snr_db=snr,layout=str(q),scope=scope,
                   source_psnr=25.+q,photo_psnr=23.+q,source_ssim=.9,photo_ssim=.8,
                   source_lpips=.1,photo_lpips=.2,total_channel_uses=10000,
                   diagnostics=dict(latent_mse=.01)) for snr in (5,10) for q in (1,2,3)]
        json_write(tmp_path/'evaluation'/label/'truck/summary.json',dict(rows=rows))
        if label.startswith('monitor'):
            json_write(tmp_path/'evaluation'/label/'progress.json',dict(step=int(label[-6:]),phase='channel_bootstrap',
                optimization_seconds=float(int(label[-6:])*3),scope='validation',selection=False))
    json_write(tmp_path/'actual_delivery/truck/summary.json',dict(results=[dict(method='2',snr_db=5,
        success_rate=0,total_channel_uses=12345)]))
    write_figures(tmp_path)
    for name in ('truck_training_time','truck_training_updates','truck_snr_source','truck_snr_photo','truck_actual_delivery'):
        assert (tmp_path/'figures'/f'{name}.pdf').stat().st_size>1000
    import csv
    with (tmp_path/'figures/training_curves.csv').open() as f:
        assert {r['scope'] for r in csv.DictReader(f)}=={'validation'}
