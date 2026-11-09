import hashlib
import json

import pytest
import torch

from gaussian_jscc import benefit_rank_gradnorm_controls as controls


def args_for(source,out,steps=2):
    return controls.parser().parse_args(['--source',str(source),'--out',str(out),
        '--updates',str(steps),'--evaluate-every','1','--cpu-threads','1'])


def test_real_cpu_training_updates_both_supervisions_and_preserves_sources(tmp_path):
    from test_benefit_isolated_positive_controls import prepared
    source=prepared(tmp_path)
    hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    out=tmp_path/'rank';result=controls.main(args_for(source,out))
    assert result['status']=='complete' and result['anchor_unchanged']
    assert result['initial_max_prediction_difference']<1e-6
    assert result['training_coverage']['draws']==4
    state=torch.load(out/'latest.pt',weights_only=True)
    assert state['balancer']['steps']==2
    assert not torch.equal(state['balancer']['logits'],torch.zeros_like(state['balancer']['logits']))
    assert result['after']['validation']['all']['records']==2
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in hashes.items())
    assert (out/'review.tar.gz').is_file() and (out/'final.pt').is_file()
    settings=json.loads((out/'settings.json').read_text())
    assert settings['temperature_fit_scope']=='training records only'
    assert len(settings['initial_reference_losses'])==len(settings['tasks'])
    assert settings['regression_factor']==4


def test_resume_restores_network_weights_task_weights_and_sampler(tmp_path,monkeypatch):
    from test_benefit_isolated_positive_controls import prepared
    source=prepared(tmp_path);args=args_for(source,tmp_path/'interrupted',3)
    original=torch.optim.AdamW.step;calls=0
    def interrupt(self,*a,**k):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('interrupted')
        return original(self,*a,**k)
    with monkeypatch.context() as m:
        m.setattr(torch.optim.AdamW,'step',interrupt)
        with pytest.raises(RuntimeError,match='interrupted'):controls.main(args)
    args.resume=True;controls.main(args)
    args.out=str(tmp_path/'straight');args.resume=False;controls.main(args)
    a=torch.load(tmp_path/'interrupted/latest.pt',weights_only=True)
    b=torch.load(tmp_path/'straight/latest.pt',weights_only=True)
    assert all(torch.equal(a['learner']['network'][k],b['learner']['network'][k]) for k in a['learner']['network'])
    assert torch.equal(a['balancer']['logits'],b['balancer']['logits'])
    assert a['sampler']['visits']==b['sampler']['visits']
    assert a['sampler']['orders']==b['sampler']['orders']
    args.resume=True;controls.main(args)
    (tmp_path/'straight/review.tar.gz').unlink()
    assert controls.main(args)['status']=='complete'


def test_training_configuration_cannot_silently_change_on_resume(tmp_path):
    from test_benefit_isolated_positive_controls import prepared
    source=prepared(tmp_path);args=args_for(source,tmp_path/'rank')
    controls.main(args)
    args.resume=True;args.gradnorm_alpha=0.
    with pytest.raises(ValueError,match='resume settings'):controls.main(args)


def test_single_tail_sample_preserves_source_importance():
    losses=dict(magnitude=torch.tensor([3.,5.]),ranking=torch.tensor([2.,7.]))
    # Fixed full-training stratum means prevent a one-record stratum batch
    # from cancelling its original importance weight.
    result=controls.aggregate_tasks(losses,[0,1],torch.tensor([2.,4.]),[0,1],{0:1.,1:2.})
    torch.testing.assert_close(result,torch.tensor([6.,4.,10.,14.]))
