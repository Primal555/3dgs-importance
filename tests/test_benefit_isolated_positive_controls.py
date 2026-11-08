import hashlib
import json

import pytest
import torch

from gaussian_jscc import benefit_isolated_positive_controls as controls
from gaussian_jscc.benefit_isolated_positive_report import compare_choices
from gaussian_jscc.benefit_feedback_audit import audit_feedback, current_exact_summary


def prepared(tmp_path):
    from test_benefit_footprint_controls import dataset
    from gaussian_jscc import benefit_footprint_controls as previous
    data = dataset(tmp_path)
    source = tmp_path/'footprint'; source.mkdir()
    data.rename(source/'data')
    manifest = json.loads((source/'data/data_manifest.json').read_text())
    manifest['replay_root'] = str(source/'data/replay')
    (source/'data/data_manifest.json').write_text(json.dumps(manifest))
    previous.main(previous.parser().parse_args(['--phase','train','--data',str(source/'data'),
        '--out',str(source/'training'),'--updates','2','--evaluate-every','1','--cpu-threads','1']))
    return source


def test_isolated_training_reuses_references_and_source_stays_readonly(tmp_path):
    source = prepared(tmp_path)
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    args = controls.parser().parse_args(['--source',str(source),'--out',str(tmp_path/'new'),
        '--updates','2','--evaluate-every','1','--cpu-threads','1'])
    result = controls.main(args)
    assert result['status']=='complete' and result['anchor_unchanged']
    assert result['initial_max_prediction_difference'] < 1e-6
    assert result['training_coverage']['draws']==4
    assert result['after']['validation']['all']['records']==2
    assert result['after']['training_probe']['all']['records']==6
    assert result['after']['against_original']['records']==2
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in hashes.items())
    assert (tmp_path/'new/review.tar.gz').is_file()
    assert (tmp_path/'new/final.pt').is_file()
    args.resume=True
    assert controls.main(args)['status']=='complete'


def test_reference_label_mismatch_rejected_before_training(tmp_path):
    source = prepared(tmp_path)
    p=source/'training/metrics/final_predictions.jsonl'
    rows=[json.loads(x) for x in p.read_text().splitlines()]
    rows[0]['rates'][0]+=1
    p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    args=controls.parser().parse_args(['--source',str(source),'--out',str(tmp_path/'new'),'--updates','2'])
    with pytest.raises(ValueError,match='reference.*(rates|labels|identity)'):
        controls.main(args)


def test_resume_matches_uninterrupted_final_weights(tmp_path,monkeypatch):
    source=prepared(tmp_path)
    args=controls.parser().parse_args(['--source',str(source),'--out',str(tmp_path/'interrupted'),
        '--updates','3','--evaluate-every','1','--cpu-threads','1'])
    original=torch.optim.AdamW.step
    calls=0
    def interrupted(self,*a,**k):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('interrupted')
        return original(self,*a,**k)
    with monkeypatch.context() as m:
        m.setattr(torch.optim.AdamW,'step',interrupted)
        with pytest.raises(RuntimeError,match='interrupted'):controls.main(args)
    args.resume=True;controls.main(args)
    args.out=str(tmp_path/'straight');args.resume=False;controls.main(args)
    a=torch.load(tmp_path/'interrupted/final.pt',weights_only=True)['learner']['network']
    b=torch.load(tmp_path/'straight/final.pt',weights_only=True)['learner']['network']
    assert all(torch.equal(a[k],b[k]) for k in a)


def test_choice_comparison_exposes_single_tail_gain_and_ordinary_harm():
    groups=[dict(source=i,step=1,rates=torch.zeros(3),views=[dict(values=torch.tensor([[0.,0.],[v,0.],[v*2,0.]]))]) for i,v in enumerate([100.,1.,1.])]
    old=[torch.tensor([[[1.,0.],[0.,0.],[2.,0.]]])]+[torch.tensor([[[0.,0.],[1.,0.],[2.,0.]]])]*2
    new=[torch.tensor([[[0.,0.],[1.,0.],[2.,0.]]])]+[torch.tensor([[[1.,0.],[0.,0.],[2.,0.]]])]*2
    r=compare_choices(groups,old,new)
    assert r['beneficial_changes']==1 and r['harmful_changes']==2
    assert r['excluding_largest_improvement']['reference_regret']==0
    assert r['excluding_largest_improvement']['candidate_regret']==1


def test_feedback_audit_keeps_missing_old_telemetry_distinct_from_zero(tmp_path):
    metrics=tmp_path/'allocation/metrics';metrics.mkdir(parents=True)
    row=dict(feedback_kind='positive',first_order_delta_rd=-1.,calibrated_delta_rd=-1.,
        measured_delta_rd=2.,applied_to_training=True)
    (metrics/'audit.jsonl').write_text(json.dumps(row)+'\n')
    (metrics/'loss.jsonl').write_text(json.dumps(dict(step=1,positive_correction_reused_actions=0,
        finite_positive_corrections=1,neural_exact_training_groups=1))+'\n')
    r=audit_feedback(tmp_path)
    assert r['historical_reuse']['actions']==0
    assert r['current_exact_replacement']['logged_actions'] is None
    assert r['observed_action_training_labels']==1
    components=torch.zeros(1,3,2);components[0,2,0]=.3
    costs=torch.tensor([[1.,1.,1.5]])
    rates=torch.tensor([[0.,.1,.2]])
    exact=torch.tensor([[False,False,True]])
    check=current_exact_summary(costs,components,rates,torch.tensor([1]),exact,.1)
    assert check['positive_current_exact_replacements']==1
    assert check['positive_current_exact_gap_max_abs_error']<1e-7
