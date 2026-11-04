from types import SimpleNamespace
import torch
from gaussian_jscc.neural_correction_learning import gap_losses
from gaussian_jscc.neural_correction_history import NetworkAwareEvidence, ContextResidualBank


def test_gap_supervision_only_constrains_observed_pair():
    prediction=torch.tensor([[[1.,2.],[4.,8.],[20.,30.]]],requires_grad=True)
    group=dict(old_tier=1,new_tier=2,delta_values=torch.tensor([3.,6.]),rate_delta=.1)
    losses=gap_losses(prediction,[group],torch.tensor(1.),.25)
    assert losses['total'].item()==0
    changed=prediction.detach().clone();changed[:,1]+=1;changed.requires_grad_()
    value=gap_losses(changed,[group],torch.tensor(1.),.25)['total'].sum()
    value.backward()
    assert changed.grad[:,0].abs().sum()>0 and changed.grad[:,1].abs().sum()>0
    assert changed.grad[:,2].abs().sum()==0


def test_gap_direction_uses_same_tie_band_as_full_state_supervision():
    prediction=torch.tensor([[[0.,0.],[5.,0.],[9.,0.]]],requires_grad=True)
    group=dict(old_tier=1,new_tier=2,delta_values=torch.tensor([.01,0.]),rate_delta=0.)
    result=gap_losses(prediction,[group],1.,.25,tie_fraction=.02)
    assert result['decision'].item()==0
    assert result['observed_gap_regression'].item()>0 and result['full_state_regression'].item()==0


def test_obsolete_variance_cannot_drive_candidate_or_teacher_selection():
    from gaussian_jscc.neural_correction_provider import NetworkCorrectionProvider
    scheduler=NetworkAwareEvidence([torch.arange(4).reshape(1,4)],rounds=1,finite_samples=0)
    scheduler.model_revision=1;scheduler.prediction_revision.fill_(1)
    scheduler.second.fill_(100);scheduler.positive_error.fill_(200);scheduler.error.fill_(200)
    scheduler.keep_error.fill_(.5)
    assert scheduler.positive_variation().min()>0
    provider=object.__new__(NetworkCorrectionProvider)
    provider.scheduler=scheduler;provider.learner=SimpleNamespace(updates=2)
    provider.prepare_step(1)
    assert scheduler.model_revision==2
    assert scheduler.positive_variation().max()==0
    torch.testing.assert_close(scheduler.positive_risk(torch.arange(4)),torch.full((4,),.5,dtype=torch.float64))
    torch.testing.assert_close(scheduler.keep_risk(),torch.full((4,),.5,dtype=torch.float64))
    assert scheduler.second.min()==100


def test_diagnostics_exclude_stale_evidence_without_losing_lifetime_counts():
    scheduler=NetworkAwareEvidence([torch.arange(4).reshape(1,4)],rounds=1,finite_samples=0)
    scheduler.configure_views(4,1);scheduler.model_revision=2
    scheduler.prediction_revision[:]=torch.tensor([2,1,1,1])
    scheduler.weight.fill_(10);scheduler.observations.fill_(20);scheduler.counts.fill_(30)
    scheduler.evidence_step.fill_(1);scheduler.positive_effective_samples.fill_(10)
    report=scheduler.diagnostics(torch.ones(4,dtype=torch.long),step=1)
    assert report['positive_history']['zero_fraction']==.75
    assert report['retained_effective_evidence_mean']==2.5
    assert report['positive_feedback_revision_rows']==1 and report['positive_feedback_revision']==2
    assert report['positive_observations']['mean']==20
    assert report['retained_candidate_evaluations_mean']==30


def test_network_revision_resets_positive_prediction_history_only():
    ids=[torch.arange(4).reshape(1,4)]
    scheduler=NetworkAwareEvidence(ids,rounds=1,finite_samples=0)
    scheduler.configure_views(4,1);scheduler.set_view_window([0])
    scheduler.current_q.fill_(1)
    selected=torch.tensor([0]);rates=torch.zeros(1,3)
    first=torch.tensor([[[1.,0.],[2.,0.],[4.,0.]]])
    scheduler.model_revision=1
    scheduler.accumulate_components(selected,first,rates,1,0.)
    scheduler.keep_mean.fill_(7);scheduler.keep_weight.fill_(9)
    keep=scheduler.keep_mean.clone();count=scheduler.observations.clone()
    scheduler.model_revision=2
    second=first*10
    scheduler.accumulate_components(selected,second,rates,2,0.)
    expected=second.double();expected=expected-expected.mean(1,keepdim=True)
    torch.testing.assert_close(scheduler.mean[0],expected[0])
    assert scheduler.weight[0]==1 and scheduler.observations[0]==count[0]+1
    assert torch.equal(scheduler.keep_mean,keep) and (scheduler.keep_weight==9).all()
    scheduler.accumulate_components(selected,second,rates,3,0.)
    assert scheduler.weight[0]>1
    state=scheduler.state_dict();other=NetworkAwareEvidence(ids,rounds=1,finite_samples=0)
    other.configure_views(4,1);other.load_state_dict(state)
    assert torch.equal(other.prediction_revision,scheduler.prediction_revision)


def test_residual_recomputed_and_context_mismatches_excluded():
    class Replay:
        training=[dict(id='one')]
        def _read(self,entry):return dict(old_tier=1,new_tier=2,delta_values=torch.tensor([3.,0.]),
                                        views=[dict(inputs={})])
    class Learner:
        value=1.
        def predict_group_views(self,groups):
            return [[torch.tensor([[0.,0.],[self.value,0.],[0.,0.]])] for _ in groups]
    bank=ContextResidualBank(1000.)
    learner=Learner();bank.bind(learner,Replay(),dict(lambda_s=1.))
    bank.set_context((0,),{0:b'a'},{0:torch.zeros(2)})
    bank.stage(1,'one',1);bank.update(torch.tensor([1]),torch.zeros(1,2),None,1,(0,))
    pos,mean,_=bank.query(torch.tensor([[0,1,2]]),None,1)
    assert pos.tolist()==[1] and mean[0,0]==2
    learner.value=2.;bank.clear_predictions()
    assert bank.query(torch.tensor([[0,1,2]]),None,1)[1][0,0]==1
    bank.set_context((1,),{0:b'a'},{0:torch.zeros(2)})
    assert len(bank.query(torch.tensor([[0,1,2]]),None,2)[0])==0
    bank.set_context((0,),{0:b'b'},{0:torch.zeros(2)})
    assert len(bank.query(torch.tensor([[0,1,2]]),None,2)[0])==0
    bank.set_context((0,),{0:b'a'},{0:torch.ones(2)*10})
    assert bank.query(torch.tensor([[0,1,2]]),None,2)[1][0,0]<.001
    bank.set_context((1,),{0:b'a'},{0:torch.zeros(2)})
    bank.stage(1,'one',2);bank.update(torch.tensor([1]),torch.zeros(1,2),None,2,(1,))
    assert len(bank.records)==2
    bank.set_context((0,),{0:b'a'},{0:torch.zeros(2)})
    assert bank.query(torch.tensor([[0,1,2]]),None,2)[0].tolist()==[1]
    bank.replay.training=[];bank.prune()
    assert len(bank.keys)==0
