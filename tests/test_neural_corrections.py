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
