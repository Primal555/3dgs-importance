import copy
import importlib

import pytest
import torch


def entry():
    return importlib.import_module('gaussian_jscc.relation_benefit_learning')


def records(n, first=0):
    return [dict(source=i, step=i, noise_trials=2,
                 values=torch.full((2,3,2), float(i)*1e-6, dtype=torch.float64)) for i in range(first, first+n)]


def test_stable_labels_survive_recent_fifo_and_validation_isolated():
    replay=entry().RelationReplay(capacity=8, anchor_capacity=2, validation_ids=[99], seed=7)
    replay.add(records(20))
    assert [r['source'] for r in replay.anchor] == [0,1]
    assert [r['source'] for r in replay.recent] == list(range(14,20))
    assert {0,1} <= {r['source'] for r in replay.sample(128)}
    with pytest.raises(ValueError, match='validation'):
        replay.add(records(1,99))
    clone=entry().RelationReplay(capacity=8,anchor_capacity=2,validation_ids=[99],seed=7)
    clone.load_state_dict(copy.deepcopy(replay.state_dict()))
    assert [r['source'] for r in replay.sample(30)] == [r['source'] for r in clone.sample(30)]


def test_replay_preserves_background_identity_and_balances_single_context_batches():
    replay=entry().RelationReplay(32,8,validation_ids=[99],seed=3)
    for context in range(4):
        replay.add([dict(g,context=context) for g in records(8)])
    assert replay.coverage()['active_groups']==32
    contexts=[]
    for _ in range(200):
        rows=replay.sample(8)
        assert len({r['context'] for r in rows})==1
        contexts.append(rows[0]['context'])
    assert set(contexts)=={0,1,2,3}
    for context in range(4):assert 25<contexts.count(context)<75
    clone=entry().RelationReplay(32,8,validation_ids=[99],seed=0)
    clone.load_state_dict(copy.deepcopy(replay.state_dict()))
    assert [(g['context'],g['source']) for g in clone.sample(20)]==[
        (g['context'],g['source']) for g in replay.sample(20)]


def test_learning_uses_float64_signed_labels_and_updates_model():
    from test_relation_benefit_model import make_bank,make_model
    bank=make_bank(); model=make_model(bank)
    replay=entry().RelationReplay(capacity=16,anchor_capacity=4,validation_ids=[8],seed=2)
    groups=records(8)
    for g in groups:
        g['values'][:,:,0]=torch.tensor([-3e-5,-1e-5,-2e-5])
        g['values'][:,:,1]=torch.tensor([1e-6,-2e-6,0.])
    replay.add(groups)
    learner=entry().RelationLearner(model,replay,bank,lr=.002)
    learner.fit_scales(groups)
    before=learner.evaluate(groups, torch.tensor([1e-6,2e-6,3e-6]))
    stats=learner.train_updates(30,4,torch.tensor([1e-6,2e-6,3e-6]))
    after=learner.evaluate(groups, torch.tensor([1e-6,2e-6,3e-6]))
    assert stats['updates']==30
    assert stats['grad_norm_max']>=stats['grad_norm_mean']
    assert stats['loss']>=0 and torch.isfinite(model.output_scale).all()
    assert after['component_rmse'][0] < before['component_rmse'][0]
    assert isinstance(groups[0]['values'],torch.Tensor) and groups[0]['values'].dtype==torch.float64
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_losses_have_correct_optimum_and_common_shift_does_not_change_positive_pairs():
    truth=torch.tensor([[[[-2e-5,0.],[-3e-5,1e-6],[-4e-5,2e-6]]]],dtype=torch.float64)
    rates=torch.tensor([1e-6,2e-6,3e-6],dtype=torch.float64)
    equal=entry().relation_losses(truth,truth,rates,torch.tensor([1e-5,1e-6]))
    assert equal['total']==0
    shifted=entry().relation_losses(truth+1e-5,truth,rates,torch.tensor([1e-5,1e-6]))
    assert shifted['common']>0
    assert shifted['relative']<1e-25


def test_rate_proxy_and_decision_use_same_explicit_units():
    proxy=entry().marginal_cost_proxy(dict(payload_complex_symbols=48*10, xyz_bytes=100,
                                         metadata_bytes=20,scale_stream_bytes=40),count=10,rates=(0,8,24,48))
    assert proxy['units']=='complex_channel_uses'
    assert proxy['costs'][0]==0
    assert proxy['costs'][3]-proxy['costs'][2]==24
    values=torch.tensor([[[0.,0.],[0.,0.],[0.,0.]], [[-2.,0.],[-3.,0.],[-4.,0.]]])
    q=entry().choose_tiers(values,torch.tensor([0.,1.,2.,3.]),beta=.1,c_ref=1.)
    assert q.tolist()==[0,3]
