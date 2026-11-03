import copy
from types import SimpleNamespace
import pytest
import torch
from gaussian_jscc.neural_dense_keep import DenseKeepNetwork, NeuralKeepTable, smoothed_gate, mixed_keep_label


def test_chunk_vjp_matches_direct_gate_and_expected_rate_gradient():
    torch.set_num_threads(1);torch.manual_seed(11)
    a=DenseKeepNetwork(69,8);b=copy.deepcopy(a)
    x=torch.randn(17,69);old=torch.linspace(.15,.95,17,dtype=torch.float64)
    noise=-torch.empty(17,2,dtype=torch.float64).exponential_().log()
    image_gradient=torch.randn(17,dtype=torch.float64);unit=torch.linspace(8.,60.,17,dtype=torch.float64)
    p,gate=smoothed_gate(a(x),old,noise,.1)
    direct=(image_gradient*gate).sum()/.03+.1*(unit*p).sum()/1000
    direct.backward()
    for start in range(0,17,5):
        sl=slice(start,start+5)
        p,g=smoothed_gate(b(x[sl]),old[sl],noise[sl],.1)
        ((image_gradient[sl]*g).sum()/.03+.1*(unit[sl]*p).sum()/1000).backward()
    for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa.grad,pb.grad,rtol=2e-5,atol=1e-6)
    assert torch.isfinite(a.output.weight.grad).all() and a.output.weight.grad.norm()>0


def test_mirror_updates_positive_tiers_without_overwriting_neural_keep():
    table=NeuralKeepTable(12)
    before=table.keep_logits.clone();tiers=table.logits.clone()
    table.update(torch.randn(12,2),torch.tensor([2,5]),torch.tensor([[.01,.02,-.01],[.03,-.02,.01]]),.1,.4,12)
    torch.testing.assert_close(table.keep_logits,before,rtol=0,atol=0)
    assert not torch.equal(table.logits,tiers)


def test_keep_label_uses_conditional_mixture_and_q0_anchor():
    deltas=torch.tensor([[.1,.2],[.2,.3],[.4,.5],[.8,.9]],dtype=torch.float64)
    p=torch.tensor([.7,.2,.1]);rates=torch.tensor([8.,24.,48.])
    label=mixed_keep_label(deltas,p,rates,.01,.003,.1,1000)
    expected=(((deltas[1:,0]-deltas[0,0])+.01*(deltas[1:,1]-deltas[0,1]))/.003+.1*rates/1000)*p
    assert float(label)==pytest.approx(float(expected.sum()))


def test_policy_sampling_matches_table_and_records_original_row_noise():
    from gaussian_jscc.neural_dense_keep import DenseKeepPolicy
    from test_successive_allocation import small_successive
    raw,geometry,_,model=small_successive();order=torch.arange(15,-1,-1)
    scene=SimpleNamespace(raw=raw[order],order=order,geometry=geometry)
    table=NeuralKeepTable(16)
    args=SimpleNamespace(keep_hidden=8,keep_network_lr=1e-4,keep_blend=.1,keep_chunk=5,
                         seed=42,snr=10.,beta=.1,keep_calibration_weight=.1)
    positions=torch.zeros(16,4,dtype=torch.long)
    policy=DenseKeepPolicy(model,scene,table,torch.zeros(16,24),args,.01,positions,torch.zeros(16,dtype=torch.bool))
    policy.prepare(dict(xyz_per_retained=10.))
    ids=order.reshape(2,8)
    torch.manual_seed(47);expected=table.sample(ids,10.)[0]
    torch.manual_seed(47);observed=policy.sample(ids,10.)[0]
    assert torch.equal(observed,expected)
    with torch.no_grad():
        logits=policy.network(policy.inputs)
        probability,gate=smoothed_gate(logits,policy.previous,policy.noise,args.keep_blend)
    torch.testing.assert_close(probability,table.keep_logits.softmax(-1)[:,1])
    assert torch.equal(gate[ids]>.5,observed>0)
