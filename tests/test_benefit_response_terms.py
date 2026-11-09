import copy
import json

import pytest
import torch


def test_response_density_is_exact_mse_decomposition_and_preserves_old_labels():
    from gaussian_jscc.pixel_benefit import pixel_density_labels, response_density_labels
    torch.manual_seed(3)
    images = torch.rand(2,4,3,26,26,dtype=torch.double)
    target = torch.rand(2,3,26,26,dtype=torch.double)
    terms = response_density_labels(images,target,.2)
    old = pixel_density_labels(images,target,.2)
    torch.testing.assert_close(torch.stack((terms[:,:,0]+terms[:,:,1],terms[:,:,2]),2),old)
    assert terms[:,:,1].min() >= 0
    direct = ((images[:,1:]-target[:,None]).square()-(images[:,:1]-target[:,None]).square()).mean(2)[...,5:21,5:21]
    torch.testing.assert_close(terms[:,:,0]+terms[:,:,1],direct)
    images[:,1:] = images[:,:1]
    assert response_density_labels(images,target,.2).abs().max() == 0


def test_exact_response_teacher_reuses_four_candidate_renders(monkeypatch):
    from gaussian_jscc.pixel_local_renderer import TileScene
    from gaussian_jscc.pixel_benefit_features import exact_values
    from gaussian_jscc.benefit_response_features import exact_response_terms
    points = torch.tensor([[7,8,1,.4,0,.4,.7,1,.3,.4,.5,5],
                           [28,4,2,.2,.01,.2,.6,1,.2,.7,.8,7]],dtype=torch.float32)
    scene = TileScene.from_projected(points,17,31,backend='cpu')
    projected = points[None].repeat(4,1,1); projected[0,0,7] = 0
    projected[2,0,8] += .2; projected[3,1,6] *= .7
    target = torch.linspace(0,1,3*17*31).reshape(3,17,31)
    original = TileScene.render_tiles
    calls = []
    def count(self,*a,**kw):
        calls.append(1); return original(self,*a,**kw)
    monkeypatch.setattr(TileScene,'render_tiles',count)
    old = exact_values(scene,torch.arange(2),projected,target,.2,.3)
    old_count = len(calls); calls.clear()
    terms = exact_response_terms(scene,torch.arange(2),projected,target,.2,.3)
    assert len(calls) == old_count
    assert terms.shape == (3,3) and terms[:,1].min() >= 0
    torch.testing.assert_close(torch.stack((terms[:,0]+terms[:,1],terms[:,2]),-1),old,rtol=1e-12,atol=1e-12)


def test_response_network_preserves_initial_value_function_and_energy_is_nonnegative():
    from gaussian_jscc.benefit_footprint_model import FootprintSplitNetwork
    from gaussian_jscc.benefit_response_model import ResponseTermsNetwork, combine_terms
    from gaussian_jscc.spatial_benefit import SpatialBenefitNetwork, CONTEXT_DIM, STATE_DIM, FEATURE_DIM
    torch.manual_seed(4)
    original = SpatialBenefitNetwork(16,0)
    old = FootprintSplitNetwork.from_spatial(original.state_dict(),0)
    new = ResponseTermsNetwork.from_spatial(original.state_dict(),0,energy_scale=.1)
    # Both region branches are zero projected at initialization.
    args=(torch.randn(2,4,FEATURE_DIM),torch.randn(2,5,CONTEXT_DIM),
          torch.randn(2,4,5,STATE_DIM),torch.ones(2,5),torch.ones(2,5,dtype=torch.bool),torch.randn(2,4,2,15,9,9))
    old.eval();new.eval()
    terms = new.forward_terms(*args)
    assert terms[...,1].min() >= 0
    torch.testing.assert_close(new(*args),old(*args).double(),rtol=1e-6,atol=1e-8)
    torch.testing.assert_close(combine_terms(terms),new(*args))
    terms.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in new.energy_head.parameters())


def test_main_loss_weight_stays_one_when_aux_balancer_changes():
    from gaussian_jscc.benefit_response_model import task_weights, assign_gradients
    from gaussian_jscc.benefit_three_task import TaskAccumulator
    p=torch.nn.Parameter(torch.zeros(2,dtype=torch.double))
    names=('value','cross','energy','structure','ranking')
    a=TaskAccumulator([p],[0],names=names)
    a.gradients=[[p.new_tensor([1.,0.])],[p.new_tensor([0.,1.])],
                 [p.new_tensor([0.,1.])],[p.new_tensor([0.,1.])],[p.new_tensor([100.,0.])]]
    requested=task_weights(torch.tensor([.03,.27,2.7],dtype=torch.double))
    applied,report=assign_gradients(a,requested)
    assert applied[0] == 1 and applied[1:4].sum() == pytest.approx(1)
    assert report['applied_rank_gradient_norm'] <= report['numeric_gradient_norm']+1e-12
    assert applied[-1] < requested[-1]


def response_dataset(tmp_path):
    from test_benefit_footprint_controls import dataset
    from gaussian_jscc.spatial_benefit_replay import SpatialReplay
    from gaussian_jscc.full_history_spatial_replay import FullHistorySpatialReplay
    from gaussian_jscc.benefit_response_features import SCHEMA
    old=dataset(tmp_path)
    old_manifest=json.loads((old/'data_manifest.json').read_text())
    original=SpatialReplay(old/'replay',max_groups=1,validation_groups=2)
    data=tmp_path/'response_data';data.mkdir()
    replay=FullHistorySpatialReplay(data/'replay',max_groups=6,validation_groups=2)
    for role in ['training','validation']:
        for entry in old_manifest[role]:
            g=copy.deepcopy(original._read(entry))
            for v in g['views']:
                energy=v['values'][:,0].abs()+.01
                v['response_terms']=torch.stack((v['values'][:,0]-energy,energy,v['values'][:,1]),-1).double()
                v['values']=torch.stack((v['response_terms'][:,0]+energy,v['response_terms'][:,2]),-1)
                v['label_schema']=SCHEMA
            replay.add(g,role=='validation')
    replay.freeze_validation()
    metadata=json.loads((old/'metadata.json').read_text());metadata['version']=SCHEMA
    (data/'metadata.json').write_text(json.dumps(metadata))
    (data/'complete.json').write_text(json.dumps(dict(status='complete')))
    (data/'data_manifest.json').write_text(json.dumps(dict(replay_root=str(replay.root),
        training=replay.training,validation=replay.validation)))
    (data/'initial.pt').write_bytes((old/'initial.pt').read_bytes())
    return data,old


def test_response_training_resume_is_exact_and_primary_weight_is_fixed(tmp_path,monkeypatch):
    import hashlib
    from gaussian_jscc import benefit_response_controls as controls
    data,source=response_dataset(tmp_path)
    hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob('*') if p.is_file()}
    def args(out):
        return controls.parser().parse_args(['--source',str(source),'--out',str(out),'--phase','train',
            '--data',str(data),'--samples','128','--evaluate-every-samples','64','--cpu-threads','1'])
    interrupted=tmp_path/'interrupted'
    original=torch.optim.AdamW.step;calls=0
    def stop(self,*a,**kw):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('interrupted')
        return original(self,*a,**kw)
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW,'step',stop)
        with pytest.raises(RuntimeError,match='interrupted'):controls.main(args(interrupted))
    resumed=args(interrupted);resumed.resume=True
    controls.main(resumed);controls.main(args(tmp_path/'straight'))
    left=torch.load(interrupted/'training/latest.pt',weights_only=True)
    right=torch.load(tmp_path/'straight/training/latest.pt',weights_only=True)
    for k,v in left['network'].items():torch.testing.assert_close(v,right['network'][k],atol=0,rtol=0)
    torch.testing.assert_close(left['balancer']['logits'],right['balancer']['logits'],atol=0,rtol=0)
    assert left['gradient_totals']==right['gradient_totals']
    assert left['sampler']['visits']==right['sampler']['visits']
    rows=[json.loads(s) for s in (interrupted/'training/metrics/optimization.jsonl').read_text().splitlines()]
    assert all(r['requested_task_weights'][0]==r['applied_task_weights'][0]==1. for r in rows)
    assert all(r['gradnorm']['names']==['cross','energy','structure'] for r in rows)
    for r in rows:
        assert r['gradnorm']['raw_weight_sum'] == pytest.approx(3.)
        torch.testing.assert_close(torch.tensor(r['auxiliary_weighted_gradient_norms']),
            torch.tensor(r['gradnorm']['weighted_gradient_norms'])/3)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in hashes.items())
    assert (interrupted/'review.tar.gz').is_file()
    assert controls.main(resumed)['status']=='complete'


def test_response_scales_reject_validation_and_old_schema(tmp_path):
    from gaussian_jscc.benefit_response_data import fit_term_scales,load_data
    data,old=response_dataset(tmp_path)
    _,_,_,groups,_,_,_=load_data(data)
    scales=fit_term_scales(groups['training'])
    assert scales.shape==(3,) and (scales>0).all()
    with pytest.raises(ValueError,match='validation'):fit_term_scales(groups['validation'])
    with pytest.raises(ValueError,match='schema'):load_data(old)
