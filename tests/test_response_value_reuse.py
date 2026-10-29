"""Exact paired-noise reference for the frozen response allocator's fast path."""
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from gaussian_jscc.budgeted_allocation_kernel import rng_state,set_rng,raw_batches
from gaussian_jscc.optimization import preserved_rng


@torch.no_grad()
def reference_candidates(model,f,q,columns,actual,state,geometry,snr,kind,**unused):
    rows=torch.arange(len(q),device=q.device);active=columns>=0
    for tier in range(4):
        alternative=q.clone();alternative[rows[active],columns[active]]=tier
        with preserved_rng(f.device):
            set_rng(state,f.device)
            pred=model.forward_tier_batches(f,f[...,:3],F.one_hot(alternative,4).to(f),
                snr,kind,paired_noise=True)[0]
        pred=torch.where((q==alternative).all(-1)[:,None,None],actual,pred)
        if unused.get('stats') is not None:
            stats=unused['stats'];stats['candidate_codec_calls']=stats.get('candidate_codec_calls',0)+1
        yield alternative,raw_batches(pred,geometry,model)


def response_model(context):
    from test_successive_allocation import small_successive
    from gaussian_jscc.model_factory import build_codec
    _,geometry,f,model=small_successive()
    if context=='both':
        model.cfg.context_variant='both'
        model=build_codec(model.cfg);model.set_scene_geometry(geometry)
        model.channel_decoder.neighbor_enabled.fill_(True)
    return geometry,f,model.eval().requires_grad_(False)


@pytest.mark.parametrize('context',['baseline','both'])
@pytest.mark.parametrize('kind',['none','awgn','rayleigh'])
@pytest.mark.parametrize('mode',['mixed','zero','highest','inactive','empty','padded'])
@torch.no_grad()
def test_candidate_reuse_matches_full_block_reference_and_reduces_calls(context,kind,mode):
    from gaussian_jscc.response_value_window import decode_candidates
    torch.set_num_threads(1);torch.manual_seed(98)
    geometry,f,model=response_model(context)
    q=torch.tensor([[0,1,2,3,1,2,3,0],[1,2,3,1,2,3,0,0]])
    columns=torch.tensor([0,2])
    if mode=='zero':q.zero_()
    if mode=='highest':q.fill_(3)
    if mode=='inactive':columns[1]=-1
    if mode=='empty':columns.fill_(-1)
    if mode=='padded':
        f=f.clone();f[1].zero_();q[1].zero_();columns[1]=-1
    state=rng_state(f.device)
    semantic=model.semantic_encoder(f,f[...,:3],q>0)
    actual=model.forward_tier_batches(f,f[...,:3],F.one_hot(q,4).to(f),10,kind,
        paired_noise=True,_semantic_y=semantic)[0]
    caller=rng_state(f.device)
    expected=list(reference_candidates(model,f,q,columns,actual,state,geometry,10,kind))
    stats={}
    with patch.object(model,'forward_tier_batches',wraps=model.forward_tier_batches) as forward, \
         patch.object(model.semantic_encoder,'forward',wraps=model.semantic_encoder.forward) as encode:
        observed=list(decode_candidates(model,f,q,columns,actual,state,geometry,10,kind,
            semantic_y=semantic,stats=stats))
    assert forward.call_count==(0 if mode=='empty' else 3)
    # One refresh per changed presence mask; positive-tier masks share semantics.
    assert encode.call_count==dict(mixed=2,zero=1,highest=1,inactive=1,empty=0,padded=1)[mode]
    assert stats['candidate_codec_calls']==forward.call_count
    assert stats['candidate_semantic_calls']==encode.call_count
    assert torch.equal(caller,rng_state(f.device))
    for (target,want),(got,raw) in zip(expected,observed):
        torch.testing.assert_close(got,target,atol=0,rtol=0)
        torch.testing.assert_close(raw,want,atol=0,rtol=0)
    # No shortened batch or changed paired-noise slot layout.
    assert all(call.args[0].shape==f.shape for call in forward.call_args_list)
