import math
import pytest
import torch
from torch.nn import functional as F

from gaussian_jscc.modular_config import ModularCodecConfig
from gaussian_jscc.model_factory import build_codec


def small_codec():
    torch.manual_seed(9)
    cfg=ModularCodecConfig(channel_variant='successive',channel_prefix_mode='full_prefix',
        channel_power_mode='block',channel_neighbor_mode='plain',latent_dim=8,channel_hidden=16,
        hidden=16,block_size=8,depth=2,grid_dim=4,levels=(2,),attention_heads=4,decoder_window=4)
    return build_codec(cfg).eval()


@torch.no_grad()
def test_selected_rows_retain_identical_awgn_slots_and_mask_changes():
    model=small_codec()
    f=torch.rand(4,8,3+len(model.attr_mean))
    q=torch.randint(1,4,(4,8)); q[1,2]=0
    torch.manual_seed(43); state=torch.get_rng_state()
    whole=model.forward_tier_batches(f,f[...,:3],F.one_hot(q,4).to(f),10.,paired_noise=True)[0]
    torch.set_rng_state(state)
    noise=torch.randn(4,8,2*model.cfg.rates[-1])*math.sqrt(.1/2)
    rows=torch.tensor([3,1,1])
    selected=model.forward_tier_batches(f[rows],f[rows,:,:3],F.one_hot(q[rows],4).to(f),10.,
                                       paired_noise=True,_channel_noise=noise[rows])[0]
    torch.testing.assert_close(selected,whole[rows],rtol=3e-5,atol=1e-6)
    changed=q.clone(); changed[1,2]=3; changed[3,4]=0
    torch.set_rng_state(state)
    oracle=model.forward_tier_batches(f,f[...,:3],F.one_hot(changed,4).to(f),10.,paired_noise=True)[0]
    actual=model.forward_tier_batches(f[rows],f[rows,:,:3],F.one_hot(changed[rows],4).to(f),10.,
                                     paired_noise=True,_channel_noise=noise[rows])[0]
    torch.testing.assert_close(actual,oracle[rows],rtol=3e-5,atol=1e-6)


@torch.no_grad()
def test_explicit_noise_is_validated_and_does_not_consume_rng():
    model=small_codec(); f=torch.rand(1,8,3+len(model.attr_mean)); choices=F.one_hot(torch.ones(1,8,dtype=torch.long),4).to(f)
    noise=torch.zeros(1,8,2*model.cfg.rates[-1]); state=torch.get_rng_state()
    model.forward_tier_batches(f,f[...,:3],choices,10.,paired_noise=True,_channel_noise=noise)
    assert torch.equal(torch.get_rng_state(),state)
    for bad in (noise[...,:2],noise+float('nan')):
        with pytest.raises(ValueError): model.forward_tier_batches(f,f[...,:3],choices,10.,paired_noise=True,_channel_noise=bad)
    with pytest.raises(ValueError): model.forward_tier_batches(f,f[...,:3],choices,10.,_channel_noise=noise)
