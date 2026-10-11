"""Experimental attribute-only progressive decoding; no source data or XYZ anchors.

Each pass sees only its received prefix. The shared local-attention trunk is
reused; later passes additionally read the preceding normalized attributes.
Corrections are zero-initialized, additive in parameter space, and gated by
actual hard received tiers. Physical constraints remain in data.to_raw.
"""
import hashlib
import json
import torch
from torch import nn


class AttributeRefinement(nn.Module):
    def __init__(self, cfg, sizes):
        super().__init__()
        self.previous = nn.Linear(cfg.attr_dim,cfg.hidden)
        self.heads = nn.ModuleDict({name:nn.Linear(cfg.hidden,size) for name,size in sizes.items() if name!='xyz'})
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, hidden):
        return torch.cat([head(hidden) for head in self.heads.values()],-1)


def decode_stages(core, received, q, snr):
    """Return cumulative predictions and two actually applied attribute deltas.

    A low-tier point can supply its available features as local context, but
    never receives a correction for an unreceived enhancement layer. Clipping
    EVERY neighbor's tier to the current stage prevents future-symbol leakage.
    """
    retained = q>0
    base_q = retained.long()
    hidden = core._decode_hidden(received,base_q,snr)
    result = torch.cat([head(hidden) for head in core.heads.values()],-1)*retained[...,None]
    stages, deltas = [result], []
    for tier, refinement in enumerate(core.refinements,start=2):
        selected = q>=tier
        if selected.any():
            hidden = core._decode_hidden(received,q.clamp_max(tier),snr,
                                         previous=refinement.previous(result[...,3:]))
            delta = refinement(hidden)*selected[...,None]
            result = torch.cat((result[...,:3],result[...,3:]+delta),-1)
        else:
            delta = torch.zeros_like(result[...,3:])
        stages.append(result)
        deltas.append(delta)
    return stages, deltas


def common_initializer_id(model):
    """Audit common codec weights/statistics; refinement-only weights are excluded.

    Full model IDs must differ because the experimental interface/weights differ.
    This signature is for paired initialization only, NOT packet model identity.
    """
    config = model.cfg.to_dict().copy()
    config.pop('decoder_mode',None)
    digest = hashlib.sha256(json.dumps(config,sort_keys=True).encode())
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith('learned.refinements.'):
            digest.update(name.encode())
            digest.update(str((tuple(value.shape),str(value.dtype))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
