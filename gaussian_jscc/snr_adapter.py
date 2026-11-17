"""Zero-initialized SNR adjustments anchored exactly at the source 10 dB condition."""
import math

import torch
from torch import nn


class AnchoredSNRAdapter(nn.Module):
    def __init__(self, hidden, outputs):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, outputs))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, snr):
        if not math.isfinite(float(snr)):
            raise ValueError('finite SNR required')
        reference = self.network[0].weight
        # One evaluation for the whole scene block: the condition is shared.
        values = reference.new_tensor([[float(snr)/20.], [.5]])
        output = self.network(values)
        return output[0]-output[1]


def adapter_parameter(name):
    return '.snr_adapter.' in name


def configure_optimizer(model, lr):
    if not model.cfg.channel_snr_adapter_hidden or not math.isfinite(lr) or lr <= 0:
        raise ValueError('enabled adapters and positive finite LR required')
    model.requires_grad_(False)
    params = []
    for name, p in model.named_parameters():
        p.grad = None
        if adapter_parameter(name):
            p.requires_grad_(True)
            params.append(p)
    if not params:
        raise ValueError('missing SNR adapter parameters')
    return torch.optim.Adam(params, lr=lr)
