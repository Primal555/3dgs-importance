"""Identity-initialized content/SNR gate for the q1 transmitter."""
import torch
from torch import nn


class ContentGate(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.content = nn.Linear(latent_dim, 32)
        self.condition = nn.Linear(1, 32)
        self.readout = nn.Linear(32, latent_dim)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, latent, snr):
        content = torch.nn.functional.gelu(self.content(latent))
        condition = 1 + self.condition(latent.new_tensor([snr / 20])).tanh()
        return .5 + self.readout(content * condition).sigmoid()


class GatedQ1Encoder(nn.Module):
    def __init__(self, mapping, latent_dim):
        super().__init__()
        self.mapping = mapping
        self.gate = ContentGate(latent_dim)
        # When set for the intervention, include the mean in the packet model identity.
        self.register_buffer('mean_gate', None)

    def forward(self, latent, snr):
        weights = self.gate(latent, snr) if self.mean_gate is None else self.mean_gate
        return self.mapping(latent * weights, snr)
