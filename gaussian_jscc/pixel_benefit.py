"""Dense, physically scaled distortion changes for a complete intervention group."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from utils.loss_utils import create_window


def pixel_density_labels(images, target, lambda_s):
    """Return q1..q3 minus q0 local MSE and weighted structural error.

    Each input patch contains the five-pixel halo around its 16-pixel core.
    Pixels outside the image must already be zero in both patches.
    """
    single = images.ndim == 4
    if single:
        images, target = images.unsqueeze(0), target.unsqueeze(0)
    if images.ndim != 5 or images.shape[1:] != (4, 3, 26, 26) or target.shape != (images.shape[0], 3, 26, 26):
        raise ValueError('expected images [T,4,3,26,26] and target [T,3,26,26]')
    if not math.isfinite(float(lambda_s)) or lambda_s < 0:
        raise ValueError('lambda_s must be finite and nonnegative')
    x, y = images.double(), target.detach().double()
    delta=x[:,1:]-x[:,:1]
    mse = (2*(x[:,:1]-y[:,None])*delta+delta.square()).mean(dim=2)[...,5:21,5:21]
    x, y = x.clamp(0, 1), y.clamp(0, 1)
    batch = x.shape[0]
    x = x.reshape(batch * 4, 3, 26, 26)
    window = create_window(11, 3).to(device=x.device, dtype=x.dtype)
    def blur(v):
        return F.conv2d(v, window, padding=5, groups=3)
    mx=blur(x); my=blur(y); vy=blur(y*y)-my*my
    def states(v):return v[:,None].expand(-1,4,-1,-1,-1).reshape_as(x)
    my,vy,y=states(my),states(vy),states(y)
    vx=blur(x*x)-mx*mx
    cov = blur(x*y)-mx*my
    structural = ((2*mx*my+.01**2)*(2*cov+.03**2) /
                  ((mx.square()+my.square()+.01**2)*(vx+vy+.03**2)))
    structural = structural.mean(dim=1).reshape(batch, 4, 26, 26)[..., 5:21, 5:21]
    result = torch.stack((mse,
                          float(lambda_s)*(structural[:, :1]-structural[:, 1:])), dim=2)
    return result[0] if single else result


def integrate_density(density, valid, height, width, d_ref):
    """Sum all tiles of one view/group, retaining tier and component axes."""
    if density.ndim == 4:
        density = density.unsqueeze(0)
    if valid.ndim == 3:
        valid = valid.unsqueeze(0)
    if height <= 0 or width <= 0 or not math.isfinite(float(d_ref)) or d_ref <= 0:
        raise ValueError('image dimensions and d_ref must be positive')
    if density.shape[1:] != (3, 2, 16, 16) or valid.shape != (density.shape[0], 1, 16, 16):
        raise ValueError('invalid density or valid shape')
    mask=valid.to(device=density.device)
    return (density.double() * mask[:, None]).sum(dim=(0, 3, 4)) / (height * width * d_ref)


def fit_density_scales(labels, valid, height, width, d_ref, floor=1e-8):
    """RMS per component, shared by all tiers; fit on training groups only."""
    mask = valid[:, None].to(labels)
    count = mask.sum() * labels.shape[1]
    if count <= 0 or floor <= 0:
        raise ValueError('at least one valid pixel and a positive floor are required')
    maps = ((labels.square()*mask).sum(dim=(0, 1, 3, 4))/count).sqrt().clamp_min(floor)
    values = integrate_density(labels, valid, height, width, d_ref).square().mean(dim=0).sqrt().clamp_min(floor)
    return maps.detach(), values.detach()


def density_loss(pred, target, valid, height, width, d_ref, map_scale, value_scale):
    """Complete group map MSE plus integrated value MSE; no pair replication."""
    mask = valid[:, None].to(pred)
    scale = torch.as_tensor(map_scale, device=pred.device, dtype=pred.dtype).reshape(1, 1, 2, 1, 1)
    value_scale = torch.as_tensor(value_scale, device=pred.device, dtype=pred.dtype).reshape(1, 2)
    if (scale <= 0).any() or (value_scale <= 0).any() or mask.sum() <= 0:
        raise ValueError('scales and valid area must be positive')
    map_loss = (((pred-target)/scale).square()*mask).sum() / (mask.sum()*3*2)
    delta = integrate_density(pred-target, valid, height, width, d_ref)
    return map_loss + (delta/value_scale).square().mean()


class PixelBenefitNetwork(nn.Module):
    """Pixel queries read every depth-ordered scene and candidate token."""
    def __init__(self, hidden=32):
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(6, hidden, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), nn.SiLU(),
                                   nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), nn.SiLU())
        self.position = nn.Linear(2, hidden, bias=False)
        self.scene_embed = nn.Linear(13, hidden)
        self.candidate_embed = nn.Linear(12, hidden)
        self.scene_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.candidate_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.query = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden*3, hidden), nn.SiLU(), nn.Linear(hidden, 2, bias=False))
        self.hidden = hidden
        self.register_buffer('output_scale', torch.ones(2))

    def set_output_scale(self, scale):
        scale = torch.as_tensor(scale, device=self.output_scale.device, dtype=self.output_scale.dtype)
        if scale.shape != (2,) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('output scale must contain two finite positive values')
        self.output_scale.copy_(scale)

    def _encode(self, tokens, mask, embedding, gru):
        batch, length, _ = tokens.shape
        mask = mask.bool()
        lengths = mask.sum(dim=1)
        if length == 0:
            return tokens.new_zeros((batch, 1, self.hidden)), mask.new_zeros((batch, 1))
        depth = tokens[:, :, 2].masked_fill(~mask, float('inf'))
        order = torch.argsort(depth, dim=1, stable=True)
        sorted_tokens = tokens.gather(1, order.unsqueeze(-1).expand_as(tokens))
        nonempty = (lengths > 0).nonzero(as_tuple=True)[0]
        output = tokens.new_zeros((batch, length, self.hidden))
        if nonempty.numel():
            embedded = embedding(sorted_tokens.index_select(0, nonempty))
            packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths[nonempty].cpu(),
                                                       batch_first=True, enforce_sorted=False)
            encoded, _ = gru(packed)
            encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=length)
            output = output.index_copy(0, nonempty, encoded)
        valid = torch.arange(length, device=tokens.device)[None, :] < lengths[:, None]
        return output, valid

    def _read(self, query, tokens, mask):
        # A zero dummy token keeps the all-masked sequence finite.
        empty = ~mask.any(dim=1)
        tokens = torch.cat((tokens, tokens.new_zeros((len(tokens), 1, self.hidden))), dim=1)
        mask = torch.cat((mask, empty[:, None]), dim=1)
        scores = torch.bmm(query, tokens.transpose(1, 2)) / math.sqrt(self.hidden)
        weights = torch.softmax(scores.masked_fill(~mask[:, None], float('-inf')), dim=-1)
        return torch.bmm(weights, tokens)

    def encode_scene(self, inputs):
        image, context, cm = inputs['image'], inputs['context'], inputs['context_mask']
        features = self.image(image)[..., 5:21, 5:21]
        axis = torch.arange(16, device=image.device, dtype=image.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        pos = torch.stack((xx, yy), dim=-1).reshape(256, 2) / 16
        batch = image.shape[0]
        pixel = features.permute(0, 2, 3, 1).reshape(batch, 256, self.hidden) + self.position(pos)
        query = self.query(pixel)
        scene, scene_mask = self._encode(context, cm, self.scene_embed, self.scene_gru)
        scene_read = self._read(query, scene, scene_mask)
        return pixel, query, scene_read

    def forward_with_scene(self, inputs, encoded):
        pixel, query, scene_read = encoded
        candidate, km, valid = inputs['candidate'], inputs['candidate_mask'], inputs['valid']
        batch, _, k, _ = candidate.shape
        tokens, token_mask = self._encode(candidate.reshape(batch*4, k, 12), km.reshape(batch*4, k),
                                          self.candidate_embed, self.candidate_gru)
        queries = query[:, None].expand(-1, 4, -1, -1).reshape(batch*4, 256, self.hidden)
        read = self._read(queries, tokens, token_mask).reshape(batch, 4, 256, self.hidden)
        p = pixel[:, None].expand(-1, 4, -1, -1)
        s = scene_read[:, None].expand(-1, 4, -1, -1)
        hidden = self.head[1](self.head[0](torch.cat((p, s, read), dim=-1)))
        delta = self.head[2](hidden[:, 1:] - hidden[:, :1])
        same_mask = (km[:, 1:] == km[:, :1]).all(dim=-1)
        same_values = ((candidate[:, 1:] == candidate[:, :1]).all(dim=-1) | ~km[:, 1:].bool()).all(dim=-1)
        delta = delta.masked_fill((same_mask & same_values)[:, :, None, None], 0.)
        delta = delta.permute(0, 1, 3, 2).reshape(batch, 3, 2, 16, 16)
        return delta * self.output_scale[None, None, :, None, None] * valid[:, None]

    def forward(self, inputs):
        return self.forward_with_scene(inputs, self.encode_scene(inputs))
