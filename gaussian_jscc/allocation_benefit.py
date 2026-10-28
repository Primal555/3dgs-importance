"""Supervised full action distortion, with an independently maskable tangent input."""
import torch
from torch import nn
from torch.nn import functional as F


FEATURE_KEYS = ('point', 'action', 'views', 'aux', 'available')


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, outputs), nn.GELU(), nn.Linear(outputs, outputs))


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mapping = mlp(width, width)

    def forward(self, x):
        return x + self.mapping(x)


class BenefitNetwork(nn.Module):
    def __init__(self, point_dim, action_dim, view_dim=12, aux_dim=15, *,
                 mse_only=False, continuous=None):
        super().__init__()
        self.dimensions = dict(point=point_dim, action=action_dim, views=view_dim, aux=aux_dim)
        self.mse_only = bool(mse_only)
        self.point_encoder, self.action_encoder = mlp(point_dim, 64), mlp(action_dim, 64)
        self.view_encoder, self.aux_encoder = mlp(view_dim, 16), mlp(aux_dim, 16)
        self.fusion = nn.Sequential(nn.Linear(177, 128), nn.GELU(), Residual(128), Residual(128))
        self.heads = nn.ModuleList([nn.Linear(128, 2), nn.Linear(128, 2)])
        for name, width in self.dimensions.items():
            self.register_buffer(name+'_mean', torch.zeros(width))
            self.register_buffer(name+'_std', torch.ones(width))
            mask = torch.ones(width, dtype=torch.bool) if continuous is None else torch.as_tensor(continuous[name]).bool()
            if mask.shape != (width,):
                raise ValueError('invalid feature normalization mask')
            self.register_buffer(name+'_continuous', mask)
        self.register_buffer('scales', torch.full((2,), 1e-6))
        self.register_buffer('ready', torch.tensor(False))

    def _normalize(self, name, value):
        return (value-getattr(self, name+'_mean')) / getattr(self, name+'_std')

    def forward(self, features, branch, *, masked=False):
        x = features
        if set(x) != set(FEATURE_KEYS):
            raise ValueError('benefit feature protocol mismatch')
        point = self.point_encoder(self._normalize('point', x['point']))
        action = self.action_encoder(self._normalize('action', x['action']))
        views = self.view_encoder(self._normalize('views', x['views']))
        views = torch.cat((views.mean(-2), views.amax(-2)), -1)
        available = torch.zeros_like(x['available']) if masked else x['available']
        # Mask both the input and encoded output, including encoder biases.
        aux = self._normalize('aux', x['aux']) * available
        aux = self.aux_encoder(aux) * available
        hidden = self.fusion(torch.cat((point, action, views, aux, available), -1))
        if isinstance(branch, str):
            if branch not in ('positive', 'keep'):
                raise ValueError('unknown benefit branch')
            index = int(branch == 'keep')
            result = self.heads[index](hidden) * self.scales[index]
        else:
            index = branch.to(device=hidden.device, dtype=torch.long)
            both = torch.stack([head(hidden) for head in self.heads], -2)
            result = both[torch.arange(len(hidden), device=hidden.device), index] * self.scales[index, None]
        if self.mse_only:
            result = result * result.new_tensor([1., 0.])
        return result

    @torch.no_grad()
    def fit(self, records):
        if not records or any(r['validation'] for r in records):
            raise ValueError('normalization requires nonempty training records, excludes validation')
        if bool(self.ready):
            raise ValueError('benefit normalization is already fixed')
        for name in self.dimensions:
            values = torch.cat([r['features'][name].reshape(-1, self.dimensions[name]).double().cpu() for r in records])
            if not torch.isfinite(values).all():
                raise ValueError('nonfinite benefit feature')
            mean, std = values.mean(0), values.std(0, correction=0)
            std = torch.where(std>1e-12,std,torch.ones_like(std))
            mask = getattr(self, name+'_continuous').cpu()
            getattr(self, name+'_mean').copy_(torch.where(mask, mean, 0.))
            getattr(self, name+'_std').copy_(torch.where(mask, std, 1.))
        for index, branch in enumerate(('positive', 'keep')):
            values = [r['target'].detach().double().cpu().abs().amax(-1) for r in records if r['branch'] == branch]
            if not values:
                raise ValueError('normalization needs both benefit branches')
            self.scales[index] = torch.quantile(torch.cat(values), .9).clamp_min(1e-12)
        self.ready.fill_(True)


def group_losses(prediction, target, rates, conditional, groups, scales):
    """One loss per action/group; a three-state keep record has unit group weight."""
    rowscale = scales[groups]
    error = (prediction-target) / rowscale[:, None]
    regression = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none').mean(-1)
    total_error = error.sum(-1)
    regression = regression + .25*F.smooth_l1_loss(total_error, torch.zeros_like(total_error), reduction='none')
    count = torch.zeros_like(scales).scatter_add_(0, groups, torch.ones_like(regression))
    result = torch.zeros_like(scales).scatter_add_(0, groups, regression) / count.clamp_min(1)
    truth = torch.zeros_like(scales).scatter_add_(0, groups, (target.sum(-1)+rates)*conditional)
    predicted = torch.zeros_like(scales).scatter_add_(0, groups, (prediction.sum(-1)+rates)*conditional)
    weight = (truth.abs()/scales).clamp_max(10.)
    direction = .1*weight*.1*F.softplus(-truth.sign()*predicted/(.1*scales))
    return result + torch.where(truth.abs()>1e-14, direction, 0.)


def benefit_loss(prediction, target, rates, conditional, scale, branch):
    if branch not in ('positive', 'keep') or len(prediction) != (1 if branch == 'positive' else 3):
        raise ValueError('invalid benefit action group')
    scale = torch.as_tensor(scale, device=prediction.device, dtype=prediction.dtype).reshape(1)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('invalid benefit target scale')
    groups = torch.zeros(len(prediction), dtype=torch.long, device=prediction.device)
    return group_losses(prediction, target.to(prediction), rates.to(prediction), conditional.to(prediction), groups, scale)[0]
