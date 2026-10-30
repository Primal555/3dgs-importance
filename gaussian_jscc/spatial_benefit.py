"""Paired neighbor encoder and calibrated absolute/relative benefit supervision."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .physical_benefit import PhysicalLearner, decision_loss
from .physical_benefit_features import FEATURE_DIM
from .physical_benefit_audit_metrics import decision_report
from .pixel_benefit_learning import summarize_pixel_values
from .spatial_benefit_features import CONTEXT_DIM, STATE_DIM


def pack_inputs(inputs, device):
    """Batch groups and cameras in one point-network execution; pad only tails."""
    if not inputs:
        raise ValueError('nonempty spatial input batch required')
    n = max(len(x['context']) for x in inputs)
    features = torch.stack([x['features'] for x in inputs])
    context = features.new_zeros(len(inputs), n, CONTEXT_DIM)
    states = features.new_zeros(len(inputs), 4, n, STATE_DIM)
    weights = features.new_zeros(len(inputs), n)
    valid = torch.zeros(len(inputs), n, dtype=torch.bool, device=features.device)
    for i, x in enumerate(inputs):
        k = len(x['context'])
        if not k:
            raise ValueError('empty spatial block')
        context[i, :k] = x['context']; states[i, :, :k] = x['states']
        weights[i, :k] = x['weights']; valid[i, :k] = True
    return tuple(x.to(device=device, dtype=torch.float32) if x.dtype != torch.bool else x.to(device)
                 for x in (features, context, states, weights, valid))


class SpatialBenefitNetwork(nn.Module):
    def __init__(self, hidden=64, dropout=.05):
        super().__init__()
        self.points = nn.Sequential(nn.Linear(CONTEXT_DIM+2*STATE_DIM+6, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden//2), nn.SiLU())
        self.state_head = nn.Sequential(nn.Linear(FEATURE_DIM+2*(hidden//2), hidden), nn.SiLU(),
                                        nn.Linear(hidden, hidden//2), nn.SiLU(), nn.Linear(hidden//2, 2))
        self.dropout = float(dropout)
        for name, size in (('input', FEATURE_DIM), ('context', CONTEXT_DIM), ('state', STATE_DIM)):
            self.register_buffer(name+'_mean', torch.zeros(size))
            self.register_buffer(name+'_scale', torch.ones(size))
        self.register_buffer('output_scale', torch.ones(2))

    def forward(self, features, context, states, weights, valid):
        # All normalizers are fitted only on training data and then frozen.
        c = torch.asinh((context-self.context_mean)/self.context_scale)
        s = torch.asinh((states-self.state_mean)/self.state_scale)
        delta = states-states[:, :1]
        # Pair residual and RGB change before any neighbor aggregation. These
        # products supplement the full paired token; they do not define labels.
        rgb = delta[..., :3]
        residual = context[..., 8:11][:, None]
        paired = torch.cat((rgb*residual, rgb.square()), -1)
        tokens = torch.cat((c[:, None].expand(-1, 4, -1, -1), s,
            torch.asinh(delta/self.state_scale), torch.asinh(paired)), -1)
        encoded = self.points(tokens)
        weighted = (encoded*weights[:, None, :, None]).sum(-2)/weights.sum(-1)[:, None, None].clamp_min(1e-12)
        maximum = encoded.masked_fill(~valid[:, None, :, None], -torch.inf).amax(-2)
        global_features = torch.asinh((features-self.input_mean)/self.input_scale)
        x = torch.cat((global_features, weighted, maximum), -1)
        if self.training and self.dropout:
            x = x*F.dropout(torch.ones_like(x[:, :1]), p=self.dropout, training=True)
        score = self.state_head(x)*self.output_scale
        result = score[:, 1:]-score[:, :1]
        identical = ((features[:, 1:]==features[:, :1]).all(-1) &
                     (states[:, 1:]==states[:, :1]).flatten(-2).all(-1))
        return result.masked_fill(identical[..., None], 0.)


def value_losses(prediction, truth, rates, scale, tie_fraction=.02):
    """Both components retain their calibrated RD units, including all six pairs."""
    scale = torch.as_tensor(scale, device=prediction.device, dtype=prediction.dtype).clamp_min(1e-8)
    error = (prediction-truth)/scale
    p = torch.cat((prediction.new_zeros(*prediction.shape[:-2], 1, 2), prediction), -2)
    t = torch.cat((truth.new_zeros(*truth.shape[:-2], 1, 2), truth), -2)
    i, j = torch.triu_indices(4, 4, offset=1, device=prediction.device)
    pair_error = ((p[..., j, :]-p[..., i, :])-(t[..., j, :]-t[..., i, :]))/scale
    return dict(absolute=error.square().mean((-2, -1)),
                common=error.mean(-2).square().mean(-1), pair=pair_error.square().mean((-2, -1)),
                decision=decision_loss(prediction, truth, rates, scale, tie_fraction))


class SpatialLearner(PhysicalLearner):
    def __init__(self, replay, device, hidden=64, lr=1e-4, dropout=.05, weight_decay=1e-4,
                 decision_weight=.25, tie_fraction=.02, common_weight=1., pair_weight=1.,
                 training_probe_groups=256):
        self.replay, self.device = replay, torch.device(device)
        self.network = SpatialBenefitNetwork(hidden, dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        self.decision_weight, self.tie_fraction = float(decision_weight), float(tie_fraction)
        self.common_weight, self.pair_weight = float(common_weight), float(pair_weight)
        self.training_probe_groups = int(training_probe_groups)
        self.scales = None; self.updates = 0

    def fit_scales(self):
        if self.scales is not None:
            raise ValueError('training scales already fixed')
        moments = {k: [torch.zeros(size, dtype=torch.float64), torch.zeros(size, dtype=torch.float64), 0.]
                   for k, size in (('input', FEATURE_DIM), ('context', CONTEXT_DIM), ('state', STATE_DIM))}
        sy = torch.zeros(2, dtype=torch.float64); count_y = 0.
        for group in self.replay.training_records():
            weight = float(group.get('importance_weight', 1.))
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError('invalid source importance')
            for view in group['views']:
                inputs = view['inputs']
                for name, key in (('input', 'features'), ('context', 'context'), ('state', 'states')):
                    x = inputs[key].double().cpu().reshape(-1, moments[name][0].numel())
                    # Each view has equal group importance, regardless of block size.
                    w = weight/len(x)
                    moments[name][0] += x.sum(0)*w; moments[name][1] += x.square().sum(0)*w
                    moments[name][2] += weight
                y = view['values'].double().cpu()
                sy += y.square().mean(0)*weight; count_y += weight
        if not count_y:
            raise ValueError('empty training replay')
        for name, (sx, sx2, count) in moments.items():
            mean = sx/count; std = (sx2/count-mean.square()).clamp_min(0).sqrt()
            std = std.clamp_min(torch.maximum(mean.abs()*1e-3, mean.new_full(mean.shape, 1e-3)))
            getattr(self.network, name+'_mean').copy_(mean)
            getattr(self.network, name+'_scale').copy_(std)
        scale = (sy/count_y).sqrt()
        scale = scale.clamp_min(max(1e-8, float(scale.max())*1e-3))
        self.scales = dict(value=scale, rd=scale.sum())
        self.network.output_scale.copy_(scale)
        return {k: v.clone() for k, v in self.scales.items()}

    def _batch_losses(self, groups):
        inputs, targets, owners = [], [], []
        sizes = [len(group['views']) for group in groups]
        if not sizes or min(sizes) < 1:
            raise ValueError('nonempty training view groups required')
        for i, group in enumerate(groups):
            for view in group['views']:
                inputs.append(view['inputs']); targets.append(view['values']); owners.append(i)
        owner = torch.tensor(owners, device=self.device)
        def mean_groups(x):
            # Views are contiguous by group. Avoid repeated-index CUDA atomics
            # so checkpoint replay does not change floating-point sum order.
            if len(set(sizes)) == 1:
                return x.reshape(len(groups), sizes[0], *x.shape[1:]).mean(1)
            return torch.stack([chunk.mean(0) for chunk in x.split(sizes)])
        pred = self.network(*pack_inputs(inputs, self.device)).double()
        truth = torch.stack(targets).to(self.device, dtype=torch.float64)
        rates = torch.stack([torch.as_tensor(g['rates']) for g in groups]).to(self.device, dtype=torch.float64)
        losses = value_losses(pred, truth, rates[owner], self.scales['rd'], self.tie_fraction)
        # Geometry/color are supervised per camera; decisions use the actual
        # mean over the same camera window as the allocator and teacher.
        losses = {k: mean_groups(v) for k, v in losses.items() if k != 'decision'}
        p, t = mean_groups(pred), mean_groups(truth)
        losses['decision'] = decision_loss(p, t, rates, self.scales['rd'], self.tie_fraction)
        losses['total'] = (losses['absolute']+self.common_weight*losses['common']+
                          self.pair_weight*losses['pair']+self.decision_weight*losses['decision'])
        pc = torch.cat((p.new_zeros(len(groups), 1), p.sum(-1)+rates), -1)
        tc = torch.cat((t.new_zeros(len(groups), 1), t.sum(-1)+rates), -1)
        losses['regret'] = (tc.gather(1, pc.argmin(-1, keepdim=True)).squeeze(1)-tc.min(-1).values).detach()
        return losses

    def train_updates(self, count, batch_groups=8):
        if self.scales is None or count < 0 or batch_groups <= 0:
            raise ValueError('fixed scales and positive batch size required')
        self.network.train(); stats = dict(updates=self.updates, loss=None)
        for _ in range(count):
            groups = self.replay.sample(batch_groups)
            evaluated = self._batch_losses(groups)
            weights = evaluated['total'].new_tensor([float(g.get('importance_weight', 1.))*
                float(g.get('replay_importance_weight', 1.)) for g in groups])
            if not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError('invalid source/replay importance')
            loss = (evaluated['total']*weights).mean()
            self.optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = nn.utils.clip_grad_norm_(self.network.parameters(), 10.)
            if not torch.isfinite(norm):
                raise ValueError('nonfinite spatial benefit gradient')
            self.optimizer.step(); self.updates += 1
            record_ids = [g['replay_record_id'] for g in groups]
            self.replay.mark_updated(record_ids)
            priorities = (evaluated['absolute'].detach().sqrt()+
                          evaluated['regret']/self.scales['rd'].to(self.device)).cpu().tolist()
            for key, priority in zip(record_ids, priorities):
                self.replay.update_priority(key, priority)
            stats = {k+'_loss': float((v.detach()*weights).mean()) for k, v in evaluated.items()
                     if k not in ('regret', 'total')}
            stats.update(loss=float(loss.detach()), train_regret=float((evaluated['regret']*weights).mean()),
                         grad_norm=float(norm), updates=self.updates)
        stats.update({'replay_'+k: v for k, v in self.replay.coverage().items()})
        return stats

    @torch.inference_mode()
    def predict(self, inputs, height, width, d_ref):
        self.network.eval()
        return self.network(*pack_inputs([inputs], self.device))[0].double()

    @torch.inference_mode()
    def predict_group_views(self, groups):
        """One forward for several candidates and all cameras, preserving order."""
        self.network.eval()
        counts = [len(views) for views in groups]
        if not counts or min(counts) < 1:
            raise ValueError('nonempty candidate view groups required')
        packed = pack_inputs([inputs for views in groups for inputs in views], self.device)
        result = self.network(*packed).double()
        return [list(values.unbind(0)) for values in result.split(counts)]

    @torch.inference_mode()
    def _report_groups(self, groups):
        rows, weights = [], []
        self.network.eval()
        for group in groups:
            views = group['views']
            pred = self.network(*pack_inputs([v['inputs'] for v in views], self.device)).double().mean(0).cpu()
            truth = torch.stack([v['values'].double().cpu() for v in views]).mean(0)
            rows.append(dict(source=group['source'], step=group['step'], rates=group['rates'],
                             truth=truth, prediction=pred))
            weights.append(float(group.get('importance_weight', 1.)))
        if not rows:
            raise ValueError('empty evaluation replay')
        report = decision_report(torch.stack([r['prediction'] for r in rows]),
            torch.stack([r['truth'] for r in rows]), torch.stack([torch.as_tensor(r['rates']) for r in rows]), weights)
        return rows, report

    def validate(self):
        rows, weighted = self._report_groups(self.replay.validation_records())
        # A deterministic uniform subset keeps the diagnostic bounded and comparable.
        order = torch.randperm(len(self.replay.training), generator=torch.Generator().manual_seed(48017))
        entries = [self.replay.training[i] for i in order[:self.training_probe_groups].tolist()]
        _, training = self._report_groups(self.replay._read(e) for e in entries)
        report = summarize_pixel_values(rows)
        report.update(network_updates=self.updates, validation_groups=len(rows),
            weighted_validation=weighted, weighted_training=training, replay_coverage=self.replay.coverage(),
            scope='paired spatial tokens; exact whole-block labels; fixed calibrated RD units')
        return report
