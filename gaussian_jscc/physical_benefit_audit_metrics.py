"""Weighted decision checks and reproducible physical-feature counterexamples."""
import math

import torch
from torch.nn import functional as F


def decision_report(prediction, target, rates, weights):
    p, t, r, w = [torch.as_tensor(x).detach().double().cpu()
                  for x in (prediction, target, rates, weights)]
    if (p.shape != t.shape or p.ndim != 3 or p.shape[1:] != (3, 2)
            or r.shape != p.shape[:2] or w.shape != p.shape[:1] or not len(w)
            or not all(torch.isfinite(x).all() for x in (p, t, r, w)) or (w <= 0).any()):
        raise ValueError('finite complete groups and positive weights required')
    w = w / w.sum()
    def mean(x):
        return float((x * w.reshape(len(w), *([1] * (x.ndim-1)))).sum(0).mean())
    def directions(actual, estimate):
        wrong = (actual.abs() > 1e-14) & (actual*estimate <= 0)
        return dict(sign_error_fraction=mean(wrong.double()),
            regret_mean=mean(torch.where(wrong, actual.abs(), 0.)),
            rd_mae=mean((estimate-actual).abs()),
            false_beneficial_fraction=mean(((actual > 1e-14) & (estimate <= 0)).double()),
            false_harmful_fraction=mean(((actual < -1e-14) & (estimate >= 0)).double()))
    actual, estimate = t.sum(-1)+r, p.sum(-1)+r
    ai, pi = actual.argmin(-1), estimate.argmin(-1)
    ids = torch.arange(len(w))
    a4 = torch.cat((torch.zeros(len(w), 1), actual), -1)
    p4 = torch.cat((torch.zeros(len(w), 1), estimate), -1)
    four = a4[ids, p4.argmin(-1)]-a4.min(-1).values
    error = (p-t).sum(-1)
    common = error.mean(-1)
    centered = error-common[:, None]
    energy = mean(error.square())
    i, j = torch.triu_indices(3, 3, offset=1)
    return dict(count=len(w), effective_sample_size=float(1/w.square().sum()),
        conditional_keep=directions(actual.mean(-1), estimate.mean(-1)),
        positive=directions(actual[:, j]-actual[:, i], estimate[:, j]-estimate[:, i]),
        three_tier_regret=mean(actual[ids, pi]-actual[ids, ai]),
        four_state_regret=mean(four),
        four_state_baselines={f'fixed_q{k}': mean(a4[:, k]-a4.min(-1).values) for k in range(4)},
        three_tier_baselines={f'fixed_q{k+1}': mean(actual[:, k]-actual.min(-1).values) for k in range(3)},
        tier_accuracy=mean((actual[ids, pi]-actual.min(-1).values <= 1e-14).double()),
        predicted_q0_fraction=mean((p4.argmin(-1) == 0).double()),
        optimal_q0_fraction=mean((a4.argmin(-1) == 0).double()),
        common_bias=mean(common), common_error_rmse=math.sqrt(mean(common.square())),
        centered_error_rmse=math.sqrt(mean(centered.square())),
        common_error_energy_fraction=mean(common.square())/energy if energy else 0.,
        target_rd_rms=math.sqrt(mean(actual.square())),
        component_rmse=[math.sqrt(mean((p[..., k]-t[..., k]).square())) for k in range(2)])


def error_oracles(prediction, target, rates, weights):
    p, t = prediction.double(), target.double()
    error = p-t
    common = error.mean(1, keepdim=True)
    candidates = dict(original=p, correct_common=p-common,
        correct_centered=p-(error-common), correct_mse=torch.stack((t[..., 0], p[..., 1]), -1),
        correct_structure=torch.stack((p[..., 0], t[..., 1]), -1))
    return {name: decision_report(value, t, rates, weights) for name, value in candidates.items()}


@torch.inference_mode()
def feature_collision_probe():
    from .pixel_local_renderer import TileScene
    from .physical_benefit_features import visibility_statistics, candidate_features
    from .pixel_benefit_features import exact_values
    # Symmetric neighbors have equal visibility but opposite target residuals.
    p = torch.tensor([[24, 8, 1, 1, 0, 1, .2, 1, .4, .4, .4, 3],
        [8, 8, 1, 1, 0, 1, .7, 1, .4, .4, .4, 3],
        [40, 8, 1, 1, 0, 1, .7, 1, .4, .4, .4, 3]], dtype=torch.float32)
    base = TileScene.from_projected(p, 16, 48, backend='cpu')
    a = p[None].repeat(4, 1, 1); a[0, 0, 7] = 0
    b = a.clone()
    a[1:, 1, 8] = .7; a[1:, 2, 8] = .1
    b[1:, 1, 8] = .1; b[1:, 2, 8] = .7
    render = lambda value: TileScene.from_projected(value, 16, 48, backend='cpu').image
    target = render(a[1])
    stats = visibility_statistics(base, target)
    raw = torch.zeros(4, 3, 14); raw[..., 7] = 1.
    features = [candidate_features(torch.zeros(24), stats, raw, value, 0, 16, 48, .01, .006)
                for value in (a, b)]
    values = [exact_values(base, torch.arange(3), value, target, .006, .01) for value in (a, b)]
    mse0 = (render(a[0]).double()-target).square().mean()
    full = [float(((render(value[1]).double()-target).square().mean()-mse0)/.01) for value in (a, b)]
    rate_cost = 1e-4  # Same known normalized communication cost in both cases.
    return dict(features_identical=bool(torch.equal(*features)),
        feature_max_abs=float((features[0]-features[1]).abs().max()),
        analytic_rate_cost=rate_cost,
        distortion_delta_a=float(values[0][0].sum()), distortion_delta_b=float(values[1][0].sum()),
        rd_a=float(values[0][0].sum())+rate_cost, rd_b=float(values[1][0].sum())+rate_cost,
        full_mse_a=full[0], full_mse_b=full[1],
        teacher_full_mse_max_abs=max(abs(float(value[0, 0])-direct) for value, direct in zip(values, full)),
        scope='synthetic identical 158D inputs; different spatial assignment of neighbor changes; not a real-scene error frequency')


@torch.inference_mode()
def codec_pairing_probe(model, features, seed=42):
    from .optimization import preserved_rng
    from .pixel_benefit_window import decode_groups
    from .allocation_benefit_features import keep_rates
    from .successive_channel import encode_scales
    device = features.device
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('paired probe requires an eval-mode frozen codec')
    q = torch.arange(features.shape[0]*features.shape[1], device=device).reshape(features.shape[:2]) % 4
    with preserved_rng(device):
        torch.manual_seed(seed)
        y = model.semantic_encoder(features, features[..., :3], q > 0)
        with preserved_rng(device):
            noise = torch.randn((*q.shape, 2*model.cfg.rates[-1]), device=device, dtype=features.dtype)
            noise *= math.sqrt(10**(-1)/2)
        implicit = model.forward_tier_batches(features, features[..., :3], F.one_hot(q, 4).to(features),
            10., 'awgn', paired_noise=True, _semantic_y=y)[0]
        explicit = model.forward_tier_batches(features, features[..., :3], F.one_hot(q, 4).to(features),
            10., 'awgn', paired_noise=True, _semantic_y=y, _channel_noise=noise)[0]
        rows = torch.tensor([0, 1, 1], device=device)
        cols = torch.tensor([0, 1, 2], device=device)
        candidates, _ = decode_groups(model, features, q, rows, cols, explicit, y, noise, 10.)
        baseline_error = float((implicit-explicit).abs().max())
        candidate_error = rate_error = 0.
        neighbor_delta, keep_delta, positive_delta = [], [], []
        rates = keep_rates(q, torch.ones_like(q, dtype=torch.bool), model.cfg.rates, xyz=24., bits=2.)
        for n, (row, col) in enumerate(zip(rows.tolist(), cols.tolist())):
            for tier in range(4):
                alt = q.clone(); alt[row, col] = tier
                full = model.forward_tier_batches(features, features[..., :3], F.one_hot(alt, 4).to(features),
                    10., 'awgn', paired_noise=True, _channel_noise=noise)[0]
                candidate_error = max(candidate_error, float((full[row]-candidates[n, tier]).abs().max()))
                others = torch.arange(q.shape[1], device=device) != col
                change = float((full[row, others, 3:]-explicit[row, others, 3:]).abs().max())
                neighbor_delta.append(change)
                (keep_delta if (tier == 0) != (int(q[row, col]) == 0) else positive_delta).append(change)
                if tier:
                    drop = q[row:row+1].clone(); drop[0, col] = 0
                    kept = drop.clone(); kept[0, col] = tier
                    scales = torch.ones((1, 3), device=device)
                    byte_delta = len(encode_scales(scales, kept))-len(encode_scales(scales, drop))
                    expected = model.cfg.rates[tier]+24.+byte_delta*8/2
                    rate_error = max(rate_error, abs(float(rates[row, col, tier-1])-expected))
        torch.testing.assert_close(implicit, explicit, atol=2e-6, rtol=2e-5)
        if candidate_error > 2e-5 or rate_error:
            raise RuntimeError(f'candidate/rate consistency failed: {candidate_error}, {rate_error}')
    return dict(baseline_noise_max_abs=baseline_error, candidate_batch_max_abs=candidate_error,
        rate_delta_max_abs=rate_error, max_other_point_attribute_change=max(neighbor_delta),
        max_keep_flip_other_point_change=max(keep_delta, default=0.),
        max_positive_shift_other_point_change=max(positive_delta, default=0.),
        scope='loaded codec on deterministic synthetic blocks; does not re-render historical labels')
