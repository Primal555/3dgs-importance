"""Read-only fixed-point XYZ diagnostics; never contributes to training loss.

All distances are in world units. A mean vector is a sampled common component,
not evidence that the entire scene undergoes a rigid translation.
"""
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from .render_validation import append_json


def vector_metrics(delta, block_ids):
    """Point-weighted orthogonal translation / block-translation decomposition."""
    d = delta.detach().cpu().double()
    ids = block_ids.detach().cpu()
    if d.ndim != 2 or d.shape[1] != 3 or not len(d) or len(ids) != len(d):
        raise ValueError('expected nonempty N x 3 vectors and N block IDs')
    if not torch.isfinite(d).all():
        raise FloatingPointError('nonfinite coordinate diagnostics')
    mean = d.mean(0)
    sse = d.square().sum()
    common_sse = len(d)*mean.square().sum()
    rows, block_sse, residual_sse = [], 0., 0.
    for index in ids.unique(sorted=True):
        values = d[ids == index]
        bias = values.mean(0)
        residual = (values-bias).square().sum()
        block_sse += float(len(values)*bias.square().sum())
        residual_sse += float(residual)
        rows.append({'block': int(index), 'points': len(values), 'mean_xyz': bias.tolist(),
                     'mean_norm': float(bias.norm()), 'rmse': float(values.square().mean().sqrt()),
                     'residual_rmse': float((residual/(3*len(values))).sqrt())})
    distance = d.norm(dim=-1)
    denominator = float(sse)
    return {'points': len(d), 'mean_xyz': mean.tolist(), 'median_xyz': d.median(0).values.tolist(),
            'mean_norm': float(mean.norm()), 'rmse': float(d.square().mean().sqrt()),
            'distance_p50': float(distance.quantile(.5)), 'distance_p95': float(distance.quantile(.95)),
            'distance_max': float(distance.max()),
            'translation_removed_rmse': float((d-mean).square().mean().sqrt()),
            'block_translation_removed_rmse': (residual_sse/(3*len(d)))**.5,
            'common_sse_fraction': float(common_sse)/denominator if denominator else 0.,
            'block_common_sse_fraction': block_sse/denominator if denominator else 0.,
            'blocks': rows}


class CenterDrift:
    def __init__(self, model, blocks, fitted, heldout, geometry, out, batch_size):
        self.model = model
        self.root = Path(out)
        self.directory = self.root/'center_drift'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.span, self.lower = geometry.span.double(), geometry.lower.double()
        self.groups, self.batches = {}, []
        offset = 0
        for split, indices in (('fit', fitted), ('heldout', heldout)):
            if not indices:
                raise ValueError('drift diagnostics require nonempty fit and heldout probes')
            targets = torch.cat([blocks[i][:, :3] for i in indices]).double()*self.span+self.lower
            ids = torch.cat([torch.full((len(blocks[i]),), i, dtype=torch.long) for i in indices])
            self.groups[split] = {'slice': slice(offset, offset+len(targets)), 'target': targets, 'ids': ids}
            offset += len(targets)
            for start in range(0, len(indices), batch_size):
                chunks = [blocks[i][:, :3] for i in indices[start:start+batch_size]]
                xyz = pad_sequence(chunks, batch_first=True)
                mask = torch.arange(xyz.shape[1])[None] < torch.tensor([len(x) for x in chunks])[:, None]
                self.batches.append((xyz, mask))
        torch.save({'geometry': geometry.to_dict(),
                    'groups': {k: {n: v[n] for n in ('target', 'ids')} for k, v in self.groups.items()}},
                   self.directory/'fixed_points.pt')
        (self.directory/'definitions.json').write_text(json.dumps({
            'units': 'world; RMSE = sqrt(mean of squared coordinate components), not mean point distance',
            'scope': 'fixed sampled training blocks and all fixed heldout blocks; not the entire scene',
            'error': 'prediction minus original XYZ; original targets and normalization never change',
            'update': 'prediction immediately after minus immediately before ONE optimizer step',
            'common_sse_fraction': 'N * ||mean(vector)||^2 / sum ||vector||^2; zero for zero vectors',
            'block_common_sse_fraction': 'sum n_b * ||block_mean||^2 / total SSE; includes global common component',
            'removed_rmse': 'oracle diagnostic only; never used to correct predictions or compute render PSNR',
            'decoder_only': 'new decoder(old encoder latent) minus old prediction',
            'encoder_under_new_decoder': 'new prediction minus new decoder(old encoder latent); '
                'ordered decomposition, includes nonlinear interactions, not independent causal attribution',
            'snapshots': 'sampled world predictions at validation steps, in fixed_points.pt point order',
        }, indent=2), encoding='utf-8')

    @torch.no_grad()
    def capture(self, old_latents=None, keep_latents=False):
        model = self.model
        device = next(model.parameters()).device
        was_training = model.training
        # Measurement must not consume the training RNG, even if stochastic
        # modules are introduced later. No codec parameters are updated here.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        values, latents = [], []
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices):
                for i, (xyz_cpu, mask_cpu) in enumerate(self.batches):
                    xyz, mask = xyz_cpu.to(device), mask_cpu.to(device)
                    z = model.learned.center_encoder(xyz*2-1, xyz, mask) if old_latents is None else old_latents[i].to(device)
                    pred = model.learned.center_decoder(z, mask)
                    values.append(pred[mask].cpu().double()*self.span+self.lower)
                    if keep_latents:
                        latents.append(z.cpu())
        finally:
            model.train(was_training)
        return {'xyz': torch.cat(values), 'latents': latents}

    def errors(self, xyz):
        return {name: vector_metrics(xyz[g['slice']]-g['target'], g['ids']) for name, g in self.groups.items()}

    def observe(self, step):
        snapshot = self.capture()
        append_json(self.root/'center_drift_validation.jsonl', {'step': step, 'errors': self.errors(snapshot['xyz'])})
        torch.save({'step': step, 'xyz': snapshot['xyz']}, self.directory/f'xyz_{step:06d}.pt')
        self.plot()

    def after_step(self, step, before, gradients, parameter_updates, lrs, clip_factor):
        after = self.capture()['xyz']
        hybrid = self.capture(old_latents=before['latents'])['xyz']
        row = {'step': step, 'lrs': lrs, 'clip_factor': clip_factor, 'module_grad_norms': gradients,
               'module_updates': parameter_updates, 'groups': {}}
        for name, g in self.groups.items():
            selection, ids = g['slice'], g['ids']
            pre, post, mixed = before['xyz'][selection], after[selection], hybrid[selection]
            error, delta = pre-g['target'], post-pre
            denom = float(error.norm()*delta.norm())
            row['groups'][name] = {
                'error_before': vector_metrics(error, ids),
                'error_after': vector_metrics(post-g['target'], ids),
                'update': vector_metrics(delta, ids),
                'decoder_only_update': vector_metrics(mixed-pre, ids),
                'encoder_under_new_decoder_update': vector_metrics(post-mixed, ids),
                'toward_target_cosine': float((-error*delta).sum())/denom if denom else None,
            }
        append_json(self.root/'center_drift_updates.jsonl', row)

    def plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        def read(name):
            path = self.root/name
            return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()] if path.exists() else []
        validation = read('center_drift_validation.jsonl')
        updates = read('center_drift_updates.jsonl')
        fig, axes = plt.subplots(3, 2, figsize=(13, 11))
        for col, name in enumerate(self.groups):
            x = [r['step'] for r in validation]
            for axis, label in enumerate('XYZ'):
                axes[0, col].plot(x, [r['errors'][name]['mean_xyz'][axis] for r in validation], label=label)
            axes[0, col].set_title(f'{name}: mean prediction error (world)')
            for key in ('rmse', 'translation_removed_rmse', 'block_translation_removed_rmse'):
                axes[1, col].plot(x, [r['errors'][name][key] for r in validation], label=key)
            axes[1, col].set_title('Raw vs oracle-debiased error (diagnostic only)')
            for key in ('rmse', 'mean_norm', 'distance_p95'):
                axes[2, col].plot([r['step'] for r in updates],
                                 [r['groups'][name]['update'][key] for r in updates], label=key)
            axes[2, col].set_title('Single optimizer-step XYZ movement (world)')
        for axis in axes.flat:
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
            axis.set_xlabel('Training step')
        fig.tight_layout()
        fig.savefig(self.directory/'drift.png', dpi=150)
        plt.close(fig)
