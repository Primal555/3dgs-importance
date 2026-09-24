"""Point-weighted center gradient accumulation, one graph resident at a time."""
import torch
from torch.nn.utils.rnn import pad_sequence

from .center_attribute_codec import center_loss


def accumulate_center_gradients(model, blocks, selections, geometry, smoothing, device, loss_kind='distance'):
    """Caller zeros gradients once and steps once after all microbatches.

    Weights use actual valid point counts (including partial blocks), not a
    blind mean of microbatch means. Selection is fixed before any backward;
    no model/Adam updates occur between microbatches. Returned loss is detached.
    """
    counts = [sum(len(blocks[i]) for i in chosen) for chosen in selections]
    total = sum(counts)
    if not counts or min(counts) < 1:
        raise ValueError('accumulation requires nonempty microbatches')
    objective = None
    losses = []
    for chosen, points in zip(selections, counts):
        group = [blocks[i] for i in chosen]
        f = pad_sequence(group, batch_first=True).to(device)
        active = torch.arange(f.shape[1], device=device)[None] < torch.tensor(
            [len(x) for x in group], device=device)[:, None]
        pred = model.learned.centers(f[..., :3], active)
        if loss_kind == 'distance':
            loss = center_loss(pred[active], f[..., :3][active], geometry, smoothing)
        else:
            loss = center_loss(pred[active], f[..., :3][active], geometry, smoothing, kind=loss_kind)
        if not bool(torch.isfinite(loss)):
            # No optimizer step has happened; never use partially accumulated gradients.
            model.zero_grad(set_to_none=True)
            raise FloatingPointError('nonfinite center microbatch loss')
        weighted = loss*(points/total)
        weighted.backward()
        value = weighted.detach()
        objective = value if objective is None else objective+value
        losses.append(float(loss.detach()))
        del f, active, pred, loss, weighted
    return objective, {'microbatches': len(selections), 'valid_points': total,
                       'points_per_microbatch': counts, 'microbatch_losses': losses,
                       'sampled_block_count': sum(map(len, selections))}
