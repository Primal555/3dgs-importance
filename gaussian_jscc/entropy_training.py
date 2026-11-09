"""Complete-pass density fitting and original-row frozen semantic caching."""
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .entropy_model import GroupedEntropyModel, fit_statistics, split_blocks


def atomic_save(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    torch.save(value, temporary); temporary.replace(path)


def tensor_hash(value):
    x = value.detach().cpu().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


@torch.no_grad()
def cache_latents(model, scene):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('semantic caching requires a frozen evaluation codec')
    device = next(model.parameters()).device
    n = len(scene.raw); y = torch.empty(n, model.cfg.latent_dim)
    blocks = torch.full((n,), -1, dtype=torch.long)
    if len(scene.groups) != len(scene.group_ids): raise ValueError('feature/index batches differ')
    offset = 0
    for features, ids in zip(scene.groups, scene.group_ids):
        ids = ids.cpu().long(); valid = ids >= 0; rows = ids[valid]
        if (ids.shape != features.shape[:2] or (rows >= n).any()
                or len(rows.unique()) != len(rows) or (blocks[rows] >= 0).any()):
            raise ValueError('original codec row mapping is not a partition')
        features = features.to(device)
        latent = model.semantic_encoder(features, features[..., :3], valid.to(device)).detach().cpu()
        if latent.shape != (*ids.shape, model.cfg.latent_dim) or not torch.isfinite(latent[valid]).all():
            raise ValueError('invalid frozen semantic output')
        y[rows] = latent[valid]
        original_blocks = torch.arange(offset, offset+len(ids))[:, None].expand_as(ids)
        blocks[rows] = original_blocks[valid]; offset += len(ids)
    if (blocks < 0).any(): raise ValueError('semantic cache omitted source rows')
    return y, blocks


@torch.no_grad()
def mean_bits(model, y, rows, batch):
    total = 0.
    for ids in rows.split(batch): total += float(model.bits(y[ids]).sum())
    return total/len(rows)


def train_density(y, block_ids, args, out, identity, *, resume=False, max_epochs=None):
    """Resume only at saved epoch boundaries; deterministic epoch permutations."""
    if (y.ndim != 2 or y.shape[1] != sum(args.groups) or len(block_ids) != len(y)
            or not torch.isfinite(y).all() or y.requires_grad or block_ids.requires_grad):
        raise ValueError('finite detached latent and matching original blocks required')
    if (min(args.epochs, args.batch_size) < 1 or not 0 < args.min_lr <= args.lr
            or not all(math.isfinite(v) and v > 0 for v in (args.lr, args.min_lr, args.grad_clip))):
        raise ValueError('invalid entropy training settings')
    out = Path(out); path = out/'checkpoints/latest/entropy.pt'
    settings = {k: getattr(args, k) for k in ('groups', 'hidden', 'mixtures', 'bin_width',
        'validation_fraction', 'seed', 'batch_size', 'epochs', 'lr', 'min_lr', 'grad_clip')}
    signature = dict(settings=settings, source=identity, latent_sha256=tensor_hash(y),
                     block_sha256=tensor_hash(block_ids))
    train, validation = split_blocks(block_ids, args.validation_fraction, args.seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        model = GroupedEntropyModel(args.groups, args.hidden, args.mixtures, args.bin_width)
    mean, std = fit_statistics(y, train)
    model.mean.copy_(mean); model.std.copy_(std); model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epoch = 0; updates = 0; logs = []
    if path.exists():
        if not resume: raise FileExistsError('entropy checkpoint exists; specify resume')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['signature'] != signature: raise ValueError('entropy resume data/settings mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        epoch, updates, logs = state['epoch'], state['updates'], state['logs']
    elif resume and (out/'metrics/entropy_training.jsonl').exists():
        raise FileNotFoundError('training logs exist without an entropy checkpoint')
    target = args.epochs if max_epochs is None else min(args.epochs, epoch+max_epochs)
    if not path.exists():
        # Save before the first update/log write so an interrupted first epoch is resumable.
        atomic_save(dict(signature=signature, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         epoch=epoch, updates=updates, logs=logs), path)
    log_path = out/'metrics/entropy_training.jsonl'; log_path.parent.mkdir(parents=True, exist_ok=True)
    # Saved logs and weights belong to the same epoch, including after an interrupted write.
    def write_logs():
        temp = log_path.with_suffix('.tmp')
        temp.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in logs), encoding='utf8')
        temp.replace(log_path)
    write_logs()
    while epoch < target:
        started = time.perf_counter(); model.train()
        lr = args.min_lr+(args.lr-args.min_lr)*(1+math.cos(math.pi*epoch/max(1,args.epochs-1)))/2
        for group in optimizer.param_groups: group['lr'] = lr
        order = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed+epoch+1))]
        norm_sum = 0.; norm_max = 0.; clipped = 0; batches = 0
        for ids in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = model.bits(y[ids]).mean()/model.dim
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite entropy NLL')
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True))
            optimizer.step(); updates += 1; batches += 1
            norm_sum += norm; norm_max = max(norm_max, norm); clipped += norm > args.grad_clip
        epoch += 1; model.eval()
        row = dict(epoch=epoch, updates=updates, train_rows=len(train), validation_rows=len(validation),
            training_draws=epoch*len(train), minimum_training_visits=epoch, lr=lr,
            train_bits_per_gaussian=mean_bits(model,y,train,args.batch_size),
            validation_bits_per_gaussian=mean_bits(model,y,validation,args.batch_size),
            grad_norm_mean=norm_sum/batches, grad_norm_max=norm_max, clipped_fraction=clipped/batches,
            seconds=time.perf_counter()-started)
        logs.append(row)
        atomic_save(dict(signature=signature, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         epoch=epoch, updates=updates, logs=logs), path)
        write_logs()
        print(f"entropy {epoch}/{args.epochs}: train={row['train_bits_per_gaussian']:.4f} bits/G, "
              f"validation={row['validation_bits_per_gaussian']:.4f}, grad={norm_sum/batches:.4g}, "
              f"clip={clipped/batches:.1%}, visits={epoch}, sec={row['seconds']:.2f}", flush=True)
    model.eval().requires_grad_(False)
    return model, logs
