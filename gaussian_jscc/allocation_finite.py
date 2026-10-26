"""Paired finite image differences; no loss-of-sign subtraction of scene MSEs."""
import torch
from torch.nn import functional as F
from .render_objective import MaskedMultiViewRenderTask, image_distortion
from .optimization import preserved_rng


class PairedMaskedRenderTask(MaskedMultiViewRenderTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline = []

    def _terms(self, scene):
        from .rendering import render
        self.baseline = []
        for camera in self.cameras:
            image = render(scene[:, :-1], camera, self.degree, self.white_background, scene[:, -1])
            reference = self.reference.get(camera, scene.device)
            self.baseline.append((image.detach(), reference.detach()))
            yield image_distortion(image, reference)

    @torch.no_grad()
    def delta_mse(self, scene):
        from .rendering import render
        if len(self.baseline) != len(self.cameras):
            raise RuntimeError('finite feedback requires a complete baseline window')
        values = []
        for camera, (base, reference) in zip(self.cameras, self.baseline):
            image = render(scene[:, :-1], camera, self.degree, self.white_background, scene[:, -1])
            delta = image.double()-base.double()
            change = (2*(base.double()-reference.double())*delta+delta.square()).mean()
            if not torch.isfinite(change): raise RuntimeError('nonfinite finite-render feedback')
            values.append(float(change))
        return sum(values)/len(values)


def correction_indices(ids, image, rate, q, counts, limit):
    """Mix high-impact, ambiguous nonzero decisions and least-checked coverage."""
    if not len(ids) or limit <= 0: return []
    image, rate, ids, q = (v.detach().cpu() for v in (image, rate, ids, q))
    action = (image+rate).clone()
    action[torch.arange(len(ids)), q[ids]-1] = torch.inf
    impact = image.abs().amax(-1)
    ambiguity = action.abs().amin(-1)
    choices, used = [], set()
    orderings = [torch.argsort(impact, descending=True, stable=True).tolist(),
                 torch.argsort(ambiguity.masked_fill(impact == 0, torch.inf), stable=True).tolist(),
                 torch.argsort(counts[ids], stable=True).tolist()]
    cursors = [0, 0, 0]
    for slot in range(min(limit, len(ids))):
        group = slot % 3
        while orderings[group][cursors[group]] in used: cursors[group] += 1
        index = orderings[group][cursors[group]]; used.add(index)
        if group == 0:
            difference = image[index].abs().clone(); difference[q[ids[index]]-1] = -1
            tier = int(difference.argmax())+1
        elif group == 1:
            tier = int(action[index].abs().argmin())+1
        else:
            tier = int(action[index].argmin())+1
        choices.append((index, tier))
    return choices


@torch.no_grad()
def correct_actions(model, features, id_batches, cache, geometry, task, scene, upstream,
                    scheduler, tier_ids, locations, images, rates, keep_image, d_ref, snr, kind):
    """Replace selected proxy costs with paired finite renders used by training."""
    from .budgeted_allocation_kernel import raw_batches, set_rng
    limit = scheduler.finite_samples
    stats = dict(finite_feedback=bool(limit), finite_corrections=0, finite_positive_corrections=0,
                 finite_keep_corrections=0, finite_nonzero_pairs=0, finite_sign_disagreements=0,
                 finite_zero_predictions=0, finite_absolute_error_mean=0.)
    if not limit: return [], stats
    if not hasattr(task, 'delta_mse'): raise ValueError('finite feedback requires paired image task')
    device = scene.device
    q_source = torch.zeros(len(scheduler.counts), dtype=torch.long)
    offsets, offset = [], 0
    for ids, entry in zip(id_batches, cache):
        valid = ids >= 0; q_source[ids[valid]] = entry[0][valid.to(device)].cpu()
        offsets.append(offset); offset += int(valid.sum())
    keep_limit = min(2, max(1, limit//4))
    positive = correction_indices(tier_ids, images, rates, q_source, scheduler.finite_counts,
                                  limit-keep_limit)
    interventions = [(locations[i], tier, i, 'positive') for i,tier in positive]
    # Rotate full-row coverage for supplemental keep checks. Full q0 gate credit
    # remains in the kernel for EVERY absent row, independently of this budget.
    ids_all = torch.arange(len(q_source))
    for closed in (True, False):
        pool = ids_all[(q_source == 0) if closed else (q_source > 0)]
        if not len(pool) or len(interventions) >= limit: continue
        # Gate influence prioritizes important corrections; every other update
        # uses least finite coverage so low-impact cases are not excluded forever.
        if int(scheduler.last.max()) % 2:
            chosen = int(pool[keep_image[pool.to(device), 1].abs().argmax().cpu()])
        else:
            chosen = int(pool[scheduler.finite_counts[pool].argmin()])
        for batch, ids in enumerate(id_batches):
            found = (ids == chosen).nonzero()
            if len(found):
                row, column = found[0].tolist()
                new = int(cache[batch][1][row, column]) if closed else 0
                interventions.append(((batch,row,column),new,None,'keep')); break
        if len(interventions)-len(positive) >= keep_limit: break
    records = []
    for (batch,row,column), new, position, feedback_kind in interventions:
        f = features[batch].to(device); ids = id_batches[batch].to(device); valid = ids >= 0
        q, _, y, _, rng = cache[batch]
        old = int(q[row,column]); source_row = int(ids[row,column])
        alt = q.clone(); alt[row,column] = new
        with preserved_rng(device):
            set_rng(rng,device)
            pred = model.forward_tier_batches(f,f[..., :3],F.one_hot(alt,4).to(f),snr,kind,
                paired_noise=True, **({'_semantic_y':y} if old > 0 and new > 0 else {}))[0]
        changed = scene.detach().clone()
        raw = raw_batches(pred,geometry,model)
        offset = offsets[batch]; n = int(valid.sum())
        changed[offset:offset+n] = torch.cat((raw[valid], (alt[valid]>0).to(raw)[:,None]),-1)
        measured = task.delta_mse(changed)
        if feedback_kind == 'positive':
            predicted = float(images[position,new-1])*d_ref
            rate_delta = float(rates[position,new-1])
            images[position,new-1] = measured/d_ref
        else:
            predicted = float(keep_image[source_row,1])*d_ref*(1 if old == 0 else -1)
            rate_delta = None
            keep_image[source_row,1] = measured/d_ref*(1 if old == 0 else -1)
        scheduler.finite_counts[source_row] += 1
        records.append(dict(source_row=source_row,old_tier=old,new_tier=new,feedback_kind=feedback_kind,
            measured_delta_mse=measured,first_order_delta_mse=predicted,rate_delta_cost=rate_delta,
            sign_agreement=(measured*predicted>0) if measured and predicted else None,
            absolute_error=abs(measured-predicted),applied_to_training=True))
    stats.update(finite_corrections=len(records),
        finite_positive_corrections=sum(r['feedback_kind']=='positive' for r in records),
        finite_keep_corrections=sum(r['feedback_kind']=='keep' for r in records),
        finite_nonzero_pairs=sum(bool(r['measured_delta_mse'] and r['first_order_delta_mse']) for r in records),
        finite_sign_disagreements=sum(r['sign_agreement'] is False for r in records),
        finite_zero_predictions=sum(r['first_order_delta_mse']==0 for r in records),
        finite_absolute_error_mean=sum(r['absolute_error'] for r in records)/max(1,len(records)))
    return records, stats
