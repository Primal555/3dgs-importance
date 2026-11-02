"""Matched frozen-network predictions and exact single-intervention labels."""
import torch

from .batched_spatial_window import BatchedSpatialWindow
from .cli import seed_all
from .direct_rd_allocation import NoReplay
from .direct_rd_decisions import HardTierMap
from .optimization import preserved_rng
from .pixel_local_renderer import capture


def select_actions(bank, proposal, q, blocks, *, per_stratum=16, seed=42):
    if per_stratum < 1: raise ValueError('positive stratum size required')
    q, blocks = q.cpu(), blocks.cpu()
    actions = []; used = set(); used_blocks = set()
    n = len(proposal['ids'])
    ranges = dict(top=range(min(per_stratum,n)),
                  middle=range(max(0,n//2-per_stratum//2),min(n,n//2+(per_stratum+1)//2)),
                  bottom=range(max(0,n-per_stratum),n))
    for name, ranks in ranges.items():
        for rank in ranks:
            source = int(proposal['ids'][rank])
            if source in used: continue
            used.add(source); used_blocks.add(int(blocks[source]))
            actions.append(dict(source=source, block=int(blocks[source]), stratum=name, rank=rank,
                old_tier=int(q[source]), new_tier=int(proposal['tiers'][rank]),
                selection_predicted_gain=float(proposal['gains'][rank])))
    pool = (bank.visits>0).nonzero(as_tuple=True)[0]
    order = torch.randperm(len(pool),generator=torch.Generator().manual_seed(seed))
    ordinary = 0
    for index in order.tolist():
        source = int(pool[index]); block = int(blocks[source])
        if source in used or block in used_blocks: continue
        costs = torch.cat((torch.zeros(1,dtype=torch.float64),bank.sums[source]/bank.visits[source]))
        alternative = costs.clone(); alternative[q[source]] = torch.inf
        tier = int(alternative.argmin())
        actions.append(dict(source=source,block=block,stratum='ordinary',rank=None,
            old_tier=int(q[source]),new_tier=tier,
            selection_predicted_gain=float(costs[q[source]]-costs[tier])))
        used.add(source); used_blocks.add(block); ordinary += 1
        if ordinary >= per_stratum: break
    return actions


class FixedSources:
    def __init__(self, groups, sources):
        count = sum(int((ids>=0).sum()) for ids in groups)
        self.validation = torch.zeros(count,dtype=torch.bool)
        self.generator = torch.Generator().manual_seed(1931)
        columns = []; rounds = 0
        for ids in groups:
            selected = torch.isin(ids.cpu(),torch.tensor(sources,dtype=torch.long)) & (ids.cpu()>=0)
            rows = [row.nonzero(as_tuple=True)[0] for row in selected]
            rounds = max(rounds,max((len(row) for row in rows),default=0)); columns.append(rows)
        self.schedule = [[torch.tensor([int(row[r]) if r<len(row) else -1 for row in rows])
                          for rows in columns] for r in range(rounds)]

    def response_rounds(self, step): return self.schedule
    def record(self, *args): pass


class DiagnosticWindow(BatchedSpatialWindow):
    def capture_inputs(self, mixed):
        result = super().capture_inputs(mixed)
        self.base_raw = result[:,:-1].detach()
        return result

    def candidate_batches(self, evaluated, entries, timings):
        if self.reuse_candidates:
            device=self.base_raw.device
            for batch in range(len(entries)):
                ids=[s for s in evaluated if int(self.positions[s,0])==batch]
                for first in range(0,len(ids),self.candidate_batch_size):
                    sources=ids[first:first+self.candidate_batch_size]
                    group=torch.tensor(sources,dtype=torch.long)
                    raw=torch.stack([self.cached_candidates[s][0] for s in sources]).to(device)
                    tiers=torch.stack([self.cached_candidates[s][1] for s in sources]).to(device)
                    yield group,batch,self.positions[group],raw,tiers
            return
        for group, batch, loc, raw, tiers in super().candidate_batches(evaluated,entries,timings):
            for i, source in enumerate(group.tolist()):
                self.cached_candidates[source]=(raw[i].detach().cpu(),tiers[i].detach().cpu())
                row = int(loc[i,1]); ids = self.scene.group_ids[batch][row]
                valid = ids>=0; tier = self.actions[source]['new_tier']
                changed = self.positions[ids[valid].cpu(),3]
                self.snapshots[source] = (changed,raw[i,tier,valid].detach().cpu())
            yield group,batch,loc,raw,tiers

    def candidate_views(self,evaluated,entries,captures,targets,labels,timings):
        self.cached_entries=entries
        for item in super().candidate_views(evaluated,entries,captures,targets,set(evaluated),timings):
            source, _, _, _, views = item
            predicted = torch.stack(self.learner.predict_group_views([[v['inputs'] for v in views]])[0]).mean(0).cpu()
            truth = torch.stack([v['values'].cpu() for v in views]).mean(0)
            zero = truth.new_zeros(1,2)
            p = torch.cat((zero,predicted)); t = torch.cat((zero,truth))
            old = int(self.table.q[source]); new = self.actions[source]['new_tier']
            self.records[source] = dict(old_tier=old,new_tier=new,
                predicted_image_delta=float((p[new]-p[old]).sum()),
                true_image_delta=float((t[new]-t[old]).sum()),
                true_mse_component_delta=float(t[new,0]-t[old,0]),
                true_structure_component_delta=float(t[new,1]-t[old,1]))
            yield item

    def intervention_raw(self,source,new_tier):
        raw = self.base_raw.clone()
        changed, alternate = self.snapshots[source]
        raw[changed.to(raw.device)] = alternate.to(raw)
        mask = (self.table.q[self.render_sources.to(self.table.q.device)]>0).to(raw)
        mask[self.positions[source,3]] = float(new_tier>0)
        return raw,mask


@torch.no_grad()
def evaluate_actions(model,scene,q,actions,cameras,learner,reference,args,*,seed,static=None):
    if not actions: raise ValueError('nonempty actions required')
    table = HardTierMap(q,next(model.parameters()).device)
    scheduler = FixedSources(scene.group_ids,[a['source'] for a in actions])
    window = DiagnosticWindow(model,scene,table,scheduler,reference,args)
    window.actions = {a['source']:a for a in actions}; window.snapshots = {}; window.records = {}
    window.cached_candidates={}; window.reuse_candidates=False
    window.learner = learner; window.static = static
    with preserved_rng(next(model.parameters()).device):
        seed_all(seed)
        result = window.collect(1,cameras,learner,NoReplay(),False)
    ids, costs = result['tier_ids'].tolist(),result['tier_costs'].cpu()
    for i,source in enumerate(ids):
        action = window.actions[source]; old,new = int(q[source]),action['new_tier']
        full = torch.cat((costs.new_zeros(1),costs[i]))
        delta = float(full[new]-full[old])
        window.records[source].update(predicted_rd_proxy_delta=delta,
            rate_proxy_delta=delta-window.records[source]['predicted_image_delta'])
    if set(window.records)!=set(window.actions): raise RuntimeError('missing paired action evaluations')
    return window.records,window


@torch.no_grad()
def evaluate_other_views(window,cameras):
    """Reuse the decoded baseline, four-state blocks and noise for view isolation."""
    device=window.base_raw.device
    old_records=window.records
    mask=(window.table.q[window.render_sources.to(device)]>0).to(window.base_raw)
    captures=[]; targets=[]
    for camera in cameras:
        tile=capture(window.base_raw,camera,window.scene.degree,mask,window.scene.white_background,backend=device.type)
        target=window.scene.reference.get(camera,device)
        window.prepare_camera(tile,target); captures.append(tile); targets.append(target)
    window.records={}; window.reuse_candidates=True
    timings=dict(candidate_decode_seconds=0.,projection_seconds=0.,feature_seconds=0.,teacher_seconds=0.,network_seconds=0.)
    for _ in window.candidate_views(list(window.actions),window.cached_entries,captures,targets,set(window.actions),timings): pass
    for source,row in window.records.items():
        rate=old_records[source]['rate_proxy_delta']
        row.update(rate_proxy_delta=rate,predicted_rd_proxy_delta=row['predicted_image_delta']+rate)
    return window.records
