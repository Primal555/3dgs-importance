"""Bounded coverage, common-condition ranking and disjoint RD acceptance."""
from collections import deque
import math
import time

import torch


def shortlist(proposal, limit):
    """Spend 3/4 on leaders; cover the remaining ranks with the other 1/4."""
    if limit < 1: raise ValueError('positive shortlist limit required')
    ids = proposal['ids'].cpu()
    if len(ids) <= limit: return ids.clone()
    head = max(1, limit-limit//4)
    tail = torch.linspace(head, len(ids)-1, limit-head).round().long()
    return torch.cat((ids[:head], ids[tail]))


def ranked_actions(ids, positive_costs, q):
    ids = ids.cpu().long(); costs = positive_costs.cpu().double()
    if costs.shape != (len(ids), 3) or not torch.isfinite(costs).all():
        raise ValueError('finite three-positive-tier costs required')
    costs = torch.cat((costs.new_zeros(len(ids), 1), costs), -1)
    best = costs.argmin(-1)
    gains = costs[torch.arange(len(ids)), q.cpu()[ids]]-costs.amin(-1)
    selected = torch.argsort(gains, descending=True, stable=True)
    selected = selected[gains[selected] > 0]
    return dict(ids=ids[selected], tiers=best[selected], gains=gains[selected])


def audit_chunks(q, proposal, baseline, measure, *, chunk_size=16, max_checks=12, min_gain=1e-6):
    if min(chunk_size,max_checks)<1 or not math.isfinite(min_gain) or min_gain<0:
        raise ValueError('invalid local audit budget')
    if len(proposal['ids'].unique()) != len(proposal['ids']):
        raise ValueError('duplicate source in audit')
    verified=proposal.get('verified')
    if verified is None:
        queue=deque((start,min(start+chunk_size,len(proposal['ids'])),False)
                    for start in range(0,len(proposal['ids']),chunk_size))
    else:
        n=int(verified.sum())
        if not verified[:n].all() or verified[n:].any():
            raise ValueError('teacher-confirmed actions must precede network-only actions')
        queue=deque((start,min(start+chunk_size,n),True) for start in range(0,n,chunk_size))
        # Keep verified actions separate. Fit network-only roots into the same
        # check budget, leaving room to split failed roots breadth-first.
        roots=max(1,max_checks-len(queue)-min(4,max_checks//3))
        width=max(chunk_size,math.ceil((len(proposal['ids'])-n)/roots))
        queue.extend((start,min(start+width,len(proposal['ids'])),False)
                     for start in range(n,len(proposal['ids']),width))
    current = q.clone(); measurement = baseline; attempts = []; commits = []
    if not math.isfinite(float(baseline['rd_score'])): raise ValueError('nonfinite baseline RD')
    while queue and len(attempts)<max_checks:
        start,end,is_verified = queue.popleft()
        ids,tiers = proposal['ids'][start:end],proposal['tiers'][start:end]
        candidate = current.clone(); candidate[ids] = tiers.to(candidate)
        row = measure(candidate); before = float(measurement['rd_score']); after = float(row['rd_score'])
        if not math.isfinite(after): raise ValueError('nonfinite candidate RD')
        accepted = before-after>=min_gain and after<before
        attempt = dict(changes=len(ids), ids=ids.tolist(), tiers=tiers.tolist(),verified=is_verified,
            predicted_gain=float(proposal['gains'][start:end].sum()),
            before_rd=before,rd_score=after,rd_delta=after-before,accepted=accepted)
        attempts.append(attempt)
        if accepted:
            current,measurement = candidate,row
            commits.append(dict(before_rd=before,after_rd=after,changes=len(ids)))
        elif end-start>1:
            middle=(start+end)//2
            # All independent root batches are visited before failed batches split.
            queue.extend(((start,middle,is_verified),(middle,end,is_verified)))
    return dict(q=current,measurement=measurement,accepted=bool(commits),
        changed=int((current!=q).sum()),attempts=attempts,commits=commits,
        deferred_batches=len(queue),budget_exhausted=bool(queue))


def matched_audit(window,bank,q,blocks,learner,seeds,cameras,meter,baseline,args):
    """A fixed budget of common-condition predictions, exact labels and checks."""
    proposal=bank.propose(q,blocks);ids=shortlist(proposal,args.rescore_groups)
    started=time.perf_counter()
    costs=window.score(ids,learner,seeds,cameras)
    reranked=ranked_actions(ids,costs,q)
    rescore_seconds=time.perf_counter()-started
    exact_ids=shortlist(reranked,args.exact_groups)
    started=time.perf_counter()
    truth=window.exact_costs(exact_ids,seeds,cameras)
    confirmed=ranked_actions(exact_ids,truth,q)
    unlabeled=~torch.isin(reranked['ids'],exact_ids)
    execution={k:torch.cat((confirmed[k],reranked[k][unlabeled])) for k in ('ids','tiers','gains')}
    execution['verified']=torch.cat((torch.ones(len(confirmed['ids']),dtype=torch.bool),
                                     torch.zeros(int(unlabeled.sum()),dtype=torch.bool)))
    lookup={int(s):i for i,s in enumerate(ids)}
    predicted=costs[torch.tensor([lookup[int(s)] for s in exact_ids],dtype=torch.long)]
    pred4=torch.cat((predicted.new_zeros(len(exact_ids),1),predicted),-1)
    true4=torch.cat((truth.new_zeros(len(exact_ids),1),truth),-1)
    choice=pred4.argmin(-1);index=torch.arange(len(exact_ids));old=q[exact_ids]
    chosen_truth=true4[index,choice]-true4[index,old]
    exact_seconds=time.perf_counter()-started
    started=time.perf_counter()
    shared_base=window.measure(q,seeds,cameras,meter)
    if abs(shared_base['rd_score']-baseline['rd_score'])>1e-5:
        raise RuntimeError('matched background disagrees with saved calibration RD')
    audited=audit_chunks(q,execution,shared_base,
        lambda tiers:window.measure(tiers,seeds,cameras,meter),
        chunk_size=args.audit_chunk,max_checks=args.max_local_checks,min_gain=args.min_rd_gain)
    audited.update(proposed=len(proposal['ids']),rescored=len(ids),
        reranked_positive=len(reranked['ids']),exact_groups=len(exact_ids),
        confirmed_actions=len(confirmed['ids']),execution_candidates=len(execution['ids']),
        network_only_actions=int(unlabeled.sum()),rescore_seconds=rescore_seconds,
        exact_seconds=exact_seconds,local_audit_seconds=time.perf_counter()-started,
        predicted_actions_rejected_by_teacher=int((chosen_truth>0).sum()),
        teacher_changed_best_tier=int((choice!=true4.argmin(-1)).sum()),
        exact_rd_prediction_mae=float((pred4-true4).abs().mean()) if len(exact_ids) else None,
        common_condition_predicted_gain=float(reranked['gains'].sum()),
        confirmed_single_gain=float(confirmed['gains'].sum()),
        single_rate_scope='existing marginal-rate proxy; joint acceptance uses exact complete packet cost')
    return audited
