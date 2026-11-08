"""Paired tier decisions, including ordinary records and concentrated gains."""
import torch


def compare_choices(groups, reference, candidate):
    if not groups or len(groups) != len(reference) or len(groups) != len(candidate):
        raise ValueError('matching nonempty decision records required')
    costs = torch.stack([torch.stack([v['values'].double().cpu() for v in g['views']]).mean(0).sum(-1)+
                         torch.as_tensor(g['rates']).double().cpu() for g in groups])
    def choices(predictions):
        return torch.stack([torch.as_tensor(p).double().cpu().mean(0).sum(-1)+
                            torch.as_tensor(g['rates']).double().cpu() for g,p in zip(groups,predictions)]).argmin(-1)
    a,b=choices(reference),choices(candidate)
    idx=torch.arange(len(groups));best=costs.min(-1).values
    ra,rb=costs[idx,a]-best,costs[idx,b]-best
    delta=rb-ra;gain=delta < -1e-14;harm=delta > 1e-14
    changed=a!=b
    def case(i):
        return dict(index=i,source=groups[i]['source'],step=groups[i]['step'],
            reference_tier=int(a[i])+1,candidate_tier=int(b[i])+1,true_best_tier=int(costs[i].argmin())+1,
            reference_regret=float(ra[i]),candidate_regret=float(rb[i]),delta_regret=float(delta[i]))
    def subset(mask):
        return dict(records=int(mask.sum()),reference_regret=float(ra[mask].mean()) if mask.any() else None,
                    candidate_regret=float(rb[mask].mean()) if mask.any() else None,
                    reference_best_hit_fraction=float((ra[mask].abs()<=1e-14).double().mean()) if mask.any() else None,
                    candidate_best_hit_fraction=float((rb[mask].abs()<=1e-14).double().mean()) if mask.any() else None)
    beneficial=delta.argsort()[gain[delta.argsort()]].tolist()
    harmful=delta.argsort(descending=True)[harm[delta.argsort(descending=True)]].tolist()
    active=torch.tensor([any(bool(v['values'].double().abs().max()>1e-14) for v in g['views']) for g in groups])
    result=dict(records=len(groups),changed_choices=int(changed.sum()),beneficial_changes=int(gain.sum()),
        harmful_changes=int(harm.sum()),neutral_changes=int((changed & ~gain & ~harm).sum()),
        total_benefit=float(-delta[gain].sum()),total_harm=float(delta[harm].sum()),
        net_benefit=float(-delta.sum()),top_benefits=[case(i) for i in beneficial[:10]],
        top_harms=[case(i) for i in harmful[:10]],active=subset(active),zero_change=subset(~active))
    for count in (1,3):
        mask=torch.ones(len(groups),dtype=torch.bool)
        removed=beneficial[:count];mask[removed]=False
        key='excluding_largest_improvement' if count==1 else 'excluding_top3_improvements'
        result[key]=dict(subset(mask),removed=[case(i) for i in removed])
    mask=torch.ones(len(groups),dtype=torch.bool);worst=int(ra.argmax());mask[worst]=False
    result['excluding_reference_worst']=dict(subset(mask),removed=case(worst))
    return result
