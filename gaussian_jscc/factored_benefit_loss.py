"""Cost-sensitive action direction, complete-group ranking, and factored regression."""
import torch
from torch.nn import functional as F


def _direction(predicted, truth, scale):
    weight=1+(truth.abs()/scale).clamp_max(10.)
    loss=weight*F.softplus(-truth.sign()*predicted/scale)
    return torch.where(truth.abs()>1e-14,loss,torch.zeros_like(loss))


def decision_loss_parts(prediction,target,rates,conditional,groups,scales,branches,common_scale,increment_scale):
    count=len(scales);sizes=torch.bincount(groups,minlength=count)
    if not torch.equal(sizes,torch.where(branches==1,3,1)):
        raise ValueError('decision supervision requires single positive actions and complete ordered keep groups')
    error=(prediction-target)/scales[groups,None]
    rows=F.smooth_l1_loss(error,torch.zeros_like(error),reduction='none').mean(-1)
    rows=rows+.25*F.smooth_l1_loss(error.sum(-1),torch.zeros_like(rows),reduction='none')
    numeric=torch.zeros_like(scales).scatter_add_(0,groups,rows)/sizes
    truth_cost=target.sum(-1)+rates;predicted_cost=prediction.sum(-1)+rates
    truth=torch.zeros_like(scales).scatter_add_(0,groups,truth_cost*conditional)
    predicted=torch.zeros_like(scales).scatter_add_(0,groups,predicted_cost*conditional)
    direction=_direction(predicted,truth,scales)
    ranking=torch.zeros_like(scales)
    keep=(branches==1).nonzero(as_tuple=True)[0]
    if len(keep):
        starts=(sizes.cumsum(0)-sizes)[keep]
        indices=starts[:,None]+torch.arange(3,device=groups.device)
        a,b=prediction[indices],target[indices]
        common_error=(a.mean(1)-b.mean(1))/common_scale
        gap_error=(a.diff(dim=1)-b.diff(dim=1))/increment_scale
        numeric=numeric.index_copy(0,keep,
            F.smooth_l1_loss(common_error,torch.zeros_like(common_error),reduction='none').mean(-1)+
            F.smooth_l1_loss(gap_error,torch.zeros_like(gap_error),reduction='none').mean((-1,-2)))
        left=torch.tensor([0,0,1],device=groups.device);right=torch.tensor([1,2,2],device=groups.device)
        actual=truth_cost[indices];estimated=predicted_cost[indices]
        differences=actual[:,left]-actual[:,right]
        losses=_direction(estimated[:,left]-estimated[:,right],differences,scales[keep,None])
        denominator=(differences.abs()>1e-14).sum(-1).clamp_min(1)
        ranking=ranking.index_copy(0,keep,losses.sum(-1)/denominator)
    return dict(numeric=numeric,direction=direction,ranking=ranking,total=.25*numeric+direction+ranking)
