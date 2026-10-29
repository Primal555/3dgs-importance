"""Independent physical-feature benefit prediction and frozen-codec allocation."""
import math

import torch

from . import pixel_benefit_allocation as base
from .physical_benefit import PhysicalLearner
from .physical_benefit_features import PhysicalWindow, visibility_statistics

FAMILY='physical_benefit_allocation_v1'


def parser():
    p=base.parser(); p.description=__doc__; p.set_defaults(hidden=64)
    p.add_argument('--neighbors',type=int,default=32)
    p.add_argument('--decision-weight',type=float,default=1.)
    p.add_argument('--tie-fraction',type=float,default=.02)
    p.add_argument('--dropout',type=float,default=.05)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    return p


def learner_factory(replay,device,args):
    return PhysicalLearner(replay,device,hidden=args.hidden,lr=args.benefit_lr,dropout=args.dropout,
        weight_decay=args.weight_decay,decision_weight=args.decision_weight,tie_fraction=args.tie_fraction)


def verify_visibility(device):
    from .pixel_local_renderer import TileScene
    # Includes a disabled point, strong occlusion, image edges and multiple tiles.
    p=torch.tensor([[0,0,1,1,0,1,.6,1,1,0,0,3],
        [0,0,2,1,0,1,.8,0,0,1,0,3],
        [15,15,1,.1,.02,.2,.99,1,1,1,0,8],
        [15,15,2,.1,.02,.2,.99,1,0,1,1,8],
        [15,15,3,.1,.02,.2,.99,1,0,0,1,8],
        [15,15,4,.1,.02,.2,.99,1,1,0,1,8]],dtype=torch.float32)
    cpu=TileScene.from_projected(p,17,31,backend='cpu')
    target=torch.linspace(0,1,3*17*31).reshape(3,17,31)
    expected=visibility_statistics(cpu,target)
    actual_scene=TileScene.from_projected(p.to(device),17,31,backend=torch.device(device).type)
    actual=visibility_statistics(actual_scene,target.to(device)).cpu()
    torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-4)
    return dict(visibility_cpu_cuda_max_abs=float((actual-expected).abs().max()),
                visibility_backend=torch.device(device).type)


def preflight(model,scene,camera,args,reference):
    result=base.preflight(model,scene,camera,args,reference)
    result.update(verify_visibility(next(model.parameters()).device))
    return result


def main(args,max_updates=None):
    for name in ('decision_weight','tie_fraction','weight_decay','dropout'):
        value=getattr(args,name)
        if not math.isfinite(value) or value<0: raise ValueError(f'invalid {name}')
    if args.dropout>=1 or args.hidden<2 or args.neighbors<1:
        raise ValueError('invalid network or neighbor settings')
    return base.main(args,max_updates,family=FAMILY,learner_factory=learner_factory,
                     window_factory=PhysicalWindow,preflight_fn=preflight)


if __name__=='__main__':
    main(parser().parse_args())
