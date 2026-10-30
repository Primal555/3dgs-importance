"""Independent spatially paired benefit learning and frozen-codec allocation."""
from functools import partial
import math

from . import pixel_benefit_allocation as base
from .physical_benefit_allocation import parser as physical_parser, preflight as physical_preflight
from .spatial_benefit import SpatialLearner
from .spatial_benefit_features import SpatialWindow
from .spatial_benefit_replay import SpatialReplay

FAMILY = 'spatial_benefit_allocation_v1'


def parser():
    p = physical_parser(); p.description = __doc__
    p.set_defaults(net_updates=64, batch_groups=8, decision_weight=.25)
    p.add_argument('--common-weight', type=float, default=1.)
    p.add_argument('--pair-weight', type=float, default=1.)
    p.add_argument('--replay-cache-mib', type=float, default=256.)
    p.add_argument('--training-probe-groups', type=int, default=256)
    return p


def learner_factory(replay, device, args):
    return SpatialLearner(replay, device, hidden=args.hidden, lr=args.benefit_lr,
        dropout=args.dropout, weight_decay=args.weight_decay, decision_weight=args.decision_weight,
        tie_fraction=args.tie_fraction, common_weight=args.common_weight, pair_weight=args.pair_weight,
        training_probe_groups=args.training_probe_groups)


def verify_spatial_inputs(device):
    import torch
    from .pixel_local_renderer import TileScene
    from .physical_benefit_features import visibility_statistics
    from .spatial_benefit_features import spatial_inputs
    p = torch.tensor([[0,0,1,1,0,1,.6,1,.2,.3,.4,3],
        [15,8,2,.1,.02,.2,.8,1,.6,.2,.7,8],
        [30,16,3,.2,0,.1,.5,0,.1,.9,.3,6]], dtype=torch.float32)
    raw = torch.zeros(4,3,14); raw[...,7]=1
    projected = p[None].repeat(4,1,1); projected[0,0,7]=0
    projected[2,1,8:11]+=.1; projected[3,1,:2]+=.75
    target = torch.linspace(0,1,3*17*31).reshape(3,17,31)
    results=[]
    for selected in ('cpu', device):
        tile=TileScene.from_projected(p.to(selected),17,31,backend=torch.device(selected).type)
        target_view=target.to(selected)
        stats=visibility_statistics(tile,target_view)
        inputs=spatial_inputs(torch.zeros(24),stats,raw.to(selected),projected.to(selected),0,
            tile,target_view,torch.arange(3,device=selected),.01,.006,cache_key='preflight')
        results.append({k:v.cpu() for k,v in inputs.items()})
    maximum=0.
    for key in results[0]:
        torch.testing.assert_close(results[0][key],results[1][key],rtol=2e-4,atol=2e-4)
        maximum=max(maximum,float((results[0][key]-results[1][key]).abs().max()))
    return dict(spatial_inputs_cpu_device_max_abs=maximum,spatial_inputs_backend=torch.device(device).type)


def preflight(model, scene, camera, args, reference):
    result=physical_preflight(model,scene,camera,args,reference)
    result.update(verify_spatial_inputs(next(model.parameters()).device))
    return result


def validate_options(args):
    for name in ('decision_weight', 'tie_fraction', 'weight_decay', 'dropout',
                 'common_weight', 'pair_weight', 'replay_cache_mib'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid {name}')
    if args.dropout >= 1 or args.hidden < 2 or args.neighbors < 1 or args.training_probe_groups < 1:
        raise ValueError('invalid network or probe settings')


def main(args, max_updates=None):
    validate_options(args)
    return base.main(args, max_updates, family=FAMILY, learner_factory=learner_factory,
        window_factory=SpatialWindow, preflight_fn=preflight,
        replay_factory=partial(SpatialReplay, cache_mib=args.replay_cache_mib))


if __name__ == '__main__':
    main(parser().parse_args())
