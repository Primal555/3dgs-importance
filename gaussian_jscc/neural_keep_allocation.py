"""Independent 700+2000 learned dense-keep and spatial positive-tier experiment."""
import math
from . import hybrid_tier_allocation as pipeline
from .neural_dense_keep import FAMILY,NeuralKeepTable
from .neural_keep_provider import NeuralAllocationProvider
from .aligned_spatial_inputs import pretrain,AlignedPositiveProvider
from .hybrid_tier_benefit import FAMILY as HYBRID_FAMILY

KEEP_SETTINGS=('keep_hidden','keep_network_lr','keep_blend','keep_chunk','keep_calibration_weight')


def parser():
    p=pipeline.parser();p.description=__doc__
    p.set_defaults(aligned_spatial_inputs=True)
    p.add_argument('--keep-estimator',choices=('neural','first_order'),default='neural')
    p.add_argument('--keep-hidden',type=int,default=64)
    p.add_argument('--keep-network-lr',type=float,default=1e-4)
    p.add_argument('--keep-blend',type=float,default=.1)
    p.add_argument('--keep-chunk',type=int,default=65536)
    p.add_argument('--keep-calibration-weight',type=float,default=.1)
    return p


def configure(options,args):
    for key in KEEP_SETTINGS:setattr(options,key,getattr(args,key))
    options.keep_estimator=args.keep_estimator;options.aligned_spatial_inputs=True


def main(args,max_updates=None):
    if (min(args.keep_hidden,args.keep_chunk)<1 or not 0<args.keep_blend<=1 or
            any(not math.isfinite(v) or v<=0 for v in (args.keep_network_lr,args.keep_calibration_weight))):
        raise ValueError('invalid learned keep settings')
    neural=args.keep_estimator=='neural'
    return pipeline.main(args,max_updates,provider_factory=NeuralAllocationProvider if neural else AlignedPositiveProvider,
        family=FAMILY if neural else HYBRID_FAMILY,table_factory=NeuralKeepTable if neural else None,
        allocation_transform=configure,pretrain_factory=pretrain)


if __name__=='__main__':main(parser().parse_args())
