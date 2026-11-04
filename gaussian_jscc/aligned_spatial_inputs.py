"""Original-PLY feature IDs and packet-order probes, used by new experiments."""
from functools import partial
import torch
from .hybrid_tier_benefit import HybridSpatialWindow,HybridPositiveProvider
from .physical_benefit_features import static_features,visibility_statistics


class AlignedSpatialWindow(HybridSpatialWindow):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.decode_batches=0

    @property
    def candidate_batch_size(self):
        size=getattr(self.args,'allocation_batch',None) or self.args.candidate_batch
        if size<1:raise ValueError('positive candidate batch required')
        return int(size)

    def prepare_camera(self,tile,target):
        if self.static is None:
            packet=static_features(self.scene.raw,self.args.neighbors,self.args.cpu_threads)
            self.static=torch.empty_like(packet);self.static[self.scene.order]=packet
        tile._physical_stats=visibility_statistics(tile,target)

    def capture_inputs(self,mixed):
        inactive=(mixed[:,-1]==0).nonzero(as_tuple=True)[0]
        if not len(inactive):return mixed
        result=mixed.clone()
        result[inactive,:-1]=self.scene.raw[inactive.cpu()].to(mixed)
        return result


class AlignedPositiveProvider(HybridPositiveProvider):
    def __init__(self,model,scene,table,scheduler,reference,args):
        super().__init__(model,scene,table,scheduler,reference,args)
        if not self.settings.get('aligned_spatial_inputs'):raise ValueError('aligned pretraining required')
        self.window=AlignedSpatialWindow(model,scene,table,self.window.scheduler,reference,self.window.args)


def pretrain(args,max_updates=None):
    from . import calibrated_spatial_benefit_allocation as old
    from .pixel_benefit_allocation import main
    from .spatial_benefit_replay import SpatialReplay
    old.spatial.validate_options(args)
    return main(args,max_updates,family='aligned_spatial_pretraining_v1',learner_factory=old.learner_factory,
        window_factory=AlignedSpatialWindow,preflight_fn=old.spatial.preflight,
        replay_factory=partial(SpatialReplay,cache_mib=args.replay_cache_mib))
