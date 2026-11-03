"""Two learned allocation heads; shared decoded scene and visibility context."""
import torch
from .aligned_spatial_inputs import AlignedPositiveProvider
from .neural_dense_keep import DenseKeepPolicy
from .physical_benefit_features import static_features
from .pixel_local_renderer import capture


class NeuralAllocationProvider(AlignedPositiveProvider):
    def __init__(self,model,scene,table,scheduler,reference,args):
        super().__init__(model,scene,table,scheduler,reference,args)
        static=static_features(scene.raw,self.settings['neighbors'],args.cpu_threads)
        self.window.static=torch.empty_like(static);self.window.static[scene.order]=static
        self.keep_policy=DenseKeepPolicy(model,scene,table,static,args,self.learner.scales['rd'],
                                         self.window.positions,self.window.scheduler.validation)

    def state_dict(self):
        return dict(super().state_dict(),keep_policy=self.keep_policy.state_dict())

    def load_state_dict(self,state):
        super().load_state_dict(state);self.keep_policy.load_state_dict(state['keep_policy'])

    def observe_context(self,scene,q,entries,captures,targets,task):
        self.keep_policy.observe(scene,self.window.render_sources,captures)
        self.keep_policy.set_tiers(q)

    @torch.no_grad()
    def predict(self,task,cache,scene,q,tier_ids,locations,**kwargs):
        if not len(tier_ids):
            mixed=self.window.capture_inputs(scene);captures=[]
            for camera in task.cameras:
                tile=capture(mixed[:,:-1],camera,self.scene.degree,mixed[:,-1],self.scene.white_background,
                             backend=scene.device.type)
                self.window.prepare_camera(tile,self.scene.reference.get(camera,scene.device));captures.append(tile)
            self.observe_context(scene,q,[],captures,[],task)
        return super().predict(task,cache,scene,q,tier_ids,locations,**kwargs)

    def result_fields(self):
        return dict(keep_estimator='dense_neural_gate',keep_network_updates=self.keep_policy.updates,
                    keep_network_trained=True,dense_keep_rows=self.keep_policy.count)
