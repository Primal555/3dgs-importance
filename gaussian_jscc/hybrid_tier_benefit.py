"""Spatial positive-tier values over the dense first-order keep kernel's scene."""
import copy
import hashlib
import math
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from .allocation_benefit_features import keep_rates
from .allocation_benefit_replay import cpu_copy
from .batched_spatial_window import BatchedSpatialWindow
from .budgeted_allocation_kernel import set_rng
from .calibrated_spatial_benefit import CalibratedSpatialLearner
from .optimization import preserved_rng
from .pixel_local_renderer import capture
from .response_value_window import ResponseScheduler
from .spatial_benefit_replay import SpatialReplay
from .transport import model_id

FAMILY = 'dense_keep_neural_tiers_v1'


def network_digest(network):
    h = hashlib.sha256()
    for name,value in sorted(network.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def clone_replay(state, destination):
    """Independent ownership and RNG; immutable initial shards may share disk blocks."""
    saved = copy.deepcopy(state); source = Path(saved['root']); destination = Path(destination).resolve()
    if source == destination: raise ValueError('online replay must be independent of pretraining')
    for kind in ('training','validation'):
        for entry in saved[kind]:
            old, new = source/entry['path'], destination/entry['path']
            if not old.is_file() or old.stat().st_size != entry['size']: raise FileNotFoundError(old)
            new.parent.mkdir(parents=True,exist_ok=True)
            if not new.exists():
                try: os.link(old,new)
                except OSError: shutil.copyfile(old,new)
            if new.stat().st_size != entry['size']: raise ValueError('cloned replay shard mismatch')
            if kind == 'training': new.with_suffix('.owned').write_text(entry['id'],encoding='ascii')
    saved['root'] = str(destination)
    return saved


class HybridSpatialWindow(BatchedSpatialWindow):
    def candidate_batches(self,*args,**kwargs):
        for batch in super().candidate_batches(*args,**kwargs):
            self.decode_batches += 1
            yield batch


class HybridPositiveProvider:
    def __init__(self, model, scene, table, scheduler, reference, args):
        self.model,self.scene,self.reference,self.args = model,scene,reference,args
        device = next(model.parameters()).device
        artifact = torch.load(args.benefit_checkpoint,map_location='cpu',weights_only=True)
        if (artifact.get('version') != FAMILY or artifact['codec_id'] != model_id(model)
                or artifact['scene_fingerprint'] != scene.fingerprint
                or any(artifact['reference'][k] != reference[k] for k in ('d_ref','c_ref','lambda_s'))):
            raise ValueError('pretrained benefit codec/scene/objective mismatch')
        self.frozen = args.network_mode == 'frozen'
        self.identity = artifact['network_sha256']
        self.settings = artifact['settings']
        aligned=bool(self.settings.get('aligned_spatial_inputs',False))
        if aligned!=bool(getattr(args,'aligned_spatial_inputs',False)):
            raise ValueError('spatial input layout mismatch between allocation and benefit network')
        replay_state = artifact['replay']
        self.replay = None
        if not self.frozen:
            self.replay = SpatialReplay(Path(args.out)/'neural_replay',max_groups=replay_state['max_groups'],
                validation_groups=replay_state['validation_groups'],cache_mib=self.settings['replay_cache_mib'])
            self.initial_replay = replay_state
        self.learner = CalibratedSpatialLearner(self.replay,device,hidden=self.settings['hidden'],
            lr=self.settings['benefit_lr'],dropout=self.settings['dropout'],
            weight_decay=self.settings['weight_decay'],decision_weight=self.settings['decision_weight'],
            tie_fraction=self.settings['tie_fraction'],common_weight=self.settings['common_weight'],
            pair_weight=self.settings['pair_weight'],training_probe_groups=self.settings['training_probe_groups'])
        self.learner.load_state_dict(artifact['learner'])
        if network_digest(self.learner.network) != self.identity: raise ValueError('benefit weights digest mismatch')
        self.initial_updates = self.learner.updates
        self.learner.network.eval().requires_grad_(not self.frozen)
        self.generator = torch.Generator().manual_seed(args.seed+77117)
        window_args = SimpleNamespace(**self.settings)
        window_args.allocation_batch = args.allocation_batch
        window_args.out = args.out
        window_class=HybridSpatialWindow
        if aligned:
            from .aligned_spatial_inputs import AlignedSpatialWindow
            window_class=AlignedSpatialWindow
        self.window = window_class(model,scene,table,ResponseScheduler(scene.group_ids,seed=args.seed),
                                           reference,window_args)
        self.resumed = False

    def initialize_replay(self):
        if self.replay is not None and not self.resumed:
            self.replay.load_state_dict(clone_replay(self.initial_replay,self.replay.root))
        self.resumed = True

    def state_dict(self):
        unchanged = network_digest(self.learner.network) == self.identity
        if self.frozen and not unchanged: raise RuntimeError('frozen benefit network changed')
        return dict(version=FAMILY,network_mode=self.args.network_mode,initial_network_sha256=self.identity,
            initial_network_updates=self.initial_updates,network_updates=self.learner.updates,
            frozen_network_verified=unchanged if self.frozen else None,
            learner=cpu_copy(self.learner.state_dict()) if not self.frozen else None,
            generator=self.generator.get_state())

    def load_state_dict(self,state):
        if (state['version']!=FAMILY or state['network_mode']!=self.args.network_mode
                or state['initial_network_sha256']!=self.identity): raise ValueError('hybrid benefit state mismatch')
        if not self.frozen: self.learner.load_state_dict(state['learner'])
        self.generator.set_state(state['generator'].cpu()); self.resumed=True

    def observe_context(self,scene,q,entries,captures,targets,task):
        pass

    @torch.no_grad()
    def predict(self,task,cache,scene,q,tier_ids,locations,*,side_proxy,step):
        started=time.perf_counter(); device=scene.device
        parts=torch.zeros(len(tier_ids),3,2,device=device,dtype=torch.float64)
        stats=dict(neural_candidates=len(tier_ids),network_mode=self.args.network_mode,
                   neural_labels=0,neural_teacher_seconds=0.,network_updates=self.learner.updates,
                   neural_candidate_decode_calls=0)
        if not len(tier_ids): return parts,stats
        if self.replay is not None: self.initialize_replay()
        entries=[]
        for features,ids,entry in zip(self.scene.groups,self.scene.group_ids,cache):
            f=features.to(device); source_ids=ids.to(device); tiers,_,y,actual,rng=entry
            with preserved_rng(device):
                set_rng(rng,device)
                noise=torch.randn((*tiers.shape,2*self.model.cfg.rates[-1]),device=device,dtype=f.dtype)
                noise*=math.sqrt(10**(-float(self.args.snr)/10)/2)
            entries.append((f,source_ids,tiers,y,actual,noise))
        captures=[]; targets=[]
        mixed=self.window.capture_inputs(scene)
        for camera in task.cameras:
            tile=capture(mixed[:,:-1],camera,self.scene.degree,mixed[:,-1],self.scene.white_background,
                         backend=device.type)
            target=self.scene.reference.get(camera,device)
            self.window.prepare_camera(tile,target); captures.append(tile); targets.append(target)
        self.observe_context(scene,q,entries,captures,targets,task)
        labels=set()
        if not self.frozen:
            training=tier_ids[~self.window.scheduler.validation[tier_ids]]
            order=torch.randperm(len(training),generator=self.generator)[:self.settings['label_groups']]
            labels=set(training[order].tolist())
        positions={int(source):i for i,source in enumerate(tier_ids.tolist())}
        pending=[]; groups=[]; timings=dict(candidate_decode_seconds=0.,projection_seconds=0.,feature_seconds=0.,
            teacher_seconds=0.,candidate_projection_calls=0,spatial_feature_batches=0)
        def flush():
            if not pending:return
            predictions=self.learner.predict_group_views([[v['inputs'] for v in views] for _,views,_ in pending])
            for (source,views,rate),predicted in zip(pending,predictions):
                values=torch.stack(predicted).mean(0)
                delta=values-values[int(q[source])-1][None]
                if self.reference['lambda_s']:
                    delta=delta.clone();delta[:,1]/=self.reference['lambda_s']
                else: delta=delta.clone();delta[:,1]=0
                parts[positions[source]]=delta
                if source in labels:
                    groups.append(dict(source=source,step=step,views=cpu_copy(views),rates=rate.cpu(),importance_weight=1.))
            pending.clear()
        # Same actual q/noise/cache as dense keep. Candidate projection is batched across all four states.
        self.window.decode_batches=0
        for source,batch,row,col,views in self.window.candidate_views(
                tier_ids.tolist(),entries,captures,targets,labels,timings):
            timings['teacher_seconds'] += sum(v.get('teacher_seconds',0.) for v in views)
            tiers=entries[batch][2];valid=entries[batch][1]>=0
            marginal=keep_rates(tiers,valid,self.model.cfg.rates,xyz=side_proxy['xyz_per_retained'],
                bits=side_proxy.get('bits_per_use',2.))[row,col]*self.args.beta/self.reference['c_ref']
            pending.append((source,views,marginal))
            if len(pending)>=self.args.allocation_batch: flush()
        flush()
        if not self.frozen:
            for group in groups:self.replay.add(group,False)
            with preserved_rng(device),torch.enable_grad():
                learned=self.learner.train_updates(self.settings['net_updates'],self.settings['batch_groups'])
            stats.update({'neural_'+k:v for k,v in learned.items()})
        self.learner.network.eval()
        stats.update(neural_labels=len(groups),neural_teacher_seconds=timings['teacher_seconds'],
            neural_total_seconds=time.perf_counter()-started,network_updates=self.learner.updates,
            neural_projection_calls=timings['candidate_projection_calls'],
            neural_feature_batches=timings['spatial_feature_batches'],
            neural_candidate_decode_calls=3*self.window.decode_batches,
            neural_candidate_semantic_calls=self.window.decode_batches,
            neural_candidate_decode_seconds=timings['candidate_decode_seconds'])
        return parts,stats
