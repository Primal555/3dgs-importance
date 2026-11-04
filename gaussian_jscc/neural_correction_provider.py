"""Delayed online learning from complete teachers and exact action corrections."""
import hashlib
import time
import torch
from .allocation_benefit_replay import cpu_copy
from .hybrid_tier_benefit import HybridPositiveProvider
from .neural_correction_history import NetworkAwareEvidence
from .neural_correction_learning import CorrectionLearner
from .optimization import preserved_rng
from .pixel_benefit_learning import summarize_pixel_values
from .render_validation import append_json

POLICY='current_network_context_residual_v2'


class NetworkCorrectionProvider(HybridPositiveProvider):
    learner_class=CorrectionLearner

    def __init__(self,model,scene,table,scheduler,reference,args):
        if not isinstance(scheduler,NetworkAwareEvidence) or args.network_mode!='online':
            raise ValueError('network corrections require online learning and network-aware evidence')
        super().__init__(model,scene,table,scheduler,reference,args)
        self.scheduler=scheduler;self.bank=scheduler.positive_corrections
        self.bank.bind(self.learner,self.replay,reference,args.allocation_batch)
        self.groups=[];self.step_data=None

    def state_dict(self):return dict(super().state_dict(),correction_policy=POLICY)

    def load_state_dict(self,state):
        if state.get('correction_policy')!=POLICY:raise ValueError('network correction policy mismatch')
        super().load_state_dict(state)

    def result_fields(self):
        return dict(neural_correction_policy=POLICY,positive_history_policy='current_network_revision',
                    positive_residual_policy='matched_context_recomputed_teacher_gap')

    def prepare_step(self,step):
        self.scheduler.model_revision=self.learner.updates

    def observe_context(self,scene,q,entries,captures,targets,task):
        self.step_data=(entries,captures,targets)
        self.block_context={}
        for b,entry in enumerate(entries):
            tiers=entry[2].cpu().numpy().astype('uint8')
            for row,values in enumerate(tiers):
                self.block_context[b,row]=hashlib.blake2b(values.tobytes(),digest_size=16).digest()
        self.view_order=sorted(range(len(task.cameras)),key=lambda i:self.scheduler.current_views[i])

    def observe_predictions(self,pending,predictions):
        # Transfer only 24 visibility descriptors per view, once per batch.
        descriptor=torch.stack([torch.stack([views[i]['inputs']['features'][0,24:48]
            for i in self.view_order]) for _,views,_ in pending])
        net=self.learner.network
        descriptor=torch.asinh((descriptor-net.input_mean[24:48])/net.input_scale[24:48]).cpu()
        for (source,_,_),value in zip(pending,descriptor):
            self.visibility[source]=value
            batch,row=self.window.positions[source,:2].tolist()
            self.blocks[source]=self.block_context[batch,row]

    def train_groups(self,groups):
        self.groups=groups
        return {}

    def predict(self,*args,**kwargs):
        self.groups=[];self.step_data=None;self.visibility={};self.blocks={}
        result=super().predict(*args,**kwargs)
        self.bank.set_context(tuple(sorted(self.scheduler.current_views)),self.blocks,self.visibility)
        return result

    @torch.no_grad()
    def observe_exact(self,records,step):
        started=time.perf_counter()
        eligible=[r for r in records if r['feedback_kind']=='positive' and
                  not self.window.scheduler.validation[r['source_row']]]
        # A bounded replay retains complete-state teachers alongside action gaps.
        eligible=eligible[:self.replay.max_groups//4]
        inputs={g['source']:g['views'] for g in self.groups}
        missing=[r['source_row'] for r in eligible if r['source_row'] not in inputs]
        decode_before=self.window.decode_batches
        timings=dict(candidate_decode_seconds=0.,projection_seconds=0.,feature_seconds=0.,teacher_seconds=0.)
        if missing:
            entries,captures,targets=self.step_data
            for source,_,_,_,views in self.window.candidate_views(missing,entries,captures,targets,set(),timings):
                inputs[source]=cpu_copy(views)
        for group in self.groups:self.replay.add(group,False)
        for r in eligible:
            source=r['source_row'];old=r['old_tier'];new=r['new_tier']
            group=dict(source=source,step=step,supervision='action_gap',old_tier=old,new_tier=new,
                views=inputs[source],delta_values=torch.tensor([r['measured_delta_mse'],
                    r['measured_delta_structure']*self.reference['lambda_s']],dtype=torch.float64)/self.reference['d_ref'],
                rate_delta=r['rate_delta_cost'],importance_weight=1.)
            record_id=self.replay.add(group,False)
            self.bank.stage(source*9+(old-1)*3+new-1,record_id,step)
        self.bank.prune()
        return dict(neural_exact_training_groups=len(eligible),
            neural_exact_reconstructed_groups=len(missing),neural_exact_input_seconds=time.perf_counter()-started,
            neural_exact_candidate_decode_calls=3*(self.window.decode_batches-decode_before),
            neural_exact_extra_teacher_renders=0)

    def finish_step(self,step):
        if self.step_data is None:return dict(network_updates=self.learner.updates)
        with preserved_rng(next(self.model.parameters()).device),torch.enable_grad():
            stats=self.learner.train_updates(self.settings['net_updates'],self.settings['batch_groups'])
        self.learner.network.eval();self.bank.clear_predictions()
        result={'neural_'+k:v for k,v in stats.items()}
        result.update(network_updates=self.learner.updates,neural_history_invalidated_rows=self.scheduler.reset_rows,
            neural_residual_records=len(self.bank.records),neural_prediction_revision=self.scheduler.model_revision)
        if step%self.args.validate_every==0:
            rows,weighted=self.learner._report_groups(self.replay.validation_records())
            report=summarize_pixel_values(rows)
            report.update(step=step,network_updates=self.learner.updates,weighted_validation=weighted,
                active_decision='q1/q2/q3 differences; q0 network metrics are auxiliary',
                scope='fixed held-out pretraining contexts; no training or on-policy replacement')
            append_json(self.args.out+'/metrics/network_validation.jsonl',report)
        self.groups=[];self.step_data=None
        return result
