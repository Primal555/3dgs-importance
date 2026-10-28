"""Audit cached benefit labels and continue only the evaluator on a fixed dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .allocation_benefit import BenefitNetwork
from .allocation_benefit_learning import BenefitLearner, pack_groups
from .allocation_benefit_report import summarize_records
from .benefit_bias_audit import cost_audit, fit_offsets, correct_offsets, fit_thresholds


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='completed learned-benefit PRETRAIN run')
    p.add_argument('--out',required=True,help='new, separate diagnostic directory')
    p.add_argument('--updates',type=int,default=1000,help='additional Adam updates; 0 runs the bias audit only')
    p.add_argument('--evaluate-every',type=int,default=100)
    p.add_argument('--device',default='cpu')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--ply',help='relocate the identical source PLY; no camera/image data required')
    p.add_argument('--fresh-factored',action='store_true',help='train a random factored evaluator on the same replay, with fresh Adam')
    return p


def static_features(state, ply=None):
    replay=state['learner']['replay']
    if not replay['static_width']:return None
    from .data import Geometry, read_ply
    from .allocation import scene_fingerprint
    path=Path(ply or state['plan']['spec']['ply'])
    raw,degree=read_ply(path);scene=state['scene_record'];weights=state['state_dict']
    if (len(raw)!=replay['static_count'] or len(raw)!=scene['source_gaussians']
            or degree!=state['config']['sh_degree'] or scene_fingerprint(raw)!=scene['fingerprint']):
        raise ValueError('source PLY identity/fingerprint mismatch')
    g=scene['geometry'];geometry=Geometry(g['lower'],g['span'],g['bits'])
    attrs=(raw[:,3:]-weights['attr_mean'])/weights['attr_std']
    static=torch.cat((geometry.normalize(raw[:,:3]),attrs),-1)
    if static.shape[1]!=replay['static_width'] or not torch.isfinite(static).all():
        raise ValueError('invalid source feature width/statistics')
    return static


def restore_learner(state, device, ply=None):
    progress=state.get('progress',{})
    if (state.get('family') not in ('learned_benefit_allocation_v1','learned_benefit_factored_v2') or progress.get('phase')!='pretrain'
            or not progress.get('pretrain_complete') or not bool(state['learner']['network']['ready'])):
        raise ValueError('diagnostics require completed learned-benefit pretraining')
    saved=state['learner'];weights=saved['network']
    Network,Learner=BenefitNetwork,BenefitLearner
    if state['family']=='learned_benefit_factored_v2':
        from .factored_benefit import FactoredBenefitNetwork,DecisionBenefitLearner
        Network,Learner=FactoredBenefitNetwork,DecisionBenefitLearner
    net=Network(*(weights[key+'_mean'].numel() for key in ('point','action','views','aux')),
        continuous={key:weights[key+'_continuous'] for key in ('point','action','views','aux')},
        mse_only=state['plan']['settings']['distortion']=='mse').to(device)
    lr,fit_windows,steps,batch,mask,mask_weight=saved['settings']
    learner=Learner(net,static_source=static_features(state,ply),capacity=saved['replay']['capacity'],
        lr=lr,fit_windows=fit_windows,updates_per_window=steps,batch_size=batch,
        mask_fraction=mask,mask_weight=mask_weight)
    learner.load_state_dict(saved)
    train=learner.replay.groups;validation=learner.validation.groups
    if {r['source'] for r in train}&{r['source'] for r in validation}:
        raise ValueError('training/validation source overlap')
    for rows in (train,validation):
        if {r['branch'] for r in rows}!={'positive','keep'}:
            raise ValueError('both branches need training and validation labels')
    return learner


def _record_digest(learner):
    digest=hashlib.sha256()
    for replay in (learner.replay,learner.validation):
        for r in replay.groups:
            digest.update(json.dumps([r[k] for k in ('source','branch','step','validation','old','new')]).encode())
            for v in [r[k] for k in ('target','rates','conditional')]+list(r['features'].values()):
                tensor=v.detach().cpu().contiguous()
                digest.update(str((tuple(tensor.shape),tensor.dtype)).encode());digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def predict_records(learner,replay):
    records=replay.records();sums=[0.,0.];counts=[0,0]
    for start in range(0,len(records),128):
        rows=records[start:start+128]
        features,target,rates,conditional,groups,branches=pack_groups(rows,learner.device)
        full=learner.predict(features,branches[groups]);masked=learner.predict(features,branches[groups],masked=True)
        losses=learner.losses(full.to(target),target,rates,conditional,groups,learner.network.scales[branches],branches)
        for branch in (0,1):
            chosen=branches==branch;sums[branch]+=float(losses[chosen].sum());counts[branch]+=int(chosen.sum())
        offset=0
        for r in rows:
            size=len(r['target']);r['prediction']=full[offset:offset+size].cpu()
            r['prediction_masked']=masked[offset:offset+size].cpu();offset+=size
    return records,sum(value/count for value,count in zip(sums,counts))/2


def _write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def _file_digest(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main(args):
    if args.updates<0 or args.evaluate_every<1 or args.cpu_threads<1:
        raise ValueError('invalid offline diagnostic budget')
    run,out=Path(args.run).resolve(),Path(args.out).resolve()
    if out.is_relative_to(run) or run.is_relative_to(out):raise ValueError('source/output must be separate')
    if out.exists():raise FileExistsError('use a new diagnostic output')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable; use --device cpu')
    torch.set_num_threads(args.cpu_threads)
    path=run/'checkpoints/latest/state.pt';source_digest=_file_digest(path)
    state=torch.load(path,map_location='cpu',weights_only=True)
    learner=restore_learner(state,device,args.ply);window=state['progress']['step']
    source_updates=learner.updates;source_baseline=None
    if args.fresh_factored:
        from .factored_benefit import fresh_factored
        train,_=predict_records(learner,learner.replay);validation,_=predict_records(learner,learner.validation)
        source_baseline=dict(training=summarize_records(train),validation=summarize_records(validation))
        del train,validation
        learner=fresh_factored(learner,state['plan']['settings'].get('seed',42))
    variant='factored' if hasattr(learner.network,'common_scale') else 'legacy'
    initial_updates=learner.updates;fixed_digest=_record_digest(learner)
    normalization={key:value.detach().cpu().clone() for key,value in learner.network.state_dict().items()
                   if key.endswith(('_mean','_std','_continuous','_scale')) or key in ('scales','ready')}
    del state
    thresholds=fit_thresholds(learner.replay.records())
    out.mkdir(parents=True)
    _write(out/'settings.json',dict(source_checkpoint=str(path),source_checkpoint_sha256=source_digest,
        fixed_records_sha256=fixed_digest,source_network_updates=source_updates,frozen_source_window=window,
        additional_updates=args.updates,evaluate_every=args.evaluate_every,device=str(device),
        model_variant=variant,optimizer_initialization='fresh Adam' if args.fresh_factored else 'source Adam retained',
        loss_and_optimizer='factored decision objective; source learning rate' if args.fresh_factored else 'source loss, Adam state and learning rate retained',
        new_labels=0,renderer_calls=0,probability_table_updated=False,normalization_refitted=False,
        thresholds_fitted_on='training labels only',magnitude_thresholds=thresholds))
    if source_baseline is not None:_write(out/'source_baseline.json',source_baseline)
    started=time.perf_counter();history=[]

    def evaluate(update):
        training,train_loss=predict_records(learner,learner.replay)
        validation,val_loss=predict_records(learner,learner.validation)
        row=dict(additional_updates=update,network_updates=learner.updates,fixed_records_sha256=fixed_digest,
            training_full_input_loss=train_loss,validation_full_input_loss=val_loss,
            training=summarize_records(training),validation=summarize_records(validation),
            elapsed_seconds=time.perf_counter()-started)
        with (out/'history.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        offsets=fit_offsets(training)
        audit=dict(additional_updates=update,training=cost_audit(training,thresholds),
            validation=cost_audit(validation,thresholds),
            training_fitted_offset=offsets,offset_scope='diagnostic only; no deployment or weight modification',
            validation_after_training_offset=summarize_records(correct_offsets(validation,offsets)))
        _write(out/('initial_bias.json' if update==0 else 'final_bias.json'),audit)
        print(f'offline {update}/{args.updates}: train_loss={train_loss:.6g}, validation_loss={val_loss:.6g}',flush=True)
        for branch in ('positive','keep'):
            v=row['validation'][branch]['stages'];t=row['training'][branch]['stages']
            print(f'  {branch}: train_regret={t["network"]["regret_mean"]:.6g}; '
                  f'validation_regret={v["first_order"]["regret_mean"]:.6g}->{v["network"]["regret_mean"]:.6g}; '
                  f'sign_error={v["network"]["sign_error_fraction"]:.2%}',flush=True)
        history.append(row)

    evaluate(0)
    for update in range(1,args.updates+1):
        # Keep the original window fixed: no synthetic aging of the recent-record pool.
        learner.train_step(window)
        if update%args.evaluate_every==0 or update==args.updates:evaluate(update)
    if not args.updates:
        (out/'final_bias.json').write_bytes((out/'initial_bias.json').read_bytes())
    if _record_digest(learner)!=fixed_digest:raise RuntimeError('fixed diagnostic labels/features changed')
    for key,value in normalization.items():
        if not torch.equal(value,learner.network.state_dict()[key].cpu()):raise RuntimeError('fixed normalization changed')
    unchanged=_file_digest(path)==source_digest
    if not unchanged:raise RuntimeError('source checkpoint changed during diagnostics')
    torch.save(dict(network={k:v.detach().cpu() for k,v in learner.network.state_dict().items()},
        source_checkpoint_sha256=source_digest,additional_updates=args.updates,
        model_variant=variant,diagnostic_only=True,probability_table_updated=False),out/'offline_final.pt')
    result=dict(status='complete',additional_updates=args.updates,source_checkpoint_unchanged=unchanged,
        fixed_records_sha256=fixed_digest,training_records=len(learner.replay.groups),
        validation_records=len(learner.validation.groups),source_network_updates=source_updates,
        initial_network_updates=initial_updates,model_variant=variant,source_baseline=source_baseline,
        before=history[0],after=history[-1],elapsed_seconds=time.perf_counter()-started,
        scope='fixed cached action distribution; does not establish deployment quality or identify a unique cause')
    _write(out/'summary.json',result);print(json.dumps({k:result[k] for k in
        ('status','additional_updates','training_records','validation_records','elapsed_seconds')},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':main(parser().parse_args())
