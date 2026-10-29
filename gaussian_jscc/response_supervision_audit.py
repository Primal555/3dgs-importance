"""Fixed-replay scale audit and paired loss-only evaluator continuation on CPU."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch

from .benefit_diagnostics import static_features,predict_records,_record_digest,_file_digest,_write
from .response_value import ResponseValueNetwork,ResponseValueLearner
from .response_value_report import summarize_response
from .visible_response import VisibleResponseNetwork,VisibleResponseLearner,supervision_scales,FAMILY


def label_statistics(records):
    keep=[r for r in records if r['branch']=='keep']
    values=torch.stack([r['target'].double().cpu() for r in keep])
    common=values.mean(1);centered=values-common[:,None]
    pairs=torch.stack((values[:,1]-values[:,0],values[:,2]-values[:,0],values[:,2]-values[:,1]),1)
    result=dict(complete_groups=len(keep),unique_points=len({r['source'] for r in keep}))
    for name,data in (('values',values),('common',common),('centered',centered),('pairs',pairs)):
        flat=data.reshape(-1,2)
        result[name]=dict(abs_quantiles={str(q):torch.quantile(flat.abs(),q,dim=0).tolist() for q in (.5,.9,.99,1.)},
            mean=flat.mean(0).tolist(),rms=flat.square().mean(0).sqrt().tolist(),
            positive_fraction=(flat>0).double().mean(0).tolist())
    return result


def restore_arm(state,static,independent):
    saved=copy.deepcopy(state['learner']);weights=saved['network']
    Network,Learner=(VisibleResponseNetwork,VisibleResponseLearner) if independent else (ResponseValueNetwork,ResponseValueLearner)
    network=Network(*(weights[k+'_mean'].numel() for k in ('point','action','views','aux')),
        continuous={k:weights[k+'_continuous'] for k in ('point','action','views','aux')},
        mse_only=state['plan']['settings']['distortion']=='mse')
    if independent:
        missing,unexpected=network.load_state_dict(weights,strict=False)
        if set(missing)!={'common_scale','centered_scale','pair_scale'} or unexpected:
            raise ValueError('unexpected source evaluator architecture')
        network.fit_supervision(saved['replay']['groups'])
        saved['network']=network.state_dict();saved['objective_variant']=FAMILY
    lr,fit,steps,batch,mask,mask_weight=saved['settings']
    learner=Learner(network,static_source=static,capacity=saved['replay']['capacity'],lr=lr,
        fit_windows=fit,updates_per_window=steps,batch_size=batch,mask_fraction=mask,mask_weight=mask_weight)
    learner.load_state_dict(saved)
    # Both arms start from identical selected weights and fresh Adam moments.
    learner.optimizer=torch.optim.Adam(learner.network.parameters(),lr=lr)
    return learner


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--updates',type=int,default=1000)
    p.add_argument('--evaluate-every',type=int,default=100)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--ply',help='identical PLY at a relocated path')
    return p


def main(args):
    if args.updates<0 or min(args.evaluate_every,args.cpu_threads)<1:raise ValueError('invalid audit budget')
    run,out=Path(args.run).resolve(),Path(args.out).resolve()
    if out.is_relative_to(run) or run.is_relative_to(out):raise ValueError('use a separate audit directory')
    if out.exists():raise FileExistsError('audit output already exists')
    path=run/'checkpoints/latest/state.pt';source_hash=_file_digest(path)
    state=torch.load(path,map_location='cpu',weights_only=True)
    if (state.get('family')!='render_response_value_v1' or not state['progress'].get('pretrain_complete')
            or state['progress']['phase']!='pretrain'):
        raise ValueError('audit requires completed response pretraining before allocation')
    torch.set_num_threads(args.cpu_threads)
    static=static_features(state,args.ply)
    arms={name:restore_arm(state,static,independent) for name,independent in (('original',False),('separate_scales',True))}
    original=arms['original'];training=original.replay.groups;validation=original.validation.groups
    if not training or not validation or {r['source'] for r in training}&{r['source'] for r in validation}:
        raise ValueError('require disjoint nonempty train/validation points')
    if any(r['validation'] for r in training) or any(not r['validation'] for r in validation):
        raise ValueError('replay split flags inconsistent')
    fixed_hash=_record_digest(original)
    if _record_digest(arms['separate_scales'])!=fixed_hash:raise RuntimeError('paired replay mismatch')
    out.mkdir(parents=True)
    settings=dict(source_checkpoint=str(path),source_checkpoint_sha256=source_hash,fixed_records_sha256=fixed_hash,
        updates=args.updates,selected_window=state['progress'].get('selected_evaluator_window'),
        optimizer='fresh Adam in both arms; original learning rate',features='identical original 677-dimensional view input',
        renderer_calls=0,probability_table_updated=False,new_labels=0,scales_fit_on='training labels only')
    _write(out/'settings.json',settings)
    audit=dict(training=label_statistics(training),validation=label_statistics(validation),
        supervision_scales={k:v.tolist() for k,v in supervision_scales(training).items()})
    _write(out/'label_scales.json',audit)
    started=time.perf_counter();history=[];window=state['progress']['step']
    def evaluate(step):
        row=dict(updates=step,arms={},elapsed_seconds=time.perf_counter()-started)
        for name,learner in arms.items():
            train,train_loss=predict_records(learner,learner.replay)
            valid,val_loss=predict_records(learner,learner.validation)
            row['arms'][name]=dict(training_loss=train_loss,validation_loss=val_loss,
                training=summarize_response(train),validation=summarize_response(valid))
            metrics=row['arms'][name]['validation']
            print(f"{name} {step}/{args.updates}: positive regret={metrics['positive']['stages']['network']['regret_mean']:.6g}; "
                  f"keep regret={metrics['keep']['stages']['network']['regret_mean']:.6g}; "
                  f"four-state regret={metrics['keep']['four_state_selection_regret']:.6g}",flush=True)
        with (out/'history.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        history.append(row)
    evaluate(0)
    for step in range(1,args.updates+1):
        for learner in arms.values():learner.train_step(window)
        if step%args.evaluate_every==0 or step==args.updates:evaluate(step)
    if any(_record_digest(a)!=fixed_hash for a in arms.values()):raise RuntimeError('fixed replay changed')
    if _file_digest(path)!=source_hash:raise RuntimeError('source checkpoint changed')
    for name,learner in arms.items():
        torch.save(dict(network=learner.network.state_dict(),diagnostic_only=True,source_checkpoint_sha256=source_hash),out/(name+'.pt'))
    result=dict(status='complete',before=history[0],after=history[-1],label_scales=audit,
        source_checkpoint_unchanged=True,fixed_records_sha256=fixed_hash,elapsed_seconds=time.perf_counter()-started)
    _write(out/'summary.json',result)
    return result


if __name__=='__main__':main(parser().parse_args())
