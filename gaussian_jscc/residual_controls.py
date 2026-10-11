"""Experiment 3: Truck-only direct versus additive attribute decoder controls.

Same random common weights/statistics, once-encoded 4/12/24 symbols, reliable
16-bit coordinates, local-response bootstrap and scene RGB-MSE render training.
No allocator, no learned deletion and no silent historical-checkpoint upgrade.
"""
import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

from .appearance_controls import build_plan as appearance_plan, summarize


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='source multiscene run for paths/config/splits ONLY; weights not loaded')
    p.add_argument('--out',required=True,help='new common output directory')
    p.add_argument('--manifest',help='override paths after migration; same PLY/splits required')
    p.add_argument('--groups',nargs='+',choices=['direct','residual'],default=['direct','residual'])
    p.add_argument('--bootstrap-steps',type=int,default=2000)
    p.add_argument('--render-steps',type=int,default=1000)
    p.add_argument('--validate-every',type=int,default=250)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--test-views',type=int,default=0)
    p.add_argument('--trials',type=int,default=2)
    p.add_argument('--image-views',type=int,default=2)
    p.add_argument('--color-rois')
    p.add_argument('--device',default='cuda')
    p.add_argument('--dry-run',action='store_true')
    return p


def build_plan(args,python=None):
    if len(set(args.groups)) != len(args.groups) or not args.groups:
        raise ValueError('select distinct decoder groups')
    options = SimpleNamespace(**vars(args),suite='directions',scenes=['truck'],color_steps=2000,skip_oracle=True)
    template = appearance_plan(options,python or sys.executable)
    if template['rates'] != [0,4,12,24]:
        raise ValueError('experiment 3 holds the current 0/4/12/24 symbol budget fixed')
    fit = next(j['command'] for j in template['jobs'] if j['name']=='directions_random')
    evaluation = next(j['command'] for j in template['jobs'] if j['name']=='directions_random_test')
    jobs = []
    for mode in args.groups:
        dest = Path(args.out).resolve()/mode
        command = fit.copy()
        command[command.index('--out')+1] = str(dest)
        command[command.index('--drop')+1] = '0'
        command += ['--decoder-mode',mode]
        jobs.append({'name':'decoder_'+mode,'kind':'fit','section':'residual','mode':mode,'command':command})
        command = evaluation.copy()
        command[command.index('--out')+1] = str(dest/'test')
        command[command.index('--checkpoint')+1] = str(dest/'checkpoints/final/codec.pt')
        jobs.append({'name':'decoder_'+mode+'_test','kind':'evaluate','section':'residual','mode':mode,'command':command})
    return {**template,'jobs':jobs,'suite':'residual','initializer':None,'random_initialization':True,
            'source_run_usage':'paths, hyperparameters and camera split only; no source weights loaded',
            'all_points_retained':True,'groups':args.groups,'scope':'Truck-only paired decoder experiment',
            'comparison_boundary':'same common initialization and symbols; added parameters/compute are reported, NOT capacity-matched'}


def job_out(job):
    command = job['command']
    return Path(command[command.index('--out')+1])


def runtime_summary(dest):
    record = json.loads((dest/'training.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (dest/'loss.jsonl').read_text(encoding='utf-8').splitlines()]
    return {'decoder_mode':record['decoder_mode'],'codec_parameters':record['codec_parameters'],
        'trainable_codec_parameters':record['trainable_codec_parameters'],
        'phases':{phase:{'updates':len(selected),'mean_step_seconds':sum(r['step_seconds'] for r in selected)/len(selected)}
                  for phase in ('bootstrap','render') if (selected:=[r for r in rows if r['phase']==phase])},
        'note':'actual observed step time; independent GPUs/users can affect timing; not a FLOP-matched comparison'}


def stage_diagnostics(dest,args,plan):
    """Full-scene, paired-noise q3 attribute-stage diagnostics; no optimizer/render.

    Attribute MSE here is only diagnostic, not a new training objective. Deltas
    are measured before physical clipping and quaternion normalization.
    """
    import torch
    from .codec import channel, prefix_mask
    from .cli import device_for, seed_all
    from .data import read_ply
    from .multiscene import Scene,manifest,parser as training_parser
    from .optimization import preserved_rng
    from .transport import load_checkpoint,model_id
    device = device_for(args.device)
    model = load_checkpoint(dest/'checkpoints/final/codec.pt',device).eval()
    identity = model_id(model)
    record = json.loads((dest/'training.json').read_text(encoding='utf-8'))
    opts = training_parser().parse_args(['--out',str(dest)])
    for key in ('blocks_per_batch','net_bits_per_use'):
        setattr(opts,key,record[key])
    spec = next(s for s in manifest(plan['manifest']) if s['name']=='truck')
    raw,degree = read_ply(spec['ply'])
    scene = Scene(spec,raw,degree,model,opts,dest,with_mask=False,cameras=False)
    del raw
    attr_dim = model.cfg.attr_dim
    names = [('opacity',0,1),('scale',1,4),('rotation',4,8),('dc',8,11)]
    if attr_dim>11:
        names.append(('sh',11,attr_dim))
    error_sums = [0.,0.,0.]
    delta_sums = [0.,0.]
    group_sums = [{name:0. for name,_,_ in names} for _ in range(2)]
    count = 0
    with torch.no_grad(),preserved_rng(device):
        seed_all(plan['seed']+20000)
        for features,ids in zip(scene.groups,scene.group_ids):
            features = features.to(device)
            q = torch.where(ids.to(device)>=0,3,0)
            z = model.learned.encode(features,features[...,:3],q,plan['snr'])
            mask = prefix_mask(q.flatten(),model.cfg.rates).reshape_as(z)
            received = channel(z.reshape(-1,2),plan['snr'],plan['channel']).reshape_as(z)*mask
            stages,deltas = model.learned.decode_stages(received,q,plan['snr'])
            target = features[q>0][:,3:]
            count += len(target)
            for i,stage in enumerate(stages):
                error_sums[i] += float((stage[q>0][:,3:]-target).double().square().sum())
            for i,delta in enumerate(deltas):
                active = delta[q>0].double()
                delta_sums[i] += float(active.square().sum())
                for name,a,b in names:
                    group_sums[i][name] += float(active[:,a:b].square().sum())
    if model_id(model)!=identity:
        raise RuntimeError('stage diagnostics changed codec weights')
    result = {'source_gaussians':count,'snr':plan['snr'],'channel':plan['channel'],'seed':plan['seed']+20000,
        'scope':'q3, full scene, one full-slot paired noise draw; no visibility weighting',
        'stage_normalized_attribute_mse':[s/count/attr_dim for s in error_sums],
        'corrections':[{'tier':i+2,'normalized_rms':(s/count/attr_dim)**.5,
            'groups':{name:(group_sums[i][name]/count/(b-a))**.5 for name,a,b in names}}
            for i,s in enumerate(delta_sums)],
        'note':'normalized parameter-space diagnostic ONLY, not a loss or rendered-quality guarantee; rotation before normalization'}
    (dest/'stage_diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


def plot_prefix_gains(out,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gains = []
    for job in plan['jobs']:
        if job['kind']=='evaluate':
            for line in (job_out(job)/'prefix_gains.jsonl').read_text(encoding='utf-8').splitlines():
                gains.append(dict(json.loads(line),control=job['name']))
    (out/'prefix_gains.json').write_text(json.dumps(gains,indent=2,allow_nan=False),encoding='utf-8')
    with (out/'prefix_gains.csv').open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for row in gains for k in row)))
        writer.writeheader(); writer.writerows(gains)
    fig,axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for label in dict.fromkeys(r['control'] for r in gains):
        selected = [r for r in gains if r['control']==label]
        for ax,key in zip(axes,('source_psnr_gain_db','source_mse_reduction_per_extra_symbol','paired_view_trial_improved_fraction')):
            ax.plot([int(r['to']) for r in selected],[r[key] for r in selected],marker='o',label=label)
            ax.set(xlabel='Upgrade to tier',ylabel=key,xticks=[2,3]); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Truck: measured enhancement gains, paired heldout test views/noise')
    fig.savefig(out/'charts/truck_residual_prefix_gains.png',dpi=150); plt.close(fig)
    return gains


def main(args):
    plan = build_plan(args)
    for job in plan['jobs']:
        print(job['name']+': '+shlex.join(job['command']),flush=True)
    if args.dry_run:
        return
    out = Path(args.out); out.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    completed,initializers,runtimes = [],{},{}
    for job in plan['jobs']:
        print('Starting '+job['name'],flush=True)
        subprocess.run(job['command'],check=True)
        dest = job_out(job)
        if job['kind']=='fit':
            from .transport import load_checkpoint
            from .residual_decoder import common_initializer_id
            check = json.loads((dest/'complete.json').read_text(encoding='utf-8'))
            if not check.get('frozen_parameters_verified') or any(
                    visits.get('allocation',0) or visits.get('joint',0) for visits in check['scene_visits'].values()):
                raise RuntimeError('coordinate/statistics freeze or no-allocation audit failed')
            initializers[job['mode']] = common_initializer_id(load_checkpoint(dest/'checkpoints/initial/codec.pt','cpu'))
            if len(set(initializers.values()))!=1:
                raise RuntimeError('common weights/statistics were not paired at random initialization')
            runtimes[job['mode']] = runtime_summary(dest)
            if job['mode']=='residual':
                stage_diagnostics(dest,args,plan)
        completed.append(job['name'])
        (out/'progress.json').write_text(json.dumps({'completed_jobs':completed},indent=2),encoding='utf-8')
    rows = summarize(out,plan)
    gains = plot_prefix_gains(out,plan)
    (out/'runtime_comparison.json').write_text(json.dumps(runtimes,indent=2),encoding='utf-8')
    (out/'complete.json').write_text(json.dumps({'completed_jobs':completed,'comparison_rows':len(rows),
        'paired_initializer_verified':len(initializers)==2,'common_initializer_ids':initializers,
        'prefix_gain_rows':len(gains),'allocator_trained':False},indent=2),encoding='utf-8')
    print(f'Experiment 3 complete: {out}',flush=True)


if __name__=='__main__':
    main(parser().parse_args())
