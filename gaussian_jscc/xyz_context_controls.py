"""Truck-only paired direct decoders: baseline versus reliable XYZ context.

Both arms start randomly; source runs provide paths/config/camera splits ONLY.
Symbol budgets, channel, batches, objectives and optimizer schedule are paired.
No residual enhancements, allocation, deletion or extra coordinate messages.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

from .appearance_controls import summarize
from .residual_controls import build_plan as direct_plan, job_out, runtime_summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='source multiscene run for config/splits, NOT weights')
    p.add_argument('--out', required=True, help='new common output directory outside source run')
    p.add_argument('--manifest', help='override paths; same Truck PLY and camera split required')
    p.add_argument('--bootstrap-steps', type=int, default=2000)
    p.add_argument('--render-steps', type=int, default=1000)
    p.add_argument('--validate-every', type=int, default=250)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--test-views', type=int, default=0)
    p.add_argument('--trials', type=int, default=2)
    p.add_argument('--image-views', type=int, default=2)
    p.add_argument('--color-rois')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dry-run', action='store_true')
    return p


def build_plan(args, python=None):
    source, out = Path(args.run).resolve(), Path(args.out).resolve()
    if out == source or source in out.parents:
        raise ValueError('use a new output directory outside the source run')
    template = direct_plan(SimpleNamespace(**vars(args), groups=['direct']), python)
    fit = next(j['command'] for j in template['jobs'] if j['kind']=='fit')
    evaluate = next(j['command'] for j in template['jobs'] if j['kind']=='evaluate')
    jobs = []
    for mode in ('none','xyz'):
        dest = out/mode
        command = fit.copy()
        command[command.index('--out')+1] = str(dest)
        command += ['--decoder-geometry',mode]
        jobs.append({'name':'decoder_geometry_'+mode, 'kind':'fit', 'section':'xyz_context',
                     'geometry':mode, 'command':command})
        command = evaluate.copy()
        command[command.index('--out')+1] = str(dest/'test')
        command[command.index('--checkpoint')+1] = str(dest/'checkpoints/final/codec.pt')
        jobs.append({'name':'decoder_geometry_'+mode+'_test','kind':'evaluate',
                     'section':'xyz_context','geometry':mode,'command':command})
    return {**template, 'jobs':jobs, 'suite':'xyz_context','groups':['none','xyz'],
        'scope':'Truck-only direct decoder geometry; random paired initialization',
        'geometry_input':'dequantized reliable 16bit XYZ in scene bbox-normalized space',
        'geometry_path':'zero-output-initialized point embedding and relative XYZ/distance attention bias',
        'comparison_boundary':'same common weights/communication budget; NOT parameter/FLOP matched'}


def common_initializer_id(model):
    config = model.cfg.to_dict().copy()
    config.pop('decoder_geometry',None)
    digest = hashlib.sha256(json.dumps(config,sort_keys=True).encode())
    for name,value in sorted(model.state_dict().items()):
        if name.startswith('learned.xyz_embedding.') or '.xyz_bias.' in name:
            continue
        digest.update(name.encode())
        digest.update(str((tuple(value.shape),str(value.dtype))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def plot_ssim(out, plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for job in plan['jobs']:
        if job['kind']!='fit':
            continue
        path = job_out(job)/'scenes/truck/validation.jsonl'
        history = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        for tier in ('1','2','3'):
            entries = [(h,next(r for r in h['layouts'] if r['layout']==tier)) for h in history]
            for ax,metric in zip(axes,('source_ssim','photo_ssim')):
                ax.plot([h['step'] for h,r in entries],[r[metric] for h,r in entries],
                        label=f"{job['geometry']}, q{tier}")
                ax.set(xlabel='Actual updates',ylabel=metric); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.savefig(out/'charts/truck_xyz_context_ssim.png',dpi=150)
    plt.close(fig)


def main(args):
    plan = build_plan(args)
    for job in plan['jobs']:
        print(job['name']+': '+shlex.join(job['command']),flush=True)
    if args.dry_run:
        return
    out = Path(args.out); out.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    completed, identities, runtimes = [], {}, {}
    for job in plan['jobs']:
        print('Starting '+job['name'],flush=True)
        subprocess.run(job['command'],check=True)
        if job['kind']=='fit':
            from .transport import load_checkpoint
            dest = job_out(job)
            check = json.loads((dest/'complete.json').read_text(encoding='utf-8'))
            if not check.get('frozen_parameters_verified') or any(
                    visits.get('allocation',0) or visits.get('joint',0) for visits in check['scene_visits'].values()):
                raise RuntimeError('coordinate/statistics freeze or no-allocation audit failed')
            record = json.loads((dest/'training.json').read_text(encoding='utf-8'))
            if record['checkpoint'] is not None or record['drop'] != 0 or record['decoder_mode'] != 'direct':
                raise RuntimeError('geometry control must start randomly and retain all points with direct decoding')
            initial = load_checkpoint(dest/'checkpoints/initial/codec.pt','cpu')
            identities[job['geometry']] = common_initializer_id(initial)
            if len(set(identities.values()))!=1:
                raise RuntimeError('geometry arms did not share common random weights/statistics')
            runtimes[job['geometry']] = {**runtime_summary(dest),
                                       'decoder_geometry':initial.cfg.decoder_geometry}
        completed.append(job['name'])
        (out/'progress.json').write_text(json.dumps({'completed_jobs':completed},indent=2),encoding='utf-8')
    rows = summarize(out,plan)
    plot_ssim(out,plan)
    (out/'runtime_comparison.json').write_text(json.dumps(runtimes,indent=2),encoding='utf-8')
    (out/'complete.json').write_text(json.dumps({'completed_jobs':completed,'comparison_rows':len(rows),
        'paired_initializer_verified':len(identities)==2,'common_initializer_ids':identities,
        'allocator_trained':False,'coordinate_budget_unchanged':True},indent=2),encoding='utf-8')
    print(f'XYZ context controls complete: {out}',flush=True)


if __name__=='__main__':
    main(parser().parse_args())
