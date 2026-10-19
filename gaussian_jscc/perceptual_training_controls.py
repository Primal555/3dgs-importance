"""Two loss arms from common pre-render semantics; no allocator or MSE rerun."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from .modular_config import ModularCodecConfig
from .modular_checkpoint import initialize_bootstrap_run, validate_scene
from .modular_experiment import json_write
from .perceptual_objective import PAPER_WEIGHTS, WEIGHT_REFERENCES


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='current trained 8/24/48 run: metadata/config ONLY, never its weights')
    p.add_argument('--bootstrap',required=True,help='common completed semantic_bootstrap checkpoint before any render updates')
    p.add_argument('--out',required=True)
    p.add_argument('--manifest')
    p.add_argument('--loss',choices=['both','mse_ssim','mse_lpips'],default='both')
    p.add_argument('--semantic-render-steps',type=int,default=1000)
    p.add_argument('--channel-bootstrap-steps',type=int,default=2000)
    p.add_argument('--channel-render-steps',type=int,default=2000)
    p.add_argument('--semantic-lr',type=float,default=1e-4)
    p.add_argument('--channel-lr',type=float,default=1e-4)
    p.add_argument('--ssim-weight',type=float,default=PAPER_WEIGHTS['mse_ssim'])
    p.add_argument('--lpips-weight',type=float,default=PAPER_WEIGHTS['mse_lpips'])
    p.add_argument('--channel-latent-weight',type=float,default=1.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--validate-every',type=int,default=500)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume-suite',action='store_true')
    p.add_argument('--sequential',action='store_true',help='optional low-memory fallback; default is same-GPU parallel')
    p.add_argument('--dry-run',action='store_true',help='verify checkpoint/config/split and print both commands; no rendering/writes')
    return p


def build_plan(args):
    if min(args.semantic_render_steps,args.channel_bootstrap_steps,args.channel_render_steps,
           args.validate_every,args.save_every)<1:
        raise ValueError('all phase budgets and save/validation intervals must be positive')
    if any(not math.isfinite(v) or v<=0 for v in (args.semantic_lr,args.channel_lr,args.ssim_weight,
            args.lpips_weight,args.channel_latent_weight)):
        raise ValueError('learning rates and objective weights must be finite and positive')
    run=Path(args.run).resolve(); bootstrap=Path(args.bootstrap).resolve(); out=Path(args.out).resolve()
    source=json.loads((run/'training.json').read_text(encoding='utf-8'))
    cfg=ModularCodecConfig.from_dict(source['codec_config'])
    if cfg.rates!=(0,8,24,48) or cfg.channel_variant!='progressive':
        raise ValueError('suite requires the current progressive 8/24/48 source config')
    # CPU-only structural/weight preflight, no rasterizer, no training, no GPU allocation.
    # Full source weights and probability/deployment tables are never read.
    model,progress,scene=initialize_bootstrap_run(bootstrap,'cpu',cfg=cfg)
    del model
    validate_scene(source['scene'],scene)
    with bootstrap.open('rb') as stream:bootstrap_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    jobs=[]
    for kind,weight in (('mse_ssim',args.ssim_weight),('mse_lpips',args.lpips_weight)):
        command=[sys.executable,'-u','-m','gaussian_jscc.modular_experiment',
            '--run',str(run),'--out',str(out/kind),'--init-bootstrap',str(bootstrap),'--use-source-config',
            '--semantic-bootstrap-steps','0','--semantic-render-steps',str(args.semantic_render_steps),
            '--channel-bootstrap-steps',str(args.channel_bootstrap_steps),'--channel-render-steps',str(args.channel_render_steps),
            '--image-objective',kind,'--image-aux-weight',str(weight),
            '--semantic-encoder-lr',str(args.semantic_lr),'--semantic-decoder-lr',str(args.semantic_lr),
            '--channel-encoder-lr',str(args.channel_lr),'--channel-decoder-lr',str(args.channel_lr),
            '--channel-latent-weight',str(args.channel_latent_weight),
            '--train-snrs','10','--test-snrs','10','--seed',str(args.seed),
            '--validate-every',str(args.validate_every),'--save-every',str(args.save_every),
            '--color-metrics','--lpips','--device',args.device]
        if args.manifest:command+=['--manifest',str(Path(args.manifest).resolve())]
        jobs.append(dict(name=kind,command=command))
    plan=dict(version=1,experiment='perceptual_from_attribute_bootstrap',
                source_run=str(run),bootstrap=str(bootstrap),bootstrap_sha256=bootstrap_hash,
                shared_attribute_updates=progress['completed']['semantic_bootstrap'],
                codec_config=cfg.to_dict(),scene=scene,jobs=jobs,
                source_usage='metadata/config only; semantic weights from shared attribute checkpoint; fresh channel modules',
                losses=dict(semantic_render='arm image objective',channel_bootstrap='latent MSE',
                            channel_render='arm image objective + weighted latent MSE'),
                weight_references=WEIGHT_REFERENCES,
                allocation='not trained; q1/q2/q3/mixed diagnostics only',
                mse_baseline='reuse historical evidence; not a matched new three-arm end-to-end experiment')
    # Canonical JSON types are essential: cfg tuples become lists on disk.
    # A saved suite must compare equal after loading for exact resume.
    return json.loads(json.dumps(plan))


def run_jobs(jobs,out,*,resume=False,parallel=True):
    """Separate child logs/PIDs; default launches both before waiting on either.

    A failed arm does not kill a healthy arm. Collect every exit status, then
    fail the suite explicitly. Closing a log viewer does not stop either child.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    state_path=out/'processes.json'
    states=json.loads(state_path.read_text(encoding='utf-8')) if resume and state_path.exists() else {}
    for job in jobs:
        old=states.get(job['name'],{})
        if old.get('status')=='running' and os.name=='posix':
            try:os.kill(old['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError(f"Recorded {job['name']} PID {old['pid']} is still running; refusing duplicate launch")
    processes={};failures={}

    def launch(job):
        name=job['name'];log=out/(name+'.log')
        with log.open('ab' if resume else 'wb') as stream:
            child=subprocess.Popen(job['command'],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            processes[name]=child
            states[name]=dict(pid=child.pid,log=str(log),status='running',returncode=None)
        (out/(name+'.pid')).write_text(str(child.pid)+'\n',encoding='utf-8')
        json_write(state_path,states)
        print(f'{name}: PID={child.pid}; log={log}',flush=True)

    def collect(name,code):
        states[name].update(status='complete' if code==0 else 'failed',returncode=code)
        if code:failures[name]=code
        json_write(state_path,states)
        print(f'{name}: exited with code {code}; log={states[name]["log"]}',flush=True)

    def supervision_error(name,exc):
        # A child may already exist when publishing its PID/state fails. Keep
        # its Popen in `processes` and reap it normally instead of abandoning it.
        message=f'{type(exc).__name__}: {exc}'
        failures[name]=message
        if name not in states:
            states[name]=dict(pid=None,log=str(out/(name+'.log')),status='launch_failed',returncode=None)
        states[name]['supervision_error']=message
        try:json_write(state_path,states)
        except OSError:pass  # Report the original error after all started children are reaped.

    def launch_safely(job):
        try:launch(job)
        except Exception as exc:supervision_error(job['name'],exc)

    def collect_safely(name,code):
        try:collect(name,code)
        except Exception as exc:supervision_error(name,exc)

    if parallel:
        for job in jobs:launch_safely(job)
        pending=set(processes)
        while pending:
            for name in tuple(pending):
                code=processes[name].poll()
                if code is not None:collect_safely(name,code);pending.remove(name)
            if pending:time.sleep(1)
    else:
        for job in jobs:
            launch_safely(job)
            if job['name'] in processes:collect_safely(job['name'],processes[job['name']].wait())
    if failures:raise RuntimeError(f'Failed training arms: {failures}; see per-arm logs in {out}')
    return states


def main(args):
    plan=build_plan(args)
    if args.dry_run:
        print(json.dumps(plan,indent=2));return plan
    out=Path(args.out).resolve(); record=out/'suite.json'
    if args.resume_suite:
        if not record.exists():raise FileNotFoundError('resume-suite requires original suite.json')
        if json.loads(record.read_text(encoding='utf-8'))!=plan:
            raise ValueError('suite configuration, source config or shared bootstrap changed')
    else:
        if out.exists():raise FileExistsError('use a NEW suite output directory or --resume-suite')
        json_write(record,plan)
    selected=[job for job in plan['jobs'] if args.loss in ('both',job['name'])]
    pending=[]
    for job in selected:
        arm=out/job['name']; command=list(job['command'])
        if (arm/'complete.json').exists():
            if not args.resume_suite:raise FileExistsError('completed arm already exists')
            print(f"Skipping completed {job['name']}",flush=True);continue
        if arm.exists():
            latest=arm/'checkpoints/latest/codec.pt'
            if not args.resume_suite or not latest.exists():
                raise FileNotFoundError('incomplete arm has no latest checkpoint; use a new output, no silent restart')
            i=command.index('--init-bootstrap');del command[i:i+2]
            command+=['--resume',str(latest)]
        print(f"\n=== {job['name']}: shared bootstrap {plan['shared_attribute_updates']} updates; then "
              f"{args.semantic_render_steps}+{args.channel_bootstrap_steps}+{args.channel_render_steps}; out={arm} ===",flush=True)
        pending.append(dict(name=job['name'],command=command))
    if pending:
        try:run_jobs(pending,out,resume=args.resume_suite,parallel=not args.sequential)
        except Exception:
            json_write(out/'progress.json',dict(status='failed',detail='see processes.json and per-arm logs'))
            raise
    completed=[job['name'] for job in plan['jobs'] if (out/job['name']/'complete.json').exists()]
    json_write(out/'progress.json',dict(completed=completed,status='complete' if len(completed)==2 else 'partial'))
    if len(completed)==2:json_write(out/'complete.json',dict(status='complete',completed=completed,bootstrap_sha256=plan['bootstrap_sha256']))
    return plan


if __name__=='__main__':main(parser().parse_args())
