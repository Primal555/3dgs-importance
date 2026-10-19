"""Independent MSE+LPIPS+GAN arm from the shared pre-render bootstrap."""
import argparse
import json
from pathlib import Path

from . import modular_experiment
from .adversarial_objective import adversarial_settings
from .perceptual_training_controls import build_plan as perceptual_plan


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='current 8/24/48 metadata/config source, not its weights')
    p.add_argument('--bootstrap',required=True,help='shared completed attribute bootstrap, BEFORE render training')
    p.add_argument('--out',required=True)
    p.add_argument('--manifest')
    p.add_argument('--semantic-render-steps',type=int,default=1000)
    p.add_argument('--channel-bootstrap-steps',type=int,default=2000)
    p.add_argument('--channel-render-steps',type=int,default=2000)
    p.add_argument('--semantic-lr',type=float,default=1e-4)
    p.add_argument('--channel-lr',type=float,default=1e-4)
    p.add_argument('--lpips-weight',type=float,default=1.)
    p.add_argument('--gan-weight',type=float,default=.001)
    p.add_argument('--discriminator-lr',type=float,default=1e-4)
    p.add_argument('--discriminator-betas',type=float,nargs=2,default=[.5,.999])
    p.add_argument('--channel-latent-weight',type=float,default=1.)
    p.add_argument('--validate-every',type=int,default=500)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true',help='exact resume in the same output, including D/Adam')
    p.add_argument('--dry-run',action='store_true',help='CPU preflight only; no output writes/training')
    return p


def build_plan(args):
    settings=adversarial_settings(args.gan_weight,args.discriminator_lr,args.discriminator_betas)
    # Reuse the exact common initializer/hash/config/split preflight. Select a
    # command only; neither existing loss arm is launched or modified.
    base=argparse.Namespace(**vars(args),ssim_weight=1/3)
    plan=perceptual_plan(base)
    job=next(j for j in plan['jobs'] if j['name']=='mse_lpips')
    command=list(job['command'])
    command[command.index('--out')+1]=str(Path(args.out).resolve())
    command[command.index('--image-objective')+1]='mse_gan'
    command+=['--gan-weight',str(args.gan_weight),'--discriminator-lr',str(args.discriminator_lr),
              '--discriminator-betas',*[str(b) for b in args.discriminator_betas]]
    if args.resume:
        checkpoint=Path(args.out).resolve()/'checkpoints/latest/codec.pt'
        if not checkpoint.is_file():raise FileNotFoundError('exact resume requires checkpoints/latest/codec.pt')
        index=command.index('--init-bootstrap');del command[index:index+2]
        command+=['--resume',str(checkpoint)]
    plan.update(experiment='adversarial_from_attribute_bootstrap',jobs=[dict(name='mse_gan',command=command)],
                adversarial=settings,
                losses=dict(semantic_render='MSE + LPIPS weight * LPIPS + GAN weight * softplus(-D(fake))',
                            channel_bootstrap='latent MSE; D paused',
                            channel_render='same image objective + weighted latent MSE'))
    return plan


def main(args,max_updates=None):
    plan=build_plan(args)
    print(json.dumps(plan,indent=2),flush=True)
    if args.dry_run:return plan
    command=plan['jobs'][0]['command']
    # One process, one output folder; nohup PID is the actual training process.
    options=modular_experiment.parser().parse_args(command[4:])
    return modular_experiment.main(options,max_updates=max_updates)


if __name__=='__main__':main(parser().parse_args())
