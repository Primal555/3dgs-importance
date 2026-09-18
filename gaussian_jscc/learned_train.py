"""Render-targeted JSCC with optional feature/local-response pretraining."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .codec import CodecConfig, GaussianCodec
from .data import read_ply, Geometry, morton_order, to_features
from .learned_training import discrete_joint_step, hard_layout
from .optimization import clip_codec_gradients, preserved_rng, update_stats
from .render_objective import bootstrap_loss, MultiViewRenderTask, split_cameras
from .render_validation import append_json, validate_render
from .training import full_scene_step
from .transport import load_checkpoint, save_checkpoint
from .position_delivery import training_position_cost
from .local_response import local_response_loss


def add_parser(sub):
    p = sub.add_parser('train-learned', aliases=['train'], description=__doc__)
    p.set_defaults(func=train)
    for key in ('ply', 'out'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--init', help='learned_joint weights/statistics; fresh optimizer, NOT exact resume')
    p.add_argument('--allocation-init', help='matching route2.pt; requires --init and --joint-steps')
    p.add_argument('--existence-prior', help='.npy probabilities in original input PLY row order')
    p.add_argument('--source')
    p.add_argument('--device', default='cuda')
    p.add_argument('--snr', type=float, default=10.)
    p.add_argument('--channel', choices=['awgn', 'none'], default='awgn')
    p.add_argument('--position-delivery', choices=['learned','float32','quantized'], default='learned',
                   help='explicit modes replace XYZ output with a charged reliable side stream; attributes still use JSCC')
    p.add_argument('--position-bits', type=int, default=12, help='quantized XYZ bits per axis (1..16)')
    p.add_argument('--position-net-bits-per-use', type=float, default=2.,
                   help='assumed digital net information bits/complex use for training charts; not a tested FEC')
    p.add_argument('--bootstrap-steps', '--steps', dest='bootstrap_steps', type=int, default=0,
                   help='optional feature or local-response initialization, NOT the final scene objective')
    p.add_argument('--bootstrap-objective', choices=['feature','local-response'], default='feature',
                   help='local-response: isolated orthographic RGB responses; requires explicit XYZ delivery')
    p.add_argument('--local-response-views', type=int, default=4)
    p.add_argument('--render-steps', type=int, default=1000)
    p.add_argument('--joint-steps', type=int, default=0)
    p.add_argument('--block-size', type=int, default=256)
    p.add_argument('--blocks-per-batch', type=int, default=32)
    p.add_argument('--decoder-window', type=int, default=32)
    p.add_argument('--attention-heads', type=int, default=4)
    p.add_argument('--hidden', type=int, default=96)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--grid-dim', type=int, default=16)
    p.add_argument('--levels', nargs='+', type=int, default=[4,8])
    p.add_argument('--rates', nargs=4, type=int, default=[0,8,16,32])
    p.add_argument('--lr', type=float, default=1e-4, help='bootstrap learning rate')
    p.add_argument('--render-lr', type=float, default=1e-4, help='render and joint codec learning rate')
    p.add_argument('--mask-lr', type=float, default=1e-3)
    p.add_argument('--mask-samples', type=int, default=2)
    p.add_argument('--beta', type=float, default=.001, help='joint-only normalized payload penalty; not a hard cap')
    p.add_argument('--drop', type=float, default=0., help='optional random q0 in mixed codec layouts')
    p.add_argument('--power-floor', type=float, default=.01)
    p.add_argument('--clip-mode', choices=['none','global','branch'], default='none')
    p.add_argument('--clip-norm', type=float, default=10., help='only used if clipping enabled; empirical threshold')
    p.add_argument('--render-backward', choices=['replay','checkpoint'], default='replay')
    p.add_argument('--training-data-device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--resolution', type=int, default=2)
    p.add_argument('--images', default='images')
    p.add_argument('--white-background', action='store_true')
    p.add_argument('--train-views', type=int, default=0, help='spaced training view subset; 0 uses all non-validation views')
    p.add_argument('--views-per-step', type=int, default=2)
    p.add_argument('--validate-every', type=int, default=100)
    p.add_argument('--validation-blocks', type=int, default=8, help='bootstrap diagnostic only')
    p.add_argument('--validation-views', type=int, default=4)
    p.add_argument('--validation-trials', type=int, default=2)
    p.add_argument('--patience', type=int, default=8, help='render-validation checks without improvement; 0 disables')
    p.add_argument('--min-delta', type=float, default=1e-5, help='absolute image MSE improvement for patience')
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu-threads', type=int, default=4)


def train(args):
    from .cli import seed_all, device_for
    from .allocation import GaussianTierMask, scene_fingerprint
    from .route2 import save_joint
    device = device_for(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(args.cpu_threads)
    if min(args.bootstrap_steps,args.render_steps,args.joint_steps,args.patience,args.train_views) < 0 or args.bootstrap_steps+args.render_steps+args.joint_steps < 1:
        raise ValueError('invalid stage lengths/view count')
    if min(args.validate_every,args.validation_blocks,args.validation_views,args.validation_trials,
           args.save_every,args.blocks_per_batch,args.views_per_step,args.cpu_threads,args.local_response_views) < 1:
        raise ValueError('counts must be positive')
    if not 0 <= args.drop < 1 or args.mask_samples < 2:
        raise ValueError('invalid drop probability or mask-samples')
    for key in ('lr','render_lr','mask_lr','clip_norm','power_floor','position_net_bits_per_use'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key) <= 0:
            raise ValueError(f'{key} must be positive and finite')
    for key in ('beta','min_delta'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key) < 0:
            raise ValueError(f'{key} must be nonnegative and finite')
    if not math.isfinite(args.snr):
        raise ValueError('SNR must be finite')
    if args.bootstrap_steps and args.bootstrap_objective == 'local-response' and args.position_delivery == 'learned':
        raise ValueError('local-response bootstrap requires explicit XYZ delivery; no XYZ learning objective is included')
    needs_render = bool(args.render_steps or args.joint_steps)
    if needs_render and (not args.source or not args.device.startswith('cuda')):
        raise ValueError('render/joint stages require CUDA and --source; CPU bootstrap checks set both to 0')
    if args.allocation_init and (not args.init or not args.joint_steps):
        raise ValueError('allocation-init requires init and joint-steps')
    if args.existence_prior and (not args.joint_steps or args.allocation_init):
        raise ValueError('existence-prior requires joint-steps and no allocation-init')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'Use a new output directory: {out}')
    seed_all(args.seed)
    original, degree = read_ply(args.ply)
    fingerprint = scene_fingerprint(original) if args.joint_steps else None
    if args.init:
        model = load_checkpoint(args.init,device).train()
        if model.cfg.architecture != 'learned_joint' or model.cfg.sh_degree != degree:
            raise ValueError('requires matching learned_joint weights; omit --init for old architectures')
        if model.cfg.position_delivery != args.position_delivery or model.cfg.position_bits != args.position_bits:
            raise ValueError('initializer position delivery/bits must match explicit construction flags')
        print('Loaded learned_joint weights/statistics; fresh optimizer. Legacy auxiliary weights are NOT used.',flush=True)
        print('Architecture, rates and feature statistics come from the checkpoint; position-delivery flags must match it.',flush=True)
    else:
        cfg = CodecConfig(architecture='learned_joint',loss_profile='learned_v1',sh_degree=degree,
                          hidden=args.hidden,grid_dim=args.grid_dim,depth=args.depth,levels=tuple(args.levels),
                          planes=False,rates=tuple(args.rates),block_size=args.block_size,
                          decoder_window=args.decoder_window,attention_heads=args.attention_heads,power_floor=args.power_floor,
                          position_delivery=args.position_delivery,position_bits=args.position_bits)
        model = GaussianCodec(cfg).to(device)
        model.attr_mean.copy_(original[:,3:].mean(0).to(device))
        model.attr_std.copy_(original[:,3:].std(0,unbiased=False).clamp_min(.01).to(device))
        if not args.bootstrap_steps:
            print('Training from random weights without bootstrap: valid, but render gradients may be poorly conditioned.',flush=True)
    if args.joint_steps and model.cfg.position_delivery != 'learned':
        raise ValueError('position delivery ablation disables joint mask training: its rate penalty must include XYZ cost first')
    if args.allocation_init:
        from .route2 import load_mask
        mask = load_mask(args.allocation_init,original,model,device).train()
    else:
        prior = np.load(args.existence_prior,allow_pickle=False) if args.existence_prior else None
        mask = GaussianTierMask(len(original),existence_prior=prior).to(device) if args.joint_steps else None
    geometry = Geometry.fit(original[:,:3],model.cfg.morton_bits)
    order = torch.from_numpy(morton_order(geometry.quantize(original[:,:3]).numpy()).astype(np.int64))
    raw = original[order]
    del original
    cache_device = device if args.training_data_device == 'cuda' else torch.device('cpu')
    blocks, ids = [], []
    with torch.no_grad():
        for start in range(0,len(raw),model.cfg.block_size):
            f,_ = to_features(raw[start:start+model.cfg.block_size].to(device),geometry,model)
            blocks.append(f.to(cache_device))
            ids.append(order[start:start+len(f)].to(cache_device))
    validation_indices = sorted(set(np.linspace(0,len(blocks)-1,min(args.validation_blocks,len(blocks)),dtype=int).tolist()))
    training_indices = [i for i in range(len(blocks)) if i not in validation_indices] or list(range(len(blocks)))
    groups = [pad_sequence(blocks[i:i+args.blocks_per_batch],batch_first=True) for i in range(0,len(blocks),args.blocks_per_batch)]
    group_ids = [pad_sequence(ids[i:i+args.blocks_per_batch],batch_first=True,padding_value=-1) for i in range(0,len(blocks),args.blocks_per_batch)]
    cameras = val_cameras = reference = None
    train_indices = view_indices = []
    if needs_render:
        from .rendering import load_cameras, RenderReference
        all_cameras = load_cameras(args.source,args.resolution,args.white_background,args.images,'train')
        cameras,val_cameras,train_indices,view_indices = split_cameras(all_cameras,args.validation_views,args.train_views)
        if args.views_per_step > len(cameras):
            raise ValueError('views-per-step exceeds available training views')
        reference = RenderReference(raw,degree,args.white_background,'source')
        del all_cameras
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    mask_optimizer = torch.optim.Adam(mask.parameters(),lr=args.mask_lr) if mask is not None else None
    out.mkdir(parents=True,exist_ok=False)
    record = {k:v for k,v in vars(args).items() if k != 'func'}
    record.update(objective='render_mse_v1',codec_config=model.cfg.to_dict(),fixed_snr=True,
                  initialization={'mode':'checkpoint' if args.init else 'random',
                                  'checkpoint':str(Path(args.init).resolve()) if args.init else None,
                                  'seed':args.seed,'bootstrap_steps':args.bootstrap_steps,
                                  'feature_statistics':'checkpoint' if args.init else 'computed from input PLY'},
                  source_gaussians=len(raw),train_view_indices=train_indices,validation_view_indices=view_indices,
                  train_view_names=[str(getattr(c,'image_name',i)) for i,c in enumerate(cameras or [])],
                  validation_view_names=[str(getattr(c,'image_name',i)) for i,c in enumerate(val_cameras or [])],
                  bootstrap_validation_blocks=validation_indices,
                  bootstrap_validation_overlap=not bool(set(range(len(blocks)))-set(validation_indices)),
                  scope='scene-trained codec; held-out camera validation, NOT unseen-scene or final test evidence',
                  target='original full PLY rendered images; photographs are evaluation references only',
                  loss_design='mean multiview RGB MSE only during render; no parameter/projection auxiliary',
                  bootstrap_design=('isolated orthographic Gaussian response RGB MSE; shared center, random directions, '
                                    'source+detached-prediction footprint probes, black+white backgrounds; no scene occlusion'
                                    if args.bootstrap_objective == 'local-response' else
                                    'uniform SmoothL1 on normalized features; explicit XYZ modes supervise attributes only'),
                  mask_gradient='REINFORCE image MSE; exact expected normalized payload penalty',
                  budget='per-Gaussian payload; optional Lagrange penalty, NOT a hard cap',
                  metadata='reliable global bbox + per-row tier syntax unchanged; chart payload excludes metadata',
                  clipping='none by default; any threshold is an explicit empirical hyperparameter')
    record.update(position_delivery=model.cfg.position_delivery,
                  position_protocol=('XYZ learned from JSCC payload; no coordinate side stream'
                                     if model.cfg.position_delivery == 'learned' else
                                     'normalized XYZ for q>0 only; reliable side stream; no learned XYZ residual'),
                  comparison='same JSCC payload, NOT equal total rate when side stream is enabled',
                  position_net_bits_per_use=args.position_net_bits_per_use)
    (out/'training.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(f'render_mse_v1: {args.channel} {args.snr:g} dB; rates={model.cfg.rates}; full scene={len(raw)}; '
          f'clip={args.clip_mode}; position={model.cfg.position_delivery}; '
          f'train/val views={len(cameras or [])}/{len(val_cameras or [])}',flush=True)

    def initialization_loss(pred, target):
        if args.bootstrap_objective == 'local-response':
            return local_response_loss(pred,target,geometry,model,args.local_response_views)
        start = 0 if model.cfg.position_delivery == 'learned' else 3
        return bootstrap_loss(pred[:,start:],target[:,start:]), {}

    @torch.no_grad()
    def validate_bootstrap(step):
        # This diagnostic is never substituted for render quality or used to
        # select the semantic codec. Keep it in a separate file/plot.
        with preserved_rng(device):
            seed_all(args.seed+9000)
            model.eval()
            values = []
            for tier in (1,2,3,None):
                losses = []
                for i in validation_indices:
                    f = blocks[i].to(device)
                    q = hard_layout(ids[i].to(device),tier,0.)
                    pred = model(f,f[:,:3],q,args.snr,args.channel)
                    losses.append(float(initialization_loss(pred,f)[0]))
                values.append({'layout':'mixed' if tier is None else str(tier),'loss':sum(losses)/len(losses)})
            model.train()
            append_json(out/'bootstrap_validation.jsonl',{'step':step,'objective':args.bootstrap_objective,'layouts':values})

    def validation(step,phase):
        return validate_render(model,groups,group_ids,raw,geometry,val_cameras,reference,args.snr,
                               args.channel,args.validation_trials,args.seed,out,step,phase,
                               mask=mask if phase == 'joint' else None,white_background=args.white_background,beta=args.beta,
                               position_net_bits_per_use=args.position_net_bits_per_use)

    def save(suffix,phase,step):
        if phase == 'joint':
            save_joint(out,suffix,model,mask,fingerprint,step,record)
        else:
            save_checkpoint(out/f'codec{suffix}.pt',model,step,record)

    step, selections, last_phase = 0, {}, None
    # Record actual render quality of the initializer, before changing weights.
    initial_validation = validation(0,'initial') if needs_render else None
    if args.bootstrap_steps:
        validate_bootstrap(0)
    for phase,maximum in [('bootstrap',args.bootstrap_steps),('render',args.render_steps),('joint',args.joint_steps)]:
        if not maximum:
            continue
        last_phase = phase
        for group in optimizer.param_groups:
            group['lr'] = args.lr if phase == 'bootstrap' else args.render_lr
        # Reset Adam moments across genuinely different objectives; preserve
        # moments across render -> joint, whose distortion is unchanged.
        if phase == 'render' and args.bootstrap_steps:
            optimizer.state.clear()
        best, patience_best, stale = float('inf'),float('inf'),0
        if phase != 'bootstrap':
            initial = initial_validation if phase == 'render' and step == 0 else validation(step,phase)
            best = patience_best = initial['score']
            selections[phase] = {'step':step,'score':best}
            save('_best_'+phase,phase,step)
        for local_step in range(1,maximum+1):
            started = time.perf_counter()
            step += 1
            optimizer.zero_grad(set_to_none=True)
            if mask_optimizer is not None:
                mask_optimizer.zero_grad(set_to_none=True)
            tier = (1,2,3,None)[(local_step-1)%4]
            stats = {'layout':'mixed' if tier is None else str(tier)}
            if phase == 'bootstrap':
                selected = [training_indices[i] for i in torch.randint(len(training_indices),(args.blocks_per_batch,)).tolist()]
                f = pad_sequence([blocks[i] for i in selected],batch_first=True).to(device)
                gi = pad_sequence([ids[i] for i in selected],batch_first=True,padding_value=-1).to(device)
                q = hard_layout(gi,tier,args.drop)
                if not (q>0).any():
                    q[gi>=0] = 1
                choices = torch.nn.functional.one_hot(q,4).to(f)
                pred,_,_ = model.forward_tier_batches(f,f[...,:3],choices,args.snr,args.channel)
                loss, local_stats = initialization_loss(pred[q>0],f[q>0])
                stats.update(local_stats,bootstrap_objective=args.bootstrap_objective)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite bootstrap loss; no optimizer step performed')
                loss.backward()
                stats['bootstrap_loss'] = float(loss.detach())
            else:
                view_ids = torch.randperm(len(cameras))[:args.views_per_step].tolist()
                task = MultiViewRenderTask([cameras[i] for i in view_ids],reference,degree,args.white_background)
                if phase == 'render':
                    qs = [hard_layout(gi,tier,args.drop) for gi in group_ids]
                    if not any((q>0).any() for q in qs):
                        qs = [hard_layout(gi,1,0.) for gi in group_ids]
                    loss, details = full_scene_step(model,list(zip(groups,qs)),geometry,args.snr,args.channel,
                                                    task,attr_weight=0.,mode=args.render_backward)
                    stats.update(details,**task.stats)
                    stats['symbols_per_source_gaussian'] = sum(float(torch.tensor(model.cfg.rates,device=q.device)[q].sum()) for q in qs)/len(raw)
                    stats.update(training_position_cost(model.cfg,details['retained_gaussians'],len(raw),
                                 stats['symbols_per_source_gaussian']*len(raw),args.position_net_bits_per_use))
                else:
                    loss,details = discrete_joint_step(model,mask,groups,group_ids,geometry,args.snr,args.channel,
                                                       task,beta=args.beta,auxiliary_weight=0.,samples=args.mask_samples,
                                                       mode=args.render_backward)
                    stats.update(details)
                    stats['layout'] = 'learned_mask'
                stats.update(image_mse=stats['render_loss'],training_view_indices=view_ids,views_per_step=len(view_ids))
            norm,gradient_stats = clip_codec_gradients(model,args.clip_norm,args.clip_mode)
            before = {name:p.detach().clone() for name,p in model.named_parameters()}
            if phase == 'joint':
                mask_norm = torch.stack([p.grad.norm() for p in mask.parameters() if p.grad is not None]).norm()
                if not torch.isfinite(mask_norm):
                    raise RuntimeError('nonfinite mask gradient; no optimizer steps performed')
                stats['mask_grad_norm'] = float(mask_norm)
                mask_optimizer.step()
            optimizer.step()
            updates = update_stats(model,before)
            update_norm = math.sqrt(sum(v['update_norm']**2 for v in updates.values()))
            row = {'step':step,'phase':phase,'objective':'render_mse_v1','loss':float(loss.detach()),
                   'snr':args.snr,'lr':optimizer.param_groups[0]['lr'],'grad_norm':float(norm),
                   'update_norm':update_norm,'updates':updates,'step_seconds':time.perf_counter()-started,
                   **stats,**gradient_stats}
            append_json(out/'loss.jsonl',row)
            if local_step == 1 or local_step%10 == 0:
                print(f'{phase} {local_step}/{maximum}: loss={row["loss"]:.6f}, grad={float(norm):.4g}, '
                      f'update={update_norm:.4g}, sec={row["step_seconds"]:.2f}',flush=True)
            if step%args.save_every == 0:
                save(f'_{step}',phase,step)
            if local_step%args.validate_every == 0 or local_step == maximum:
                if phase == 'bootstrap':
                    validate_bootstrap(step)
                    # Track actual scene quality during initialization too; the
                    # local objective is not a substitute for held-out renders.
                    if needs_render:
                        validation(step,phase)
                else:
                    score = validation(step,phase)['score']
                    # Best checkpoint tracks EVERY improvement; patience has
                    # its own tolerance and must not discard a better model.
                    if score < best:
                        best = score
                        selections[phase] = {'step':step,'score':best}
                        save('_best_'+phase,phase,step)
                    if score < patience_best-args.min_delta:
                        patience_best,stale = score,0
                    else:
                        stale += 1
                    if args.patience and stale >= args.patience:
                        print(f'{phase}: stopping after {stale} validation checks without sufficient MSE improvement.',flush=True)
                        break
        save('_end_'+phase,phase,step)
    # Final and best are deliberately different files; do not call the last
    # weights "best", silently restore them, or mismatch a codec/mask pair.
    save('',last_phase,step)
    (out/'selection.json').write_text(json.dumps({'best_by_phase':selections,'codec.pt':'last executed step; not necessarily best',
                                                'evaluation':'best_render or matched best_joint pair; final held-out test still required'},indent=2),encoding='utf-8')
    print(f'Saved last codec: {out/"codec.pt"}; render-selected checkpoints: {selections}',flush=True)
    from .plots import safe_plot
    safe_plot('training',out)
