"""Render-targeted JSCC plus optional/isolated bootstrap experiments."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .codec import CodecConfig, GaussianCodec
from .data import read_ply, Geometry, morton_order, to_features, fit_feature_statistics
from .learned_training import discrete_joint_step, hard_layout
from .optimization import clip_codec_gradients, preserved_rng, update_stats, ValidationLRSchedule
from .render_objective import bootstrap_loss, MultiViewRenderTask, split_cameras
from .render_validation import append_json, validate_render
from .training import full_scene_step
from .transport import load_checkpoint, save_checkpoint
from .position_delivery import training_position_cost
from .local_response import local_response_loss
from .spatial_response import spatial_response_loss, DEFAULT_BANDWIDTHS
from .transformer_decoder import capture_xyz_stages


def add_parser(sub):
    p = sub.add_parser('train-learned', aliases=['train'], description=__doc__)
    p.set_defaults(func=train)
    for key in ('ply', 'out'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--init', help='learned codec weights/statistics; fresh optimizer, NOT exact resume')
    p.add_argument('--architecture', choices=['learned_joint', 'learned_split', 'learned_split_logcov'], default=None,
                   help='random start defaults to learned_joint; init inherits checkpoint unless explicitly checked')
    p.add_argument('--allocation-init', help='matching route2.pt; requires --init and --joint-steps')
    p.add_argument('--context-mode', choices=['window','multiscale_self'], default=None,
                   help='multiscale_self adds pointwise paths and pooled context; random start required when changing mode')
    p.add_argument('--encoder-attention', choices=['window','geometric_point'], default=None)
    p.add_argument('--encoder-neighbors', type=int, default=None)
    p.add_argument('--decoder-attention',choices=['window','feature_point','transformer_trunk'],default=None)
    p.add_argument('--decoder-depth',type=int,default=None,help='Transformer trunk layers, >=3; independent of encoder depth')
    p.add_argument('--decoder-memory', choices=['none','received'], default=None)
    p.add_argument('--decoder-refinement', choices=['none','progressive'], default=None)
    p.add_argument('--decoder-neighbors',type=int,default=None)
    p.add_argument('--xyz-decoder',choices=['additive','block_center','context_center','residual_center','symbol_skip'],default=None)
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
    p.add_argument('--bootstrap-tier', type=int, choices=[1,2,3], default=None,
                   help='isolated bootstrap-only test: fix all retained points to this tier, including validation')
    p.add_argument('--bootstrap-objective', choices=['feature','local-response','spatial-response'], default='feature',
                   help='local-response requires explicit XYZ; spatial-response is a learned-XYZ bootstrap-only experiment')
    p.add_argument('--local-response-views', type=int, default=4)
    p.add_argument('--spatial-bandwidths', nargs='+', type=float, default=list(DEFAULT_BANDWIDTHS),
                   help='fixed position-only kernel widths in bbox-diagonal units; native shape remains unblurred')
    p.add_argument('--spatial-fine-weight', type=float, default=1.,
                   help='teacher-radius bounded-slope position loss weight; 0 is the v2 objective ablation')
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
    p.add_argument('--lr', type=float, default=2e-4, help='bootstrap learning rate')
    p.add_argument('--render-lr', type=float, default=2e-4, help='render and joint codec learning rate')
    p.add_argument('--lr-schedule', choices=['plateau','constant'], default='constant')
    p.add_argument('--lr-factor', type=float, default=.5)
    p.add_argument('--lr-patience', type=int, default=3, help='consecutive bad validation checks before LR reduction')
    p.add_argument('--lr-threshold', type=float, default=.005, help='relative validation improvement required by LR scheduler')
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--mask-lr', type=float, default=1e-3)
    p.add_argument('--mask-samples', type=int, default=2)
    p.add_argument('--beta', type=float, default=.001, help='joint-only normalized payload penalty; not a hard cap')
    p.add_argument('--drop', type=float, default=0., help='optional random q0 in mixed codec layouts')
    p.add_argument('--power-floor', type=float, default=.01)
    p.add_argument('--clip-mode', choices=['none','global','branch'], default='none')
    p.add_argument('--clip-norm', type=float, default=10., help='only used if clipping enabled; empirical threshold')
    p.add_argument('--render-backward', choices=['direct','replay','checkpoint'], default='replay',
                   help='replay recomputes codec batches with the same channel RNG to save memory; direct retains all activations')
    p.add_argument('--training-data-device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--resolution', type=int, default=2)
    p.add_argument('--images', default='images')
    p.add_argument('--white-background', action='store_true')
    p.add_argument('--train-views', type=int, default=0, help='spaced training view subset; 0 uses all non-validation views')
    p.add_argument('--views-per-step', type=int, default=2)
    p.add_argument('--validate-every', type=int, default=100)
    p.add_argument('--validation-blocks', type=int, default=8, help='bootstrap diagnostic only')
    p.add_argument('--validation-region-size', type=int, default=None,
                   help='optional fixed Morton-region size, divisible by block-size; validation-blocks counts regions, preserving heldout points across block sizes')
    p.add_argument('--validation-views', type=int, default=4)
    p.add_argument('--validation-trials', type=int, default=2)
    p.add_argument('--patience', type=int, default=8, help='render-validation checks without improvement; 0 disables')
    p.add_argument('--min-delta', type=float, default=1e-5, help='absolute image MSE improvement for patience')
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu-threads', type=int, default=4)


def bootstrap_split(point_count, block_size, count, region_size=None):
    """Fixed point regions for context-size experiments, no holdout leakage."""
    total = (point_count+block_size-1)//block_size
    if region_size is None:
        validation = sorted(set(np.linspace(0,total-1,min(count,total),dtype=int).tolist()))
        training = [i for i in range(total) if i not in validation] or list(range(total))
        return training, validation
    if region_size < block_size or region_size % block_size:
        raise ValueError('validation-region-size must be a positive multiple of block-size')
    regions = point_count//region_size
    if regions <= count:
        raise ValueError('fixed-region validation requires more complete regions than validation-blocks')
    selected = np.linspace(0,regions-1,count,dtype=int).tolist()
    ratio = region_size//block_size
    validation = [r*ratio+j for r in selected for j in range(ratio)]
    return [i for i in range(total) if i not in validation], validation


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
           args.save_every,args.blocks_per_batch,args.views_per_step,args.cpu_threads,args.local_response_views,args.lr_patience) < 1:
        raise ValueError('counts must be positive')
    if not 0 <= args.drop < 1 or args.mask_samples < 2:
        raise ValueError('invalid drop probability or mask-samples')
    if args.bootstrap_tier is not None and (args.render_steps or args.joint_steps or not args.bootstrap_steps or args.drop):
        raise ValueError('bootstrap-tier requires bootstrap-only training and drop=0')
    for key in ('lr','render_lr','mask_lr','clip_norm','power_floor','position_net_bits_per_use','min_lr'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key) <= 0:
            raise ValueError(f'{key} must be positive and finite')
    for key in ('beta','min_delta'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key) < 0:
            raise ValueError(f'{key} must be nonnegative and finite')
    if not math.isfinite(args.snr):
        raise ValueError('SNR must be finite')
    if not 0 < args.lr_factor < 1 or not 0 <= args.lr_threshold < 1:
        raise ValueError('invalid LR factor or relative threshold')
    active_lrs = ([args.lr] if args.bootstrap_steps else []) + ([args.render_lr] if args.render_steps or args.joint_steps else [])
    if args.lr_schedule == 'plateau' and args.min_lr > min(active_lrs):
        raise ValueError('min-lr must not exceed any active phase starting LR')
    if args.bootstrap_steps and args.bootstrap_objective == 'local-response' and args.position_delivery == 'learned':
        raise ValueError('local-response bootstrap requires explicit XYZ delivery; no XYZ learning objective is included')
    spatial_test = args.bootstrap_objective == 'spatial-response'
    if spatial_test:
        if args.position_delivery != 'learned' or args.render_steps or args.joint_steps or not args.bootstrap_steps:
            raise ValueError('spatial-response test requires learned XYZ and bootstrap-only training')
        if not args.spatial_bandwidths or any(not math.isfinite(s) or s <= 0 for s in args.spatial_bandwidths):
            raise ValueError('spatial bandwidths must be positive and finite')
        if not math.isfinite(args.spatial_fine_weight) or args.spatial_fine_weight<0:
            raise ValueError('spatial-fine-weight must be finite and nonnegative')
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
        if model.cfg.sh_degree != degree:
            raise ValueError('initializer SH degree must match the input PLY')
        if args.architecture is not None and args.architecture != model.cfg.architecture:
            raise ValueError('initializer architecture mismatch; a new architecture requires random initialization')
        if args.context_mode is not None and args.context_mode != model.cfg.context_mode:
            raise ValueError('initializer context mode mismatch; start the new structure from random weights')
        if args.encoder_attention is not None and args.encoder_attention != model.cfg.encoder_attention:
            raise ValueError('initializer encoder attention mismatch; start from random weights')
        if args.encoder_neighbors is not None and args.encoder_neighbors != model.cfg.encoder_neighbors:
            raise ValueError('initializer encoder neighbors mismatch')
        if args.decoder_attention is not None and args.decoder_attention!=model.cfg.decoder_attention:
            raise ValueError('initializer decoder attention mismatch; start from random weights')
        if args.decoder_neighbors is not None and args.decoder_neighbors!=model.cfg.decoder_neighbors:
            raise ValueError('initializer decoder neighbors mismatch')
        if args.decoder_depth is not None and args.decoder_depth!=model.cfg.decoder_depth:
            raise ValueError('initializer decoder depth mismatch; start from random weights')
        if args.decoder_memory is not None and args.decoder_memory!=model.cfg.decoder_memory:
            raise ValueError('initializer decoder memory mismatch; start from random weights')
        if args.decoder_refinement is not None and args.decoder_refinement!=model.cfg.decoder_refinement:
            raise ValueError('initializer decoder refinement mismatch; start from random weights')
        if args.xyz_decoder is not None and args.xyz_decoder != model.cfg.xyz_decoder:
            raise ValueError('initializer XYZ decoder mismatch; start from random weights')
        if model.cfg.position_delivery != args.position_delivery or model.cfg.position_bits != args.position_bits:
            raise ValueError('initializer position delivery/bits must match explicit construction flags')
        print(f'Loaded {model.cfg.architecture} weights/statistics; fresh optimizer. Legacy auxiliary weights are NOT used.',flush=True)
        print('Architecture, rates and feature statistics come from the checkpoint; position-delivery flags must match it.',flush=True)
    else:
        cfg = CodecConfig(architecture=args.architecture or 'learned_joint',loss_profile='learned_v1',sh_degree=degree,
                          context_mode=args.context_mode or 'window',
                          encoder_attention=args.encoder_attention or 'window',
                          encoder_neighbors=16 if args.encoder_neighbors is None else args.encoder_neighbors,
                          decoder_attention=args.decoder_attention or 'window',
                          decoder_neighbors=16 if args.decoder_neighbors is None else args.decoder_neighbors,
                          decoder_depth=4 if args.decoder_depth is None else args.decoder_depth,
                          decoder_memory=args.decoder_memory or 'none',
                          decoder_refinement=args.decoder_refinement or 'none',
                          xyz_decoder=args.xyz_decoder or 'additive',
                          hidden=args.hidden,grid_dim=args.grid_dim,depth=args.depth,levels=tuple(args.levels),
                          planes=False,rates=tuple(args.rates),block_size=args.block_size,
                          decoder_window=args.decoder_window,attention_heads=args.attention_heads,power_floor=args.power_floor,
                          position_delivery=args.position_delivery,position_bits=args.position_bits)
        model = GaussianCodec(cfg).to(device)
        fit_feature_statistics(original, model)
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
    training_indices, validation_indices = bootstrap_split(len(raw), model.cfg.block_size, args.validation_blocks,
                                                          args.validation_region_size)
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
                  bootstrap_validation_points=sum(len(blocks[i]) for i in validation_indices),
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
    if spatial_test:
        record.update(objective='spatial_response_v3',
                      target='decoupled position, native shape and centered appearance; NOT scene rendering',
                      loss_design='(coarse_position + fine_weight * teacher_radius_pseudo_huber + native_shape_negative_log_overlap + centered_RGB_response) / 3; empirical weights',
                      fine_position_design='3D distance; teacher max-axis transition radius; divided by min(spatial_bandwidths); slope bounded by inverse reference bandwidth in bbox-diagonal coordinates',
                      bootstrap_design='XYZ-only fixed kernels; unblurred coincident-center covariance; source/native footprint RGB probes',
                      scope='bootstrap-only; held-out spatial blocks from the same scene, NOT held-out rendered views',
                      diagnostic_aggregation='equal block/trial means; XYZ RMSE is mean per-block RMSE, not pooled global RMSE',
                      spatial_bandwidth_units='fraction of global source bbox diagonal; position-only cold-start scales, NOT shape smoothing',
                      initialization_success='requires native scale diagnostics AND offline fixed-view render quality; bootstrap loss alone is insufficient',
                      position_side_stream_bits=0)
    if model.cfg.architecture == 'learned_split_logcov':
        record.update(shape_representation='symmetric log covariance (xx,xy,xz,yy,yz,zz)',
                      renderer_shape='matrix_exp -> packed covariance -> cov3D_precomp; no eigenvectors',
                      export_shape='eigh -> scale/quaternion only for no-grad PLY export')
        if spatial_test:
            record.update(objective='spatial_logcov_v1',
                          loss_design='(coarse_position + fine_weight * teacher_radius_pseudo_huber + logcov_Frobenius_squared/9 + centered_RGB_response) / 3; empirical equal group weights',
                          bootstrap_design='unchanged XYZ kernels and centered RGB; replaces native shape overlap by physical logcov MSE')
    if model.cfg.context_mode == 'multiscale_self':
        record.update(encoder_attention=model.cfg.encoder_attention, encoder_neighbors=model.cfg.encoder_neighbors,
                      encoder_graph='source kNN within each block/pooled scale; geometry enters weights and values'
                      if model.cfg.encoder_attention == 'geometric_point' else 'Morton windows with relative attention bias')
        record.update(context_design='pointwise paths plus tanh-gated fine/x4/x16 pooled context within each block',
                      context_gate_initialization='tanh(0.1), trainable engineering initialization, NOT a loss weight',
                      receiver_context='packet-slot features only; no source XYZ, centroid or pooling features cross channel',
                      unchanged='logcov representation, spatial_logcov_v1 objective, per-Gaussian symbol budgets and normalization')
    record.update(tier_protocol=f'fixed q{args.bootstrap_tier}' if args.bootstrap_tier is not None else 'cycle q1/q2/q3/mixed',
                  channel_protocol='identity channel; SNR is conditioning only' if args.channel=='none' else 'AWGN at configured SNR')
    record['xyz_decoder_design'] = ('received-feature mean -> learned block center + zero-mean own offsets + gated zero-mean Context offsets; no source center'
                                   if model.cfg.xyz_decoder=='block_center' else 'own absolute XYZ + gated Context correction')
    if model.cfg.xyz_decoder=='residual_center':
        record['xyz_decoder_design']='unchanged additive XYZ + zero-initialized block translation residual from both received-feature paths; no source center'
    if model.cfg.xyz_decoder=='symbol_skip':
        record['xyz_decoder_design']='unchanged additive XYZ + zero-initialized learned linear readout of each received symbol vector; no coordinate side stream'
    if model.cfg.xyz_decoder=='context_center':
        record['xyz_decoder_design']='block_center with pooled received own AND geometry Context features for centroid prediction; new Context weights start at zero; zero-mean offsets; no source center'
    record['decoder_attention_design']=('received-feature cosine kNN, grouped relation attention; no source XYZ or slot sinusoid; x4/x16 pooling still uses packet slots'
                                        if model.cfg.decoder_attention=='feature_point' else 'shifted sequence-window multihead attention with slot sinusoid')
    if model.cfg.decoder_attention=='transformer_trunk':
        record['decoder_attention_design']='received-only full attention within each codec block; pre-norm residual Transformer main path; no kNN, slot IDs, pooling or gated output bypass'
        record['xyz_decoder_design']='separately normalized shallow/middle/deep Transformer features -> concatenation -> nonlinear XYZ readout; no source coordinates'
        record['decoder_xyz_taps']=[i+1 for i in getattr(model.learned.dec_trunk,'tap_indices',())]
        record['decoder_memory_design'] = ('per-layer cross-attention to fixed received payload embeddings; zero-start output projections; memory is not detached; no new symbols or localization token'
                                            if model.cfg.decoder_memory=='received' else 'none')
        record['context_design']='encoder retains gated fine/x4/x16 context; receiver replaced by block Transformer trunk'
        record['context_gate_initialization']='encoder only: tanh(0.1); no receiver Context gate'
        record['receiver_context']='full attention over received tokens within codec block; no source geometry, slot embedding or hard neighbors'
        if model.cfg.decoder_refinement == 'progressive':
            record['xyz_decoder_design']='first-layer learned XYZ then per-point additive refinement after each subsequent Transformer layer; same received embedding at every refinement; predicted absolute/centered XYZ features and soft predicted squared-distance attention bias; no detach or extra transmission'
            record['decoder_attention_design']='full block attention; learned per-head softplus distance precision in normalized bbox coordinates, no hard kNN'
            record['decoder_refinement_stages']=model.cfg.decoder_depth
            record['refinement_initialization']='initial XYZ output std=.02,bias=.5; delta output std=.002,bias=0; learnable log_precision=0; engineering initialization, NOT extra loss weights'
    (out/'training.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    if args.bootstrap_tier is not None:
        print(f'Fixed bootstrap/validation q{args.bootstrap_tier}: {model.cfg.rates[args.bootstrap_tier]} complex symbols/G; {record["channel_protocol"]}.',flush=True)
    print(f'{record["objective"]}: {args.channel} {args.snr:g} dB; rates={model.cfg.rates}; full scene={len(raw)}; '
          f'clip={args.clip_mode}; position={model.cfg.position_delivery}; '
          f'train/val views={len(cameras or [])}/{len(val_cameras or [])}',flush=True)
    backward_description = {
        'replay': 'recompute codec batches with matching channel RNG; retain one batch of codec activations.',
        'direct': 'retain the entire codec graph; no automatic recomputation fallback.',
        'checkpoint': 'checkpoint codec batches and recompute during backward.',
    }
    if spatial_test:
        print(f'Bootstrap-only: ordinary minibatch backward; no scene replay. LR schedule: {args.lr_schedule}.',flush=True)
    else:
        print(f'Backward: {args.render_backward}; LR schedule: {args.lr_schedule}. '
              + backward_description[args.render_backward],flush=True)

    def initialization_loss(pred, target):
        if spatial_test:
            return spatial_response_loss(pred,target,geometry,model,args.local_response_views,args.spatial_bandwidths,
                                         fine_weight=args.spatial_fine_weight)
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
            for tier in ((args.bootstrap_tier,) if args.bootstrap_tier is not None else (1,2,3,None)):
                losses = []
                measurements = []
                symbols, points = 0, 0
                xyz_squared_sum, xyz_coordinates = 0., 0
                stage_squared_sums = {}
                for trial in range(args.validation_trials if spatial_test else 1):
                    for i in validation_indices:
                        f = blocks[i].to(device)
                        q = hard_layout(ids[i].to(device),tier,0.)
                        if spatial_test:
                            symbols += int(torch.tensor(model.cfg.rates,device=q.device)[q].sum())
                            points += len(q)
                        with capture_xyz_stages(model) as stages:
                            pred = model(f,f[:,:3],q,args.snr,args.channel)
                        for depth, xyz in enumerate(stages):
                            error = (xyz.reshape(-1,3)-f[:,:3]).double()*geometry.span.to(f).double()
                            stage_squared_sums[depth] = stage_squared_sums.get(depth,0.)+float(error.square().sum())
                        value, diagnostics = initialization_loss(pred,f)
                        if spatial_test:
                            delta=(pred[:,:3]-f[:,:3]).double()*geometry.span.to(pred).double()
                            xyz_squared_sum += float(delta.square().sum())
                            xyz_coordinates += delta.numel()
                            common=delta.mean(0,keepdim=True)
                            diagnostics.update(block_xyz_common_mse=float(common.square().mean()),
                                               block_xyz_relative_mse=float((delta-common).square().mean()))
                        losses.append(float(value))
                        measurements.append(diagnostics)
                entry = {'layout':'mixed' if tier is None else str(tier),'loss':sum(losses)/len(losses)}
                if spatial_test:
                    entry.update({key:sum(r[key] for r in measurements)/len(measurements) for key in measurements[0]})
                    entry.update(position_side_stream_bits=0,validation_blocks=len(validation_indices),
                                 validation_trials=args.validation_trials,
                                 xyz_pooled_world_rmse=math.sqrt(xyz_squared_sum/xyz_coordinates),
                                 symbols_per_gaussian=symbols/points)
                    if stage_squared_sums:
                        entry['xyz_stage_pooled_world_rmse'] = [math.sqrt(stage_squared_sums[d]/xyz_coordinates) for d in sorted(stage_squared_sums)]
                values.append(entry)
            model.train()
            append_json(out/'bootstrap_validation.jsonl',{'step':step,'objective':args.bootstrap_objective,
                                                         'loss_version':record['objective'],'layouts':values,
                                                         'channel':args.channel,'snr_conditioning':args.snr})
            if spatial_test:
                print(f'Spatial validation step={step}: '+', '.join(
                    f'q{v["layout"]} XYZ NRMSE={v["xyz_nrmse_bbox"]:.6g}'
                    f' radius_ratio_p50={v["max_axis_ratio_p50"]:.3g}' for v in values),flush=True)
            return sum(v['loss'] for v in values)/len(values)

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
    initial_bootstrap_score = validate_bootstrap(0) if args.bootstrap_steps else None
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
        lr_schedule = ValidationLRSchedule(optimizer,args.lr_schedule,args.lr_factor,args.lr_patience,
                                           args.lr_threshold,args.min_lr)

        def schedule_lr(score, baseline=False):
            event = lr_schedule.observe(score)
            event.update(step=step,phase=phase,baseline=baseline,
                         monitor='local_validation_loss' if phase == 'bootstrap' else 'render_validation_score')
            append_json(out/'lr_schedule.jsonl',event)
            if event['reduced']:
                print(f'{phase}: validation plateau, LR {event["lr_before"][0]:g} -> '
                      f'{event["lr_after"][0]:g}',flush=True)
            return event['reduced']

        best, patience_best, stale = float('inf'),float('inf'),0
        if phase != 'bootstrap':
            initial = initial_validation if phase == 'render' and step == 0 else validation(step,phase)
            best = patience_best = initial['score']
            selections[phase] = {'step':step,'score':best}
            save('_best_'+phase,phase,step)
            schedule_lr(best,baseline=True)
        else:
            schedule_lr(initial_bootstrap_score,baseline=True)
            if spatial_test:
                best = initial_bootstrap_score
                selections[phase] = {'step':0,'score':best,'criterion':'mean_layout_spatial_response'}
                save('_best_bootstrap',phase,0)
        for local_step in range(1,maximum+1):
            started = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            step += 1
            optimizer.zero_grad(set_to_none=True)
            if mask_optimizer is not None:
                mask_optimizer.zero_grad(set_to_none=True)
            tier = args.bootstrap_tier if args.bootstrap_tier is not None else (1,2,3,None)[(local_step-1)%4]
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
                stats.update(tier_counts=torch.bincount(q[gi>=0],minlength=4).tolist(),
                             symbols_per_source_gaussian=float(torch.tensor(model.cfg.rates,device=q.device)[q[gi>=0]].float().mean()))
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
            if hasattr(model.learned,'context_diagnostics'):
                stats['context_gates'] = model.learned.context_diagnostics()
            update_norm = math.sqrt(sum(v['update_norm']**2 for v in updates.values()))
            row = {'step':step,'phase':phase,'objective':record['objective'],'loss':float(loss.detach()),
                   'snr':args.snr,'lr':optimizer.param_groups[0]['lr'],'grad_norm':float(norm),
                   'update_norm':update_norm,'updates':updates,'step_seconds':time.perf_counter()-started,
                   **stats,**gradient_stats}
            if device.type == 'cuda':
                row.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                           peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20)
            append_json(out/'loss.jsonl',row)
            if local_step == 1 or local_step%10 == 0:
                spatial_note = (f', pos={stats["spatial_position_response"]:.4g}'
                                f', fine={stats["spatial_fine_position_response"]:.4g}'
                                f', shape={stats["spatial_shape_objective"]:.4g}'
                                f', appearance={stats["spatial_appearance_response"]:.4g}'
                                f', radius_ratio_p50={stats["max_axis_ratio_p50"]:.3g}'
                                if spatial_test else '')
                print(f'{phase} {local_step}/{maximum}: loss={row["loss"]:.6f}, grad={float(norm):.4g}, '
                      f'update={update_norm:.4g}, lr={row["lr"]:g}, sec={row["step_seconds"]:.2f}'
                      + spatial_note,flush=True)
            if step%args.save_every == 0:
                save(f'_{step}',phase,step)
            if local_step%args.validate_every == 0 or local_step == maximum:
                if phase == 'bootstrap':
                    bootstrap_score = validate_bootstrap(step)
                    schedule_lr(bootstrap_score)
                    if spatial_test and bootstrap_score < best:
                        best = bootstrap_score
                        selections[phase] = {'step':step,'score':best,'criterion':'mean_layout_spatial_response'}
                        save('_best_bootstrap',phase,step)
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
                    if schedule_lr(score):
                        # Let the reduced LR optimize before early stopping.
                        stale = 0
                    if args.patience and stale >= args.patience:
                        print(f'{phase}: stopping after {stale} validation checks without sufficient MSE improvement.',flush=True)
                        break
        save('_end_'+phase,phase,step)
    # Final and best are deliberately different files; do not call the last
    # weights "best", silently restore them, or mismatch a codec/mask pair.
    save('',last_phase,step)
    (out/'selection.json').write_text(json.dumps({'best_by_phase':selections,'codec.pt':'last executed step; not necessarily best',
                                                'evaluation':('best_bootstrap selects local spatial response, NOT render quality' if spatial_test else
                                                              'best_render or matched best_joint pair; final held-out test still required')},indent=2),encoding='utf-8')
    print(f'Saved last codec: {out/"codec.pt"}; selected checkpoints: {selections}',flush=True)
    from .plots import safe_plot
    safe_plot('training',out)
