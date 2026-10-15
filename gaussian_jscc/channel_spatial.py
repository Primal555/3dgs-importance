"""Receiver-only, stage-causal residual denoising within original codec blocks."""
import math
import torch
from torch import nn
from .pointnet_context import gather, metric_coordinates


class ChannelSpatialDenoiser(nn.Module):
    def __init__(self,cfg,tier):
        super().__init__()
        self.mode=cfg.channel_denoiser_mode
        self.neighbors=cfg.channel_denoiser_neighbors
        self.tier=tier; self.symbols=cfg.rates[tier]
        h=cfg.channel_denoiser_hidden
        self.input=nn.Sequential(nn.Linear(cfg.latent_dim+4*cfg.rates[-1]+1,h),nn.GELU(),nn.Linear(h,h))
        self.query=nn.Linear(h,h,bias=False); self.key=nn.Linear(h,h,bias=False)
        self.value=nn.Linear(h,h)
        self.relative=nn.Sequential(nn.Linear(4,h),nn.GELU(),nn.Linear(h,h))
        self.fuse=nn.Sequential(nn.Linear(2*h,h),nn.GELU())
        self.readout=nn.Linear(h,cfg.latent_dim)
        nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)
        self.register_buffer('scene_span',torch.ones(3),persistent=False)
        self.geometry_ready=False

    def set_scene_geometry(self,geometry):
        span=geometry.span.to(self.scene_span)
        if span.shape!=(3,) or not torch.isfinite(span).all() or (span<=0).any():
            raise ValueError('receiver denoiser requires finite positive XYZ bbox span')
        self.scene_span.copy_(span); self.geometry_ready=True

    @torch.no_grad()
    def neighborhood(self,xyz,active):
        if not self.geometry_ready: raise ValueError('receiver denoiser requires XYZ bbox metadata')
        if xyz is None or xyz.shape!=(*active.shape,3) or not torch.isfinite(xyz[active]).all():
            raise ValueError('receiver denoiser requires matching finite delivered XYZ')
        xyz=metric_coordinates(xyz,self.scene_span,active)
        n=xyz.shape[1]; count=min(self.neighbors,n)
        if self.mode=='pointwise':
            ids=torch.arange(n,device=xyz.device)[None,:,None].expand(len(xyz),-1,count)
            valid=active[...,None].expand_as(ids)
            return ids,valid,xyz.new_zeros(*ids.shape,4)
        distances=torch.cdist(xyz.float(),xyz.float(),compute_mode='donot_use_mm_for_euclid_dist')
        distances=distances.masked_fill(~active[:,None,:],float('inf'))
        # Coincident centers may otherwise evict self through top-k tie breaks.
        diagonal=torch.arange(n,device=xyz.device)
        distances[:,diagonal,diagonal]=torch.where(active,-1.,float('inf'))
        d,ids=distances.topk(count,dim=-1,largest=False,sorted=True)
        valid=active[...,None]&torch.isfinite(d)
        # Kth-neighbor length normalizes offsets; distances themselves retain
        # world-space ordering. Geometry is detached and never re-estimated.
        radius=d.masked_fill(~valid,0).amax(-1,keepdim=True).clamp_min(1e-6)
        relative=(gather(xyz,ids)-xyz[:,:,None])/radius[...,None]
        relative=relative.masked_fill(~valid[...,None],0)
        return ids,valid,torch.cat((relative,relative.norm(dim=-1,keepdim=True)),-1)

    def forward(self,previous,received,receive_mask,xyz,active,snr):
        if not math.isfinite(float(snr)): raise ValueError('SNR must be finite')
        if received.shape!=receive_mask.shape or received.shape[:2]!=active.shape or previous.shape[:2]!=active.shape:
            raise ValueError('invalid receiver denoiser shapes')
        ids,valid,relative=self.neighborhood(xyz,active)
        # Even in mixed maps no neighboring point can expose a future prefix.
        stage_mask=receive_mask.bool().clone(); stage_mask[...,2*self.symbols:]=False
        stage_mask &= active[...,None]
        signal=received.masked_fill(~stage_mask,0)
        own=previous.masked_fill(~active[...,None],0)
        inputs=torch.cat((own,signal,stage_mask.to(signal),signal.new_full((*active.shape,1),float(snr)/20)),-1)
        h=self.input(inputs).masked_fill(~active[...,None],0)
        keys=gather(self.key(h),ids); values=gather(self.value(h),ids)+self.relative(relative)
        logits=(self.query(h)[:,:,None]*keys).sum(-1)/math.sqrt(h.shape[-1])
        logits=logits.masked_fill(~valid,-torch.finfo(logits.dtype).max)
        weights=logits.softmax(-1)*valid
        context=(values.masked_fill(~valid[...,None],0)*weights[...,None]).sum(2)
        correction=self.readout(self.fuse(torch.cat((h,context),-1)))
        return correction.masked_fill(~active[...,None],0)


def enable_channel_denoising(model,mode,*,hidden=64,neighbors=16):
    from .modular_config import ModularCodecConfig
    if model.cfg.channel_denoiser_mode!='none': raise ValueError('receiver denoiser already enabled')
    values=model.cfg.to_dict()
    values.update(channel_denoiser_mode=mode,channel_denoiser_hidden=hidden,channel_denoiser_neighbors=neighbors)
    cfg=ModularCodecConfig.from_dict(values)
    # Preserve the shared initialization/random stream across control modes.
    reference=next(model.parameters())
    with torch.random.fork_rng(devices=[]):
        blocks=nn.ModuleList(ChannelSpatialDenoiser(cfg,t) for t in (1,2,3)).to(reference)
    model.channel_decoder.denoisers=blocks
    model.cfg.channel_denoiser_mode=mode; model.cfg.channel_denoiser_hidden=hidden
    model.cfg.channel_denoiser_neighbors=neighbors


@torch.no_grad()
def diagnose_context(model,scene,out,*,points,trials,batch_size,seed,snr):
    """Sample outputs only AFTER full-block spatial decoding, never regroup y."""
    import csv
    import hashlib
    from pathlib import Path
    from .codec import channel,prefix_mask
    from .latent_error_diagnostics import error_moments
    from .modular_experiment import json_write
    from .optimization import preserved_rng
    from .position_delivery import delivered_positions
    from .transport import model_id
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    total=sum(int((ids>=0).sum()) for ids in scene.group_ids)
    if total<1:raise ValueError('diagnostics requires active scene points')
    selected=torch.randperm(total,generator=torch.Generator().manual_seed(seed))[:min(points,total)].sort().values
    device=next(model.parameters()).device;mode=model.training;identity=model_id(model)
    targets=[];signals=[];raw_signals=[];predictions=[[[] for _ in range(trials+1)] for _ in range(3)]
    offset=0
    try:
        with preserved_rng(device):
            model.eval();model.set_scene_geometry(scene.geometry)
            for group,(features,ids) in enumerate(zip(scene.groups,scene.group_ids)):
                valid=ids>=0;count=int(valid.sum())
                local=selected[(selected>=offset)&(selected<offset+count)]-offset;offset+=count
                if not local.numel():continue
                f=features.to(device);active=valid.to(device);local=local.to(device)
                y=model.semantic_encoder(f,f[...,:3],active)
                values={};hook=model.channel_encoder.readout.register_forward_hook(lambda m,a,o:values.update(raw=o))
                try:z=model.channel_encoder(y,active,snr)
                finally:hook.remove()
                targets.append(y[active][local].cpu());signals.append(z[active][local].cpu())
                raw_signals.append(values['raw'][active][local].cpu())
                side=delivered_positions(f[...,:3],active.long(),model.cfg)
                for tier in (1,2,3):
                    q=active.long()*tier;mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(z)
                    for trial in range(-1,trials):
                        generator=torch.Generator(device=device).manual_seed(seed+1000+max(trial,0)+1000003*group)
                        received=channel(z.reshape(-1,2),snr,'none' if trial<0 else 'awgn',generator).reshape_as(z)
                        yh=model.channel_decoder(received,mask,q,snr,delivered_xyz=side)
                        predictions[tier-1][trial+1].append(yh[active][local].cpu())
    finally:model.train(mode)
    if model_id(model)!=identity:raise RuntimeError('diagnostic changed model state')
    target=torch.cat(targets);signal=torch.cat(signals);raw=torch.cat(raw_signals)
    reports=[];energies=[]
    for tier,outputs in enumerate(predictions,1):
        decoded=[torch.cat(v) for v in outputs]
        record=error_moments(target,decoded[0],torch.stack(decoded[1:]));record.update(tier=tier,symbols=model.cfg.rates[tier])
        reports.append(record)
        print(f"context diagnostic q{tier}: finite NMSE={record['summary']['noiseless_nmse']:.6g}, "
              f"10dB NMSE={record['summary']['noisy_nmse']:.6g}",flush=True)
    for start,end in zip(model.cfg.rates[:-1],model.cfg.rates[1:]):
        pre=raw[:,2*start:2*end].double().square().sum(-1)/(end-start)
        post=signal[:,2*start:2*end].double().square().sum(-1)/(end-start)
        energies.append(dict(start=start,end=end,raw_power_mean=float(pre.mean()),raw_power_median=float(pre.median()),
            normalized_power_mean=float(post.mean()),normalized_power_max=float(post.max()),
            radial_log_gain_median=float((model.cfg.power_floor/(pre+model.cfg.power_floor)).median())))
    result=dict(version=1,sample_points=len(target),population_points=total,
        sample_sha256=hashlib.sha256(selected.numpy().tobytes()).hexdigest(),trials=trials,batch_size=batch_size,
        actual_snr=snr,tx_condition_snr=snr,rx_condition_snr=snr,parameters_unchanged=True,
        tiers=reports,segment_power=energies,
        scope='original full codec blocks; delivered quantized XYZ; uniform point selection AFTER receiver context',
        batching='diagnostic_batch does not regroup points; original scene codec batches retained',
        variance_definition='finite-trial population variance with divisor T',
        warning='finite-trial mean error includes estimation noise; no per-dimension semantic label')
    json_write(out/'latent_errors.json',result)
    rows=[dict(tier=r['tier'],symbols=r['symbols'],**d) for r in reports for d in r['dimensions']]
    with (out/'latent_dimensions.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result
