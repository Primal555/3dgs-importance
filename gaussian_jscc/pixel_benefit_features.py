"""Unpooled pixel tiles, complete occlusion context and paired exact labels."""
import time
from collections import OrderedDict

import torch

from .pixel_benefit import pixel_density_labels
from .pixel_local_renderer import _range_entries, _rectangles


def pixel_coordinates(tiles, width, halo=5):
    axis=torch.arange(-halo,16+halo,device=tiles.device)
    y,x=torch.meshgrid(axis,axis,indexing='ij'); gx=(width+15)//16
    return (tiles[:,None,None].remainder(gx)*16+x,
            tiles[:,None,None].div(gx,rounding_mode='floor')*16+y)


def crops(image, tiles, halo=5):
    h,w=image.shape[-2:]; x,y=pixel_coordinates(tiles,w,halo)
    valid=(x>=0)&(y>=0)&(x<w)&(y<h)
    values=image[:,y.clamp(0,h-1),x.clamp(0,w-1)].permute(1,0,2,3)
    return values*valid[:,None]


def _surrounding_tiles(tiles,h,w):
    gx,gy=(w+15)//16,(h+15)//16
    dy,dx=torch.meshgrid(torch.arange(-1,2,device=tiles.device),torch.arange(-1,2,device=tiles.device),indexing='ij')
    x=tiles[:,None].remainder(gx)+dx.flatten(); y=tiles[:,None].div(gx,rounding_mode='floor')+dy.flatten()
    good=(x>=0)&(y>=0)&(x<gx)&(y<gy)
    return y*gx+x,good


def _relative(tokens,tile,width):
    result=tokens.clone(); gx=(width+15)//16
    result[...,0]-=(tile%gx)*16; result[...,1]-=(tile//gx)*16
    result[...,2]=result[...,2].clamp_min(0).log1p()
    return result


def _intersects(p, tile, height, width):
    # Rasterizer radius selects whole tiles; its alpha tail inside those tiles
    # can extend beyond the radius. Use the identical tile bounds for tokens.
    low,high=_rectangles(p.reshape(-1,12),height,width)
    gx=(width+15)//16; x=tile%gx; y=tile//gx
    good=((low[:,0]<x+2)&(high[:,0]>x-1)&(low[:,1]<y+2)&(high[:,1]>y-1)
          &(high>low).all(-1))
    return good.reshape(p.shape[:-1])


@torch.no_grad()
def build_group(scene, changed_ids, projected, target, lambda_s, d_ref, *, with_labels):
    """Build one complete view; features depend only on the current scene.

    Exact intervention images are used exclusively in ``density``. The area
    includes every changed footprint, the SSIM output halo, and its read halo.
    Candidate pixels outside the changed footprint reuse the cached baseline.
    """
    start=time.perf_counter(); device=scene.projected.device
    ids=torch.as_tensor(changed_ids,device=device,dtype=torch.long)
    projected=projected.to(scene.projected); target=target.to(scene.image)
    affected=scene.affected_tiles(ids,projected)
    tiles=scene.affected_tiles(ids,projected,halo_tiles=1)
    h,w=scene.height,scene.width; nt=len(tiles); k=len(ids)
    image=crops(scene.image,tiles); reference=crops(target,tiles)
    inputs={'image':torch.cat((image,image-reference),1),
            'valid':crops(scene.image.new_ones(1,h,w),tiles,halo=0)}
    # Neighboring tile lists cover the five-pixel input halo. Deduplicate IDs
    # before depth ordering; contributors hidden by early termination remain.
    surrounding,good=_surrounding_tiles(tiles,h,w)
    cache_key=(target.data_ptr(),target._version,tuple(target.shape),scene.image._version,scene.projected._version)
    if getattr(scene,'_pixel_cache_key',None)!=cache_key:
        scene._pixel_cache_key=cache_key; scene._pixel_context_cache=OrderedDict()
    cache=scene._pixel_context_cache
    all_context=[]; hits=0
    tile_list=tiles.cpu().tolist()
    for i,tile in enumerate(tile_list):
        if tile in cache:
            rows,context=cache.pop(tile); hits+=1
        else:
            _,entries=_range_entries(scene.tile_ranges[surrounding[i,good[i]]])
            rows=scene.tile_ids[entries].unique(sorted=True)
            rows=rows[torch.argsort(scene.projected[rows,2],stable=True)]
            context=_relative(scene.projected[rows],tile,w)
        cache[tile]=(rows,context)
        while len(cache)>128: cache.popitem(last=False)
        all_context.append(torch.cat((context,torch.isin(rows,ids)[:,None].to(context)),1))
    inputs['context']=tuple(all_context)
    inputs['context_mask']=tuple(torch.ones(len(c),dtype=torch.bool,device=device) for c in all_context)
    inputs['candidate']=scene.image.new_empty(nt,4,k,12)
    inputs['candidate_mask']=torch.zeros(nt,4,k,dtype=torch.bool,device=device)
    for i,(tile,context) in enumerate(zip(tile_list,all_context)):
        inputs['candidate'][i]=_relative(projected,tile,w)
        inputs['candidate_mask'][i]=_intersects(projected,tile,h,w)
    result=dict(inputs=inputs,height=h,width=w,d_ref=float(d_ref),tiles=tiles,cache_hits=hits)
    block_key=tuple(ids.cpu().tolist())
    result['prediction_cache_keys']=tuple((id(scene),cache_key,block_key,tile) for tile in tile_list)
    feature_seconds=time.perf_counter()-start
    if with_labels:
        start=time.perf_counter()
        if nt:
            # Construct just the requested halo pixels by looking up changed
            # tile slots. No full-image clone is made for any intervention.
            lookup=torch.full((len(scene.tile_ranges),),-1,device=device,dtype=torch.long)
            lookup[affected]=torch.arange(len(affected),device=device)
            states=[]
            for q in range(4):
                rendered=scene.image.new_empty(len(affected),3,16,16)
                for first in range(0,len(affected),128):
                    last=min(first+128,len(affected))
                    rendered[first:last]=scene.render_tiles(ids,projected[q],affected[first:last])
                states.append(rendered)
            labels=[]
            for first in range(0,nt,16):
                last=min(first+16,nt); x,y=pixel_coordinates(tiles[first:last],w)
                inside=(x>=0)&(y>=0)&(x<w)&(y<h)
                safe_x=x.clamp(0,w-1); safe_y=y.clamp(0,h-1)
                slots=lookup[safe_y.div(16,rounding_mode='floor')*((w+15)//16)+safe_x.div(16,rounding_mode='floor')]
                changed=inside&(slots>=0); candidates=[]
                for rendered in states:
                    replacement=rendered[slots.clamp_min(0),:,safe_y.remainder(16),safe_x.remainder(16)].permute(0,3,1,2)
                    candidates.append(torch.where(changed[:,None],replacement,image[first:last]))
                labels.append(pixel_density_labels(torch.stack(candidates,1),reference[first:last],lambda_s).cpu())
            result['density']=torch.cat(labels)
        else:
            result['density']=scene.image.new_zeros((0,3,2,16,16),dtype=torch.float64)
        result['teacher_seconds']=time.perf_counter()-start
    result['feature_seconds']=feature_seconds
    return result
