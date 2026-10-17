"""Actual deployed tier contributions, using the received scene's splat geometry."""
from pathlib import Path
import torch
from torch.nn import functional as F

# Background is NOT q0: a deleted Gaussian has no received projection.
TIER_COLORS=((0,0,0),(45,115,220),(240,155,45),(160,85,200))


def contribution_labels(weights, threshold=1e-5):
    if weights.ndim!=3 or weights.shape[0]!=3 or not torch.isfinite(weights).all() or threshold<0:
        raise ValueError('tier contributions must be finite [3,H,W]')
    labels=weights.argmax(0)+1
    return torch.where(weights.sum(0)>threshold,labels,0)


def colorize_labels(labels):
    if labels.ndim!=2 or ((labels<0)|(labels>3)).any():
        raise ValueError('expected tier labels [H,W] in 0..3')
    palette=torch.tensor(TIER_COLORS,device=labels.device,dtype=torch.float32)/255
    return palette[labels.long()].permute(2,0,1)


@torch.no_grad()
def render_tier_map(raw, tiers, camera, degree):
    from .rendering import render
    tiers=torch.as_tensor(tiers,device=raw.device)
    if tiers.shape!=(len(raw),) or ((tiers<1)|(tiers>3)).any():
        raise ValueError('tier map requires received rows and their matching positive tiers')
    # Rasterize ALL tiers together once: per-channel RGB accumulates each tier's
    # alpha/transmittance contribution with identical cross-tier occlusion.
    colors=F.one_hot(tiers.long()-1,3).to(raw)
    weights=render(raw,camera,degree,False,colors=colors)
    labels=contribution_labels(weights)
    return colorize_labels(labels),labels


def save_tier_panel(path, photo, source, received, tier_image, rates, counts):
    from PIL import Image, ImageDraw
    images=[photo[:3],source,received,tier_image]
    pixels=(torch.cat(images,2).detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
    panel=Image.fromarray(pixels)
    canvas=Image.new('RGB',(panel.width,panel.height+64),'white'); canvas.paste(panel,(0,64))
    draw=ImageDraw.Draw(canvas)
    for i,title in enumerate(('Photo','Source PLY','Received / deployed map','Dominant tier contribution')):
        draw.text((i*panel.width//4+4,4),title,fill='black')
    x=4
    for tier in range(1,4):
        label=f'q{tier}: {rates[tier]} symbols; {counts[tier]:,} points'
        draw.rectangle((x,24,x+12,36),fill=TIER_COLORS[tier]); draw.text((x+18,24),label,fill='black')
        x+=max(190,len(label)*7)
    draw.text((4,44),f'q0 omitted: {counts[0]:,} points; black = uncovered background, NOT q0. Pixel color = largest accumulated tier contribution.',fill='black')
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); canvas.save(path)
