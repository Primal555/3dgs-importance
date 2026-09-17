"""Fixed-view/noise validation for render-first training; no auxiliary objective."""
import json
from pathlib import Path
import torch
from .learned_training import decode_batches, hard_layout
from .optimization import preserved_rng
from .render_objective import image_metrics


def append_json(path, row):
    # Reject NaN/Inf rather than silently write non-standard JSON.
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, allow_nan=False)+'\n')


def save_panel(path, photo, reference, decoded):
    from PIL import Image, ImageDraw
    images = [photo, reference, decoded, (decoded-reference).abs().mean(0, keepdim=True).expand(3,-1,-1)*4]
    pixels = (torch.cat(images, 2).detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
    panel = Image.fromarray(pixels)
    canvas = Image.new('RGB', (panel.width, panel.height+24), 'white')
    canvas.paste(panel, (0,24))
    draw = ImageDraw.Draw(canvas)
    for i,label in enumerate(('Photo', 'Source PLY', 'Received', 'Abs error x4')):
        draw.text((i*panel.width//4+4,5), label, fill='black')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def validate_render(model, groups, group_ids, raw, geometry, cameras, reference,
                    snr, channel, trials, seed, out, step, phase, mask=None,
                    white_background=False, beta=0.):
    from .cli import seed_all
    from .rendering import render
    device = next(model.parameters()).device
    was_training = model.training
    entries = []
    # Uniform/mixed codec conditions stay visible even when a learned mask is
    # enabled. The mask is an additional, separately scored deployment layout.
    layouts = (1,2,3,None) + (('mask',) if mask is not None else ())
    try:
        with preserved_rng(device):
            model.eval()
            for index, tier in enumerate(layouts):
                label = 'mixed' if tier is None else str(tier)
                seed_all(seed+10000+index*1000)
                qs = [torch.where(ids.to(device)>=0,
                                  mask.scores(ids.clamp_min(0).to(device),snr).argmax(-1),0)
                      if tier == 'mask' else hard_layout(ids,tier,0.) for ids in group_ids]
                # Layout is fixed across trials. Padding is never a source row.
                flat_q = torch.cat([q[ids.to(q.device)>=0].cpu() for q,ids in zip(qs,group_ids)])
                lengths = torch.tensor(model.cfg.rates)[flat_q]
                observations, xyz_errors, trial_scores = [], [], []
                for trial in range(trials):
                    seed_all(seed+20000+index*1000+trial)
                    scene = decode_batches(model,groups,qs,snr,channel,geometry)
                    if len(scene):
                        xyz_errors.append(float((scene[:,:3]-raw[flat_q>0,:3].to(device)).square().mean().sqrt()))
                    view_scores = []
                    for view, camera in enumerate(cameras):
                        decoded = render(scene,camera,model.cfg.sh_degree,white_background)
                        target = reference.get(camera,device)
                        photo = camera.original_image[:3].to(device)
                        source_metrics = image_metrics(decoded,target)
                        photo_metrics = image_metrics(decoded,photo)
                        baseline_metrics = image_metrics(target,photo)
                        row = {'trial':trial,'view_index':view,
                               'view':str(getattr(camera,'image_name',view)),
                               **{'source_'+k:v for k,v in source_metrics.items()},
                               **{'photo_'+k:v for k,v in photo_metrics.items()},
                               **{'reference_photo_'+k:v for k,v in baseline_metrics.items()}}
                        observations.append(row)
                        view_scores.append(source_metrics['mse'])
                        if trial == 0:
                            save_panel(Path(out)/'validation_images'/f'{step:06d}'/f'{label}_view{view:02d}.png',
                                       photo,target,decoded)
                    trial_scores.append(sum(view_scores)/len(view_scores))
                    del scene
                keys = [k for k in observations[0] if k not in ('trial','view_index','view')]
                entry = {'layout':label, **{k:sum(r[k] for r in observations)/len(observations) for k in keys},
                         'noise_trial_mse_std':float(torch.tensor(trial_scores).std(unbiased=False)),
                         'xyz_rmse_retained':sum(xyz_errors)/len(xyz_errors) if xyz_errors else None,
                         'symbols_per_source_gaussian':float(lengths.float().mean()),
                         'tier_counts':torch.bincount(flat_q,minlength=4).tolist(),
                         'views':observations}
                entries.append(entry)
    finally:
        model.train(was_training)
    codec_score = sum(e['source_mse'] for e in entries[:4])/4
    # Joint checkpoint selection evaluates ACTUAL hard deployment, not the
    # expected soft allocation rate used in score-function training.
    selected = entries[-1] if mask is not None else None
    score = selected['source_mse']+beta*selected['symbols_per_source_gaussian']/model.cfg.rates[-1] if selected else codec_score
    result = {'step':step,'phase':phase,'score':score,
              'score_definition':'hard_mask_source_mse_plus_normalized_payload' if selected else 'mean_layout_source_mse',
              'codec_score':codec_score,'snr':snr,'channel':channel,'trials':trials,
              'validation_views':len(cameras),'layouts':entries,
              'metrics_note':'unclipped MSE/PSNR; displayed-RGB SSIM; no projection/parameter loss; payload excludes metadata'}
    append_json(Path(out)/'validation.jsonl',result)
    print(f'validation step={step}: source MSE={codec_score:.6f}; '+', '.join(
        f'q{e["layout"]} PSNR={e["source_psnr"]:.2f}' for e in entries),flush=True)
    return result
