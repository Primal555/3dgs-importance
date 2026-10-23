"""Per-scene endpoint gains; unseen scenes are never pooled into training macro."""
import json
from pathlib import Path

from .channel_audit import write_csv
from .modular_experiment import json_write


def summarize(out,specs,heldout):
    out=Path(out);rows=[]
    heldouts=heldout if isinstance(heldout,list) else [heldout]
    keys=('step','path','layout','source_psnr','source_ssim','source_lpips','photo_psnr','photo_ssim',
          'photo_lpips','latent_mse','latent_nmse','latent_target_power','total_channel_uses',
          'payload_complex_symbols','xyz_bytes','metadata_bytes','scale_stream_bytes','views','trials',
          'source_color_r_bias','source_color_g_bias','source_color_b_bias')
    for spec in specs+heldouts:
        for label in ('initial','final'):
            path=out/'scenes'/spec['name']/'evaluation'/label/'metrics/validation.jsonl'
            for line in path.read_text(encoding='utf-8').splitlines():
                if not line.strip():continue
                r=json.loads(line);row={k:r.get(k) for k in keys}
                row.update(scene=spec['name'],role=spec['role'],endpoint=label,
                    generalization='zero_shot_unseen_scene' if spec['role']=='heldout' else 'training_scene_unseen_views')
                previous=next((x for x in rows if x['scene']==spec['name'] and x['endpoint']=='initial'
                               and x['path']==r['path'] and x['layout']==r['layout']),None)
                row['source_psnr_change_db']=r['source_psnr']-previous['source_psnr'] if previous else None
                rows.append(row)
    json_write(out/'comparison/summary.json',dict(rows=rows,
        heldout='no optimizer updates, statistics fitting, calibration or intermediate model selection',
        mixed='fixed random positive q1/q2/q3 map, not learned allocation',
        protocol='initial/final use identical test cameras and paired noise; only training scenes validate mid-run'))
    write_csv(out/'comparison/endpoints.csv',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'charts';folder.mkdir(exist_ok=True)
    for selected,name in ((heldouts,'heldout_quality'),(specs,'training_scene_quality')):
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
        for ax,metric in zip(axes,('source_psnr','source_ssim','source_lpips')):
            for spec in selected:
                for endpoint,style in (('initial','--'),('final','-')):
                    rr=[next(r for r in rows if r['scene']==spec['name'] and r['endpoint']==endpoint
                             and r['path']=='awgn' and r['layout']==str(q)) for q in (1,2,3)]
                    ax.plot([8,24,48],[r[metric] for r in rr],style,marker='o',label=spec['name']+'/'+endpoint)
            ax.set(xlabel='Complex symbols per Gaussian',title=metric,ylabel=metric.split('_',1)[1])
            ax.grid(alpha=.25);ax.legend(fontsize=7)
        fig.savefig(folder/(name+'.png'),dpi=170);plt.close(fig)
