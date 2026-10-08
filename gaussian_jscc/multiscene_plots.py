"""Static, reproducible PNG/PDF figures; never pool heldout and trained scenes.

Chart contract: phase-separated learning curves answer convergence; deployment
stacked bars answer resource mix; labelled RD points answer quality/cost; paired
prefix bars answer enhancement gains. Small samples remain labelled observations,
not fitted trends. Blue/gold plus neutrals; stacked tier identity additionally olive.
Underlying per-view metrics and protocol assumptions stay in JSONL/CSV companions.
"""
import argparse
import json
from pathlib import Path
import numpy as np

BLUE, GOLD, GREY, OLIVE = '#497CA6', '#D5A63B', '#92989D', '#8C995D'
COLORS = [GREY, BLUE, GOLD, OLIVE]


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()] if Path(path).exists() else []


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.labelcolor':'#30363B','text.color':'#30363B','axes.titlecolor':'#30363B'})
    return plt


def finish(fig, folder, name):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    fig.tight_layout(rect=(0,0,1,.95))
    for ext in ('png','pdf'):
        fig.savefig(folder/f'{name}.{ext}',dpi=160,bbox_inches='tight')
    pyplot().close(fig)


def plot_training(out):
    out=Path(out)
    rows=read_rows(out/'loss.jsonl')
    if not rows:
        return
    plt=pyplot()
    scenes=list(dict.fromkeys(r['scene'] for r in rows))
    phases=list(dict.fromkeys(r['phase'] for r in rows))
    fig,axes=plt.subplots(len(scenes),len(phases),figsize=(4*len(phases),2.9*len(scenes)),squeeze=False)
    for i,scene in enumerate(scenes):
        for j,phase in enumerate(phases):
            data=[r for r in rows if r['scene']==scene and r['phase']==phase]
            ax=axes[i,j]
            ax.plot([r['scene_step'] for r in data],[r['loss'] for r in data],color=BLUE,
                    alpha=.8 if len(data)<8 else .4,lw=.7,
                    marker='o' if len(data)<8 else None,linestyle='none' if len(data)<8 else '-')
            window=min(25,len(data))
            if len(data)>=8:
                smooth=np.convolve([r['loss'] for r in data],np.ones(window)/window,mode='valid')
                ax.plot([r['scene_step'] for r in data][window-1:],smooth,color=BLUE,lw=1.5)
            ax.set(title=f'{scene} / {phase}',xlabel='Optimizer updates for this scene',ylabel='Phase objective (own scale)')
            ax.grid(alpha=.15)
    fig.suptitle('Multi-scene training objectives\nIndependent phase scales; no cross-beta loss ranking')
    finish(fig,out/'charts','training_by_scene_phase')
    for scene in scenes:
        folder=out/'scenes'/scene
        validation=read_rows(folder/'validation.jsonl')
        if validation:
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            labels=list(dict.fromkeys(e['layout'] for v in validation for e in v['layouts']))
            for k,label in enumerate(labels):
                series=[(v['step'],e) for v in validation for e in v['layouts'] if e['layout']==label]
                for ax,metric in zip(axes,('source_psnr','photo_psnr')):
                    ax.plot([v[0] for v in series],[v[1][metric] for v in series],
                            color=(BLUE if k%2==0 else GOLD),linestyle=['-','--',':','-.'][k%4],
                            marker=['o','s','^','x','d'][k%5],markersize=3,label=label)
                    ax.set(xlabel='Shared optimizer updates',ylabel='Mean view/trial PSNR (dB)',title=metric)
                    ax.grid(alpha=.15)
            axes[1].legend(title='Deployment layout',fontsize=8)
            fig.suptitle(f'{scene}: held-out training-camera validation\nNot final test evidence; shared codec changes across scene visits')
            finish(fig,out/'charts',scene+'_validation')
        history=read_rows(folder/'allocation_history.jsonl')
        if history:
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            n=len(history[0]['hard_tier_shares'])
            for q in range(n):
                axes[0].plot([r['step'] for r in history],[r['hard_tier_shares'][q] for r in history],
                             color=COLORS[q%4],marker=['o','s','^','d'][q%4],label=f'q{q}')
            axes[0].set(ylabel='Share of source Gaussians',xlabel='Shared updates',ylim=(0,1),title='Actual ten-draw deployment')
            rates=np.array(history[0]['rates'])
            for field,color,style in [('hard_tier_counts',BLUE,'-'),('expected_tier_counts',GOLD,'--')]:
                axes[1].plot([r['step'] for r in history],[np.dot(r[field],rates)/r['source_gaussians'] for r in history],
                             color=color,linestyle=style,marker='o',label=field)
            axes[1].set(ylabel='Complex payload symbols / source Gaussian',xlabel='Shared updates',title='Expected vs deployed payload')
            for ax in axes:
                ax.legend(fontsize=8)
                ax.grid(alpha=.15)
            fig.suptitle(f'{scene}: allocation convergence\nPayload only here; complete packet cost is in final evaluation')
            finish(fig,out/'charts',scene+'_allocation')


def plot_experiments(out):
    out=Path(out)
    rows=read_rows(out/'results.jsonl')
    if not rows:
        return
    plt=pyplot()
    for scene in dict.fromkeys(r['scene'] for r in rows):
        selected=[r for r in rows if r['scene']==scene]
        for snr in dict.fromkeys(r['snr'] for r in selected):
            data=[r for r in selected if r['snr']==snr]
            title=f'{scene} / {data[0]["role"]} / {snr:g} dB / {data[0]["test_views"]} test views / {data[0]["trials"]} noise trials'
            fig,axes=plt.subplots(1,2,figsize=(14,5))
            for i,r in enumerate(data):
                for ax,metric in zip(axes,('source_psnr','photo_psnr')):
                    ax.plot(r['total_channel_uses']/1e6,r[metric],marker=['o','s','^','d','x'][i%5],
                            color=BLUE if r['layout'].isdigit() else GOLD,linestyle='none',label=r['layout'])
                    ax.set(xlabel='Total complex channel uses (millions)',ylabel='Mean PSNR (dB)',title=metric)
                    ax.grid(alpha=.15)
            axes[1].legend(fontsize=7,loc='upper left',bbox_to_anchor=(1,1))
            fig.suptitle(title+'\nXYZ + complete metadata included; shared model excluded; labelled operating points, not a fitted curve')
            finish(fig,out/'charts',f'{scene}_snr{snr:g}_rate_distortion')
            fig,axes=plt.subplots(1,2,figsize=(14,max(4,len(data)*.32)))
            y=np.arange(len(data))
            base=np.zeros(len(data))
            for q in range(len(data[0]['tier_counts'])):
                values=np.array([r['tier_counts'][q]/r['source_gaussians'] for r in data])
                axes[0].barh(y,values,left=base,color=COLORS[q%4],edgecolor='white',label=f'q{q}')
                base+=values
            base=np.zeros(len(data))
            for key,color,label in [('payload_complex_symbols',BLUE,'JSCC payload'),('coordinate_channel_uses',GOLD,'XYZ'),('metadata_channel_uses',GREY,'Metadata')]:
                values=np.array([r[key]/1e6 for r in data])
                axes[1].barh(y,values,left=base,color=color,edgecolor='white',label=label)
                base+=values
            for ax in axes:
                ax.set_yticks(y,[r['layout'] for r in data])
                ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.12))
            axes[0].set(xlim=(0,1),xlabel='Share of ALL source Gaussians',title='Actual deployed tiers')
            axes[1].set(xlabel='Complex channel uses (millions)',title='Measured stream cost + payload')
            fig.suptitle(title+f'\nReliable digital side streams assumed at {data[0]["net_bits_per_use_assumption"]:g} net bits / complex use')
            finish(fig,out/'charts',f'{scene}_snr{snr:g}_deployment_cost')
        gains=[r for r in read_rows(out/'prefix_gains.jsonl') if r['scene']==scene]
        if len({r['snr'] for r in selected})>1:
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            labels=[label for label in dict.fromkeys(r['layout'] for r in selected) if not label.startswith('shuffle')]
            for i,label in enumerate(labels):
                data=sorted((r for r in selected if r['layout']==label),key=lambda r:r['snr'])
                for ax,metric in zip(axes,('photo_psnr','photo_ssim')):
                    ax.plot([r['snr'] for r in data],[r[metric] for r in data],color=BLUE if i%2==0 else GOLD,
                            linestyle=['-','--',':','-.'][i%4],marker=['o','s','^','d'][i%4],label=label)
                    ax.set(xlabel='Evaluation SNR (dB)',ylabel=metric,title=metric)
                    ax.grid(alpha=.15)
            axes[1].legend(fontsize=8)
            fig.suptitle(f'{scene}: SNR mismatch response\nFixed {selected[0]["training_snr"]:g} dB training; fixed deployed tier map, no SNR reallocation')
            finish(fig,out/'charts',scene+'_snr_sweep')
        if gains:
            fig,ax=plt.subplots(figsize=(max(7,len(gains)*.7),4))
            ax.bar(np.arange(len(gains)),[r['source_psnr_gain_db'] for r in gains],color=BLUE)
            ax.axhline(0,color='#30363B',lw=.8)
            ax.set_xticks(np.arange(len(gains)),[f'{r["from_complex_symbols"]:g}->{r["to_complex_symbols"]:g}\n{r["snr"]:g} dB' for r in gains])
            ax.set(ylabel='Source-reference PSNR gain (dB)',xlabel='Prefix lengths (complex symbols / retained Gaussian)')
            fig.suptitle(f'{scene}: incremental prefix benefit\nMatched views and point/slot noise; negative gains are retained')
            finish(fig,out/'charts',scene+'_prefix_gains')


def compare_runs(runs,out):
    """Compare beta/stage runs without ranking their incomparable objective values."""
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    rows=[dict(row,run=str(Path(run))) for run in runs for row in read_rows(Path(run)/'results.jsonl') if row['layout']=='mask']
    if not rows:
        raise ValueError('no results.jsonl found')
    with (out/'results.jsonl').open('w',encoding='utf-8') as handle:
        for r in rows:
            # Distinguish runs in legends but retain the original label as evidence.
            r['original_layout']=r['layout']
            r['layout']=Path(r['run']).parent.name+'/'+Path(r['run']).name+'/'+r['layout']
            handle.write(json.dumps(r)+'\n')
    plot_experiments(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    compare_runs(a.runs,a.out)
