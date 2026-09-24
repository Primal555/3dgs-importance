"""Read-only phase analysis; writes new reports, never changes training records.

Reports trends, not an automatic/conclusive convergence detector. Source PSNR
is averaged over observations/layouts, NOT calculated from the averaged MSE.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics as st


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def analyze(root, window=500):
    root = Path(root)
    losses = read_rows(root/'loss.jsonl')
    validations = read_rows(root/'validation.jsonl')
    local = {r['step']:st.mean(e.get('loss',e.get('feature_loss')) for e in r['layouts'])
             for r in read_rows(root/'bootstrap_validation.jsonl')}
    config = json.loads((root/'training.json').read_text(encoding='utf-8'))
    checkpoints, windows, phases = [], [], []
    for phase in dict.fromkeys(r['phase'] for r in losses):
        train = [r for r in losses if r['phase']==phase]
        start = min(r['step'] for r in train)-1
        end = max(r['step'] for r in train)
        # Last pre-phase validation is the phase's baseline; de-duplicate the
        # repeated stage boundary evaluation without dropping later phases.
        indexed = {r['step']:r for r in validations if r['phase']==phase and start<=r['step']<=end}
        baseline = [r for r in validations if r['step']<=start]
        if start not in indexed and baseline:
            candidate = max(baseline,key=lambda r:r['step'])
            if candidate['step']==start:
                indexed[start] = candidate
        points = []
        for step,r in sorted(indexed.items()):
            layouts = [e for e in r['layouts'] if e['layout'] in ('1','2','3','mixed')]
            point = {'phase':phase,'step':step,'phase_step':step-start,
                     'source_psnr':st.mean(e['source_psnr'] for e in layouts),
                     'source_mse':st.mean(e['source_mse'] for e in layouts),
                     'photo_psnr':st.mean(e['photo_psnr'] for e in layouts),
                     'local_validation_loss':local.get(step) if phase=='bootstrap' else None}
            point.update({f'q{e["layout"]}_source_psnr':e['source_psnr'] for e in layouts})
            points.append(point)
        checkpoints.extend(points)
        previous = None
        for lower in range(0,end-start,window):
            upper = min(lower+window,end-start)
            subset = [p for p in points if lower<p['phase_step']<=upper]
            if not subset:
                continue
            item = {'phase':phase,'from_phase_step':lower,'to_phase_step':upper,'checks':len(subset),
                    'mean_source_psnr':st.mean(p['source_psnr'] for p in subset),
                    'median_source_psnr':st.median(p['source_psnr'] for p in subset),
                    'mean_source_mse':st.mean(p['source_mse'] for p in subset),
                    'mean_photo_psnr':st.mean(p['photo_psnr'] for p in subset)}
            item['mean_psnr_gain_vs_previous_window'] = (item['mean_source_psnr']-previous['mean_source_psnr']
                                                        if previous else None)
            windows.append(item)
            previous = item
        times = [r['step_seconds'] for r in train]
        summary = {'phase':phase,'updates':len(train),'start_step':start,'end_step':end,
                   'mean_step_seconds':st.mean(times),'median_step_seconds':st.median(times),
                   'recorded_training_seconds':sum(times),'lr_values':sorted(set(r['lr'] for r in train))}
        if points:
            best = min(points,key=lambda p:p['source_mse'])
            summary.update(initial=points[0],final=points[-1],best_by_source_mse=best)
        phases.append(summary)
    return {'source':str(root.resolve()),'window_steps':window,'phases':phases,
            'settings':{k:config.get(k) for k in ('position_bits','prefix_mode','bootstrap_objective','lr','render_lr',
                                                  'bootstrap_steps','render_steps','blocks_per_batch','views_per_step')},
            'notes':['Validation/source-render PSNR, not unseen-scene generalization evidence.',
                     'Time sums cover recorded training updates, not validation/saving/loading/wall time.',
                     'Window means describe one run; no automatic plateau cutoff or significance claim.',
                     'Phase-local steps: render step 2000 may be global step 7000.']}, checkpoints, windows


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--out',type=Path,help='new directory; default TRAINING/phase_analysis')
    parser.add_argument('--window',type=int,default=500)
    args = parser.parse_args()
    if args.window<1:
        parser.error('--window must be positive')
    summary,points,windows = analyze(args.training,args.window)
    out = args.out or args.training/'phase_analysis'
    out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    write_csv(out/'checkpoints.csv',points)
    write_csv(out/'window_gains.csv',windows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    phases = [p['phase'] for p in summary['phases']]
    fig,axes = plt.subplots(2,len(phases),figsize=(7*len(phases),8),squeeze=False)
    fig.suptitle('Phase-local progress | fixed validation, not a convergence guarantee')
    for col,phase in enumerate(phases):
        subset = [p for p in points if p['phase']==phase]
        for key in ('q1_source_psnr','q2_source_psnr','q3_source_psnr','qmixed_source_psnr','source_psnr'):
            axes[0,col].plot([p['phase_step'] for p in subset],[p.get(key) for p in subset],
                             linewidth=2 if key=='source_psnr' else 1,alpha=1 if key=='source_psnr' else .55,
                             label='mean layouts' if key=='source_psnr' else key.split('_')[0])
        axes[0,col].set(title=phase,ylabel='PSNR vs source render (dB)',xlabel='Updates within this phase')
        axes[0,col].legend()
        bins = [w for w in windows if w['phase']==phase and w['mean_psnr_gain_vs_previous_window'] is not None]
        axes[1,col].bar([b['to_phase_step'] for b in bins],
                        [b['mean_psnr_gain_vs_previous_window'] for b in bins],width=args.window*.7)
        axes[1,col].axhline(0,color='black',linewidth=.6)
        axes[1,col].set(xlabel='End of phase-local window',ylabel='Mean PSNR gain vs previous window (dB)',
                        title=f'Non-overlapping {args.window}-update windows (not single checkpoints)')
    fig.tight_layout()
    fig.savefig(out/'phase_progress.png',dpi=160)
    fig.savefig(out/'phase_progress.svg')
    plt.close(fig)
    print(json.dumps(summary,indent=2))
    print('Saved analysis:',out)


if __name__=='__main__':
    main()
