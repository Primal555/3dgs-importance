"""Read-only rate sweep exports: actual symbols/costs, quality and marginal gains."""
import csv
import json
from pathlib import Path
from .modular_experiment import json_write


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _csv(path,rows):
    if not rows: return
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def summarize(out,plan):
    out=Path(out); folder=out/'comparison'; folder.mkdir(exist_ok=True)
    results=[]; common=[]; best=[]; model_info={}
    for arm,rates in plan['arms'].items():
        run=out/arm
        config=json.loads((run/'training.json').read_text(encoding='utf-8'))
        done=json.loads((run/'complete.json').read_text(encoding='utf-8'))
        if done['status']!='complete': raise ValueError('cannot summarize unfinished arm: '+arm)
        model_info[arm]={'parameters':config['parameters'],'refinement_stages':len(rates),
                         'updates':plan['expected_updates'][arm]}
        logs=read_rows(run/'metrics/loss.jsonl')
        if len(logs)!=plan['expected_updates'][arm] or [r['step'] for r in logs]!=list(range(1,len(logs)+1)):
            raise ValueError('missing/duplicate optimizer updates: '+arm)
        model_info[arm]['mean_step_seconds']=sum(r['step_seconds'] for r in logs)/len(logs)
        model_info[arm]['updates_by_symbols']={str(b):sum(r['symbols']==b for r in logs) for b in rates}
        if any(n!=plan['updates_per_rate'] for n in model_info[arm]['updates_by_symbols'].values()):
            raise ValueError('per-prefix update quota mismatch: '+arm)
        tests=read_rows(run/'metrics/test.jsonl')
        for r in tests:
            uniform=r['layout'] not in ('clean','mixed')
            row={'arm':arm,'path':r['path'],'layout':r['layout'],
                 'symbols':rates[int(r['layout'])-1] if uniform else None,
                 'actual_snr':r['actual_snr'],'views':r['views'],'trials':r['trials']}
            for key in ('source_psnr','photo_psnr','source_ssim','photo_ssim','source_lpips','photo_lpips',
                        'source_mse','photo_mse','latent_mse','latent_nmse','symbols_per_gaussian',
                        'payload_complex_symbols','xyz_bytes','metadata_bytes','coordinate_channel_uses',
                        'metadata_channel_uses','total_channel_uses','net_bits_per_use_assumption',
                        'source_color_r_bias','source_color_g_bias','source_color_b_bias',
                        'source_color_rg_error_rmse','source_color_gb_error_rmse'):
                row[key]=r.get(key)
            for key in ('source_psnr','photo_psnr','source_ssim','photo_ssim','source_lpips','photo_lpips'):
                row[key+'_std_trials']=r.get(key+'_std_trials')
            results.append(row)
        validation=read_rows(run/'metrics/validation.jsonl')
        for tier,budget in enumerate(rates,1):
            rr=[r for r in validation if r['path']=='awgn' and r['actual_snr']==10 and r['layout']==str(tier)]
            quality=max(rr,key=lambda r:r['source_psnr']); latent=min(rr,key=lambda r:r['latent_mse'])
            best.append(dict(arm=arm,symbols=budget,best_source_psnr=quality['source_psnr'],
                best_psnr_step=quality['step'],lowest_latent_mse=latent['latent_mse'],lowest_latent_step=latent['step'],
                scope='observed validation metrics, not separately saved best-model checkpoints'))
    for arm in plan['arms']:
        for snr in plan['eval_snrs']:
            group=sorted((r for r in results if r['arm']==arm and r['path']=='awgn' and r['actual_snr']==snr
                          and r['symbols'] is not None),key=lambda r:r['symbols'])
            previous=None
            for r in group:
                r['source_psnr_gain_over_previous_db']=r['source_psnr']-previous['source_psnr'] if previous else None
                r['extra_symbols']=r['symbols']-previous['symbols'] if previous else None
                r['source_psnr_gain_per_extra_symbol']=r['source_psnr_gain_over_previous_db']/r['extra_symbols'] if previous else None
                r['psnr_dominated_by_lower_budget']=any(x['symbols']<r['symbols'] and x['source_psnr']>=r['source_psnr'] for x in group)
                previous=r
    if 'three_tier' in plan['arms'] and 'fine_tiers' in plan['arms']:
        for a in results:
            if a['arm']!='three_tier' or a['path']!='awgn' or a['symbols'] is None: continue
            b=next((r for r in results if r['arm']=='fine_tiers' and r['path']=='awgn' and
                    r['actual_snr']==a['actual_snr'] and r['symbols']==a['symbols']),None)
            if b is None: continue
            row=dict(symbols=a['symbols'],actual_snr=a['actual_snr'],views=a['views'],trials=a['trials'])
            for key in ('source_psnr','photo_psnr','source_ssim','photo_ssim','source_lpips','photo_lpips','latent_mse'):
                row['three_tier_'+key]=a.get(key); row['fine_tiers_'+key]=b.get(key)
                row['fine_minus_three_'+key]=b[key]-a[key] if a.get(key) is not None and b.get(key) is not None else None
            common.append(row)
    _csv(folder/'rate_distortion.csv',results)
    _csv(folder/'common_budget_comparison.csv',common)
    _csv(folder/'best_observed_validation.csv',best)
    summary=dict(status='complete',plan=plan,model_info=model_info,final_test=results,common_budgets=common,
        caveats=['Both semantics are frozen; latent MSE training does not guarantee monotonically improving render quality.',
                 'More tiers add decoder refinement stages; parameter count and compute are not matched.',
                 'No allocator, q0 deletion or semantic/render training. Mixed is random positive-tier diagnostic.',
                 'Payload is complex channel uses, not float-file MB. Digital coordinate/metadata conversion is an explicit net-bits/use assumption.',
                 'Each arm shares its own encoder across prefixes; different arms do not share a transmitted symbol stream.'])
    json_write(folder/'summary.json',summary)
    plot_results(folder,results,model_info)
    plot_history(folder,out,plan)
    return summary


def plot_results(folder,rows,model_info):
    # Chart contract: discrete measured budgets, explicit markers/arms, two
    # reference images in separate panels. No rate interpolation or quality guarantee.
    # Binary arm comparison uses blue/orange; noiseless references dashed.
    # Files are experiment-native Matplotlib output, not a separate HTML report.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    snrs=sorted({r['actual_snr'] for r in rows if r['path']=='awgn'})
    selected=10. if 10. in snrs else snrs[0]
    colors={'three_tier':'#356d9b','fine_tiers':'#d88931'}
    fig,axes=plt.subplots(3,2,figsize=(12,11),layout='constrained')
    for i,(metric,label) in enumerate((('psnr','PSNR (dB, higher)'),('ssim','SSIM (higher)'),('lpips','LPIPS/VGG v0.1 (lower)'))):
        for ax,ref in zip(axes[i],('source','photo')):
            key=ref+'_'+metric
            for arm in model_info:
                for path in ('awgn','finite_noiseless'):
                    rr=sorted((r for r in rows if r['arm']==arm and r['path']==path and r['symbols'] is not None
                               and (path!='awgn' or r['actual_snr']==selected) and r.get(key) is not None),key=lambda r:r['symbols'])
                    if rr: ax.errorbar([r['symbols'] for r in rr],[r[key] for r in rr],
                                   yerr=[r.get(key+'_std_trials') or 0 for r in rr],capsize=2,color=colors[arm],
                                   linestyle='-' if path=='awgn' else '--',marker='o' if arm=='three_tier' else 's',
                                   label=arm+(' AWGN' if path=='awgn' else ' noiseless'))
            clean=next((r for r in rows if r['path']=='clean' and r.get(key) is not None),None)
            if clean: ax.axhline(clean[key],color='#444444',linestyle=':',label='shared clean')
            ax.set(title='vs '+('Source PLY' if ref=='source' else 'Photos'),xlabel='Complex symbols per retained Gaussian',ylabel=label)
            ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
            else: ax.text(.5,.5,'Metric not enabled',ha='center',va='center',transform=ax.transAxes)
    fig.suptitle(f'Test views; physical AWGN {selected:g} dB; same frozen semantic pair')
    fig.savefig(Path(folder)/'quality_vs_symbols.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for arm in model_info:
        for path in ('awgn','finite_noiseless'):
            rr=sorted((r for r in rows if r['arm']==arm and r['path']==path and r['symbols'] is not None
                       and (path!='awgn' or r['actual_snr']==selected)),key=lambda r:r['symbols'])
            for ax,key in zip(axes,('latent_mse','latent_nmse')):
                ax.plot([r['symbols'] for r in rr],[r[key] for r in rr],color=colors[arm],marker='o',
                        linestyle='-' if path=='awgn' else '--',label=arm+'/'+path)
                ax.set(xlabel='Complex symbols per retained Gaussian',ylabel=key,yscale='log'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Frozen target y; latent MSE and rendering quality must be checked separately')
    fig.savefig(Path(folder)/'latent_vs_symbols.png',dpi=150); plt.close(fig)


def plot_history(folder,out,plan):
    import matplotlib.pyplot as plt
    # Each budget's training loss uses its own update index, avoiding a misleading
    # eight-versus-three comparison at unequal numbers of exposures per budget.
    fig,axes=plt.subplots(2,len(plan['arms']),figsize=(6*len(plan['arms']),8),squeeze=False,layout='constrained')
    palette=plt.get_cmap('tab10')
    for column,(arm,rates) in enumerate(plan['arms'].items()):
        logs=read_rows(Path(out)/arm/'metrics/loss.jsonl')
        validation=read_rows(Path(out)/arm/'metrics/validation.jsonl')
        for index,budget in enumerate(rates):
            rr=[r for r in logs if r['symbols']==budget]
            axes[0,column].plot([r['budget_update'] for r in rr],[r['latent_mse'] for r in rr],
                               color=palette(index%10),label=f'{budget} symbols',alpha=.8)
            rr=[r for r in validation if r['path']=='awgn' and r['layout']==str(index+1) and r['actual_snr']==10]
            axes[1,column].plot([r['step'] for r in rr],[r['source_psnr'] for r in rr],
                               color=palette(index%10),marker='o',label=f'{budget} symbols')
        axes[0,column].set(title=arm+' / minibatch latent MSE',xlabel='Own-budget optimizer updates',ylabel='MSE (lower)',yscale='log')
        axes[1,column].set(title=arm+' / fixed validation views, 10 dB',xlabel='All optimizer updates in arm',ylabel='Source PSNR (dB, higher)')
        for ax in axes[:,column]: ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(Path(folder)/'training_history.png',dpi=150); plt.close(fig)
