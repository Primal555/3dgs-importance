"""Plot changing-SNR deliveries with gaps for failures and explicit byte costs."""
import csv
from pathlib import Path

from .modular_experiment import json_write


def write_figures(out,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);destination=out/'figures';destination.mkdir(parents=True,exist_ok=True)
    flat=[]
    for r in report['results']:
        row={k:v for k,v in r.items() if k not in ('metrics','metrics_std')}
        row.update(r['metrics'] or {})
        row.update({k+'_std':v for k,v in (r['metrics_std'] or {}).items()})
        flat.append(row)
    with (destination/'dynamic_trace.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in flat))))
        writer.writeheader();writer.writerows(flat)
    plan=report['plan']
    for scene,method in sorted({(r['scene'],r['method']) for r in flat}):
        fig,axes=plt.subplots(3,2,figsize=(12,9))
        axes=axes.flatten()
        axes[0].plot(range(1,len(plan['snr_trace'])+1),plan['snr_trace'],'k.-',label='Actual SNR')
        axes[0].set_ylabel('Channel SNR (dB)');axes[0].legend()
        for endpoint,color in (('initial','tab:gray'),('final','tab:blue')):
            subset=sorted([r for r in flat if (r['scene'],r['method'],r['endpoint'])==
                           (scene,method,endpoint)],key=lambda r:r['trace_step'])
            x=[r['trace_step'] for r in subset]
            label='Fixed 10 dB codec' if endpoint=='initial' else 'Multi-SNR codec'
            for axis,metric,ylabel,scale in zip(axes[1:],
                    ('source_psnr','photo_psnr','success_rate','total_channel_uses','cumulative_channel_uses'),
                    ('Source PSNR (dB)','Photo PSNR (dB)','Complete delivery success rate',
                     'Channel uses per delivery (M)','Cumulative channel uses per trace (M)'),
                    (1,1,1,1e6,1e6)):
                # A failed delivery has no reconstructed-scene PSNR; leave a gap.
                axis.plot(x,[r.get(metric,float('nan'))/scale for r in subset],'.-',color=color,label=label)
                axis.set_ylabel(ylabel);axis.legend(fontsize=8)
        axes[3].set_ylim(-.02,1.02)
        for axis in axes: axis.set_xlabel('Delivery index');axis.grid(alpha=.25)
        fig.suptitle(f'{scene} q{method}: {plan["trials"]} paired trials; frozen models; '
                     'quality conditional on successful delivery')
        fig.tight_layout()
        for ext in ('png','pdf'): fig.savefig(destination/f'{scene}_q{method}_dynamic.{ext}',dpi=180)
        plt.close(fig)
    json_write(destination/'definitions.json',dict(
        time_axis=plan['time_axis'],channel=plan['channel'],
        quality='mean across successful trials using the same original test cameras at every index',
        paired_quality='summary.json also reports differences restricted to trials where both codecs succeed',
        failures='quality gaps; included in success rate, per-delivery cost and cumulative cost',
        trials='independent noise per trace index; matching seeds across codecs; no repeated-result reuse',
        costs='actual serialized side files plus transmitted attribute complex symbols; no ARQ',
        mcs='same public SNR-to-MCS rule for both codecs',optimizer_updates=0))
