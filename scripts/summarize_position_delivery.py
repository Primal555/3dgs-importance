"""Collect paired render-only XYZ-delivery histories without claiming causality."""
import argparse
import csv
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def summarize(root):
    root = Path(root)
    flattened, summaries, paired = [], {}, None
    for mode in ('learned', 'float32', 'quantized'):
        folder = root/mode
        config = json.loads((folder/'training.json').read_text(encoding='utf-8'))
        if config['codec_config'].get('position_delivery', 'learned') != mode:
            raise ValueError('Position-delivery mode does not match directory: '+mode)
        if config['initialization']['mode'] != 'random' or config['bootstrap_steps'] or config['joint_steps']:
            raise ValueError('Comparison requires random initialization and render-only training')
        # Fail rather than silently compare differing rates, views or schedules.
        codec = {k:v for k,v in config['codec_config'].items() if k not in ('position_delivery','position_bits')}
        contract = {k:config[k] for k in ('seed','ply','source','snr','channel','render_steps','render_lr',
                    'resolution','white_background','blocks_per_batch','views_per_step','train_view_indices',
                    'validation_view_indices','validation_trials','validate_every','position_net_bits_per_use')}
        contract['codec'] = codec
        if paired is not None and paired != contract:
            raise ValueError('Mismatched experiment settings: '+mode)
        paired = contract
        losses, validation = read_rows(folder/'loss.jsonl'), read_rows(folder/'validation.jsonl')
        best = min(validation, key=lambda r:r['codec_score'])
        summaries[mode] = {
            'initial_source_mse':validation[0]['codec_score'],
            'last_source_mse':validation[-1]['codec_score'],
            'best_source_mse':best['codec_score'], 'best_step':best['step'],
            'zero_gradient_fraction':sum(r['grad_norm']==0 for r in losses)/len(losses),
            'zero_attribute_head_gradient_fraction':sum(r['gradient_groups'].get('attribute_heads',{}).get('before',0)==0 for r in losses)/len(losses),
            'xyz_head_trained':mode=='learned',
            'last_layouts':[{k:v for k,v in entry.items() if k!='views'} for entry in validation[-1]['layouts']],
        }
        for row in validation:
            for entry in row['layouts']:
                flattened.append({'mode':mode,'step':row['step'],'phase':row['phase'],
                                  **{k:v for k,v in entry.items() if k not in ('views','tier_counts')}})
    result = {'scope':'matched payload, not matched total channel use; assumed reliable XYZ side stream',
              'interpretation':'Improvement supports a position-path bottleneck, NOT proof of multi-loss gradient conflict.',
              'reference':'source_mse compares decoded renders to full original PLY renders; photo metrics are separate',
              'runs':summaries}
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    with (root/'validation_comparison.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    print(json.dumps({k:{n:v for n,v in r.items() if n!='last_layouts'} for k,r in summaries.items()},indent=2))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    summarize(parser.parse_args().directory)
