"""Matched center-only decoder experiment; current trunk versus old light XYZ branch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
CASES = ('transformer', 'historical_light')
INTERACTION_CASES = ('block_attention', 'self_only')


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def summarize(root):
    import torch
    experiment = json.loads((root/'experiment.json').read_text(encoding='utf-8')) if (root/'experiment.json').exists() else {}
    interaction = experiment.get('comparison') == 'interaction'
    cases = INTERACTION_CASES if interaction else CASES
    results, histories, audits = {}, {}, {}
    for case in cases:
        directory = root/case
        info = json.loads((directory/'training.json').read_text(encoding='utf-8'))
        if interaction:
            expected_scope = 'self' if case == 'self_only' else 'block'
            if info['arguments'].get('center_attention_scope', 'block') != expected_scope or info['arguments'].get('center_decoder_kind', 'transformer') != 'transformer':
                raise ValueError(f'{case} does not use the expected Transformer attention scope')
        summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        loss = read_rows(directory/'loss.jsonl')
        validation = read_rows(directory/'validation.jsonl')
        if summary['status'] != 'complete' or summary['completed']['center'] != info['arguments']['center_steps']:
            raise ValueError(f'{case} is incomplete; resume that run before summarizing')
        if any(r['phase'] != 'center' for r in loss):
            raise ValueError('this comparison must not contain attribute or joint optimization')
        batch_hash = hashlib.sha256(json.dumps([r['stats']['sampled_blocks'] for r in loss]).encode()).hexdigest()
        state = torch.load(directory/'codec_0.pt', map_location='cpu', weights_only=True)['state_dict']
        shared = hashlib.sha256()
        for name, value in sorted(state.items()):
            if interaction or not name.startswith('learned.center_decoder.'):
                shared.update(name.encode())
                shared.update(value.cpu().contiguous().numpy().tobytes())
        audits[case] = {'sampled_blocks_sha256': batch_hash, 'shared_initial_weights_sha256': shared.hexdigest(),
                       'shared_arguments': {k: v for k, v in info['arguments'].items()
                                            if k not in ('out', 'resume', 'center_attention_scope' if interaction else 'center_decoder_kind')},
                       'fingerprint': info['fingerprint'], 'fitted_blocks': info['fitted_blocks'],
                       'heldout_blocks': info['heldout_blocks'], 'fitted_probe_blocks': info['fitted_probe_blocks'],
                       'validation_views': info['validation_views']}
        selected = next(r for r in validation if r['step'] == summary['best_steps']['center'])
        results[case] = {'decoder_parameters': info['center_decoder_parameters'],
            'encoder_parameters': info['center_encoder_parameters'], 'steps': len(loss),
            'step_seconds_mean': sum(r['seconds'] for r in loss)/len(loss),
            'grad_norm_max': max(r['grad_norm'] for r in loss),
            'last': validation[-1], 'selected': selected,
            'selection': 'center-only source PSNR' if selected['render'] else 'heldout world RMSE'}
        histories[case] = (loss, validation)
    if audits[cases[0]] != audits[cases[1]]:
        raise ValueError('paired audit failed: initialization, sampling, scene or split differs')
    report = {'protocol': 'same random shared initialization and sampled blocks; same center distance loss and LR; decoder-only change',
              'not_parameter_matched': True, 'historical_source': '97ef4cc gaussian_jscc/multiscale_codec.py XYZ decoder branch',
              'adaptation': 'clean center latent replaces JSCC packet; no tier/SNR condition; attributes and noise excluded',
              'audit_passed': True, 'audits': audits, 'results': results}
    if interaction:
        report.update(protocol='same complete random initial tensors and sampled blocks; decoder cross-token attention only',
                      not_parameter_matched=False, historical_source=None,
                      adaptation='self-only keeps V/output projections, Pre-LN, FFN and residuals; Q/K inactive; encoder Context unchanged',
                      effective_capacity_matched=False)
    (root/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for case, (loss, rows) in histories.items():
        x = [r['step'] for r in rows]
        for field, style in (('centers', '-'), ('fitted_centers', '--')):
            axes[0, 0].plot(x, [r[field]['world_rmse'] for r in rows], style, label=case+'/'+field)
        # Same non-overlapping 50-step mean; no smoothing across cases.
        chunks = [loss[i:i+50] for i in range(0, len(loss), 50)]
        axes[0, 1].plot([c[-1]['step'] for c in chunks], [sum(r['loss'] for r in c)/len(c) for c in chunks], label=case)
        axes[1, 0].plot([r['step'] for r in loss], [r['grad_norm'] for r in loss], label=case, alpha=.6)
        rendered = [r for r in rows if r['render']]
        if rendered:
            axes[1, 1].plot([r['step'] for r in rendered], [r['render']['center_only']['source_psnr'] for r in rendered], label=case)
    axes[0, 0].set_title('XYZ RMSE: heldout (solid), fitted probes (dashed)')
    axes[0, 0].set_yscale('log')
    axes[0, 1].set_title('Training center distance / 50-step mean')
    axes[1, 0].set_title('Pre-clip parameter gradient norm')
    axes[1, 0].set_yscale('symlog', linthresh=.01)
    axes[1, 1].set_title('Predicted XYZ + source attributes: source PSNR')
    for axis in axes.flat:
        if axis.lines:
            axis.legend(fontsize=6)
        else:
            axis.text(.5, .5, 'No camera/render evaluation', ha='center', transform=axis.transAxes)
        axis.grid(alpha=.2)
        axis.set_xlabel('Optimization step')
    fig.tight_layout()
    fig.savefig(root/'comparison.png', dpi=160)
    plt.close(fig)
    for case, result in results.items():
        last = result['last']
        print(f'{case}: parameters={result["decoder_parameters"]}, last fitted RMSE={last["fitted_centers"]["world_rmse"]:.6g}, '
              f'heldout RMSE={last["centers"]["world_rmse"]:.6g}', flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ply')
    p.add_argument('--source')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--blocks-per-batch', type=int, default=32)
    p.add_argument('--hidden', type=int, default=96)
    p.add_argument('--block-size', type=int, default=256)
    p.add_argument('--validate-every', type=int, default=500)
    p.add_argument('--render-every', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--resolution', type=int, default=2)
    p.add_argument('--comparison', choices=['decoder', 'interaction'], default='decoder')
    p.add_argument('--readout-norm', choices=['layernorm', 'affine'], default='layernorm')
    p.add_argument('--summarize-only', action='store_true')
    args = p.parse_args()
    root = Path(args.out).resolve()
    if args.summarize_only:
        summarize(root)
        return
    if not args.ply or min(args.steps, args.validate_every, args.render_every, args.blocks_per_batch, args.hidden, args.block_size) < 1:
        p.error('PLY and positive step/batch/model counts required')
    if args.comparison == 'decoder' and args.readout_norm != 'layernorm':
        p.error('historical lightweight comparison has no configurable tap norm')
    if root.exists() and any(x.name not in ('console.log', 'run.pid') for x in root.iterdir()):
        raise FileExistsError('choose a new output directory; existing experiments are never overwritten')
    root.mkdir(parents=True, exist_ok=True)
    (root/'experiment.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    cases = INTERACTION_CASES if args.comparison == 'interaction' else CASES
    for case in cases:
        case_out = root/case
        case_out.mkdir()
        command = [sys.executable, '-u', '-m', 'gaussian_jscc', 'train-center-attributes',
            '--ply', str(Path(args.ply).resolve()), '--out', str(case_out), '--device', args.device,
            '--center-decoder-kind', 'transformer' if args.comparison == 'interaction' else case,
            '--center-readout-norm', args.readout_norm,
            '--center-attention-scope', 'self' if case == 'self_only' else 'block',
            '--center-steps', str(args.steps),
            '--min-center-steps', str(args.steps+1), '--attribute-steps', '0', '--joint-steps', '0',
            '--center-lr', str(args.lr), '--blocks-per-batch', str(args.blocks_per_batch),
            '--hidden', str(args.hidden), '--block-size', str(args.block_size),
            '--validation-region-size', str(2*args.block_size), '--center-probe-blocks', '32',
            '--validate-every', str(args.validate_every), '--render-every', str(args.render_every),
            '--save-every', str(args.render_every), '--profile-every', '50',
            '--seed', str(args.seed), '--resolution', str(args.resolution)]
        if args.source:
            command += ['--source', str(Path(args.source).resolve())]
        print(f'Starting {case}: random initialization, center only, fixed {args.steps} steps', flush=True)
        started = time.perf_counter()
        with (case_out/'console.log').open('w', encoding='utf-8') as log:
            # Separate processes release GPU allocations between cases.
            process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if code:
            raise RuntimeError(f'{case} failed with exit {code}; see {case_out}/console.log; second case not silently run')
        print(f'{case} wall seconds including validation/save: {time.perf_counter()-started:.2f}', flush=True)
    summarize(root)


if __name__ == '__main__':
    main()
