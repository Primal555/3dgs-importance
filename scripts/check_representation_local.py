"""Real-PLY CPU integration/fitting check; NOT render or communication certification."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from gaussian_jscc.data import read_ply, prepare, write_ply
from gaussian_jscc.representation_train import add_parser, train
from gaussian_jscc.checkpoint_history import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ply', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--regions', type=int, default=32)
    p.add_argument('--block-size', type=int, default=256)
    p.add_argument('--representation-steps', type=int, default=300)
    p.add_argument('--adapter-steps', type=int, default=100)
    p.add_argument('--joint-steps', type=int, default=100)
    p.add_argument('--hidden', type=int, default=48)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--position-objective', choices=['scene-scale', 'teacher-axis'], default='scene-scale')
    p.add_argument('--axis-floor-percentile', type=float, default=1.)
    args = p.parse_args()
    if args.regions < 8 or args.block_size < 2:
        p.error('at least eight regions and block-size >=2 required')
    if args.position_objective == 'teacher-axis' and (args.adapter_steps or args.joint_steps):
        p.error('teacher-axis requires --adapter-steps 0 --joint-steps 0')
    root = Path(args.out).resolve()
    if root.exists():
        raise FileExistsError('choose a new local-check directory')
    raw, degree = read_ply(args.ply)
    raw, _, _ = prepare(raw, 16)
    count = len(raw)//args.block_size
    if count < args.regions:
        raise ValueError('not enough complete input blocks')
    indices = [int((i+.5)*count/args.regions) for i in range(args.regions)]
    sample = torch.cat([raw[i*args.block_size:(i+1)*args.block_size] for i in indices])
    root.mkdir(parents=True)
    write_ply(root/'sample.ply', sample, degree)
    (root/'sampling.json').write_text(json.dumps({**vars(args), 'indices': indices,
        'source_sha256': file_hash(args.ply), 'source_points': len(raw), 'sample_points': len(sample),
        'scope': 're-fit normalization to sampled real PLY; reduced hidden size; no GPU rasterization',
        'gates': 'permissive CPU structural gates deliberately exercise all phases; NOT quality acceptance'}, indent=2), encoding='utf-8')
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest='command'))
    train_args = parser.parse_args(['train-representation', '--ply', str(root/'sample.ply'),
        '--out', str(root/'run'), '--device', 'cpu', '--hidden', str(args.hidden),
        '--block-size', str(args.block_size), '--validation-region-size', str(args.block_size),
        '--blocks-per-batch', '4', '--validation-blocks', '4', '--latent-dim', '64',
        '--representation-steps', str(args.representation_steps), '--adapter-steps', str(args.adapter_steps),
        '--joint-steps', str(args.joint_steps), '--cpu-threads', str(args.threads),
        '--position-objective', args.position_objective, '--axis-floor-percentile', str(args.axis_floor_percentile),
        '--validate-every', '50', '--render-every', '500', '--save-every', '100',
        '--max-clean-loss', '1000000000', '--max-adapter-loss-ratio', '1000000000'])
    train(train_args)


if __name__ == '__main__':
    main()
