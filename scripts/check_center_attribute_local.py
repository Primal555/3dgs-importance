"""Real-PLY CPU center/attribute fitting check. No rasterization/PSNR certification."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from gaussian_jscc.data import read_ply, prepare, write_ply
from gaussian_jscc.center_attribute_train import add_parser, train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ply', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--attribute-steps', type=int, default=0)
    p.add_argument('--center-max-world-rmse', type=float)
    p.add_argument('--regions', type=int, default=32)
    p.add_argument('--block-size', type=int, default=256)
    args = p.parse_args()
    if args.attribute_steps and args.center_max_world_rmse is None:
        p.error('CPU A+B requires an explicit diagnostic --center-max-world-rmse gate')
    root = Path(args.out).resolve()
    if root.exists():
        raise FileExistsError('choose a new output directory')
    if min(args.steps, args.block_size) < 1 or args.regions < 8:
        p.error('positive steps/block-size and at least 8 regions required')
    raw, degree = read_ply(args.ply)
    raw, _, _ = prepare(raw, 16)
    count = len(raw)//args.block_size
    if count < args.regions:
        raise ValueError('not enough complete blocks')
    indices = [int((i+.5)*count/args.regions) for i in range(args.regions)]
    sample = torch.cat([raw[i*args.block_size:(i+1)*args.block_size] for i in indices])
    root.mkdir(parents=True)
    write_ply(root/'sample.ply', sample, degree)
    (root/'sampling.json').write_text(json.dumps({**vars(args), 'indices': indices,
        'points': len(sample), 'scope': 'real-PLY subset, hidden48, clean CPU fit; not full-scene render evidence'}, indent=2), encoding='utf-8')
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest='command'))
    train(parser.parse_args(['train-center-attributes', '--ply', str(root/'sample.ply'),
        '--out', str(root/'run'), '--device', 'cpu', '--hidden', '48', '--center-steps', str(args.steps),
        '--attribute-steps', str(args.attribute_steps), '--joint-steps', '0', '--block-size', str(args.block_size),
        '--validation-region-size', str(args.block_size), '--blocks-per-batch', '4',
        '--validation-blocks', '4', '--validate-every', '50', '--save-every', '100',
        *(['--center-max-world-rmse', str(args.center_max_world_rmse)] if args.center_max_world_rmse is not None else [])]))


if __name__ == '__main__':
    main()
