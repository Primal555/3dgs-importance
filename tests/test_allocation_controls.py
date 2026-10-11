"""Command planning is tested without running a server renderer."""
import json
from pathlib import Path
import tempfile
import unittest

from gaussian_jscc.allocation_controls import build_plan, parser


class AllocationControlTests(unittest.TestCase):
    def fixture(self, root):
        source = root/'source'
        for stage in ('final', 'end_render'):
            folder = source/'checkpoints'/stage
            folder.mkdir(parents=True)
            (folder/'codec.pt').touch()
        (source/'checkpoints/final/train.pt').touch()
        manifest = root/'manifest.json'
        manifest.write_text(json.dumps({'scenes': [
            {'name': 'train', 'role': 'train', 'ply': 'train.ply', 'source': 'images'}]}))
        (root/'train.ply').touch()
        (root/'images').mkdir()
        (source/'training.json').write_text(json.dumps({'manifest': str(manifest),
            'selected_scenes': [{'name': 'train', 'role': 'train'}], 'seed': 42, 'snr': 10.}))
        return parser().parse_args(['--run', str(source), '--out', str(root/'controls'),
                                    '--allocation-steps', '1000', '--test-views', '8'])

    def test_only_four_frozen_fits_and_baseline_is_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            plan = build_plan(args, python='python')
            fits = [job for job in plan['jobs'] if job['kind'] == 'fit']
            self.assertEqual(len(fits), 4)
            self.assertEqual([(job['sampling'], job['beta']) for job in fits],
                             [('deployment', .01), ('single', .01),
                              ('deployment', .003), ('deployment', .001)])
            for job in fits:
                command = job['command']
                self.assertIn('--allocation-only', command)
                for option in ('--bootstrap-steps', '--render-steps', '--joint-steps'):
                    self.assertEqual(command[command.index(option)+1], '0')
                self.assertEqual(command[command.index('--allocation-steps')+1], '1000')
                self.assertEqual(command[command.index('--seed')+1], '42')
                self.assertEqual(command[command.index('--checkpoint')+1],
                                 str(Path(args.run)/'checkpoints/end_render/codec.pt'))
            evaluations = [job for job in plan['jobs'] if job['kind'] == 'evaluate']
            self.assertEqual(len(evaluations), 5)
            first = evaluations[0]['command']
            self.assertIn('--draw-counts', first)
            offset = first.index('--draw-counts')
            self.assertEqual(first[offset+1:offset+4], ['10', '5', '3'])

    def test_missing_allocation_fails_before_any_training(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            (Path(args.run)/'checkpoints/final/train.pt').unlink()
            with self.assertRaises(FileNotFoundError):
                build_plan(args)


if __name__ == '__main__':
    unittest.main()
