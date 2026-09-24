import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gaussian_jscc.center_attribute_train import train
from gaussian_jscc.data import write_ply
from gaussian_jscc.transport import load_checkpoint, model_id
from test_center_attribute import setup, args_for


class LaterPhaseTests(unittest.TestCase):
    def test_selected_center_frozen_B_joint_C_and_exact_resume(self):
        torch.set_num_threads(1)
        raw, _, _, _ = setup(40)
        def cameras(source, resolution, white_background, images, split):
            return [SimpleNamespace(image_name=split, original_image=torch.zeros(3, 12, 12))]
        def raster(scene, *a, **kw):
            rgb = (scene[:, :3].mean(0)*.02 + scene[:, 3].sigmoid().mean()).sigmoid()
            return rgb[:, None, None].expand(3, 12, 12)
        with tempfile.TemporaryDirectory() as tmp, \
                patch('gaussian_jscc.center_attribute_train.plot_run'), \
                patch('gaussian_jscc.rendering.load_cameras', cameras), \
                patch('gaussian_jscc.rendering.render', raster):
            root = Path(tmp)
            ply = root/'source.ply'
            write_ply(ply, raw, 0)
            source = root/'center'
            train(args_for(ply, source, ['--source', 'mock', '--min-center-steps', '99']))
            original_bytes = (source/'training_state.pt').read_bytes()
            summary = json.loads((source/'summary.json').read_text())
            best = load_checkpoint(source/f'codec_selected_center_{summary["best_steps"]["center"]}.pt', 'cpu')
            def request(out):
                return args_for(ply, out, ['--after-center-run', str(source), '--source', 'mock',
                    '--attribute-steps', '3', '--joint-steps', '3', '--later-phase-policy', 'budget',
                    '--attribute-min-improvement', '.99', '--joint-center-lr', '1e-5',
                    '--joint-attribute-lr', '1e-4', '--render-backward', 'replay'])
            train(request(root/'full'))
            initial = load_checkpoint(root/'full/codec_center_initial.pt', 'cpu')
            self.assertEqual(model_id(initial), model_id(best))
            frozen = load_checkpoint(root/'full/codec_6.pt', 'cpu')
            for key, value in best.state_dict().items():
                if 'center_encoder.' in key or 'center_decoder.' in key:
                    torch.testing.assert_close(value, frozen.state_dict()[key], atol=0, rtol=0)
            logs = [json.loads(x) for x in (root/'full/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in logs], ['attribute']*3+['joint']*3)
            self.assertTrue(all(r['module_grad_norms']['center_decoder'] == 0 for r in logs[:3]))
            self.assertTrue(any(r['module_grad_norms']['center_decoder'] > 0 for r in logs[3:]))
            final = json.loads((root/'full/summary.json').read_text())
            self.assertEqual(final['completed'], {'center': 3, 'attribute': 3, 'joint': 3})
            self.assertEqual(final['status'], 'complete')
            original = torch.optim.Adam.step
            count = [0]
            def crash(opt, *a, **kw):
                count[0] += 1
                if count[0] == 5:
                    raise RuntimeError('interrupt joint')
                return original(opt, *a, **kw)
            with patch.object(torch.optim.Adam, 'step', crash):
                with self.assertRaisesRegex(RuntimeError, 'interrupt joint'):
                    train(request(root/'resume'))
            args = args_for(ply, root/'resume')
            args.resume = str(root/'resume/training_state.pt')
            train(args)
            self.assertEqual(model_id(load_checkpoint(root/'full/codec.pt', 'cpu')),
                             model_id(load_checkpoint(root/'resume/codec.pt', 'cpu')))
            self.assertEqual((source/'training_state.pt').read_bytes(), original_bytes)


if __name__ == '__main__':
    unittest.main()
