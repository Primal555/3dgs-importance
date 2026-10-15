"""Warm-start all trained modules, then restore y before joint adaptation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import test_modular_experiment as fixture
import test_render_first as rendering


class ChannelJointFinetuneTests(unittest.TestCase):
    def options(self, source, out):
        from gaussian_jscc.modular_experiment import parser
        return parser().parse_args([
            '--run', str(source), '--out', str(out), '--device', 'cpu',
            '--channel-bootstrap-steps', '4', '--channel-render-steps', '4',
            '--semantic-bootstrap-steps', '1', '--semantic-render-steps', '1',
            '--semantic-decoder-lr', '1e-5', '--train-snrs', '10',
            '--validate-every', '4', '--save-every', '2', '--color-metrics'])

    def test_trained_spatial_codec_is_not_reset_and_phase_updates_are_scoped(self):
        # Catches reinitializing channels/config, inheriting neighbor-only flags,
        # or updating semantic weights during the y-only stage.
        from gaussian_jscc.modular_experiment import main
        from gaussian_jscc.channel_spatial import enable_channel_denoising, enable_neighbor_refinement
        from gaussian_jscc.modular_checkpoint import restore_training_checkpoint, save_training_checkpoint, capture_rng
        from gaussian_jscc.transport import load_checkpoint
        with tempfile.TemporaryDirectory() as t, \
                patch('gaussian_jscc.rendering.render', side_effect=rendering.synthetic_render), \
                patch('gaussian_jscc.rendering.load_cameras', return_value=rendering.RenderFirstTests().cameras()):
            root = Path(t); source = fixture.source_fixture(root); trained = root/'trained'
            opts = fixture.ModularExperimentTests().args(source, trained)
            opts.train_snrs = [10.]; opts.rates = [8, 24, 48]
            opts.channel_variant = 'progressive'; opts.channel_hidden = 32
            opts.context_variant = 'both'; opts.context_centers = [4, 2]
            opts.context_neighbors = [3, 4]
            main(opts)
            checkpoint = trained/'checkpoints/final/codec.pt'
            model, channel_opt, progress, _, record = restore_training_checkpoint(checkpoint, 'cpu')
            from gaussian_jscc.progressive_channel import enable_full_prefix
            enable_full_prefix(model, channel_opt)
            enable_channel_denoising(model, 'pointwise', hidden=16, neighbors=3)
            enable_neighbor_refinement(model, 'plain', hidden=16, neighbors=3)
            progress.update(channel_neighbor_only=True, freeze_semantic=True)
            from gaussian_jscc.modular_training import configure_phase
            progress.update(phase='channel_bootstrap', phase_complete=True)
            opt = configure_phase(model, 'channel_bootstrap', {}, freeze_semantic=True, neighbor_only=True)
            save_training_checkpoint(checkpoint, model, opt, progress, capture_rng(), record)
            metadata = json.loads((trained/'training.json').read_text())
            metadata['codec_config'] = model.cfg.to_dict()
            (trained/'training.json').write_text(json.dumps(metadata))
            original = checkpoint.read_bytes()
            out = root/'joint'; opts = self.options(trained, out)
            opts.init_codec = str(checkpoint)
            main(opts)
            initial = load_checkpoint(out/'checkpoints/initial/codec.pt', 'cpu')
            bootstrap = load_checkpoint(out/'checkpoints/end_channel_bootstrap/codec.pt', 'cpu')
            final = load_checkpoint(out/'checkpoints/final/codec.pt', 'cpu')
            self.assertEqual(initial.cfg, model.cfg)
            for n, v in model.state_dict().items():
                torch.testing.assert_close(v, initial.state_dict()[n], rtol=0, atol=0)
            for module in ('semantic_encoder', 'semantic_decoder'):
                for n, v in getattr(initial, module).state_dict().items():
                    torch.testing.assert_close(v, getattr(bootstrap, module).state_dict()[n], rtol=0, atol=0)
            for n, v in initial.semantic_encoder.state_dict().items():
                torch.testing.assert_close(v, final.semantic_encoder.state_dict()[n], rtol=0, atol=0)
            rows = [json.loads(l) for l in (out/'metrics/loss.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in rows], ['channel_bootstrap']*4+['channel_render']*4)
            self.assertEqual([r['layout'] for r in rows], ['1','2','3','mixed']*2)
            for row in rows:
                self.assertGreater(row['updates']['channel_encoder']['update_norm'], 0)
                self.assertGreater(row['updates']['channel_decoder']['update_norm'], 0)
                self.assertEqual(row['updates']['semantic_encoder']['update_norm'], 0)
                if row['phase'] == 'channel_render':
                    self.assertGreater(row['updates']['semantic_decoder']['update_norm'], 0)
                    self.assertEqual(row['module_lrs']['semantic_decoder'], 1e-5)
                    self.assertAlmostEqual(row['loss'], row['render_loss']+row['weighted_latent_loss'], places=6)
            meta = json.loads((out/'training.json').read_text())
            self.assertFalse(meta['random_initialization']); self.assertTrue(meta['source_weights_loaded'])
            self.assertEqual(meta['source_step'], 8)
            self.assertEqual(checkpoint.read_bytes(), original)
            self.assertEqual(len(list((out/'images/final').glob('*.png'))), 5)

            # Exact resume must infer the FULL saved config, not CLI defaults,
            # and keep Adam/RNG within either stage and at the boundary.
            for cut in (3, 4, 6):
                resumed = root/f'resume{cut}'
                a = self.options(trained, resumed); a.init_codec = str(checkpoint)
                main(a, max_updates=cut)
                a = self.options(trained, resumed); a.resume = str(resumed/'checkpoints/latest/codec.pt')
                main(a)
                recovered = load_checkpoint(resumed/'checkpoints/final/codec.pt', 'cpu')
                for n, v in final.state_dict().items():
                    torch.testing.assert_close(v, recovered.state_dict()[n], rtol=0, atol=0)

    def test_full_initializer_rejects_untrained_channel(self):
        from gaussian_jscc.modular_checkpoint import initialize_codec_run, save_training_checkpoint, capture_rng
        from test_modular_training import setup
        model, _, _, _ = setup()
        with tempfile.TemporaryDirectory() as t:
            path = Path(t)/'teacher.pt'
            save_training_checkpoint(path, model, None,
                dict(step=4, phase='semantic_render', channel_trained=False), capture_rng(), {})
            with self.assertRaisesRegex(ValueError, 'trained channel'):
                initialize_codec_run(path, 'cpu')


if __name__ == '__main__': unittest.main()
