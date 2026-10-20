"""SSIM communication controls must select the intended weights, LRs and phases."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class SSIMLowLRLauncherTests(unittest.TestCase):
    def launch(self, root, *, resume=False, checkpoint=True,
               script='test_ssim_low_lr_truck.sh', initializer='end_semantic_render'):
        git_bash = Path('C:/softwares/Git/bin/bash.exe')
        bash = str(git_bash) if git_bash.is_file() else shutil.which('bash')
        if not bash:
            self.skipTest('Bash unavailable')
        run, out = root/'source run', root/'new out'
        run.mkdir(exist_ok=True)
        (run/'training.json').touch()
        initial = run/'checkpoints'/initializer/'codec.pt'
        latest = out/'checkpoints/latest/codec.pt'
        target = latest if resume else initial
        if checkpoint:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        capture = root/'arguments'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='2', PYTHON_BIN='capture_python',
                   CAPTURE=capture.as_posix(), RESUME='1' if resume else '0')
        command = ('capture_python() { printf "%s\\0" "$@" > "$CAPTURE"; }; '
                   'export -f capture_python; '
                   'bash "$3" "$1" "$2"')
        result = subprocess.run([bash, '-c', command, 'test', run.as_posix(), out.as_posix(),
                                 'scripts/'+script],
                                cwd=Path(__file__).resolve().parents[1], env=env,
                                capture_output=True, text=True)
        return result, capture, initial, latest

    def test_fresh_run_passes_low_lrs_and_preserves_ssim_schedule(self):
        # Catches accidentally inheriting 1e-4 or restarting semantic training.
        from gaussian_jscc.modular_experiment import parser
        with tempfile.TemporaryDirectory() as directory:
            result, capture, initial, _ = self.launch(Path(directory))
            self.assertEqual(result.returncode, 0, result.stderr)
            argv = capture.read_text().split('\0')[:-1]
            self.assertEqual(argv[:3], ['-u', '-m', 'gaussian_jscc.modular_experiment'])
            args = parser().parse_args(argv[3:])
            self.assertEqual(args.init_semantic, initial.as_posix())
            self.assertIsNone(args.init_codec)
            self.assertIsNone(args.resume)
            self.assertEqual((args.channel_encoder_lr, args.channel_decoder_lr,
                              args.semantic_decoder_lr), (1e-5, 1e-5, 1e-5))
            self.assertEqual((args.semantic_bootstrap_steps, args.semantic_render_steps,
                              args.channel_bootstrap_steps, args.channel_render_steps),
                             (0, 0, 2000, 2000))
            self.assertEqual(args.image_objective, 'mse_ssim')
            self.assertAlmostEqual(args.image_aux_weight, 1/3)
            self.assertEqual(args.channel_latent_weight, 1.)
            self.assertEqual(args.train_snrs, [10.])
            self.assertEqual(args.test_snrs, [10.])
            self.assertTrue(args.use_source_config)
            self.assertTrue(args.color_metrics)
            self.assertTrue(args.lpips)

    def test_resume_selects_saved_optimizer_instead_of_reinitializing_channel(self):
        from gaussian_jscc.modular_experiment import parser
        with tempfile.TemporaryDirectory() as directory:
            result, capture, _, latest = self.launch(Path(directory), resume=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = parser().parse_args(capture.read_text().split('\0')[3:-1])
            self.assertEqual(args.resume, latest.as_posix())
            self.assertIsNone(args.init_semantic)

    def test_missing_initializer_stops_before_starting_python(self):
        with tempfile.TemporaryDirectory() as directory:
            result, capture, _, _ = self.launch(Path(directory), checkpoint=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(capture.exists())
            self.assertIn('Missing semantic initializer', result.stderr)

    def test_joint_only_preserves_trained_channels_and_uses_asymmetric_lrs(self):
        # Catches resetting trained channels, repeating bootstrap, or using 1e-4
        # for the semantic decoder when the source metadata contains that LR.
        from gaussian_jscc.modular_experiment import parser
        with tempfile.TemporaryDirectory() as directory:
            result, capture, initial, _ = self.launch(Path(directory),
                script='test_ssim_joint_lr_truck.sh', initializer='end_channel_bootstrap')
            self.assertEqual(result.returncode, 0, result.stderr)
            args = parser().parse_args(capture.read_text().split('\0')[3:-1])
            self.assertEqual(args.init_codec, initial.as_posix())
            self.assertIsNone(args.init_semantic)
            self.assertIsNone(args.resume)
            self.assertEqual((args.channel_encoder_lr, args.channel_decoder_lr,
                              args.semantic_decoder_lr), (1e-4, 1e-4, 1e-5))
            self.assertEqual((args.semantic_bootstrap_steps, args.semantic_render_steps,
                              args.channel_bootstrap_steps, args.channel_render_steps),
                             (0, 0, 0, 2000))
            self.assertEqual(args.image_objective, 'mse_ssim')
            self.assertAlmostEqual(args.image_aux_weight, 1/3)
            self.assertEqual(args.channel_latent_weight, 1.)
            self.assertEqual(args.train_snrs, [10.])
            self.assertEqual(args.test_snrs, [10.])
            self.assertTrue(args.color_metrics)
            self.assertTrue(args.lpips)

    def test_joint_resume_restores_latest_instead_of_source_checkpoint(self):
        from gaussian_jscc.modular_experiment import parser
        with tempfile.TemporaryDirectory() as directory:
            result, capture, _, latest = self.launch(Path(directory), resume=True,
                script='test_ssim_joint_lr_truck.sh', initializer='end_channel_bootstrap')
            self.assertEqual(result.returncode, 0, result.stderr)
            args = parser().parse_args(capture.read_text().split('\0')[3:-1])
            self.assertEqual(args.resume, latest.as_posix())
            self.assertIsNone(args.init_codec)

    def test_joint_missing_channel_checkpoint_does_not_start_random_training(self):
        with tempfile.TemporaryDirectory() as directory:
            result, capture, _, _ = self.launch(Path(directory), checkpoint=False,
                script='test_ssim_joint_lr_truck.sh', initializer='end_channel_bootstrap')
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(capture.exists())
            self.assertIn('Missing channel initializer', result.stderr)


if __name__ == '__main__':
    unittest.main()
