"""Exercise actual Bash argument construction without starting CUDA training."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def launch(self, overrides=None, script='scripts/test_render_first.sh'):
        bash = 'C:/softwares/Git/bin/bash.exe' if os.name=='nt' else shutil.which('bash')
        if not bash or not Path(bash).exists():
            self.skipTest('Bash unavailable')
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'input.ply').touch()
            (folder/'sparse').mkdir()
            (folder/'codec.pt').touch()
            env=os.environ.copy()
            for key in ('INITIALIZATION','INIT','STEPS','BOOTSTRAP_STEPS','CUDA_VISIBLE_DEVICES'):
                env.pop(key,None)
            env.update(PYTHON_BIN='/bin/echo',PLY=(folder/'input.ply').as_posix(),SCENE=folder.as_posix())
            env.update(overrides or {})
            if env.get('INIT')=='fixture':
                env['INIT']=(folder/'codec.pt').as_posix()
            return subprocess.run([bash,script,(folder/'run').as_posix()],
                                  cwd=root,env=env,capture_output=True,text=True,encoding='utf-8',timeout=15)

    def test_default_random_no_pretraining(self):
        result=self.launch()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('Initialization: random',result.stdout)
        self.assertIn('--bootstrap-steps 0',result.stdout)
        self.assertNotIn('--init ',result.stdout)

    def test_inherited_old_initializer_and_steps_are_ignored(self):
        result=self.launch({'INIT':'nonexistent-old-checkpoint.pt','STEPS':'2000'})
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('ignoring inherited INIT',result.stdout)
        self.assertIn('--bootstrap-steps 0',result.stdout)
        self.assertNotIn('--init ',result.stdout)

    def test_checkpoint_requires_explicit_mode(self):
        result=self.launch({'INITIALIZATION':'checkpoint','INIT':'fixture'})
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--init ',result.stdout)
        self.assertIn('--bootstrap-steps 0',result.stdout)
        rejected=self.launch({'INITIALIZATION':'checkpoint'})
        self.assertNotEqual(rejected.returncode,0)
        self.assertIn('requires an existing INIT',rejected.stderr)

    def test_position_comparison_requires_gpu_selection(self):
        result=self.launch(script='scripts/test_position_delivery.sh')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Set CUDA_VISIBLE_DEVICES',result.stderr)

    def test_position_comparison_forces_three_random_render_only_runs(self):
        result=self.launch({'CUDA_VISIBLE_DEVICES':'2','INITIALIZATION':'checkpoint',
                            'INIT':'nonexistent.pt','BOOTSTRAP_STEPS':'2000','JOINT_STEPS':'100'},
                           script='scripts/test_position_delivery.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.count('Initialization: random'),3)
        self.assertEqual(result.stdout.count('--bootstrap-steps 0'),3)
        self.assertEqual(result.stdout.count('--joint-steps 0'),3)
        self.assertNotIn('--init ',result.stdout)
        for mode in ('learned','float32','quantized'):
            self.assertIn('--position-delivery '+mode,result.stdout)

    def test_restored_quantized_baseline_locked_parameters(self):
        result=self.launch({'CUDA_VISIBLE_DEVICES':'2', 'INIT':'obsolete.pt',
                            'BOOTSTRAP_STEPS':'5000', 'JOINT_STEPS':'1000',
                            'RENDER_LR':'0.002', 'POSITION_DELIVERY':'learned'},
                           script='scripts/train_quantized12_render_only.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        for argument in ('--position-delivery quantized', '--position-bits 12',
                         '--bootstrap-steps 0', '--joint-steps 0', '--render-lr 0.0001',
                         '--render-backward replay', '--rates 0 8 16 32',
                         '--snr 10', '--channel awgn', '--clip-mode none'):
            self.assertIn(argument,result.stdout)
        self.assertNotIn('--init ',result.stdout)

    def test_restored_quantized_checkpoint_and_steps_override(self):
        result=self.launch({'CUDA_VISIBLE_DEVICES':'2','INITIALIZATION':'checkpoint',
                            'INIT':'fixture','RENDER_STEPS':'20000'},
                           script='scripts/train_quantized12_render_only.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--init ',result.stdout)
        self.assertIn('--render-steps 20000',result.stdout)
        self.assertIn('NOT exact resume',result.stdout)

    def test_quantized16_compressed_launcher(self):
        result=self.launch({'CUDA_VISIBLE_DEVICES':'2','INIT':'old12.pt','RENDER_STEPS':'10000'},
                           script='scripts/train_quantized16_render_only.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        for argument in ('--position-bits 16','--position-compression delta_zlib',
                         '--position-delivery quantized','--render-steps 10000',
                         '--render-lr 0.0001','--bootstrap-steps 0','--joint-steps 0'):
            self.assertIn(argument,result.stdout)
        self.assertNotIn('--init ',result.stdout)


if __name__=='__main__':
    unittest.main()
