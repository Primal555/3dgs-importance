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
            for key in ('INITIALIZATION','INIT','STEPS','BOOTSTRAP_STEPS','CUDA_VISIBLE_DEVICES',
                        'LR','RENDER_LR','LR_SCHEDULE','RENDER_BACKWARD','BLOCKS_PER_BATCH'):
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

    def test_pure_render_baseline_requires_gpu(self):
        result=self.launch(script='scripts/train_quantized12_render_only.sh')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Set CUDA_VISIBLE_DEVICES',result.stderr)

    def test_pure_render_baseline_ignores_two_stage_environment(self):
        result=self.launch({'CUDA_VISIBLE_DEVICES':'1','BOOTSTRAP_STEPS':'2000',
                            'BOOTSTRAP_OBJECTIVE':'local-response','JOINT_STEPS':'100',
                            'RENDER_LR':'0.00001','TRAIN_VIEWS':'12','POSITION_DELIVERY':'learned',
                            'POSITION_BITS':'8','CLIP_MODE':'global','INIT':'missing.pt'},
                           script='scripts/train_quantized12_render_only.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        for flag in ('--bootstrap-steps 0','--joint-steps 0','--render-steps 5000',
                     '--position-delivery quantized','--position-bits 12','--render-lr 0.0001',
                     '--train-views 0','--clip-mode none','--blocks-per-batch 64'):
            self.assertIn(flag,result.stdout)
        self.assertNotIn('--init ',result.stdout)
        self.assertNotIn('local-response',result.stdout)

    def test_pure_render_weight_continuation_is_explicit(self):
        result=self.launch({'CUDA_VISIBLE_DEVICES':'1','INITIALIZATION':'checkpoint',
                            'INIT':'fixture','RENDER_STEPS':'4000'},
                           script='scripts/train_quantized12_render_only.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--init ',result.stdout)
        self.assertIn('--render-steps 4000',result.stdout)
        self.assertIn('NOT exact resume',result.stdout)
        rejected=self.launch({'CUDA_VISIBLE_DEVICES':'1','INITIALIZATION':'checkpoint'},
                             script='scripts/train_quantized12_render_only.sh')
        self.assertNotEqual(rejected.returncode,0)

    def test_local_response_two_stage_launcher(self):
        result=self.launch({'CUDA_VISIBLE_DEVICES':'2','INIT':'nonexistent.pt'},
                           script='scripts/test_local_response.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        for flag in ('--bootstrap-steps 2000','--render-steps 300',
                     '--bootstrap-objective local-response','--position-delivery quantized',
                     '--position-bits 12','--lr 0.0002','--render-lr 0.0002','--joint-steps 0',
                     '--render-backward replay','--lr-schedule constant','--lr-patience 3'):
            self.assertIn(flag,result.stdout)
        self.assertNotIn('--init ',result.stdout)
        self.assertNotEqual(self.launch(script='scripts/test_local_response.sh').returncode,0)

    def test_render_continuation_from_bootstrap(self):
        result=self.launch({'CUDA_VISIBLE_DEVICES':'2','INITIALIZATION':'checkpoint',
                            'INIT':'fixture','BOOTSTRAP_STEPS':'0','RENDER_STEPS':'1000',
                            'POSITION_DELIVERY':'quantized','POSITION_BITS':'12',
                            'BLOCKS_PER_BATCH':'64','JOINT_STEPS':'0'},
                           script='scripts/train_codec_learned.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        for flag in ('--init ', '--bootstrap-steps 0','--render-steps 1000',
                     '--position-delivery quantized','--position-bits 12',
                     '--blocks-per-batch 64','--render-backward replay',
                     '--lr 0.0002','--render-lr 0.0002','--lr-schedule constant'):
            self.assertIn(flag,result.stdout)

    def test_direct_remains_explicit_opt_in(self):
        result=self.launch({'RENDER_BACKWARD':'direct'},script='scripts/train_codec_learned.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--render-backward direct',result.stdout)

    def test_lr_settings_can_be_overridden(self):
        result=self.launch({'CUDA_VISIBLE_DEVICES':'2','LR_PATIENCE':'5',
                            'LR_FACTOR':'0.3','MIN_LR':'0.000002'},script='scripts/test_local_response.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        for flag in ('--lr-patience 5','--lr-factor 0.3','--min-lr 0.000002'):
            self.assertIn(flag,result.stdout)

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


if __name__=='__main__':
    unittest.main()
