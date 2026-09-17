"""Exercise actual Bash argument construction without starting CUDA training."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def launch(self, overrides=None):
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
            for key in ('INITIALIZATION','INIT','STEPS','BOOTSTRAP_STEPS'):
                env.pop(key,None)
            env.update(PYTHON_BIN='/bin/echo',PLY=(folder/'input.ply').as_posix(),SCENE=folder.as_posix())
            env.update(overrides or {})
            if env.get('INIT')=='fixture':
                env['INIT']=(folder/'codec.pt').as_posix()
            return subprocess.run([bash,'scripts/test_render_first.sh',(folder/'run').as_posix()],
                                  cwd=root,env=env,capture_output=True,text=True,timeout=15)

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


if __name__=='__main__':
    unittest.main()
