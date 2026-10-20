import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ContinuationLauncherTests(unittest.TestCase):
    def launch(self,root,*,resume=False,gpu=True):
        path=Path('C:/softwares/Git/bin/bash.exe')
        bash=str(path) if path.is_file() else shutil.which('bash')
        if not bash:self.skipTest('Bash unavailable')
        source=root/'source run';out=root/'new out';capture=root/'arguments'
        (source/'checkpoints/final').mkdir(parents=True)
        (source/'training.json').write_text('{}');(source/'checkpoints/final/codec.pt').touch()
        if resume:
            (out/'checkpoints/latest').mkdir(parents=True);(out/'checkpoints/latest/codec.pt').touch()
        env=dict(os.environ,PYTHON_BIN='capture_python',CAPTURE=capture.as_posix(),RESUME='1' if resume else '0')
        if gpu:env['CUDA_VISIBLE_DEVICES']='2'
        else:env.pop('CUDA_VISIBLE_DEVICES',None)
        cmd=('capture_python() { printf "%s\\0" "$@" > "$CAPTURE"; }; export -f capture_python; '
             'bash scripts/continue_ssim_joint_truck.sh "$1" "$2"')
        result=subprocess.run([bash,'-c',cmd,'test',source.as_posix(),out.as_posix()],env=env,
            cwd=Path(__file__).resolve().parents[1],text=True,capture_output=True)
        return result,capture

    def test_launcher_extends_1000_with_250_validation_and_preserves_resume_mode(self):
        from gaussian_jscc.modular_joint_continue import parser
        for resume in (False,True):
            with tempfile.TemporaryDirectory() as t:
                result,capture=self.launch(Path(t),resume=resume)
                self.assertEqual(result.returncode,0,result.stderr)
                argv=capture.read_text().split('\0')[:-1]
                self.assertEqual(argv[:3],['-u','-m','gaussian_jscc.modular_joint_continue'])
                opts=parser().parse_args(argv[3:])
                self.assertEqual((opts.additional_steps,opts.validate_every,opts.save_every),(1000,250,250))
                self.assertEqual(opts.resume,resume)

    def test_missing_gpu_never_launches_python(self):
        with tempfile.TemporaryDirectory() as t:
            result,capture=self.launch(Path(t),gpu=False)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('CUDA_VISIBLE_DEVICES',result.stderr)
            self.assertFalse(capture.exists())


if __name__=='__main__':unittest.main()
