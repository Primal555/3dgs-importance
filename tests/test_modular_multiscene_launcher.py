import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ModularMultisceneLauncherTests(unittest.TestCase):
    def launch(self, root, *, resume=False, gpu=True, manifest=True):
        bash=Path('C:/softwares/Git/bin/bash.exe')
        binary=str(bash) if bash.is_file() else shutil.which('bash')
        if not binary:self.skipTest('Bash unavailable')
        source=root/'pruned scenes.json';out=root/'new output';capture=root/'arguments'
        if manifest:source.write_text('{}')
        if resume:
            checkpoint=out/'checkpoints/latest/codec.pt'
            checkpoint.parent.mkdir(parents=True);checkpoint.touch()
        env=dict(os.environ,PYTHON_BIN='capture_python',CAPTURE=capture.as_posix(),RESUME='1' if resume else '0')
        if gpu:env['CUDA_VISIBLE_DEVICES']='2'
        else:env.pop('CUDA_VISIBLE_DEVICES',None)
        command=('capture_python() { printf "%s\\0" "$@" > "$CAPTURE"; }; '
                 'export -f capture_python; bash scripts/train_modular_multiscene_ssim.sh "$1" "$2" --no-lpips')
        result=subprocess.run([binary,'-c',command,'test',source.as_posix(),out.as_posix()],
            cwd=Path(__file__).resolve().parents[1],env=env,text=True,capture_output=True)
        return result,capture,out

    def test_fresh_launch_requires_no_source_weights_and_forwards_stage_defaults(self):
        from gaussian_jscc.modular_multiscene import parser,phase_lrs
        with tempfile.TemporaryDirectory() as t:
            result,capture,out=self.launch(Path(t))
            self.assertEqual(result.returncode,0,result.stderr)
            argv=capture.read_text().split('\0')[:-1]
            self.assertEqual(argv[:3],['-u','-m','gaussian_jscc.modular_multiscene'])
            args=parser().parse_args(argv[3:])
            self.assertIsNone(args.resume)
            self.assertEqual(args.scenes,['train','truck','playroom'])
            self.assertEqual((args.semantic_bootstrap_steps,args.semantic_render_steps,args.channel_bootstrap_steps,args.channel_render_steps),
                             (5000,1000,2000,2000))
            self.assertEqual(phase_lrs(args,'semantic_render')['semantic_decoder'],1e-4)
            self.assertEqual(phase_lrs(args,'channel_render')['semantic_decoder'],1e-5)
            self.assertEqual(phase_lrs(args,'channel_render')['channel_encoder'],1e-4)
            self.assertFalse(args.lpips)
            self.assertFalse(out.exists())

    def test_resume_forwards_exact_checkpoint(self):
        from gaussian_jscc.modular_multiscene import parser
        with tempfile.TemporaryDirectory() as t:
            result,capture,out=self.launch(Path(t),resume=True)
            self.assertEqual(result.returncode,0,result.stderr)
            args=parser().parse_args(capture.read_text().split('\0')[3:-1])
            self.assertEqual(args.resume,(out/'checkpoints/latest/codec.pt').as_posix())

    def test_missing_gpu_or_manifest_does_not_launch_python(self):
        for settings in (dict(gpu=False),dict(manifest=False)):
            with self.subTest(settings=settings),tempfile.TemporaryDirectory() as t:
                result,capture,_=self.launch(Path(t),**settings)
                self.assertNotEqual(result.returncode,0)
                self.assertFalse(capture.exists())
                self.assertIn('CUDA_VISIBLE_DEVICES' if settings.get('gpu') is False else 'Missing pruned manifest',result.stderr)


if __name__=='__main__':unittest.main()
