import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ChannelAuditLauncherTests(unittest.TestCase):
    def test_launcher_passes_two_checkpoints_read_only_and_requires_gpu(self):
        from gaussian_jscc.channel_audit import parser
        binary=Path('C:/softwares/Git/bin/bash.exe')
        bash=str(binary) if binary.exists() else shutil.which('bash')
        if not bash:self.skipTest('Bash unavailable')
        for gpu in (True,False):
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);capture=root/'argv'
                current=root/'new run';reference=root/'old run';out=root/'audit'
                for run in (current,reference):
                    (run/'checkpoints/final').mkdir(parents=True)
                    (run/'checkpoints/final/codec.pt').touch();(run/'training.json').write_text('{}')
                env=dict(os.environ,PYTHON_BIN='capture_python',CAPTURE=capture.as_posix())
                if gpu:env['CUDA_VISIBLE_DEVICES']='2'
                else:env.pop('CUDA_VISIBLE_DEVICES',None)
                command=('capture_python() { printf "%s\\0" "$@" > "$CAPTURE"; }; export -f capture_python; '
                         'bash scripts/audit_channel_codecs_truck.sh "$1" "$2" "$3" --trials 8')
                result=subprocess.run([bash,'-c',command,'test',current.as_posix(),reference.as_posix(),out.as_posix()],
                    cwd=Path(__file__).resolve().parents[1],env=env,text=True,capture_output=True)
                if not gpu:
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('CUDA_VISIBLE_DEVICES',result.stderr);self.assertFalse(capture.exists())
                    continue
                self.assertEqual(result.returncode,0,result.stderr)
                argv=capture.read_text().split('\0')[:-1]
                self.assertEqual(argv[:3],['-u','-m','gaussian_jscc.channel_audit'])
                opts=parser().parse_args(argv[3:])
                self.assertEqual(Path(opts.run),current);self.assertEqual(Path(opts.reference_run),reference)
                self.assertEqual(opts.trials,8);self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
