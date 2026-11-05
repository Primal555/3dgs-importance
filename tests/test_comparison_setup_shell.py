"""Exercise installer shell behavior without downloading or compiling packages."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


BASH = shutil.which('bash') if os.name != 'nt' else 'C:/softwares/Git/bin/bash.exe'
INSTALLER = Path(__file__).resolve().parents[1] / 'scripts/setup_comparison_baselines.sh'


@unittest.skipUnless(BASH and Path(BASH).is_file(), 'requires Bash')
class ComparisonSetupShellTests(unittest.TestCase):
    def run_installer(self, failure=''):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files = {
                'scripts/setup.sh': INSTALLER.read_text(encoding='utf-8'),
                'comparison_baselines/MesonGS/mesongs.py': '',
                'bin/conda': '''#!/usr/bin/env bash
if [[ "$1" == info ]]; then printf '%s/conda\\n' "$PWD"; fi
''',
                'conda/etc/profile.d/conda.sh': '''conda() {
  if [[ "$-" == *u* ]]; then echo 'unexpected nounset during Conda hook' >&2; fi
  echo "${CONDA_BACKUP_CXX}" >/dev/null
  if [[ "$1" == install && "${TEST_FAILURE:-}" == conda ]]; then return 37; fi
  command conda "$@"
}
''',
                'conda/envs/gs_mesongs/bin/python': '''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$PWD/python_calls"
if [[ "$*" == *submodules/diff-gaussian-rasterization* && "${TEST_FAILURE:-}" == extension ]]; then exit 41; fi
''',
                'conda/envs/gs_mesongs/bin/nvcc': '#!/usr/bin/env bash\nexit 0\n',
            }
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding='utf-8', newline='\n')
                path.chmod(0o755)
            env = os.environ.copy()
            for name in ('DIGITAL_ENV', 'MESONGS_ENV', 'MESONGS_CUDA_HOME', 'CONDA_BACKUP_CXX'):
                env.pop(name, None)
            env['TEST_FAILURE'] = failure
            result = subprocess.run(
                [BASH, '-c', 'export PATH="$PWD/bin:$PATH"; bash scripts/setup.sh mesongs'],
                cwd=root, env=env, capture_output=True, text=True,
            )
            calls = (root / 'python_calls').read_text() if (root / 'python_calls').exists() else ''
            return result, calls

    def test_conda_hook_with_unset_backup_does_not_abort_extension_install(self):
        result, calls = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('unexpected nounset', result.stderr)
        for module in ('diff-gaussian-rasterization', 'simple-knn', 'weighted_distance'):
            self.assertIn('submodules/' + module, calls)

    def test_conda_failure_stops_before_extension_compilation(self):
        result, calls = self.run_installer('conda')
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertNotIn('submodules/', calls)

    def test_extension_build_failure_stops_remaining_builds(self):
        result, calls = self.run_installer('extension')
        self.assertEqual(result.returncode, 41, result.stderr)
        self.assertNotIn('submodules/simple-knn', calls)


if __name__ == '__main__':
    unittest.main()
