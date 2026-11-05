"""Native child processes must use their selected Conda runtime libraries."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from comparison_baselines.run import _subprocess_environment


class ProcessEnvironmentTests(unittest.TestCase):
    def test_target_runtime_precedes_parent_and_system_libraries(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'gs_mesongs'
            (target / 'bin').mkdir(parents=True)
            (target / 'lib').mkdir()
            python = target / 'bin/python'
            original = os.pathsep.join(['/parent/maskgs/lib', '/usr/lib'])
            with patch.dict(os.environ, {'LD_LIBRARY_PATH': original}):
                child = _subprocess_environment(str(python))
                self.assertEqual(child['LD_LIBRARY_PATH'].split(os.pathsep),
                                 [str(target / 'lib'), '/parent/maskgs/lib', '/usr/lib'])
                self.assertEqual(os.environ['LD_LIBRARY_PATH'], original)
                self.assertEqual(child['PYTHONUNBUFFERED'], '1')

    def test_bare_interpreter_uses_path_resolution_without_empty_search_entries(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            (target / 'lib').mkdir()
            with patch('comparison_baselines.run.shutil.which', return_value=str(target / 'bin/python')), \
                    patch.dict(os.environ, {'LD_LIBRARY_PATH': ''}):
                child = _subprocess_environment('python')
            self.assertEqual(child['LD_LIBRARY_PATH'], str(target / 'lib'))

    def test_missing_runtime_directory_preserves_existing_search_path(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {'LD_LIBRARY_PATH': '/original/lib'}):
            child = _subprocess_environment(str(Path(folder) / 'bin/python'))
            self.assertEqual(child['LD_LIBRARY_PATH'], '/original/lib')


if __name__ == '__main__':
    unittest.main()
