import io
import pickle
import unittest
from unittest.mock import patch

import torch

from gaussian_jscc.budgeted_allocation_state import load_checkpoint


class AllocationCheckpointBytesTests(unittest.TestCase):
    def checkpoint(self):
        stream = io.BytesIO()
        torch.save({'context': {(7, (1, 2), b'\x00\xff'): {'block': b'\x00\xff'}},
                    'step': 2000, 'weights': torch.arange(3)}, stream)
        stream.seek(0)
        return stream

    def test_legacy_restricted_loader_accepts_bytes_and_restores_allowlist(self):
        module = torch.serialization._weights_only_unpickler
        allowed = dict(module._get_allowed_globals())
        allowed.pop('_codecs.encode', None)
        lookup = lambda: allowed
        with patch.object(module, '_get_allowed_globals', lookup), \
                patch.object(torch.serialization, 'safe_globals', None):
            with self.assertRaises(pickle.UnpicklingError):
                torch.load(self.checkpoint(), weights_only=True)
            state = load_checkpoint(self.checkpoint())
            self.assertEqual(state['context'][7, (1, 2), b'\x00\xff']['block'], b'\x00\xff')
            self.assertEqual(state['step'], 2000)
            self.assertIs(module._get_allowed_globals, lookup)
            self.assertNotIn('_codecs.encode', allowed)

    def test_modern_reader_preserves_binary_keys_and_tensors(self):
        state = load_checkpoint(self.checkpoint())
        torch.testing.assert_close(state['weights'], torch.arange(3))
        self.assertIn((7, (1, 2), b'\x00\xff'), state['context'])

    def test_legacy_allowlist_is_restored_after_read_failure(self):
        module = torch.serialization._weights_only_unpickler
        lookup = module._get_allowed_globals
        with patch.object(torch.serialization, 'safe_globals', None):
            with self.assertRaises(Exception):
                load_checkpoint(io.BytesIO(b'invalid checkpoint'))
        self.assertIs(module._get_allowed_globals, lookup)


if __name__ == '__main__':
    unittest.main()
