"""Normalize PyTorch device aliases before constructing Sionna PHY blocks."""
import unittest
from unittest.mock import patch

import torch

from comparison_baselines.digital_link import _resolve_device


class LinkDeviceTests(unittest.TestCase):
    def test_cuda_alias_uses_current_logical_device(self):
        with patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'current_device', return_value=2):
            self.assertEqual(str(_resolve_device(torch, 'cuda')), 'cuda:2')

    def test_explicit_cuda_index_is_preserved(self):
        with patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'current_device', side_effect=AssertionError('unexpected query')):
            self.assertEqual(str(_resolve_device(torch, 'cuda:0')), 'cuda:0')

    def test_cpu_alias_is_normalized(self):
        self.assertEqual(str(_resolve_device(torch, 'cpu:0')), 'cpu')

    def test_unavailable_cuda_still_fails(self):
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaises(RuntimeError):
                _resolve_device(torch, 'cuda')


if __name__ == '__main__':
    unittest.main()
