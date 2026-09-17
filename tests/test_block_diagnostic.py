"""Algebra/accounting checks for the local diagnostic, not quality thresholds."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from gaussian_jscc.block_geometry import BlockGeometry
from scripts.diagnose_block_geometry import decomposition, pilot_control, train_short


class BlockDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(11)
        self.model=BlockGeometry((0,4,8,16),16)
        self.blocks=[torch.rand(32,3)*.2 + .4, torch.rand(16,3)]
        self.span=torch.tensor([2.,3.,4.])

    def test_error_decomposition_reconciles_including_cross_terms(self):
        result=decomposition(self.model,self.blocks,self.span,[0,1],1,42)
        self.assertLess(abs(result['additive_mse_residual']),1e-6)
        self.assertLess(result['max_coordinate_reconciliation'],1e-6)
        self.assertAlmostEqual(result['mse']['offset'],result['mse']['oracle_reference'],places=6)
        self.assertEqual(result['n'],48)

    def test_pilot_control_has_equal_power_and_noiseless_identity(self):
        with patch('scripts.diagnose_block_geometry.channel',side_effect=lambda z,*args:z):
            for tier in (1,2,3):
                result=pilot_control(self.model,self.blocks,self.span,[0,1],tier,42)
                self.assertLess(result['rmse'],1e-6)
                self.assertLess(result['maximum_power_deviation'],1e-6)
                self.assertLess(result['max_abs_relative_gain_error'],1e-6)

    def test_short_training_does_not_change_initializer(self):
        before=copy.deepcopy(self.model.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            result=train_short(self.model,self.blocks,self.span,[0,1],1,'current_local',Path(tmp))
            self.assertEqual(len(result['logs']),1)
            self.assertEqual([r['step'] for r in result['evaluations']],[0,1])
        for key,value in before.items():
            torch.testing.assert_close(value,self.model.state_dict()[key],atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
