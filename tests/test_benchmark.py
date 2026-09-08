"""End-to-end isolated codec benchmark tests."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from gaussian_jscc.benchmark import parameter_metrics
from gaussian_jscc.data import write_ply
from gaussian_jscc.transport import save_checkpoint
from test_gaussian_jscc import fixture


class CodecBenchmarkTests(unittest.TestCase):
    def test_group_metrics_are_zero_for_identical_gaussians(self):
        raw, _ = fixture(n=12)
        metrics = parameter_metrics(raw, raw.clone())
        for key, value in metrics.items():
            if value is not None:
                self.assertAlmostEqual(value, 0., places=6, msg=key)

    def test_cli_isolates_positive_tiers_and_removes_temporary_packets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, model = fixture(n=12, degree=0)
            ply = root / "source.ply"
            checkpoint = root / "codec.pt"
            output = root / "benchmark"
            write_ply(ply, raw, 0)
            save_checkpoint(checkpoint, model, 0)
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            result = subprocess.run(
                [sys.executable, "-m", "gaussian_jscc", "benchmark-codec",
                 "--ply", str(ply), "--checkpoint", str(checkpoint), "--out", str(output),
                 "--device", "cpu", "--tiers", "1", "3", "--snrs", "0", "10",
                 "--channels", "none", "awgn", "--trials", "1"],
                env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 8)
            for row in rows:
                self.assertEqual(row["retained_gaussians"], len(raw))
                self.assertEqual(row["tier_counts"][0], 0)
                self.assertEqual(row["payload_complex_symbols"],
                                 len(raw) * model.cfg.rates[row["tier"]])
                self.assertIn("rotation_angle_mean_deg", row)
                self.assertTrue(torch.isfinite(torch.tensor(row["position_rmse"])))
            self.assertFalse(any(output.glob("packet_*")))
            if importlib.util.find_spec("matplotlib"):
                self.assertTrue((output / "charts" / "codec_parameter_errors_vs_snr.png").is_file())


if __name__ == "__main__":
    unittest.main()
