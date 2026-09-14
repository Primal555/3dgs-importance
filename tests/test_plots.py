"""Static chart generation from saved training, evaluation and allocation data."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gaussian_jscc.plots import plot_allocation, plot_evaluation, plot_training


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "requires matplotlib")
class PlotTests(unittest.TestCase):
    def test_geometry_first_components_and_codec_gradient_chart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"step": step, "phase": "attribute", "loss": 1 / step,
                     "geometry_loss": .8 / step, "shape_loss": .1 / step,
                     "opacity_loss": .01 / step, "dc_loss": .02 / step,
                     "sh_loss": .03 / step, "grad_norm": .5 / step} for step in range(1, 6)]
            (root / "loss.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            plot_training(root)
            self.assertTrue((root / "charts" / "training_physical_losses.png").is_file())
            self.assertTrue((root / "charts" / "training_objectives.png").is_file())

    def test_all_statistical_chart_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training"
            training.mkdir()
            rows = []
            for step in range(1, 17):
                row = {"step": step, "phase": "codec_warmup" if step < 5 else "joint_render",
                       "snr": float(step), "loss": 1 / step,
                       "codec_grad_norm": .5 / step, "mask_grad_norm": 0. if step < 5 else .1 / step}
                if step >= 5:
                    row.update(distortion=.8 / step, aux_loss=.6 / step,
                               expected_symbols_per_gaussian=24 - step / 2,
                               sampled_tier_counts=[step, 20 - step, 8, 4], temperature=1 / np.sqrt(step))
                rows.append(row)
            (training / "loss.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            plot_training(training)
            self.assertTrue((training / "charts" / "training_objectives.png").is_file())
            self.assertTrue((training / "charts" / "training_rate_and_tiers.svg").is_file())

            evaluation = root / "evaluation"
            evaluation.mkdir()
            results = []
            for snr in (0, 5, 10, 15, 20):
                for trial in range(3):
                    results.append({"label": f"map_snr{snr}_trial{trial}",
                                    "snr_db": snr, "source_gaussians": 100,
                                    "tier_counts": [10, 25, 35, 30],
                                    "payload_complex_symbols": 1800,
                                    "metadata_channel_uses": 100 / (snr + 1),
                                    "total_channel_uses": 1800 + 100 / (snr + 1),
                                    "total_uses_per_source_gaussian": 18 + 1 / (snr + 1),
                                    "position_rmse": .1 / (snr + 1), "attribute_mse": .2 / (snr + 1),
                                    "received_psnr": 20 + snr / 2 + trial / 10,
                                    "received_ssim": .7 + snr / 100,
                                    "received_lpips": .3 - snr / 100,
                                    "reference_psnr": 31., "reference_ssim": .93,
                                    "reference_lpips": .08,
                                    "position_seed_rmse": .08 / (snr + 1),
                                    "position_seed_nrmse_bbox_diagonal": .008 / (snr + 1),
                                    "position_nrmse_bbox_diagonal": .01 / (snr + 1),
                                    "seed_position_error_only_psnr": 22 + snr / 2 + trial / 10,
                                    "seed_position_error_only_ssim": .74 + snr / 100,
                                    "position_error_only_psnr": 21 + snr / 2 + trial / 10,
                                    "position_error_only_ssim": .72 + snr / 100,
                                    "attribute_error_only_psnr": 28 + snr / 4 + trial / 10,
                                    "attribute_error_only_ssim": .86 + snr / 200})
            (evaluation / "results.json").write_text(json.dumps(results), encoding="utf-8")
            plot_evaluation(evaluation)
            for name in ("quality_vs_snr", "channel_uses_vs_snr", "tier_mix_vs_snr",
                         "rate_distortion", "gaussian_errors_vs_snr",
                         "hybrid_ablation_quality_vs_snr", "position_seed_vs_final"):
                self.assertTrue((evaluation / "charts" / f"{name}.png").is_file(), name)

            allocation = root / "allocation"
            allocation.mkdir()
            rng = np.random.default_rng(4)
            probabilities = rng.dirichlet(np.ones(4), size=200).astype(np.float32)
            tiers = probabilities.argmax(1).astype(np.uint8)
            np.save(allocation / "probabilities.npy", probabilities)
            np.save(allocation / "tiers.npy", tiers)
            plot_allocation(allocation, xyz=rng.normal(size=(200, 3)))
            for name in ("allocation_tier_composition", "allocation_probability_distributions",
                         "allocation_spatial_projections"):
                self.assertTrue((allocation / "charts" / f"{name}.svg").is_file(), name)
            self.assertTrue((allocation / "charts" / "charts_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
