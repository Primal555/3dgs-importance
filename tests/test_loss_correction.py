"""Regression contracts for the observed scale-collapse objective defect."""

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gaussian_jscc.codec import CodecConfig
from gaussian_jscc.data import prepare, to_features, write_ply
from gaussian_jscc.losses import (PROFILES, add_arguments, configure_training, objective_stats,
                                  physical_terms, physical_terms_v1, reconstruction_loss)
from gaussian_jscc.benchmark import parameter_metrics
from gaussian_jscc.plots import _phase_segments, _rolling
from gaussian_jscc.transport import load_checkpoint, model_id, save_checkpoint, transmit, receive
from test_gaussian_jscc import fixture


class LossCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def scene(self):
        raw, model = fixture(n=8)
        raw, geometry, _ = prepare(raw, 16)
        f, _ = to_features(raw, geometry, model)
        return raw, model, geometry, f

    def test_scale_expansion_and_collapse_are_symmetric_and_have_gradients(self):
        raw, model, geometry, target = self.scene()
        values = []
        for sign in (-1, 1):
            offset = torch.tensor(sign * math.log(100.), requires_grad=True)
            # Construct normalized outputs with the requested physical log scale.
            pred = target.clone()
            pred[:, 4:7] = target[:, 4:7] + offset / model.attr_std[1:4]
            terms = physical_terms(pred, target, geometry, model)
            loss = terms["scale"].mean() + terms["shape"].mean()
            loss.backward()
            self.assertGreater(float(offset.grad) * sign, 1.)
            self.assertGreater(float(loss.detach()), 1.)
            values.append(loss.detach())
        torch.testing.assert_close(*values)
        collapsed = target.clone()
        collapsed[:, 4:7] -= math.log(100) / model.attr_std[1:4]
        self.assertLess(float(physical_terms_v1(collapsed, target, geometry, model)["shape"].mean()), .3)

    def test_guard_gradients_survive_renderer_clamps_and_sigmoid_saturation(self):
        _, model, geometry, target = self.scene()
        for value in (-40., 40.):
            pred = target.clone()
            pred[:, 4:7] = (value - model.attr_mean[1:4]) / model.attr_std[1:4]
            pred[:, 3] = (value - model.attr_mean[0]) / model.attr_std[0]
            pred.requires_grad_()
            loss = reconstruction_loss(pred, target, geometry, model)
            loss.backward()
            self.assertTrue(torch.isfinite(pred.grad).all())
            self.assertGreater(float(pred.grad[:, 4:7].abs().sum()), 0.)
            self.assertGreater(float(pred.grad[:, 3].abs().sum()), 0.)

    def test_equivalent_axis_permutations_do_not_pay_shape_or_scale_loss(self):
        raw, model, geometry, _ = self.scene()
        raw[:, 4:7] = torch.tensor([-3., -4., -5.])
        permuted = raw.clone()
        permuted[:, 4:7] = raw[:, [5, 4, 6]]
        permuted[:, 7:11] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
        terms = physical_terms(to_features(permuted, geometry, model)[0],
                               to_features(raw, geometry, model)[0], geometry, model)
        self.assertLess(float(terms["shape"].max()), 1e-10)
        self.assertEqual(float(terms["scale"].max()), 0.)
        metrics = parameter_metrics(raw, permuted)
        self.assertLess(metrics["log_covariance_rmse"], 2e-6)
        self.assertGreater(metrics["rotation_angle_mean_deg"], 80.)

    def test_old_geometry_config_hash_and_packet_survive_loading(self):
        raw, model, _, _ = self.scene()
        historical = model.cfg.to_dict()
        historical.pop("loss_profile")
        historical.pop("scale_weight")
        for key, value in PROFILES["physical_v1"].items():
            if key != "scale_weight":
                historical[key] = value
        cfg = CodecConfig.from_dict(historical)
        self.assertEqual(cfg.loss_profile, "physical_v1")
        self.assertEqual(cfg.to_dict(), historical)
        model.cfg = cfg
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Deliberately build the OLD serialized format, not a new roundtrip.
            torch.save(dict(version=3, config=historical, state_dict=model.state_dict(), step=21000), root / "old.pt")
            loaded = load_checkpoint(root / "old.pt", "cpu")
            self.assertEqual(model_id(model), model_id(loaded))
            transmit(model, raw, torch.ones(len(raw), dtype=torch.long), 10., "none", 42, root / "packet")
            torch.testing.assert_close(receive(model, root / "packet"), receive(loaded, root / "packet"))

    def test_training_migration_explicit_weights_and_contribution_accounting(self):
        _, model, geometry, f = self.scene()
        model.cfg.loss_profile = "physical_v1"
        before = {k: v.clone() for k, v in model.state_dict().items()}
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        configure_training(model, parser.parse_args(["--shape-weight", "0.4"]))
        self.assertEqual(model.cfg.loss_profile, "balanced_v2")
        self.assertEqual(model.cfg.shape_weight, .4)
        self.assertEqual(model.cfg.scale_weight, 1.)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(before[k], v, atol=0, rtol=0)
        loss, terms = reconstruction_loss(f + .2, f, geometry, model, return_terms=True)
        values = {k + "_loss": float(v.mean()) for k, v in terms.items()}
        values["aux_loss"] = float(loss)
        stats = objective_stats(values, model, .7)
        component_sum = sum(v for k, v in stats.items() if k.endswith("_contribution") and k != "aux_contribution")
        self.assertAlmostEqual(component_sum, stats["aux_contribution"], places=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new.pt"
            save_checkpoint(path, model, 1)
            loaded = load_checkpoint(path, "cpu")
            self.assertEqual(model_id(model), model_id(loaded))

    def test_signed_scale_metrics_and_phase_segmentation(self):
        raw, _, _, _ = self.scene()
        recovered = raw.clone()
        recovered[:, 4:7] -= math.log(2)
        self.assertAlmostEqual(parameter_metrics(raw, recovered)["log_volume_bias"], -3 * math.log(2), places=5)
        rows = [dict(step=1, phase="attribute", loss_profile="physical_v1", loss=10.),
                dict(step=2, phase="attribute", loss_profile="balanced_v2", loss=5.),
                dict(step=3, phase="render", loss_profile="balanced_v2", loss=1.)]
        segments = _phase_segments(rows)
        self.assertEqual(len(segments), 3)
        for _, group in segments:
            self.assertEqual(list(_rolling([r["loss"] for r in group], 101)), [group[0]["loss"]])

    def test_cli_phase_transition_migrates_old_model_and_preserves_auxiliary(self):
        from gaussian_jscc.cli import main
        raw, model, _, _ = self.scene()
        model.cfg.loss_profile = "physical_v1"
        for key, value in PROFILES["physical_v1"].items():
            setattr(model.cfg, key, value)
        camera = SimpleNamespace(original_image=torch.full((3, 8, 8), .5))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_ply(root / "source.ply", raw, 3)
            save_checkpoint(root / "old.pt", model, 21000)
            argv = ["gaussian_jscc", "train", "--ply", str(root / "source.ply"),
                    "--init", str(root / "old.pt"), "--out", str(root / "run"),
                    "--source", "mock-scene", "--device", "cuda", "--training-data-device", "cpu",
                    "--steps", "1", "--render-steps", "1", "--save-every", "1",
                    "--lr", "0.00005", "--render-lr", "0.00001"]
            def render(scene, *args):
                return scene[:, :3].mean(0).sigmoid()[:, None, None].expand(3, 8, 8)
            with patch.object(sys, "argv", argv), \
                    patch("gaussian_jscc.cli.device_for", return_value=torch.device("cpu")), \
                    patch("gaussian_jscc.rendering.load_cameras", return_value=[camera]), \
                    patch("gaussian_jscc.rendering.render", side_effect=render), \
                    patch("gaussian_jscc.plots.safe_plot"):
                main()
            rows = [json.loads(line) for line in (root / "run" / "loss.jsonl").read_text().splitlines()]
            self.assertEqual([r["phase"] for r in rows], ["attribute", "render"])
            self.assertEqual([r["learning_rate"] for r in rows], [5e-5, 1e-5])
            for row in rows:
                self.assertEqual(row["loss_profile"], "balanced_v2")
                self.assertEqual(row["auxiliary_weight"], 1.)
                terms = sum(row[k + "_contribution"] for k in ("geometry", "shape", "scale", "opacity", "dc", "sh"))
                self.assertAlmostEqual(row["loss"], terms + row.get("render_loss", 0.), places=4)
            new = load_checkpoint(root / "run" / "codec.pt", "cpu")
            self.assertEqual(new.cfg.rates, model.cfg.rates)
            self.assertEqual(new.cfg.geometry_rates, model.cfg.geometry_rates)
            torch.testing.assert_close(new.attr_mean, model.attr_mean, rtol=0, atol=0)
            torch.testing.assert_close(new.attr_std, model.attr_std, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
