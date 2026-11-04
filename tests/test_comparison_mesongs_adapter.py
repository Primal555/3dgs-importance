"""CPU checks for the MesonGS archive and receiver adapter."""
import ast
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

from comparison_baselines import mesongs_adapter as adapter


REPO = Path(__file__).resolve().parents[1] / "comparison_baselines" / "MesonGS"


class MesonGSAdapterTests(unittest.TestCase):
    def official_geometry_functions(self):
        import torch
        source = (REPO / "scene" / "gaussian_model.py").read_text(encoding="utf-8")
        names = {"d1halfing_fast", "octreecodes", "decode_oct", "ToEulerAngles_FT",
                 "quantize_tensor", "torch_vanilla_quant_ave"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GaussianModel")
        functions.append(next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "save_npz"))
        quant_tree = ast.parse((REPO / "utils" / "quant_utils.py").read_text(encoding="utf-8"))
        functions.append(next(node for node in quant_tree.body if isinstance(node, ast.FunctionDef) and node.name == "split_length"))
        namespace = {"np": np, "torch": torch}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(REPO / "scene" / "gaussian_model.py"), "exec"), namespace)
        return namespace

    def test_uses_json_values_and_rejects_unknown_scene(self):
        self.assertEqual(adapter.load_config(REPO, "c1", "playroom")["percent"], 0.0)
        self.assertEqual(adapter.load_config(REPO, "c3", "flowers")["n_block"], 57)
        self.assertEqual(adapter.load_config(REPO, "c3", "flowers")["percent"], 0.5)
        with self.assertRaises(ValueError):
            adapter.load_config(REPO, "c1", "unknown")

    def test_zero_prune_removes_no_points_even_with_equal_importance(self):
        np.testing.assert_array_equal(adapter.pruning_mask(0, np.zeros(4)), np.zeros(4, bool))
        np.testing.assert_array_equal(adapter.pruning_mask(.5, np.arange(4)), [True, True, False, False])

    def test_zip_traversal_and_unexpected_members_are_rejected(self):
        for name in ("../escape", "/escape", "C:/escape", "bins/../escape", "extra.txt"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                archive = Path(directory) / "bad.zip"
                with zipfile.ZipFile(archive, "w") as stream:
                    stream.writestr(name, "x")
                with self.assertRaises(ValueError):
                    adapter.read_archive(archive)
                self.assertFalse((Path(directory).parent / "escape").exists())

    def test_encoder_requires_cuda_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "playroom"
            source.mkdir()
            ply, output = Path(directory) / "public.ply", Path(directory) / "encoded"
            ply.write_bytes(b"placeholder")
            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    adapter.encode(source=source, ply=ply, out=output, repo=REPO)
            self.assertFalse(output.exists())

    def test_cli_background_is_explicit_and_defaults_to_black(self):
        arguments = ["encode", "--source", "photos", "--ply", "public.ply",
                     "--out", "encoded", "--repo", str(REPO)]
        for flags, expected in (([], False), (["--white-background"], True),
                                (["--no-white-background"], False)):
            with self.subTest(flags=flags), patch.object(adapter, "encode", return_value={}) as encode:
                with patch("sys.stdout", new_callable=io.StringIO):
                    adapter.main(arguments + flags)
                self.assertIs(encode.call_args.kwargs["white_background"], expected)

    def test_python_packaging_keeps_only_codec_members(self):
        arrays, _ = self.fixture_arrays()
        with tempfile.TemporaryDirectory() as directory:
            codec = Path(directory) / "codec with spaces"
            bins = codec / "bins"
            bins.mkdir(parents=True)
            for name, contents in arrays.items():
                np.savez_compressed(bins / name, **contents)
            adapter._ArchiveOS(codec).system("upstream zip invocation")
            restored = adapter.read_archive(codec / "bins.zip")
            np.testing.assert_array_equal(restored["ntk.npz"]["ntk"], arrays["ntk.npz"]["ntk"])

    def fixture_arrays(self):
        # Two leaves: forward RAHT stores their average and difference.
        features = np.array([[.2, .1, -.2, .3, .4, .5, .6],
                             [.4, -.1, .2, -.3, .6, .5, .4]], np.float32)
        root = (features[0] + features[1]) / np.sqrt(2)
        ac = (-features[0] + features[1]) / np.sqrt(2)
        # Store exact chosen uint8 AC, then compare against its dequantized value.
        quantized = np.round(ac / .01 + 128).clip(0, 255).astype(np.uint8)
        arrays = {
            "oct.npz": {"points": np.array([0, 1], np.int64), "params": np.array([0, 1, 0, 1, 0, 1])},
            "ntk.npz": {"ntk": np.array([1, 0], np.int32)},
            "um.npz": {"umap": np.arange(90, dtype=np.float32).reshape(2, 45) / 100},
            "orgb.npz": {"f": root, "i": quantized},
            "ct.npz": {"i": np.array([10, 20, 30, 40, 50, 60], np.uint8)},
            "t.npz": {"t": np.array([1, 1] + [.01, 128] * 7 + [.1, 0] * 3)},
        }
        restored_ac = (quantized.astype(np.float32) - 128) * .01
        expected = np.stack([(root - restored_ac) / np.sqrt(2), (root + restored_ac) / np.sqrt(2)])
        return arrays, expected

    def make_archive(self, path, arrays):
        with zipfile.ZipFile(path, "w") as stream:
            for name, contents in arrays.items():
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **contents)
                stream.writestr(name, buffer.getvalue())

    def test_archive_alone_recovers_standard_ply_with_real_inverse_raht(self):
        from plyfile import PlyData
        arrays, expected = self.fixture_arrays()
        with tempfile.TemporaryDirectory() as directory:
            archive, output = Path(directory) / "received.zip", Path(directory) / "decoded.ply"
            self.make_archive(archive, arrays)
            adapter.decode(archive=archive, ply=output, repo=REPO)
            vertex = PlyData.read(output, mmap=False)["vertex"]
            self.assertEqual(len(vertex.data), 2)
            np.testing.assert_allclose(vertex["opacity"], expected[:, 0], atol=1e-6)
            np.testing.assert_allclose(np.stack([vertex[f"f_dc_{i}"] for i in range(3)], -1), expected[:, 4:], atol=1e-6)
            np.testing.assert_allclose(vertex["z"], [0, .5])
            np.testing.assert_allclose(vertex["scale_0"], [1, 2])
            # Codebook SH order is (coefficient, color); PLY is (color, coefficient).
            self.assertAlmostEqual(vertex["f_rest_1"][0], float(arrays["um.npz"]["umap"][1, 3]))
            np.testing.assert_allclose(sum(vertex[f"rot_{i}"] ** 2 for i in range(4)), 1, atol=1e-6)

    def test_inverse_matches_official_forward_tree_for_many_leaves_and_blocks(self):
        from plyfile import PlyData
        rng = np.random.default_rng(21)
        raht = adapter._raht_module(REPO)
        codes = np.sort(rng.choice(512, size=20, replace=False)).astype(np.int64)
        voxel = np.stack([codes // 64, codes // 8 % 8, codes % 8], -1)
        weights, values, order = raht.copyAsort(voxel)
        forward = raht.haar3D_param(3, weights.copy(), values.copy())
        # Apply the official forward tree to a basis; its transpose is an
        # independent reference for the inverse of this orthogonal transform.
        matrix = np.eye(len(codes), dtype=np.float32)[order]
        for level in range(9):
            left_index, right_index = forward["iLeft_idx"][level], forward["iRight_idx"][level]
            left, right = matrix[left_index].copy(), matrix[right_index].copy()
            w1 = np.asarray(forward["iW1"][level], np.float32)[:, None]
            w2 = np.asarray(forward["iW2"][level], np.float32)[:, None]
            matrix[left_index], matrix[right_index] = w1 * left + w2 * right, -w2 * left + w1 * right
        original = rng.normal(0, .15, (len(codes), 7)).astype(np.float32)
        coefficients = matrix @ original
        quantized = np.round(coefficients[1:] / .01 + 128).astype(np.uint8)
        transmitted = np.concatenate([coefficients[:1], (quantized.astype(np.float32) - 128) * .01])
        expected = matrix.T @ transmitted
        arrays, _ = self.fixture_arrays()
        arrays["oct.npz"]["points"] = codes
        arrays["ntk.npz"]["ntk"] = np.zeros(len(codes), np.int32)
        arrays["orgb.npz"] = {"f": coefficients[0], "i": quantized.T.reshape(-1)}
        arrays["ct.npz"]["i"] = np.zeros(3 * len(codes), np.uint8)
        arrays["t.npz"]["t"] = np.array([3, 3] + [.01, 128] * 21 + [.1, 0] * 9)
        with tempfile.TemporaryDirectory() as directory:
            archive, output = Path(directory) / "received.zip", Path(directory) / "decoded.ply"
            self.make_archive(archive, arrays)
            adapter.decode(archive=archive, ply=output, repo=REPO)
            vertex = PlyData.read(output, mmap=False)["vertex"]
            np.testing.assert_allclose(vertex["opacity"], expected[:, 0], atol=1e-6)
            np.testing.assert_allclose(np.stack([vertex[f"f_dc_{i}"] for i in range(3)], -1), expected[:, 4:], atol=1e-6)

    def test_official_endpoint_codes_decode_without_clipping(self):
        from plyfile import PlyData
        official = self.official_geometry_functions()
        for depth in (4, 20):
            with self.subTest(depth=depth):
                points = np.array([[0, 0, 0], [1, .5, .5], [.5, 1, 1]], np.float64)
                if depth == 20:
                    points = np.concatenate([points, np.full((1, 3), 1 - .5 / 2 ** depth)])
                result = official["octreecodes"](points, depth)
                codes, bounds = result[0], np.asarray(result[1:7])
                self.assertEqual(codes.dtype, np.dtype("int64"))
                self.assertGreater(int(codes.max()), 2 ** (3 * depth))
                if depth == 4:
                    np.testing.assert_array_equal(codes, [0, 2320, 4232])
                expected_xyz, voxel = official["decode_oct"](bounds, codes, depth)
                raht = adapter._raht_module(REPO)
                weights, values, order = raht.copyAsort(voxel)
                forward = raht.haar3D_param(depth, weights.copy(), values.copy())
                matrix = np.eye(len(codes), dtype=np.float32)[order]
                for level in range(3 * depth):
                    li, ri = forward["iLeft_idx"][level], forward["iRight_idx"][level]
                    left, right = matrix[li].copy(), matrix[ri].copy()
                    w1 = np.asarray(forward["iW1"][level], np.float32)[:, None]
                    w2 = np.asarray(forward["iW2"][level], np.float32)[:, None]
                    matrix[li], matrix[ri] = w1 * left + w2 * right, -w2 * left + w1 * right
                coefficients = matrix @ np.arange(len(codes) * 7, dtype=np.float32).reshape(len(codes), 7) / 100
                quantized = np.round(coefficients[1:] / .01 + 128).astype(np.uint8)
                transmitted = np.concatenate([coefficients[:1], (quantized.astype(np.float32) - 128) * .01])
                expected_features = matrix.T @ transmitted
                arrays, _ = self.fixture_arrays()
                arrays["oct.npz"] = {"points": codes, "params": bounds}
                arrays["ntk.npz"]["ntk"] = np.zeros(len(codes), np.int32)
                arrays["orgb.npz"] = {"f": coefficients[0], "i": quantized.T.reshape(-1)}
                arrays["ct.npz"]["i"] = np.zeros(3 * len(codes), np.uint8)
                arrays["t.npz"]["t"][0] = depth
                with tempfile.TemporaryDirectory() as directory:
                    archive, output = Path(directory) / "received.zip", Path(directory) / "decoded.ply"
                    self.make_archive(archive, arrays)
                    adapter.decode(archive=archive, ply=output, repo=REPO)
                    vertex = PlyData.read(output, mmap=False)["vertex"]
                    actual_xyz = np.stack([vertex[key] for key in ("x", "y", "z")], -1)
                    np.testing.assert_array_equal(actual_xyz, expected_xyz.astype(np.float32))
                    np.testing.assert_allclose(vertex["opacity"], expected_features[:, 0], atol=1e-6)

    def test_official_save_npz_serialization_matches_receiver_contract(self):
        import torch
        official = self.official_geometry_functions()
        points = np.array([[0, 0, 0], [1, .5, .5], [.5, 1, 1]])
        result = official["octreecodes"](points, 4)
        codes, bounds = result[0], np.asarray(result[1:7])
        expected_xyz, voxel = official["decode_oct"](bounds, codes, 4)
        raht = adapter._raht_module(REPO)
        weights, values, order = raht.copyAsort(voxel)
        forward = raht.haar3D_param(4, weights.copy(), values.copy())
        model = SimpleNamespace(
            depth=4, n_block=1, oct=codes, oct_param=bounds, reorder=order, res=forward,
            _feature_indices=torch.tensor([0, 1, 0]),
            _features_rest=torch.arange(90, dtype=torch.float32).reshape(2, 45) / 100,
            get_ori_rotation=torch.tensor([[1., 0, 0, 0], [1., .1, .2, .3], [1., -.1, .3, .2]]),
            get_origin_opacity=torch.tensor([[.1], [.2], [.3]]),
            get_features_dc=torch.arange(9, dtype=torch.float32).reshape(3, 1, 3) / 100,
            get_ori_scaling=torch.full((3, 3), -.3),
            qas=[SimpleNamespace(scale=torch.tensor(.01), zero_point=torch.tensor(128.), bit=8) for _ in range(10)],
        )
        with tempfile.TemporaryDirectory() as directory:
            codec = Path(directory) / "official_codec"
            official["os"] = adapter._ArchiveOS(codec)
            official["transform_batched_torch"] = raht.transform_batched_torch
            # Only device placement is redirected; the official forward RAHT,
            # quantization, NPZ field construction and dtype casts execute.
            with patch.object(torch.Tensor, "cuda", lambda tensor: tensor), patch("sys.stdout", new_callable=io.StringIO):
                official["save_npz"](model, str(codec), SimpleNamespace(), per_block_quant=True)
            arrays = adapter.read_archive(codec / "bins.zip")
            self.assertEqual(arrays["ntk.npz"]["ntk"].dtype, np.dtype("int32"))
            self.assertEqual(arrays["um.npz"]["umap"].shape, (2, 45))
            self.assertEqual(arrays["t.npz"]["t"].shape, (22,))
            self.assertEqual(arrays["orgb.npz"]["i"].shape, (14,))
            self.assertEqual(arrays["ct.npz"]["i"].shape, (9,))
            output = Path(directory) / "decoded.ply"
            adapter.decode(archive=codec / "bins.zip", ply=output, repo=REPO)
            from plyfile import PlyData
            vertex = PlyData.read(output, mmap=False)["vertex"]
            np.testing.assert_array_equal(np.stack([vertex[key] for key in ("x", "y", "z")], -1), expected_xyz.astype(np.float32))

    def test_euler_quaternion_matches_upstream_rotation_convention(self):
        euler = np.array([[.4, -.7, 1.2], [0, np.pi / 2, 0]])
        w, x, y, z = adapter.euler_to_quaternion(euler).T
        actual = np.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
                           2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
                           2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], -1).reshape(-1, 3, 3)
        for index, (roll, pitch, yaw) in enumerate(euler):
            rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
            ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
            rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            np.testing.assert_allclose(actual[index], rz @ ry @ rx, atol=2e-7)

    def test_archive_rejects_invalid_shapes_indices_and_quantization(self):
        for field in ("indices", "shape", "depth", "scale", "duplicate_oct", "out_of_bounds_oct"):
            arrays, _ = self.fixture_arrays()
            if field == "indices": arrays["ntk.npz"]["ntk"][0] = 2
            if field == "shape": arrays["ct.npz"]["i"] = np.zeros(3, np.uint8)
            if field == "depth": arrays["t.npz"]["t"][0] = 100
            if field == "scale": arrays["t.npz"]["t"][2] = float("nan")
            if field == "duplicate_oct": arrays["oct.npz"]["points"][:] = 0
            if field == "out_of_bounds_oct": arrays["oct.npz"]["points"][1] = 14
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                archive = Path(directory) / "received.zip"
                self.make_archive(archive, arrays)
                with self.assertRaises(ValueError): adapter.read_archive(archive)


if __name__ == "__main__":
    unittest.main()
