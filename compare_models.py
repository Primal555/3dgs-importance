"""Compare a vanilla 3D-GS baseline with a MaskGaussian model directory."""

import argparse
import json
import re
from pathlib import Path


ITERATION_RE = re.compile(r"iteration_(\d+)$")
METHOD_RE = re.compile(r"ours_(\d+)$")


def latest_numbered_directory(parent, pattern):
    candidates = []
    if parent.exists():
        for path in parent.iterdir():
            if not path.is_dir():
                continue
            match = pattern.match(path.name)
            if match:
                candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError("No iteration directory found under {}".format(parent))
    return max(candidates, key=lambda item: item[0])


def resolve_model(model_dir):
    model_dir = Path(model_dir).expanduser().resolve()
    iteration, iteration_dir = latest_numbered_directory(
        model_dir / "point_cloud", ITERATION_RE
    )
    ply_path = iteration_dir / "point_cloud.ply"
    if not ply_path.is_file():
        raise FileNotFoundError("Missing point cloud: {}".format(ply_path))
    return model_dir, iteration, ply_path


def read_ply_vertex_count(path):
    with Path(path).open("rb") as ply_file:
        first_line = ply_file.readline().decode("ascii", errors="strict").strip()
        if first_line != "ply":
            raise ValueError("Not a PLY file: {}".format(path))
        while True:
            raw_line = ply_file.readline()
            if not raw_line:
                raise ValueError("PLY header has no end_header: {}".format(path))
            line = raw_line.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError("PLY header has no vertex element: {}".format(path))


def read_metrics(model_dir, iteration):
    results_path = model_dir / "results.json"
    if not results_path.is_file():
        return None
    with results_path.open("r", encoding="utf-8") as results_file:
        results = json.load(results_file)
    preferred_key = "ours_{}".format(iteration)
    if preferred_key in results:
        return results[preferred_key]
    numbered = []
    for key, value in results.items():
        match = METHOD_RE.match(key)
        if match:
            numbered.append((int(match.group(1)), value))
    return max(numbered, key=lambda item: item[0])[1] if numbered else None


def metric_delta(baseline_metrics, mask_metrics):
    if baseline_metrics is None or mask_metrics is None:
        return None
    shared = sorted(set(baseline_metrics) & set(mask_metrics))
    return {
        key: mask_metrics[key] - baseline_metrics[key]
        for key in shared
        if isinstance(baseline_metrics[key], (int, float))
        and isinstance(mask_metrics[key], (int, float))
    }


def latest_render_set(model_dir):
    _, method_dir = latest_numbered_directory(model_dir / "test", METHOD_RE)
    renders_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    if not renders_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError("Render or ground-truth directory missing in {}".format(method_dir))
    return renders_dir, gt_dir


def labeled_panel(image, label):
    from PIL import Image, ImageDraw

    label_height = 28
    panel = Image.new("RGB", (image.width, image.height + label_height), "white")
    panel.paste(image.convert("RGB"), (0, label_height))
    ImageDraw.Draw(panel).text((8, 7), label, fill="black")
    return panel


def create_visuals(baseline_dir, mask_dir, output_dir, difference_scale):
    from PIL import Image, ImageChops, ImageEnhance

    baseline_renders, gt_dir = latest_render_set(baseline_dir)
    mask_renders, _ = latest_render_set(mask_dir)
    names = sorted(
        path.name
        for path in baseline_renders.iterdir()
        if path.is_file() and (mask_renders / path.name).is_file() and (gt_dir / path.name).is_file()
    )
    if not names:
        raise FileNotFoundError("No matching rendered test images were found")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        with Image.open(gt_dir / name) as gt_src, Image.open(
            baseline_renders / name
        ) as baseline_src, Image.open(mask_renders / name) as mask_src:
            gt = gt_src.convert("RGB")
            baseline = baseline_src.convert("RGB")
            mask = mask_src.convert("RGB")
            if gt.size != baseline.size or baseline.size != mask.size:
                raise ValueError("Image sizes differ for {}".format(name))
            difference = ImageEnhance.Brightness(
                ImageChops.difference(baseline, mask)
            ).enhance(difference_scale)
            panels = [
                labeled_panel(gt, "Ground truth"),
                labeled_panel(baseline, "Vanilla 3D-GS"),
                labeled_panel(mask, "MaskGaussian"),
                labeled_panel(difference, "|Vanilla - Mask| x{}".format(difference_scale)),
            ]
            composite = Image.new(
                "RGB", (sum(panel.width for panel in panels), panels[0].height), "white"
            )
            x_offset = 0
            for panel in panels:
                composite.paste(panel, (x_offset, 0))
                x_offset += panel.width
            composite.save(output_dir / name)
    return len(names)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Gaussian count, model size, metrics, and rendered views"
    )
    parser.add_argument("--baseline", "-b", required=True, help="Vanilla model directory")
    parser.add_argument("--mask", "-m", required=True, help="MaskGaussian model directory")
    parser.add_argument("--output", type=Path, help="Summary JSON path")
    parser.add_argument("--make_visuals", action="store_true")
    parser.add_argument("--visual_dir", type=Path)
    parser.add_argument("--difference_scale", type=float, default=4.0)
    args = parser.parse_args()

    baseline_dir, baseline_iteration, baseline_ply = resolve_model(args.baseline)
    mask_dir, mask_iteration, mask_ply = resolve_model(args.mask)
    baseline_count = read_ply_vertex_count(baseline_ply)
    mask_count = read_ply_vertex_count(mask_ply)
    if baseline_count <= 0:
        raise ValueError("Vanilla baseline contains no Gaussians: {}".format(baseline_ply))
    baseline_metrics = read_metrics(baseline_dir, baseline_iteration)
    mask_metrics = read_metrics(mask_dir, mask_iteration)

    reduction = 1.0 - mask_count / baseline_count
    summary = {
        "comparison_type": "independently_trained_representation_size_reduction",
        "baseline": {
            "model_dir": str(baseline_dir),
            "iteration": baseline_iteration,
            "ply": str(baseline_ply),
            "gaussians": baseline_count,
            "ply_bytes": baseline_ply.stat().st_size,
            "metrics": baseline_metrics,
        },
        "maskgaussian": {
            "model_dir": str(mask_dir),
            "iteration": mask_iteration,
            "ply": str(mask_ply),
            "gaussians": mask_count,
            "ply_bytes": mask_ply.stat().st_size,
            "metrics": mask_metrics,
        },
        "gaussian_reduction_ratio": reduction,
        "gaussian_reduction_percent": reduction * 100.0,
        "ply_size_reduction_ratio": 1.0 - mask_ply.stat().st_size / baseline_ply.stat().st_size,
        "metric_delta_mask_minus_baseline": metric_delta(baseline_metrics, mask_metrics),
    }

    if args.make_visuals:
        visual_dir = args.visual_dir or (mask_dir / "comparison_vs_baseline")
        summary["comparison_images"] = {
            "directory": str(visual_dir.resolve()),
            "count": create_visuals(
                baseline_dir, mask_dir, visual_dir.resolve(), args.difference_scale
            ),
        }

    output_path = args.output or (mask_dir / "comparison_vs_baseline.json")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=False)

    print("Vanilla Gaussians : {:,}".format(baseline_count))
    print("Mask Gaussians    : {:,}".format(mask_count))
    print("Gaussian reduction: {:.2f}%".format(reduction * 100.0))
    if baseline_metrics is None or mask_metrics is None:
        print("Metrics not found. Run render.py and metrics.py for both models first.")
    else:
        print("Metric delta (Mask - Vanilla): {}".format(summary["metric_delta_mask_minus_baseline"]))
    print("Summary written to {}".format(output_path))


if __name__ == "__main__":
    main()
