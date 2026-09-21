"""Reproducible statistical figures for Gaussian JSCC experiments.

Charts are exported as both PNG (quick inspection/slides) and SVG (papers).
The plotting dependency is deliberately imported lazily so codec training and
packet decoding remain usable in minimal receiver environments.
"""

import csv
import json
from pathlib import Path

import numpy as np


TIER_NAMES = ("Drop", "Low", "Medium", "High")
TIER_COLORS = ("#9AA1A8", "#2F6B9A", "#D8A72E", "#D96C2F")
INK = "#17202A"
GRID = "#DDE2E6"
REFERENCE = "#59636E"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": REFERENCE, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": .7,
        "grid.alpha": .75, "axes.axisbelow": True,
        "legend.frameon": False, "savefig.bbox": "tight",
    })
    return plt


def _read_jsonl(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("png", "svg"):
        path = stem.with_suffix("." + extension)
        fig.savefig(path, dpi=180 if extension == "png" else None)
        outputs.append(str(path))
    return outputs


def _finish(fig, stem):
    plt = _plt()
    fig.tight_layout()
    outputs = _save(fig, stem)
    plt.close(fig)
    return outputs


def _numeric(rows, key):
    return np.asarray([np.nan if row.get(key) is None else float(row[key]) for row in rows], dtype=float)


def _rolling(values, window):
    values = np.asarray(values, dtype=float)
    window = min(window, len(values))
    if window <= 1:
        return values.copy()
    valid = np.isfinite(values).astype(float)
    clean = np.where(np.isfinite(values), values, 0.)
    kernel = np.ones(window)
    total = np.convolve(clean, kernel, mode="same")
    count = np.convolve(valid, kernel, mode="same")
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def _phase_segments(rows):
    """Never join/smooth across either an objective or sampling-phase change."""
    segments = []
    for row in rows:
        key = (row.get("phase", "unknown"), row.get("loss_profile", "unrecorded"))
        if not segments or segments[-1][0] != key:
            segments.append((key, []))
        segments[-1][1].append(row)
    return segments


def _trace(axis, rows, key, label, color, style="-", raw=True):
    values = _numeric(rows, key)
    if not np.isfinite(values).any():
        return
    x = _numeric(rows, "step")
    window = max(1, min(101, len(rows) // 40))
    if raw:
        axis.plot(x, values, color=color, alpha=.16, linewidth=.6)
    axis.plot(x, _rolling(values, window), color=color, linestyle=style,
              linewidth=1.6, label=label, marker="o" if len(rows) == 1 else None)


def _manifest(out, kind, source, charts, notes):
    payload = {"kind": kind, "source": str(Path(source).resolve()),
               "charts": charts, "notes": notes}
    (Path(out) / "charts_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def plot_training(training_dir, output_dir=None):
    """Plot codec/route2 traces without connecting unlike training objectives."""
    training_dir = Path(training_dir)
    out = Path(output_dir) if output_dir else training_dir / "charts"
    out.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(training_dir / "loss.jsonl")
    if rows[0].get('objective') == 'render_mse_v1':
        from .render_plots import plot_render_training
        return plot_render_training(training_dir, output_dir)
    charts = []
    plt = _plt()

    if rows[0].get('objective') in ('spatial_response_v1', 'spatial_response_v2', 'spatial_response_v3', 'spatial_logcov_v1'):
        validation_rows = _read_jsonl(training_dir / 'bootstrap_validation.jsonl')
        entries = [dict(entry,step=r['step']) for r in validation_rows for entry in r['layouts']]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        fig.suptitle('Learned XYZ bootstrap: fixed validation blocks\nBlock/trial means; no coordinate side stream or scene rendering')
        fields = (('loss','Spatial response loss'),('xyz_nrmse_bbox','XYZ RMSE / bbox diagonal'),
                  ('xyz_distance_p95_world','Mean block P95 point distance (world units)'),
                  ('spatial_appearance_response','Appearance response error'))
        for axis, (key,label) in zip(axes.flat,fields):
            for tier,color,style in zip(('1','2','3','mixed'),
                                        (TIER_COLORS[1],TIER_COLORS[2],TIER_COLORS[3],REFERENCE),
                                        ('-','--','-.',':')):
                group = [r for r in entries if r['layout']==tier]
                axis.plot([r['step'] for r in group],_numeric(group,key),style,color=color,label='q'+tier)
            axis.set(xlabel='Bootstrap step',ylabel=label)
            axis.legend(fontsize=8)
        charts += _finish(fig,out/'bootstrap_position_validation')
        _write_csv(out/'bootstrap_position_validation.csv',entries,
                   ['step','layout','loss','xyz_rmse_world','xyz_nrmse_bbox',
                    'xyz_distance_p50_world','xyz_distance_p95_world','spatial_geometry_response',
                    'spatial_appearance_response','symbols_per_gaussian','position_side_stream_bits',
                    'spatial_position_response','spatial_native_shape_response','spatial_logcov_shape_mse',
                    'decoded_anisotropy_p50','source_anisotropy_p50','decoded_near_sphere_fraction',
                    'spatial_coarse_position_response','spatial_fine_position_response',
                    'spatial_fine_weight','spatial_fine_contribution',
                    'max_axis_ratio_p05','max_axis_ratio_p50','max_axis_ratio_p95',
                    'max_axis_ratio_gt10_fraction','max_axis_ratio_lt0_1_fraction',
                    'decoded_max_axis_p50_world','source_max_axis_p50_world',
                    'xyz_distance_over_source_radius_p50','decoded_alpha_p50'])
        if rows[0].get('objective') == 'spatial_logcov_v1':
            fig, axes = plt.subplots(2, 2, figsize=(11, 7))
            fig.suptitle('Log-covariance: fixed-block shape diagnostics (not render quality)')
            fields = (('spatial_logcov_shape_mse','Physical log-covariance Frobenius MSE / 9'),
                      ('decoded_anisotropy_p50','Decoded max/min axis ratio: mean block median'),
                      ('decoded_near_sphere_fraction','Decoded max/min axis ratio < 1.5: fraction'),
                      ('max_axis_ratio_p50','Decoded/source max radius: mean block median'))
            for axis, (key, label) in zip(axes.flat, fields):
                for tier in ('1','2','3','mixed'):
                    group = [r for r in entries if r['layout']==tier]
                    axis.plot([r['step'] for r in group],_numeric(group,key),label='q'+tier)
                if key == 'decoded_anisotropy_p50':
                    teacher = [r for r in entries if r['layout']=='3']
                    axis.plot([r['step'] for r in teacher],_numeric(teacher,'source_anisotropy_p50'),
                              '--',color=REFERENCE,label='Source')
                axis.set(xlabel='Bootstrap step',ylabel=label)
                axis.legend(fontsize=8)
            charts += _finish(fig,out/'bootstrap_logcov_shape')

    segments = _phase_segments(rows)
    fig, axes = plt.subplots(2, len(segments), figsize=(6 * len(segments), 7), squeeze=False)
    fig.suptitle("Training objectives by phase\nIndependent axes; smoothing stays within each phase", fontsize=12)
    for col, ((phase, profile), group) in enumerate(segments):
        top, bottom = axes[:, col]
        _trace(top, group, "loss", "Total objective", TIER_COLORS[1])
        render_key = "render_loss" if any("render_loss" in r for r in group) else "distortion"
        _trace(top, group, render_key, "Render / task term", TIER_COLORS[3], "--")
        aux_key = "aux_contribution" if any("aux_contribution" in r for r in group) else "aux_loss"
        _trace(top, group, aux_key,
               "Weighted reconstruction" if aux_key == "aux_contribution" else "Raw auxiliary (weight not shown)",
               REFERENCE, ":")
        _trace(top, group, "rate_loss", "Weighted rate term", TIER_COLORS[2], "-.")
        top.set(title=f"{phase} | {profile}", ylabel="Objective value", xlabel="Optimization step")
        top.margins(y=.25)
        top.legend(fontsize=8)
        for key, label, color in (("codec_grad_norm", "Codec", TIER_COLORS[1]),
                                  ("grad_norm", "Codec", TIER_COLORS[1]),
                                  ("mask_grad_norm", "Tier mask", TIER_COLORS[3])):
            _trace(bottom, group, key, label, color)
        bottom.set_yscale("symlog", linthresh=1e-5)
        bottom.set(title="Gradient norms before clipping", ylabel="L2 norm (symlog)", xlabel="Optimization step")
        if bottom.lines:
            bottom.legend()
    charts += _finish(fig, out / "training_objectives")

    physical_keys = ("geometry_loss", "shape_loss", "scale_loss", "opacity_loss", "dc_loss", "sh_loss")
    if any(np.isfinite(_numeric(rows, key)).any() for key in physical_keys):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        fig.suptitle("Unweighted reconstruction components\nSeparate units/scales; local-block and full-scene sampling differ")
        for axis, key in zip(axes.flat, physical_keys):
            for index, ((phase, profile), group) in enumerate(segments):
                _trace(axis, group, key, f"{phase} | {profile}",
                       (TIER_COLORS[1], TIER_COLORS[3], REFERENCE)[index % 3],
                       ("-", "--", ":")[index % 3], raw=False)
            axis.set(title=key.removesuffix("_loss"), xlabel="Optimization step", ylabel="Unweighted term")
            if axis.lines:
                axis.legend(fontsize=7)
            else:
                axis.text(.5, .5, "Not recorded", ha="center", transform=axis.transAxes)
        charts += _finish(fig, out / "training_physical_losses")

    if any("geometry_contribution" in r for r in rows):
        fig, axes = plt.subplots(1, len(segments), figsize=(6 * len(segments), 4.5), squeeze=False)
        fig.suptitle("Weighted reconstruction contributions\nScalar objective contributions, not gradient magnitudes")
        colors = (TIER_COLORS[1], TIER_COLORS[3], TIER_COLORS[2], REFERENCE, "#687F45", "#AD668B")
        for axis, ((phase, _), group) in zip(axes.flat, segments):
            for i, (key, color) in enumerate(zip(physical_keys, colors)):
                name = key.removesuffix("_loss")
                _trace(axis, group, name + "_contribution", name, color,
                       ("-", "--", ":")[i % 3], raw=False)
            axis.set(title=phase, xlabel="Optimization step", ylabel="Weighted term")
            if axis.lines:
                axis.legend(ncol=2, fontsize=8)
        charts += _finish(fig, out / "training_weighted_contributions")

    gradient_rows = [r for r in rows if r.get("gradient_groups")]
    if gradient_rows:
        groups = sorted({g for r in gradient_rows for g in r["gradient_groups"]})
        fig, axes = plt.subplots(2, len(groups), figsize=(3.5*len(groups), 6), squeeze=False)
        fig.suptitle("Disjoint parameter groups: clipping and actual updates\nBranch caps do not impose the same global cap")
        for col, group in enumerate(groups):
            selected = [r for r in gradient_rows if group in r["gradient_groups"]]
            for key, style in (("before", "-"), ("after", "--")):
                axes[0,col].plot([r['step'] for r in selected],
                                 [r['gradient_groups'][group][key] for r in selected],
                                 style, label=key, linewidth=.8)
            axes[0,col].set(title=group, yscale='symlog', ylabel='Gradient L2 norm')
            axes[0,col].legend()
            selected = [r for r in selected if group in r.get('updates', {})]
            if selected:
                axes[1,col].plot([r['step'] for r in selected],
                                 [r['updates'][group]['relative_update'] for r in selected], linewidth=.8)
            axes[1,col].set(xlabel='Step', ylabel='Relative parameter update', yscale='symlog')
        charts += _finish(fig, out / 'training_gradient_groups')

    gate_rows = [r for r in rows if r.get('context_gates')]
    if gate_rows:
        fig,axis = plt.subplots(figsize=(9,4))
        for key in gate_rows[0]['context_gates']:
            axis.plot([r['step'] for r in gate_rows],[r['context_gates'][key] for r in gate_rows],label=key)
        axis.set(title='Learned context gates | not loss weights or importance scores',
                 xlabel='Optimization step',ylabel='tanh(gate)',ylim=(-1.05,1.05))
        axis.legend()
        charts += _finish(fig,out/'training_context_gates')
        flattened = [dict(step=r['step'],**r['context_gates']) for r in gate_rows]
        _write_csv(out/'training_context_gates.csv',flattened,list(flattened[0]))

    position_path = training_dir / 'position_evaluation.json'
    if position_path.exists():
        evaluations = json.loads(position_path.read_text(encoding='utf-8'))
        conditions = sorted({(r['channel'],r['snr']) for r in evaluations})
        if conditions:
            fig, axes = plt.subplots(2,len(conditions),figsize=(4*len(conditions),6),squeeze=False)
            fig.suptitle('Fixed-block position recovery by tier\nSame source blocks and noise seeds; not held-out-scene performance')
            for col,(channel,snr) in enumerate(conditions):
                for tier in (1,2,3):
                    selected=[r for r in evaluations if (r['channel'],r['snr'],r['tier'])==(channel,snr,tier)]
                    for axis,key in zip(axes[:,col],('position_rmse','distance_p95')):
                        axis.plot([r['step'] for r in selected],[r[key] for r in selected],
                                  marker='.',label=f'q{tier}',color=TIER_COLORS[tier])
                axes[0,col].set(title=f'{channel}, conditioning SNR {snr:g} dB',ylabel='XYZ RMSE (scene units)')
                axes[1,col].set(xlabel='Step',ylabel='Euclidean distance P95 (scene units)')
                axes[0,col].legend()
            charts += _finish(fig,out / 'training_fixed_positions')
            _write_csv(out / 'position_evaluation.csv',evaluations,list(evaluations[0]))

    joint_rows = [row for row in rows if row.get("sampled_tier_counts") is not None]
    if joint_rows:
        joint_steps = _numeric(joint_rows, "step")
        expected = _numeric(joint_rows, "expected_symbols_per_gaussian")
        temperatures = _numeric(joint_rows, "temperature")
        counts = np.asarray([row["sampled_tier_counts"] for row in joint_rows], dtype=float)
        if counts.ndim != 2 or counts.shape[1] != 4 or (counts < 0).any():
            raise ValueError("sampled_tier_counts must contain four nonnegative values")
        denominator = counts.sum(1, keepdims=True)
        shares = np.divide(counts, denominator, out=np.zeros_like(counts), where=denominator > 0)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(joint_steps, expected, color=TIER_COLORS[1], alpha=.2, linewidth=.7)
        axes[0].plot(joint_steps, _rolling(expected, max(1, min(101, len(joint_rows) // 40))),
                     color=TIER_COLORS[1], linewidth=2, label="Expected payload")
        axes[0].set_title("Expected JSCC payload")
        axes[0].set_ylabel("Complex symbols / source Gaussian")
        lines = axes[0].lines[-1:]
        if np.isfinite(temperatures).any():
            temp_axis = axes[0].twinx()
            temp_axis.grid(False)
            temp_axis.plot(joint_steps, temperatures, color=REFERENCE, linestyle="--", linewidth=1,
                           label="Gumbel temperature")
            temp_axis.set_ylabel("Temperature", color=REFERENCE)
            lines += temp_axis.lines
        axes[0].legend(lines, [line.get_label() for line in lines], loc="upper right")
        axes[1].stackplot(joint_steps, shares.T, labels=TIER_NAMES, colors=TIER_COLORS, alpha=.9)
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Sampled tier composition")
        axes[1].set_ylabel("Share of sampled Gaussians")
        axes[1].set_xlabel("Optimization step")
        axes[1].legend(ncol=4, loc="upper center")
        charts += _finish(fig, out / "training_rate_and_tiers")

    flat = []
    for row in rows:
        item = {key: value for key, value in row.items() if not isinstance(value, (list, dict))}
        counts = row.get("sampled_tier_counts")
        for index, name in enumerate(TIER_NAMES):
            item[f"tier_{name.lower()}_count"] = counts[index] if counts is not None else None
        flat.append(item)
    fields = list(dict.fromkeys(key for item in flat for key in item))
    _write_csv(out / "training_chart_data.csv", flat, fields)
    return _manifest(out, "training", training_dir / "loss.jsonl", charts,
                     ["Smoothing is per contiguous phase/profile, window=min(101, max(1, phase_rows//40)).",
                      "Objective panels have independent axes; losses are not comparable across changed objectives.",
                      "Missing historical weighted contributions are not inferred; raw terms use different units.",
                      "Tier composition records sampled hard actions (possibly sample averages), not deployment argmax counts."])


def _evaluation_series(row):
    label = str(row.get("label", "allocation"))
    return label.split("_snr", 1)[0] if "_snr" in label else "allocation"


def _group_by_series_snr(rows):
    groups = {}
    for row in rows:
        if row.get("snr_db") is None:
            raise ValueError("evaluation records must contain snr_db")
        groups.setdefault((_evaluation_series(row), float(row["snr_db"])), []).append(row)
    return sorted(groups.items())


def _mean_std(group, key):
    values = np.asarray([float(row[key]) for row in group if row.get(key) is not None], dtype=float)
    return ((float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.)
            if len(values) else (np.nan, np.nan))


def plot_evaluation(evaluation_dir, output_dir=None):
    """Plot multi-SNR quality, rate, allocation and Gaussian error summaries."""
    evaluation_dir = Path(evaluation_dir)
    out = Path(output_dir) if output_dir else evaluation_dir / "charts"
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((evaluation_dir / "results.json").read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("results.json must contain a nonempty list")
    grouped = _group_by_series_snr(data)
    summary = []
    for (series, snr), group in grouped:
        row = {"series": series, "snr_db": snr, "trials": len(group)}
        for key in ("received_psnr", "received_ssim", "received_lpips",
                    "reference_psnr", "reference_ssim", "reference_lpips",
                    "received_vs_reference_psnr", "received_vs_reference_ssim",
                    "received_vs_reference_lpips", "received_vs_reference_l1",
                    "psnr_delta_received_minus_reference", "ssim_delta_received_minus_reference",
                    "lpips_delta_received_minus_reference", "total_uses_per_source_gaussian",
                    "position_rmse", "attribute_mse", "position_nrmse_bbox_diagonal",
                    "position_seed_rmse", "position_seed_nrmse_bbox_diagonal",
                    "opacity_alpha_mae", "log_scale_rmse", "rotation_angle_mean_deg",
                    "sorted_log_scale_rmse", "log_covariance_rmse", "log_volume_bias",
                    "rotation_angle_p95_deg", "dc_rmse", "sh_rest_rmse",
                    "all_parameter_rmse"):
            row[key + "_mean"], row[key + "_std"] = _mean_std(group, key)
        for variant in ("seed_position_error_only", "position_error_only",
                        "attribute_error_only"):
            for metric in ("psnr", "ssim", "lpips", "vs_reference_psnr",
                           "vs_reference_ssim", "vs_reference_lpips", "vs_reference_l1"):
                key = f"{variant}_{metric}"
                row[key + "_mean"], row[key + "_std"] = _mean_std(group, key)
        for key in ("payload_complex_symbols", "metadata_channel_uses", "total_channel_uses"):
            row[key + "_mean"], row[key + "_std"] = _mean_std(group, key)
        tier_arrays = [record.get("tier_counts") for record in group if record.get("tier_counts") is not None]
        if tier_arrays:
            mean_counts = np.asarray(tier_arrays, dtype=float).mean(0)
            for index, name in enumerate(TIER_NAMES):
                row[f"tier_{name.lower()}_count_mean"] = float(mean_counts[index])
                row[f"tier_{name.lower()}_share"] = float(mean_counts[index] / mean_counts.sum())
        summary.append(row)
    fields = sorted({key for row in summary for key in row})
    _write_csv(out / "evaluation_chart_data.csv", summary, fields)
    charts = []
    plt = _plt()
    series_names = sorted({row["series"] for row in summary})
    series_rows = {series: sorted((row for row in summary if row["series"] == series),
                                  key=lambda row: row["snr_db"])
                   for series in series_names}
    palette = (TIER_COLORS[1], TIER_COLORS[3], TIER_COLORS[2], "#6F7F3F", "#B55A8A")
    markers = ("o", "s", "^", "D", "v")

    available = [("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("lpips", "LPIPS (lower is better)")]
    available = [(key, label) for key, label in available
                 if any(np.isfinite(row.get(f"received_{key}_mean", np.nan)) for row in summary)]
    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4), squeeze=False)
        for axis, (key, label) in zip(axes[0], available):
            for index, series in enumerate(series_names):
                rows = series_rows[series]
                snrs = np.asarray([row["snr_db"] for row in rows])
                received = np.asarray([row[f"received_{key}_mean"] for row in rows])
                error = np.asarray([row[f"received_{key}_std"] for row in rows])
                axis.errorbar(snrs, received, yerr=error, color=palette[index % len(palette)],
                              marker=markers[index % len(markers)], capsize=3, linewidth=1.8,
                              label="Received" if len(series_names) == 1 else series)
            rows = series_rows[series_names[0]]
            snrs = np.asarray([row["snr_db"] for row in rows])
            reference = np.asarray([row[f"reference_{key}_mean"] for row in rows])
            if np.isfinite(reference).any():
                axis.plot(snrs, reference, color=REFERENCE, linestyle="--", marker="s",
                          fillstyle="none", label="Input PLY reference")
            axis.set_title(label.split(" (")[0])
            axis.set_xlabel("SNR (dB)")
            axis.set_ylabel(label)
            axis.legend()
        charts += _finish(fig, out / "quality_vs_snr")

    hybrid_variants = (("position_error_only", "Decoded XYZ + source attributes"),
                       ("attribute_error_only", "Source XYZ + decoded attributes"),
                       ("received", "Fully decoded"))
    if any(np.isfinite(row.get("position_error_only_psnr_mean", np.nan)) for row in summary):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
        for column, (variant, title) in enumerate(hybrid_variants):
            for row_index, (metric, label) in enumerate((("psnr", "PSNR (dB)"),
                                                          ("ssim", "SSIM"))):
                axis = axes[row_index, column]
                for index, series in enumerate(series_names):
                    rows = series_rows[series]
                    snrs = np.asarray([row["snr_db"] for row in rows])
                    means = np.asarray([row[f"{variant}_{metric}_mean"] for row in rows])
                    stds = np.asarray([row[f"{variant}_{metric}_std"] for row in rows])
                    axis.errorbar(snrs, means, yerr=stds,
                                  color=palette[index % len(palette)],
                                  marker=markers[index % len(markers)], capsize=3,
                                  linewidth=1.8, label=series)
                rows = series_rows[series_names[0]]
                axis.plot([row["snr_db"] for row in rows],
                          [row[f"reference_{metric}_mean"] for row in rows],
                          color=REFERENCE, linestyle="--", marker="s", fillstyle="none",
                          label="Input PLY reference")
                axis.set_title(title)
                axis.set_xlabel("SNR (dB)")
                axis.set_ylabel(label)
                if row_index == 0 and column == 0:
                    axis.legend()
        charts += _finish(fig, out / "hybrid_ablation_quality_vs_snr")

    if any(np.isfinite(row.get("seed_position_error_only_psnr_mean", np.nan))
           for row in summary):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
        comparisons = (("position_nrmse_bbox_diagonal", "position_seed_nrmse_bbox_diagonal",
                        "Position NRMSE / bbox diagonal"),
                       ("position_error_only_psnr", "seed_position_error_only_psnr",
                        "Position-only render PSNR (dB)"),
                       ("position_error_only_ssim", "seed_position_error_only_ssim",
                        "Position-only render SSIM"))
        for axis, (final_key, seed_key, label) in zip(axes[0], comparisons):
            for index, series in enumerate(series_names):
                rows = series_rows[series]
                snrs = np.asarray([row["snr_db"] for row in rows])
                color = palette[index % len(palette)]
                axis.errorbar(snrs, [row[seed_key + "_mean"] for row in rows],
                              yerr=[row[seed_key + "_std"] for row in rows],
                              color=color, linestyle="--", marker="x", capsize=3,
                              label=f"{series} seed")
                axis.errorbar(snrs, [row[final_key + "_mean"] for row in rows],
                              yerr=[row[final_key + "_std"] for row in rows],
                              color=color, linestyle="-", marker="o", capsize=3,
                              label=f"{series} final")
            axis.set_title(label)
            axis.set_xlabel("SNR (dB)")
            axis.set_ylabel(label)
        axes[0, 0].legend(ncol=2, fontsize=8)
        charts += _finish(fig, out / "position_seed_vs_final")

    direct = (("received_vs_reference_psnr", "Codec render PSNR (dB)"),
              ("received_vs_reference_ssim", "Codec render SSIM"),
              ("received_vs_reference_l1", "Codec render L1 (lower is better)"),
              ("received_vs_reference_lpips", "Codec render LPIPS (lower is better)"))
    direct = [(key, label) for key, label in direct
              if any(np.isfinite(row.get(key + "_mean", np.nan)) for row in summary)]
    if direct:
        columns = min(2, len(direct))
        rows_count = int(np.ceil(len(direct) / columns))
        fig, axes = plt.subplots(rows_count, columns, figsize=(6 * columns, 4 * rows_count),
                                 squeeze=False)
        for axis, (key, label) in zip(axes.flat, direct):
            for index, series in enumerate(series_names):
                rows = series_rows[series]
                snrs = np.asarray([row["snr_db"] for row in rows])
                means = np.asarray([row[key + "_mean"] for row in rows])
                stds = np.asarray([row[key + "_std"] for row in rows])
                axis.errorbar(snrs, means, yerr=stds, color=palette[index % len(palette)],
                              marker=markers[index % len(markers)], capsize=3, label=series)
            axis.set_title(label.split(" (")[0])
            axis.set_xlabel("SNR (dB)")
            axis.set_ylabel(label)
            axis.legend()
        for axis in axes.flat[len(direct):]:
            axis.set_visible(False)
        charts += _finish(fig, out / "codec_render_fidelity_vs_snr")

    if any(np.isfinite(row["total_uses_per_source_gaussian_mean"]) for row in summary):
        fig, axis = plt.subplots(figsize=(8.5, 4.8))
        for index, series in enumerate(series_names):
            rows = series_rows[series]
            snrs = np.asarray([row["snr_db"] for row in rows])
            total = np.asarray([row["total_uses_per_source_gaussian_mean"] for row in rows])
            axis.plot(snrs, total, color=palette[index % len(palette)],
                      marker=markers[index % len(markers)], linewidth=2,
                      label="Total" if len(series_names) == 1 else series)
        if len(series_names) == 1:
            raw_groups = [grouped_item[1] for grouped_item in grouped]
            source_counts = np.asarray([float(group[0].get("source_gaussians", 1)) for group in raw_groups])
            payload = np.asarray([row["payload_complex_symbols_mean"] for row in summary]) / source_counts
            metadata = np.asarray([row["metadata_channel_uses_mean"] for row in summary]) / source_counts
            axis.plot(snrs, payload, color=TIER_COLORS[1], marker="s", fillstyle="none",
                      linestyle="--", label="JSCC payload")
            axis.plot(snrs, metadata, color=TIER_COLORS[2], marker="^", fillstyle="none",
                      linestyle=":", label="Reliable metadata")
        axis.set_title("Channel use by SNR")
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel("Complex channel uses / source Gaussian")
        axis.legend(ncol=3)
        charts += _finish(fig, out / "channel_uses_vs_snr")

    share_fields = [f"tier_{name.lower()}_share" for name in TIER_NAMES]
    if all(field in summary[0] for field in share_fields):
        fig, axes = plt.subplots(len(series_names), 1, figsize=(8.5, 4 * len(series_names)),
                                 squeeze=False)
        for series_index, series in enumerate(series_names):
            axis = axes[series_index, 0]
            rows = series_rows[series]
            snrs = np.asarray([row["snr_db"] for row in rows])
            shares = np.asarray([[row[field] for field in share_fields] for row in rows])
            bottom = np.zeros(len(snrs))
            width = .7 * (np.diff(snrs).min() if len(snrs) > 1 else 1.)
            for index, (name, color) in enumerate(zip(TIER_NAMES, TIER_COLORS)):
                axis.bar(snrs, shares[:, index], bottom=bottom, width=width, label=name,
                         color=color, edgecolor="white", linewidth=.6)
                bottom += shares[:, index]
            axis.set_ylim(0, 1)
            axis.set_title("Deployment tier composition by SNR" +
                           (f" — {series}" if len(series_names) > 1 else ""))
            axis.set_xlabel("SNR (dB)")
            axis.set_ylabel("Share of source Gaussians")
            axis.legend(ncol=4)
        charts += _finish(fig, out / "tier_mix_vs_snr")

    psnr_rows = [row for row in data if row.get("received_psnr") is not None and
                 row.get("total_uses_per_source_gaussian") is not None]
    if psnr_rows:
        fig, axis = plt.subplots(figsize=(8, 5))
        if len(series_names) == 1:
            unique_snrs = sorted({float(row["snr_db"]) for row in psnr_rows})
            shades = plt.cm.Blues(np.linspace(.45, .9, max(2, len(unique_snrs))))
            for color, snr in zip(shades, unique_snrs):
                group = [row for row in psnr_rows if float(row["snr_db"]) == snr]
                axis.scatter([row["total_uses_per_source_gaussian"] for row in group],
                             [row["received_psnr"] for row in group], color=color,
                             edgecolor=INK, linewidth=.35, s=45, label=f"{snr:g} dB")
        else:
            for index, series in enumerate(series_names):
                group = [row for row in psnr_rows if _evaluation_series(row) == series]
                axis.scatter([row["total_uses_per_source_gaussian"] for row in group],
                             [row["received_psnr"] for row in group],
                             color=palette[index % len(palette)], marker=markers[index % len(markers)],
                             edgecolor=INK, linewidth=.35, s=45, label=series)
        axis.set_title("Rate-distortion observations")
        axis.set_xlabel("Total complex channel uses / source Gaussian")
        axis.set_ylabel("Received PSNR (dB)")
        axis.legend(title="SNR" if len(series_names) == 1 else "Allocation",
                    ncol=min(5, len(series_names) if len(series_names) > 1 else len(unique_snrs)))
        charts += _finish(fig, out / "rate_distortion")

    error_keys = [("position_rmse", "Position RMSE"), ("attribute_mse", "Attribute MSE")]
    error_keys = [(key, label) for key, label in error_keys
                  if any(np.isfinite(row.get(key + "_mean", np.nan)) for row in summary)]
    if error_keys:
        fig, axes = plt.subplots(1, len(error_keys), figsize=(5 * len(error_keys), 4), squeeze=False)
        for axis, (key, label) in zip(axes[0], error_keys):
            for index, series in enumerate(series_names):
                rows = series_rows[series]
                snrs = np.asarray([row["snr_db"] for row in rows])
                means = np.asarray([row[key + "_mean"] for row in rows])
                stds = np.asarray([row[key + "_std"] for row in rows])
                axis.errorbar(snrs, means, yerr=stds, color=palette[index % len(palette)],
                              marker=markers[index % len(markers)], capsize=3,
                              label=series if len(series_names) > 1 else None)
            axis.set_title(label)
            axis.set_xlabel("SNR (dB)")
            axis.set_ylabel(label)
            if len(series_names) > 1:
                axis.legend()
        charts += _finish(fig, out / "gaussian_errors_vs_snr")

    parameter_keys = (("position_nrmse_bbox_diagonal", "Position NRMSE / bbox diagonal"),
                      ("opacity_alpha_mae", "Opacity alpha MAE"),
                      ("log_scale_rmse", "Log-scale RMSE"),
                      ("rotation_angle_mean_deg", "Mean rotation error (degrees)"),
                      ("dc_rmse", "DC coefficient RMSE"),
                      ("sh_rest_rmse", "Higher-order SH RMSE"),
                      ("sorted_log_scale_rmse", "Sorted log-scale RMSE"),
                      ("log_covariance_rmse", "Log-covariance RMSE"),
                      ("log_volume_bias", "Mean log-volume ratio (negative = shrink)"))
    parameter_keys = [(key, label) for key, label in parameter_keys
                      if any(np.isfinite(row.get(key + "_mean", np.nan)) for row in summary)]
    if parameter_keys:
        columns = min(3, len(parameter_keys))
        rows_count = int(np.ceil(len(parameter_keys) / columns))
        fig, axes = plt.subplots(rows_count, columns, figsize=(5 * columns, 3.8 * rows_count),
                                 squeeze=False)
        for axis, (key, label) in zip(axes.flat, parameter_keys):
            for index, series in enumerate(series_names):
                rows = series_rows[series]
                snrs = np.asarray([row["snr_db"] for row in rows])
                means = np.asarray([row[key + "_mean"] for row in rows])
                stds = np.asarray([row[key + "_std"] for row in rows])
                axis.errorbar(snrs, means, yerr=stds, color=palette[index % len(palette)],
                              marker=markers[index % len(markers)], capsize=3, label=series)
            axis.set_title(label)
            axis.set_xlabel("SNR (dB)")
            axis.set_ylabel(label)
        for axis in axes.flat[len(parameter_keys):]:
            axis.set_visible(False)
        axes.flat[0].legend(ncol=min(4, len(series_names)))
        charts += _finish(fig, out / "codec_parameter_errors_vs_snr")

    return _manifest(out, "evaluation", evaluation_dir / "results.json", charts,
                     ["Lines show trial means; error bars show sample standard deviation.",
                      "Rate includes measured JSCC payload and the configured reliable-metadata accounting model.",
                      "received_vs_reference metrics isolate communication reconstruction from source-scene error.",
                      "Hybrid ablation, when present, mixes only row-aligned XYZ versus non-position attributes."])


def plot_allocation(allocation_dir, output_dir=None, xyz=None, rates=(0, 8, 16, 32), snr=None):
    """Plot hard tier composition, probability distributions and spatial projections."""
    allocation_dir = Path(allocation_dir)
    out = Path(output_dir) if output_dir else allocation_dir / "charts"
    out.mkdir(parents=True, exist_ok=True)
    probabilities = np.load(allocation_dir / "probabilities.npy", allow_pickle=False)
    tiers = np.load(allocation_dir / "tiers.npy", allow_pickle=False)
    if probabilities.ndim != 2 or probabilities.shape[1] != 4 or tiers.shape != (len(probabilities),):
        raise ValueError("allocation arrays must have shapes [N,4] and [N]")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("allocation probabilities must be finite values in [0,1]")
    if not np.allclose(probabilities.sum(1), 1., atol=2e-4) or ((tiers < 0) | (tiers > 3)).any():
        raise ValueError("invalid categorical allocation")
    rates = np.asarray(rates, dtype=float)
    if rates.shape != (4,):
        raise ValueError("rates must contain four values")
    counts = np.bincount(tiers.astype(np.int64), minlength=4)
    shares = counts / len(tiers)
    expected = probabilities @ rates
    existence = 1 - probabilities[:, 0]
    charts = []
    plt = _plt()

    fig, axis = plt.subplots(figsize=(8, 4.8))
    bars = axis.bar(TIER_NAMES, shares, color=TIER_COLORS, edgecolor="white", linewidth=.8)
    axis.set_ylim(0, max(1., float(shares.max()) * 1.18))
    axis.set_title("Hard tier allocation" + (f" at {snr:g} dB" if snr is not None else ""))
    axis.set_ylabel("Share of source Gaussians")
    for bar, count, share in zip(bars, counts, shares):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"{share:.1%}\n{count:,}", ha="center", va="bottom", color=INK)
    charts += _finish(fig, out / "allocation_tier_composition")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(existence, bins=50, color=TIER_COLORS[1], edgecolor="white", linewidth=.25)
    axes[0].set_title("Existence probability")
    axes[0].set_xlabel("1 - P(drop)")
    axes[0].set_ylabel("Gaussian count")
    axes[1].hist(expected, bins=50, color=TIER_COLORS[2], edgecolor="white", linewidth=.25)
    axes[1].set_title("Expected payload allocation")
    axes[1].set_xlabel("Complex symbols / Gaussian")
    axes[1].set_ylabel("Gaussian count")
    charts += _finish(fig, out / "allocation_probability_distributions")

    if xyz is not None:
        xyz = np.asarray(xyz)
        if xyz.shape != (len(tiers), 3) or not np.isfinite(xyz).all():
            raise ValueError("xyz must be a finite [N,3] array matching the allocation")
        selected = []
        for tier in range(4):
            indices = np.flatnonzero(tiers == tier)
            if len(indices) > 25000:
                indices = indices[np.linspace(0, len(indices) - 1, 25000).astype(int)]
            selected.append(indices)
        selected = np.concatenate(selected) if selected else np.empty(0, dtype=int)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for axis, (a, b, labels) in zip(axes, ((0, 1, ("X", "Y")), (0, 2, ("X", "Z")),
                                                     (1, 2, ("Y", "Z")))):
            for tier, (name, color) in enumerate(zip(TIER_NAMES, TIER_COLORS)):
                idx = selected[tiers[selected] == tier]
                if len(idx):
                    axis.scatter(xyz[idx, a], xyz[idx, b], s=1.2, alpha=.4, color=color,
                                 linewidths=0, label=name)
            axis.set_xlabel(labels[0])
            axis.set_ylabel(labels[1])
            axis.set_aspect("equal", adjustable="datalim")
        axes[0].legend(markerscale=5, ncol=2)
        fig.suptitle("Spatial distribution of hard tiers (up to 25k points per tier)", color=INK)
        charts += _finish(fig, out / "allocation_spatial_projections")

    summary = [{"tier": name, "symbol_length": float(rates[index]), "count": int(counts[index]),
                "share": float(shares[index]), "snr_db": snr}
               for index, name in enumerate(TIER_NAMES)]
    _write_csv(out / "allocation_chart_data.csv", summary, list(summary[0]))
    return _manifest(out, "allocation", allocation_dir, charts,
                     ["Existence probability is defined as 1 - P(drop).",
                      "Spatial plots use deterministic per-tier subsampling and do not represent density proportions."])


def safe_plot(kind, source, **kwargs):
    """Best-effort automatic plotting; never invalidate a completed experiment."""
    try:
        result = {"training": plot_training, "evaluation": plot_evaluation,
                  "allocation": plot_allocation}[kind](source, **kwargs)
        print(f"Saved {kind} charts to {Path(kwargs.get('output_dir') or source) / ('charts' if not kwargs.get('output_dir') else '')}")
        return result
    except Exception as exc:  # plotting is a post-processing convenience
        print(f"WARNING: automatic {kind} charts were not generated: {exc}")
        return None


def plot_command(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    made = 0
    if args.training:
        plot_training(args.training, out / "training")
        made += 1
    if args.evaluation:
        plot_evaluation(args.evaluation, out / "evaluation")
        made += 1
    if args.allocation:
        xyz = rates = snr = None
        info_path = Path(args.allocation) / "allocation.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            snr = info.get("snr_db")
            rates = info.get("rates")
        if args.ply:
            from .data import read_ply
            raw, _ = read_ply(args.ply)
            xyz = raw[:, :3].numpy()
        if args.checkpoint:
            import torch
            from .transport import load_checkpoint
            rates = load_checkpoint(args.checkpoint, torch.device("cpu")).cfg.rates
        plot_allocation(args.allocation, out / "allocation", xyz=xyz,
                        rates=rates or (0, 8, 16, 32), snr=snr)
        made += 1
    if not made:
        raise ValueError("provide at least one of --training, --evaluation or --allocation")
    print(f"Saved statistical charts to {out}")


def add_parser(sub):
    parser = sub.add_parser("plot-stats", help="render route2 statistical charts from saved outputs")
    parser.set_defaults(func=plot_command)
    parser.add_argument("--training", help="directory containing loss.jsonl")
    parser.add_argument("--evaluation", help="directory containing results.json")
    parser.add_argument("--allocation", help="directory containing probabilities.npy and tiers.npy")
    parser.add_argument("--ply", help="matching PLY for allocation spatial projections")
    parser.add_argument("--checkpoint", help="matching codec.pt for the allocation rate table")
    parser.add_argument("--out", required=True, help="new chart output directory")
