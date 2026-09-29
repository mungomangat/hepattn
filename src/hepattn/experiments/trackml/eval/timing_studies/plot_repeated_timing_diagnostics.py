import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _resolve_model_name(timing_dir: Path, model_name: str | None) -> str:
    if model_name is not None:
        return model_name

    times_paths = sorted(timing_dir.glob("*_times.npy"))
    if len(times_paths) != 1:
        raise ValueError(
            f"Expected exactly one '*_times.npy' file in {timing_dir}, found {len(times_paths)}. "
            "Pass --model-name to disambiguate."
        )
    return times_paths[0].name.removesuffix("_times.npy")


def _save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=200)
    fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def _style_axis(axis) -> None:
    axis.set_axisbelow(True)
    axis.grid(zorder=0, alpha=0.22, linestyle="--", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _load_optional_array(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.asarray(np.load(path))


def _timing_summary(repeat_times_ms: np.ndarray, aggregated_times_ms: np.ndarray) -> dict:
    medians = np.nanmedian(repeat_times_ms, axis=1)
    means = np.nanmean(repeat_times_ms, axis=1)
    stds = np.nanstd(repeat_times_ms, axis=1)
    p16 = np.nanpercentile(repeat_times_ms, 16, axis=1)
    p84 = np.nanpercentile(repeat_times_ms, 84, axis=1)
    spreads = p84 - p16

    return {
        "num_events": int(repeat_times_ms.shape[0]),
        "num_repeats": int(repeat_times_ms.shape[1]),
        "aggregated_times_match_event_medians": bool(np.allclose(aggregated_times_ms, medians)),
        "median_time_ms_mean": float(medians.mean()),
        "median_time_ms_median": float(np.nanmedian(medians)),
        "median_time_ms_min": float(medians.min()),
        "median_time_ms_max": float(medians.max()),
        "mean_time_ms_mean": float(means.mean()),
        "std_within_event_ms_mean": float(stds.mean()),
        "std_within_event_ms_median": float(np.nanmedian(stds)),
        "spread_16_84_ms_mean": float(spreads.mean()),
        "spread_16_84_ms_median": float(np.nanmedian(spreads)),
        "spread_16_84_ms_max": float(spreads.max()),
        "top5_by_spread": [
            {
                "event_index": int(idx),
                "median_ms": float(medians[idx]),
                "mean_ms": float(means[idx]),
                "std_ms": float(stds[idx]),
                "spread_16_84_ms": float(spreads[idx]),
                "min_ms": float(repeat_times_ms[idx].min()),
                "max_ms": float(repeat_times_ms[idx].max()),
            }
            for idx in np.argsort(spreads)[-5:][::-1]
        ],
    }


def _plot_eventwise_median(
    output_dir: Path,
    model_name: str,
    repeat_times_ms: np.ndarray,
) -> None:
    event_index = np.arange(repeat_times_ms.shape[0])
    medians = np.nanmedian(repeat_times_ms, axis=1)
    p16 = np.nanpercentile(repeat_times_ms, 16, axis=1)
    p84 = np.nanpercentile(repeat_times_ms, 84, axis=1)
    lower = medians - p16
    upper = p84 - medians

    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.errorbar(
        event_index,
        medians,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        markersize=3.4,
        elinewidth=0.8,
        capsize=2.0,
        color="#0072B2",
        ecolor="#88bde6",
        markerfacecolor="#0072B2",
        markeredgecolor="white",
        markeredgewidth=0.4,
        alpha=0.9,
    )
    ax.set_xlabel("Event index")
    ax.set_ylabel("Inference time [ms]")
    ax.set_title("Per-event median inference time with 16th-84th percentile bars")
    _style_axis(ax)
    _save_figure(fig, output_dir / f"{model_name}_median_errorbars")


def _plot_eventwise_mean_std(
    output_dir: Path,
    model_name: str,
    repeat_times_ms: np.ndarray,
) -> None:
    event_index = np.arange(repeat_times_ms.shape[0])
    means = np.nanmean(repeat_times_ms, axis=1)
    stds = np.nanstd(repeat_times_ms, axis=1)

    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.errorbar(
        event_index,
        means,
        yerr=stds,
        fmt="o",
        markersize=3.2,
        elinewidth=0.8,
        capsize=2.0,
        color="#009E73",
        ecolor="#7fcdbb",
        markerfacecolor="#009E73",
        markeredgecolor="white",
        markeredgewidth=0.4,
        alpha=0.9,
    )
    ax.set_xlabel("Event index")
    ax.set_ylabel("Inference time [ms]")
    ax.set_title("Per-event mean inference time with ±1 standard deviation")
    _style_axis(ax)
    _save_figure(fig, output_dir / f"{model_name}_mean_std_errorbars")


def _plot_time_vs_hits_std(
    output_dir: Path,
    model_name: str,
    repeat_times_ms: np.ndarray,
    hit_counts: np.ndarray | None,
) -> bool:
    if hit_counts is None:
        return False

    hit_counts = np.asarray(hit_counts, dtype=np.int64)
    if hit_counts.shape != (repeat_times_ms.shape[0],):
        raise ValueError(
            f"Expected hit counts shape {(repeat_times_ms.shape[0],)}, found {hit_counts.shape}."
        )

    means = repeat_times_ms.mean(axis=1)
    stds = repeat_times_ms.std(axis=1)
    order = np.argsort(hit_counts)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.errorbar(
        hit_counts[order],
        means[order],
        yerr=stds[order],
        fmt="o",
        markersize=3.4,
        elinewidth=0.8,
        capsize=2.0,
        color="#D55E00",
        ecolor="#fdbb84",
        markerfacecolor="#D55E00",
        markeredgecolor="white",
        markeredgewidth=0.4,
        alpha=0.9,
    )
    if len(np.unique(hit_counts)) >= 2:
        slope, intercept = np.polyfit(hit_counts, means, deg=1)
        fit_x = np.linspace(hit_counts.min(), hit_counts.max(), 200)
        fit_y = slope * fit_x + intercept
        ax.plot(fit_x, fit_y, color="#7f2704", linewidth=1.6, label="Linear fit")
        ax.legend(frameon=False)
    ax.set_xlabel("Number of hits per event")
    ax.set_ylabel("Inference time [ms]")
    ax.set_title("Inference time vs. event hit count with ±1 standard deviation")
    _style_axis(ax)
    _save_figure(fig, output_dir / f"{model_name}_time_vs_hits_std_errorbars")
    return True


def _plot_time_vs_hits_binned(
    output_dir: Path,
    model_name: str,
    repeat_times_ms: np.ndarray,
    hit_counts: np.ndarray | None,
    num_bins: int = 20,
) -> bool:
    if hit_counts is None:
        return False

    hit_counts = np.asarray(hit_counts, dtype=np.int64)
    if hit_counts.shape != (repeat_times_ms.shape[0],):
        raise ValueError(
            f"Expected hit counts shape {(repeat_times_ms.shape[0],)}, found {hit_counts.shape}."
        )

    event_means = repeat_times_ms.mean(axis=1)
    order = np.argsort(hit_counts)
    hit_sorted = hit_counts[order]
    time_sorted = event_means[order]

    num_bins = max(1, min(int(num_bins), len(hit_sorted)))
    edges = np.linspace(0, len(hit_sorted), num_bins + 1, dtype=int)

    bin_hit_means = []
    bin_time_means = []
    bin_time_stds = []
    bin_counts = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        if right <= left:
            continue
        hits_bin = hit_sorted[left:right]
        time_bin = time_sorted[left:right]
        bin_hit_means.append(float(hits_bin.mean()))
        bin_time_means.append(float(time_bin.mean()))
        bin_time_stds.append(float(time_bin.std()))
        bin_counts.append(int(len(time_bin)))

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.errorbar(
        bin_hit_means,
        bin_time_means,
        yerr=bin_time_stds,
        fmt="o",
        markersize=4.0,
        elinewidth=0.8,
        capsize=2.0,
        color="#E69F00",
        ecolor="#fddc8c",
        markerfacecolor="#E69F00",
        markeredgecolor="white",
        markeredgewidth=0.4,
        alpha=0.95,
    )
    if len(np.unique(bin_hit_means)) >= 2:
        slope, intercept = np.polyfit(bin_hit_means, bin_time_means, deg=1)
        fit_x = np.linspace(min(bin_hit_means), max(bin_hit_means), 200)
        fit_y = slope * fit_x + intercept
        ax.plot(fit_x, fit_y, color="#a6611a", linewidth=1.6, label="Linear fit")
        ax.legend(frameon=False)
    ax.set_xlabel("Number of hits per event")
    ax.set_ylabel("Inference time [ms]")
    ax.set_title(f"Inference time vs. event hit count in {len(bin_hit_means)} bins")
    _style_axis(ax)
    _save_figure(fig, output_dir / f"{model_name}_time_vs_hits_binned_20")
    return True


def _plot_time_vs_queries_std(
    output_dir: Path,
    model_name: str,
    repeat_times_ms: np.ndarray,
    query_counts: np.ndarray | None,
) -> bool:
    if query_counts is None:
        return False

    query_counts = np.asarray(query_counts, dtype=np.int64)
    if query_counts.shape != (repeat_times_ms.shape[0],):
        raise ValueError(
            f"Expected query counts shape {(repeat_times_ms.shape[0],)}, found {query_counts.shape}."
        )

    means = repeat_times_ms.mean(axis=1)
    stds = repeat_times_ms.std(axis=1)
    order = np.argsort(query_counts)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.errorbar(
        query_counts[order],
        means[order],
        yerr=stds[order],
        fmt="o",
        markersize=3.4,
        elinewidth=0.8,
        capsize=2.0,
        color="#6A3D9A",
        ecolor="#cab2d6",
        markerfacecolor="#6A3D9A",
        markeredgecolor="white",
        markeredgewidth=0.4,
        alpha=0.9,
    )
    if len(np.unique(query_counts)) >= 2:
        slope, intercept = np.polyfit(query_counts, means, deg=1)
        fit_x = np.linspace(query_counts.min(), query_counts.max(), 200)
        fit_y = slope * fit_x + intercept
        ax.plot(fit_x, fit_y, color="#4a1486", linewidth=1.6, label="Linear fit")
        ax.legend(frameon=False)
    ax.set_xlabel("Number of queries per event")
    ax.set_ylabel("Inference time [ms]")
    ax.set_title("Inference time vs. query count with ±1 standard deviation")
    _style_axis(ax)
    _save_figure(fig, output_dir / f"{model_name}_time_vs_queries_std_errorbars")
    return True


def _plot_time_vs_queries_binned(
    output_dir: Path,
    model_name: str,
    repeat_times_ms: np.ndarray,
    query_counts: np.ndarray | None,
    num_bins: int = 20,
) -> bool:
    if query_counts is None:
        return False

    query_counts = np.asarray(query_counts, dtype=np.int64)
    if query_counts.shape != (repeat_times_ms.shape[0],):
        raise ValueError(
            f"Expected query counts shape {(repeat_times_ms.shape[0],)}, found {query_counts.shape}."
        )

    event_means = repeat_times_ms.mean(axis=1)
    order = np.argsort(query_counts)
    query_sorted = query_counts[order]
    time_sorted = event_means[order]

    num_bins = max(1, min(int(num_bins), len(query_sorted)))
    edges = np.linspace(0, len(query_sorted), num_bins + 1, dtype=int)

    bin_query_means = []
    bin_time_means = []
    bin_time_stds = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        if right <= left:
            continue
        queries_bin = query_sorted[left:right]
        time_bin = time_sorted[left:right]
        bin_query_means.append(float(queries_bin.mean()))
        bin_time_means.append(float(time_bin.mean()))
        bin_time_stds.append(float(time_bin.std()))

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.errorbar(
        bin_query_means,
        bin_time_means,
        yerr=bin_time_stds,
        fmt="o",
        markersize=4.0,
        elinewidth=0.8,
        capsize=2.0,
        color="#7B3294",
        ecolor="#c2a5cf",
        markerfacecolor="#7B3294",
        markeredgecolor="white",
        markeredgewidth=0.4,
        alpha=0.95,
    )
    if len(np.unique(bin_query_means)) >= 2:
        slope, intercept = np.polyfit(bin_query_means, bin_time_means, deg=1)
        fit_x = np.linspace(min(bin_query_means), max(bin_query_means), 200)
        fit_y = slope * fit_x + intercept
        ax.plot(fit_x, fit_y, color="#4d004b", linewidth=1.6, label="Linear fit")
        ax.legend(frameon=False)
    ax.set_xlabel("Number of queries per event")
    ax.set_ylabel("Inference time [ms]")
    ax.set_title(f"Inference time vs. query count in {len(bin_query_means)} bins")
    _style_axis(ax)
    _save_figure(fig, output_dir / f"{model_name}_time_vs_queries_binned_20")
    return True


def _plot_time_vs_hits_queries_2d(
    output_dir: Path,
    model_name: str,
    repeat_times_ms: np.ndarray,
    hit_counts: np.ndarray | None,
    query_counts: np.ndarray | None,
) -> bool:
    if hit_counts is None or query_counts is None:
        return False

    hit_counts = np.asarray(hit_counts, dtype=np.int64)
    query_counts = np.asarray(query_counts, dtype=np.int64)
    expected_shape = (repeat_times_ms.shape[0],)
    if hit_counts.shape != expected_shape:
        raise ValueError(f"Expected hit counts shape {expected_shape}, found {hit_counts.shape}.")
    if query_counts.shape != expected_shape:
        raise ValueError(f"Expected query counts shape {expected_shape}, found {query_counts.shape}.")

    means = repeat_times_ms.mean(axis=1)

    fig, ax = plt.subplots(figsize=(7.8, 5.8), constrained_layout=True)
    scatter = ax.scatter(
        hit_counts,
        query_counts,
        c=means,
        s=34.0,
        cmap="viridis",
        alpha=0.85,
        edgecolors="white",
        linewidths=0.35,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Mean inference time [ms]")
    ax.set_xlabel("Number of hits per event")
    ax.set_ylabel("Number of queries per event")
    ax.set_title("Event hit/query counts coloured by mean inference time")
    _style_axis(ax)
    _save_figure(fig, output_dir / f"{model_name}_time_vs_hits_queries_2d")
    return True


def _plot_repeat_trace_envelope(
    output_dir: Path,
    model_name: str,
    repeat_times_ms: np.ndarray,
) -> None:
    event_index = np.arange(repeat_times_ms.shape[0])
    medians = np.nanmedian(repeat_times_ms, axis=1)
    p16 = np.nanpercentile(repeat_times_ms, 16, axis=1)
    p84 = np.nanpercentile(repeat_times_ms, 84, axis=1)

    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    for repeat_idx in range(repeat_times_ms.shape[1]):
        label = "Individual repeats" if repeat_idx == 0 else None
        ax.plot(event_index, repeat_times_ms[:, repeat_idx], color="#bdbdbd", linewidth=0.8, alpha=0.35, label=label)

    ax.fill_between(event_index, p16, p84, color="#9ecae1", alpha=0.35, label="16th-84th percentile band")
    ax.plot(event_index, medians, color="#08519c", linewidth=1.8, label="Per-event median")
    ax.set_xlabel("Event index")
    ax.set_ylabel("Inference time [ms]")
    ax.set_title("Repeated timing traces by event")
    _style_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    _save_figure(fig, output_dir / f"{model_name}_repeat_trace_envelope")


def _plot_spread_histogram(
    output_dir: Path,
    model_name: str,
    repeat_times_ms: np.ndarray,
) -> None:
    spreads = np.nanpercentile(repeat_times_ms, 84, axis=1) - np.nanpercentile(repeat_times_ms, 16, axis=1)
    stds = np.nanstd(repeat_times_ms, axis=1)

    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    ax.hist(spreads, bins=20, alpha=0.7, color="#D55E00", label="16th-84th percentile spread")
    ax.hist(stds, bins=20, alpha=0.5, color="#0072B2", label="Within-event standard deviation")
    ax.set_xlabel("Spread [ms]")
    ax.set_ylabel("Number of events")
    ax.set_title("Distribution of repeat-to-repeat timing variability")
    _style_axis(ax)
    ax.legend(frameon=False)
    _save_figure(fig, output_dir / f"{model_name}_repeat_spread_histogram")


def _plot_time_vs_sweep_position(
    output_dir: Path,
    model_name: str,
    repeat_times_ms: np.ndarray,
    sweep_positions: np.ndarray | None,
) -> bool:
    if sweep_positions is None:
        return False
    if sweep_positions.shape != repeat_times_ms.shape:
        raise ValueError(
            f"Expected sweep-position shape {repeat_times_ms.shape}, found {sweep_positions.shape}."
        )

    valid = (sweep_positions >= 0) & np.isfinite(repeat_times_ms)
    if not np.any(valid):
        return False
    positions = sweep_positions[valid]
    times = repeat_times_ms[valid]
    unique_positions = np.unique(positions)
    position_means = np.array([times[positions == position].mean() for position in unique_positions])
    position_stds = np.array([times[positions == position].std() for position in unique_positions])

    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.scatter(positions, times, s=10, color="#9ecae1", alpha=0.25, edgecolors="none", label="Individual timings")
    ax.errorbar(
        unique_positions,
        position_means,
        yerr=position_stds,
        fmt="o-",
        color="#08519c",
        elinewidth=0.8,
        capsize=2.0,
        markersize=3.4,
        label="Mean $\\pm$ 1 std",
    )
    ax.set_xlabel("Event position within timed sweep")
    ax.set_ylabel("Inference time [ms]")
    ax.set_title("Inference time versus event position within sweep")
    _style_axis(ax)
    ax.legend(frameon=False)
    _save_figure(fig, output_dir / f"{model_name}_time_vs_sweep_position")
    return True


def _plot_memory_envelope(
    output_dir: Path,
    model_name: str,
    repeat_peak_allocated_bytes: np.ndarray | None,
    repeat_peak_reserved_bytes: np.ndarray | None,
) -> None:
    if repeat_peak_allocated_bytes is None and repeat_peak_reserved_bytes is None:
        return

    event_index = None
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)

    if repeat_peak_allocated_bytes is not None:
        allocated_mb = repeat_peak_allocated_bytes.astype(np.float64) / (1024**2)
        event_index = np.arange(allocated_mb.shape[0])
        alloc_med = np.median(allocated_mb, axis=1)
        alloc_p16 = np.percentile(allocated_mb, 16, axis=1)
        alloc_p84 = np.percentile(allocated_mb, 84, axis=1)
        ax.fill_between(event_index, alloc_p16, alloc_p84, color="#9ecae1", alpha=0.35)
        ax.plot(event_index, alloc_med, color="#0072B2", linewidth=1.8, label="Allocated median")

    if repeat_peak_reserved_bytes is not None:
        reserved_mb = repeat_peak_reserved_bytes.astype(np.float64) / (1024**2)
        event_index = np.arange(reserved_mb.shape[0])
        reserved_med = np.median(reserved_mb, axis=1)
        reserved_p16 = np.percentile(reserved_mb, 16, axis=1)
        reserved_p84 = np.percentile(reserved_mb, 84, axis=1)
        ax.fill_between(event_index, reserved_p16, reserved_p84, color="#fdbb84", alpha=0.25)
        ax.plot(event_index, reserved_med, color="#D55E00", linewidth=1.8, label="Reserved median")

    ax.set_xlabel("Event index")
    ax.set_ylabel("Peak GPU memory [MB]")
    ax.set_title("Per-event repeated peak GPU memory")
    _style_axis(ax)
    ax.legend(frameon=False)
    _save_figure(fig, output_dir / f"{model_name}_memory_errorbars")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot diagnostics for repeated inference timings.")
    parser.add_argument("timing_dir", type=Path, help="Directory containing timing .npy files.")
    parser.add_argument("--model-name", type=str, default=None, help="Base model name used in file stems.")
    args = parser.parse_args()

    timing_dir = args.timing_dir.resolve()
    model_name = _resolve_model_name(timing_dir, args.model_name)

    aggregated_times_ms = np.asarray(np.load(timing_dir / f"{model_name}_times.npy"), dtype=np.float64)
    repeat_times_ms = np.asarray(np.load(timing_dir / f"{model_name}_repeat_times_ms.npy"), dtype=np.float64)
    if repeat_times_ms.ndim != 2:
        raise ValueError(
            f"Expected repeated timings to be a 2D array in {timing_dir / f'{model_name}_repeat_times_ms.npy'}, "
            f"found shape {repeat_times_ms.shape}."
        )
    if aggregated_times_ms.shape != (repeat_times_ms.shape[0],):
        raise ValueError(
            f"Expected aggregated timings shape {(repeat_times_ms.shape[0],)}, found {aggregated_times_ms.shape}."
        )

    repeat_peak_allocated_bytes = _load_optional_array(timing_dir / f"{model_name}_repeat_peak_allocated_bytes.npy")
    repeat_peak_reserved_bytes = _load_optional_array(timing_dir / f"{model_name}_repeat_peak_reserved_bytes.npy")
    hit_counts = _load_optional_array(timing_dir / f"{model_name}_dims.npy")
    query_counts = _load_optional_array(timing_dir / f"{model_name}_query_counts.npy")
    sweep_positions = _load_optional_array(timing_dir / f"{model_name}_sweep_positions.npy")
    metadata_path = timing_dir / f"{model_name}_timing_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)

    _apply_style()
    _plot_eventwise_median(timing_dir, model_name, repeat_times_ms)
    _plot_eventwise_mean_std(timing_dir, model_name, repeat_times_ms)
    _plot_repeat_trace_envelope(timing_dir, model_name, repeat_times_ms)
    _plot_spread_histogram(timing_dir, model_name, repeat_times_ms)
    _plot_memory_envelope(timing_dir, model_name, repeat_peak_allocated_bytes, repeat_peak_reserved_bytes)
    has_hits_plot = _plot_time_vs_hits_std(timing_dir, model_name, repeat_times_ms, hit_counts)
    has_hits_binned_plot = _plot_time_vs_hits_binned(timing_dir, model_name, repeat_times_ms, hit_counts)
    has_queries_plot = _plot_time_vs_queries_std(timing_dir, model_name, repeat_times_ms, query_counts)
    has_queries_binned_plot = _plot_time_vs_queries_binned(timing_dir, model_name, repeat_times_ms, query_counts)
    has_hits_queries_2d_plot = _plot_time_vs_hits_queries_2d(
        timing_dir, model_name, repeat_times_ms, hit_counts, query_counts
    )
    has_sweep_position_plot = _plot_time_vs_sweep_position(
        timing_dir, model_name, repeat_times_ms, sweep_positions
    )

    notes = [
        "The error bars and bands use the 16th and 84th percentiles across repeated timings for each event."
    ]
    if hit_counts is None:
        notes.append("These plots use event index on the x-axis because no '*_dims.npy' file was present.")
    else:
        notes.append("A '*_dims.npy' sidecar was present, so hit-count plots were generated.")
    if query_counts is None:
        notes.append("No '*_query_counts.npy' sidecar was present, so query-count plots were skipped.")
    else:
        notes.append("A '*_query_counts.npy' sidecar was present, so query-count plots were generated.")
    if has_sweep_position_plot:
        notes.append("A '*_sweep_positions.npy' sidecar was present, so the sweep-position diagnostic was generated.")

    summary = {
        "metadata": metadata,
        "summary": _timing_summary(repeat_times_ms=repeat_times_ms, aggregated_times_ms=aggregated_times_ms),
        "notes": notes,
        "has_time_vs_hits_plot": has_hits_plot,
        "has_time_vs_hits_binned_plot": has_hits_binned_plot,
        "has_time_vs_queries_plot": has_queries_plot,
        "has_time_vs_queries_binned_plot": has_queries_binned_plot,
        "has_time_vs_hits_queries_2d_plot": has_hits_queries_2d_plot,
        "has_time_vs_sweep_position_plot": has_sweep_position_plot,
    }
    summary_path = timing_dir / f"{model_name}_repeat_diagnostics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Saved repeated-timing diagnostics to {timing_dir}")


if __name__ == "__main__":
    main()
