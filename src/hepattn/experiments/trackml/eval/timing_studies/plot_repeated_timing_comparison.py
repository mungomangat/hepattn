"""Make paper-style multi-model timing plots from repeated inference measurements."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


@dataclass(frozen=True)
class TimingSeries:
    label: str
    architecture: str
    model: str
    colour: str
    timing_dir: Path


SERIES = (
    TimingSeries("MA-900", "DQ+MA", "Pix1.0", "#D55E00", Path("MA-900")),
    TimingSeries("LCA-900", "DQ+LSCA", "Pix1.0", "#0072B2", Path("LCA-900")),
    TimingSeries("MA-600", "DQ+MA", "Pix0.6", "#D55E00", Path("MA-600")),
    TimingSeries("LCA-600", "DQ+LSCA", "Pix0.6", "#0072B2", Path("LCA-600")),
)


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def _style_axis(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(zorder=0, alpha=0.22, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))


def _load_series(
    timing_root: Path,
    series: TimingSeries,
    variable: str,
    error_mode: str = "central68",
    summary_stat: str = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timing_dir = timing_root / series.timing_dir
    repeat_paths = sorted(timing_dir.glob("*_repeat_times_ms.npy"))
    if len(repeat_paths) != 1:
        raise ValueError(
            f"Expected exactly one '*_repeat_times_ms.npy' file for {series.label} in {timing_dir}, "
            f"found {len(repeat_paths)}."
        )
    repeat_path = repeat_paths[0]
    stem = repeat_path.name.removesuffix("_repeat_times_ms.npy")
    repeated = np.asarray(np.load(repeat_path), dtype=np.float64)
    values = np.asarray(np.load(timing_dir / f"{stem}_{variable}.npy"), dtype=np.float64)
    if repeated.ndim != 2:
        raise ValueError(f"Expected a 2D repeat array for {series.label}, found {repeated.shape}.")
    if values.shape != (repeated.shape[0],):
        raise ValueError(f"Length mismatch for {series.label}: {values.shape} and {repeated.shape}.")

    if summary_stat == "mean":
        summary = np.nanmean(repeated, axis=1)
    elif summary_stat == "median":
        summary = np.nanmedian(repeated, axis=1)
    else:
        raise ValueError(f"Unknown summary statistic: {summary_stat}.")
    if error_mode == "std":
        lower = np.nanstd(repeated, axis=1)
        upper = lower
    elif error_mode == "central68":
        lower = summary - np.nanpercentile(repeated, 16, axis=1)
        upper = np.nanpercentile(repeated, 84, axis=1) - summary
    else:
        raise ValueError(f"Unknown error mode: {error_mode}.")
    return values, summary, lower, upper


def _add_legends(ax) -> None:
    architecture_handles = [
        Line2D([0], [0], color="#D55E00", linewidth=2.4, label="DQ+MA"),
        Line2D([0], [0], color="#0072B2", linewidth=2.4, label="DQ+LSCA"),
    ]
    legend = ax.legend(
        handles=architecture_handles,
        title="Architecture",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.87),
        borderaxespad=0.0,
    )
    ax.add_artist(legend)


def _plot_comparison(
    timing_root: Path,
    output_dir: Path,
    variable: str,
    xlabel: str,
    stem: str,
    error_mode: str = "central68",
    summary_stat: str = "median",
) -> None:
    loaded = [
        (series, *_load_series(timing_root, series, variable, error_mode, summary_stat)) for series in SERIES
    ]
    all_values = np.concatenate([values for _, values, *_ in loaded])
    pix1_max = max(values.max() for series, values, *_ in loaded if series.model == "Pix1.0")
    pix06_min = min(values.min() for series, values, *_ in loaded if series.model == "Pix0.6")
    split = (pix1_max + pix06_min) / 2.0
    pad = 0.06 * (all_values.max() - all_values.min())

    fig, ax = plt.subplots(figsize=(7.1, 4.5), constrained_layout=True)
    ax.axvspan(all_values.min() - pad, split, color="#fafafa", zorder=-2)
    ax.axvspan(split, all_values.max() + pad, color="#f3f3f3", zorder=-2)
    ax.axvline(split, color="#e5e5e5", linewidth=0.8, zorder=-1)

    for series, values, median, lower, upper in loaded:
        order = np.argsort(values)
        ax.errorbar(
            values[order],
            median[order],
            yerr=np.vstack([lower[order], upper[order]]),
            fmt="o",
            markersize=3.8,
            color=series.colour,
            ecolor=series.colour,
            elinewidth=0.65,
            capsize=1.5,
            alpha=0.65,
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=3,
        )
        slope, intercept = np.polyfit(values, median, deg=1)
        fit_x = np.linspace(values.min(), values.max(), 200)
        ax.plot(fit_x, slope * fit_x + intercept, color=series.colour, linewidth=2.3, zorder=4)

    ylim = ax.get_ylim()
    ax.text(
        (all_values.min() + split) / 2,
        ylim[1] - 0.035 * (ylim[1] - ylim[0]),
        "Model: Pix1.0",
        ha="center",
        va="top",
    )
    ax.text(
        (split + all_values.max()) / 2,
        ylim[1] - 0.035 * (ylim[1] - ylim[0]),
        "Model: Pix0.6",
        ha="center",
        va="top",
        bbox={"facecolor": "#dddddd", "edgecolor": "none", "pad": 2.5},
    )
    ax.set_xlim(all_values.min() - pad, all_values.max() + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Inference time [ms]")
    _style_axis(ax)
    _add_legends(ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def _bin_series(
    values: np.ndarray,
    event_times: np.ndarray,
    num_bins: int,
    error_mode: str = "central68",
    summary_stat: str = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(values.min(), values.max(), num_bins + 1)
    bin_index = np.clip(np.digitize(values, edges) - 1, 0, num_bins - 1)
    binned_values = []
    binned_times = []
    binned_lower = []
    binned_upper = []
    for index in range(num_bins):
        in_bin = event_times[bin_index == index]
        if not len(in_bin):
            continue
        if summary_stat == "mean":
            bin_time = float(np.mean(in_bin))
        elif summary_stat == "median":
            bin_time = float(np.median(in_bin))
        else:
            raise ValueError(f"Unknown summary statistic: {summary_stat}.")
        binned_values.append(float(np.mean(values[bin_index == index])))
        binned_times.append(bin_time)
        if error_mode == "std":
            uncertainty = float(np.std(in_bin))
            binned_lower.append(uncertainty)
            binned_upper.append(uncertainty)
        elif error_mode == "central68":
            binned_lower.append(bin_time - float(np.percentile(in_bin, 16)))
            binned_upper.append(float(np.percentile(in_bin, 84)) - bin_time)
        else:
            raise ValueError(f"Unknown error mode: {error_mode}.")
    return tuple(np.asarray(values) for values in (binned_values, binned_times, binned_lower, binned_upper))


def _plot_binned_comparison(
    timing_root: Path,
    output_dir: Path,
    variable: str,
    xlabel: str,
    stem: str,
    num_bins: int = 20,
    error_mode: str = "central68",
    summary_stat: str = "median",
) -> None:
    loaded = [(series, *_load_series(timing_root, series, variable, summary_stat=summary_stat)) for series in SERIES]
    all_values = np.concatenate([values for _, values, *_ in loaded])
    pix1_max = max(values.max() for series, values, *_ in loaded if series.model == "Pix1.0")
    pix06_min = min(values.min() for series, values, *_ in loaded if series.model == "Pix0.6")
    split = (pix1_max + pix06_min) / 2.0
    pad = 0.06 * (all_values.max() - all_values.min())

    fig, ax = plt.subplots(figsize=(7.1, 4.5), constrained_layout=True)
    ax.axvspan(all_values.min() - pad, split, color="#fafafa", zorder=-2)
    ax.axvspan(split, all_values.max() + pad, color="#f3f3f3", zorder=-2)
    ax.axvline(split, color="#e5e5e5", linewidth=0.8, zorder=-1)

    for series, values, event_times, _, _ in loaded:
        bin_x, bin_time, bin_lower, bin_upper = _bin_series(
            values, event_times, num_bins, error_mode, summary_stat
        )
        ax.errorbar(
            bin_x,
            bin_time,
            yerr=np.vstack([bin_lower, bin_upper]),
            fmt="o",
            markersize=4.2,
            color=series.colour,
            ecolor=series.colour,
            elinewidth=0.8,
            capsize=2.0,
            alpha=0.8,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=3,
        )
        slope, intercept = np.polyfit(bin_x, bin_time, deg=1)
        fit_x = np.linspace(bin_x.min(), bin_x.max(), 200)
        ax.plot(fit_x, slope * fit_x + intercept, color=series.colour, linewidth=2.3, zorder=4)

    ylim = ax.get_ylim()
    ax.text(
        (all_values.min() + split) / 2,
        ylim[1] - 0.035 * (ylim[1] - ylim[0]),
        "Model: Pix1.0",
        ha="center",
        va="top",
    )
    ax.text(
        (split + all_values.max()) / 2,
        ylim[1] - 0.035 * (ylim[1] - ylim[0]),
        "Model: Pix0.6",
        ha="center",
        va="top",
        bbox={"facecolor": "#dddddd", "edgecolor": "none", "pad": 2.5},
    )
    ax.set_xlim(all_values.min() - pad, all_values.max() + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Inference time [ms]")
    _style_axis(ax)
    _add_legends(ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_root", type=Path, help="Directory containing MA-600, LCA-600, MA-900, and LCA-900.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (defaults to timing_root).")
    args = parser.parse_args()

    timing_root = args.timing_root.resolve()
    output_dir = (args.output_dir or timing_root).resolve()
    _apply_style()
    _plot_comparison(timing_root, output_dir, "dims", "Hits per event", "repeated_time_vs_hits")
    _plot_comparison(timing_root, output_dir, "query_counts", "Dynamic queries per event", "repeated_time_vs_queries")
    _plot_comparison(
        timing_root, output_dir, "dims", "Hits per event", "repeated_time_vs_hits_std_errorbars", error_mode="std"
    )
    _plot_comparison(
        timing_root,
        output_dir,
        "query_counts",
        "Dynamic queries per event",
        "repeated_time_vs_queries_std_errorbars",
        error_mode="std",
    )
    _plot_comparison(
        timing_root,
        output_dir,
        "dims",
        "Hits per event",
        "repeated_mean_time_vs_hits_std_errorbars",
        error_mode="std",
        summary_stat="mean",
    )
    _plot_comparison(
        timing_root,
        output_dir,
        "query_counts",
        "Dynamic queries per event",
        "repeated_mean_time_vs_queries_std_errorbars",
        error_mode="std",
        summary_stat="mean",
    )
    _plot_binned_comparison(
        timing_root, output_dir, "dims", "Hits per event", "repeated_time_vs_hits_binned_20"
    )
    _plot_binned_comparison(
        timing_root, output_dir, "query_counts", "Dynamic queries per event", "repeated_time_vs_queries_binned_20"
    )
    _plot_binned_comparison(
        timing_root,
        output_dir,
        "dims",
        "Hits per event",
        "repeated_time_vs_hits_binned_20_std_errorbars",
        error_mode="std",
    )
    _plot_binned_comparison(
        timing_root,
        output_dir,
        "query_counts",
        "Dynamic queries per event",
        "repeated_time_vs_queries_binned_20_std_errorbars",
        error_mode="std",
    )
    _plot_binned_comparison(
        timing_root,
        output_dir,
        "dims",
        "Hits per event",
        "repeated_mean_time_vs_hits_binned_20_std_errorbars",
        error_mode="std",
        summary_stat="mean",
    )
    _plot_binned_comparison(
        timing_root,
        output_dir,
        "dims",
        "Hits per event",
        "repeated_time_vs_hits_binned_10",
        num_bins=10,
    )
    _plot_binned_comparison(
        timing_root,
        output_dir,
        "query_counts",
        "Dynamic queries per event",
        "repeated_time_vs_queries_binned_10",
        num_bins=10,
    )
    _plot_binned_comparison(
        timing_root,
        output_dir,
        "dims",
        "Hits per event",
        "repeated_time_vs_hits_binned_10_std_errorbars",
        num_bins=10,
        error_mode="std",
    )
    _plot_binned_comparison(
        timing_root,
        output_dir,
        "query_counts",
        "Dynamic queries per event",
        "repeated_time_vs_queries_binned_10_std_errorbars",
        num_bins=10,
        error_mode="std",
    )
    _plot_binned_comparison(
        timing_root,
        output_dir,
        "dims",
        "Hits per event",
        "repeated_mean_time_vs_hits_binned_10_std_errorbars",
        num_bins=10,
        error_mode="std",
        summary_stat="mean",
    )
    _plot_binned_comparison(
        timing_root,
        output_dir,
        "query_counts",
        "Dynamic queries per event",
        "repeated_mean_time_vs_queries_binned_10_std_errorbars",
        num_bins=10,
        error_mode="std",
        summary_stat="mean",
    )
    _plot_binned_comparison(
        timing_root,
        output_dir,
        "query_counts",
        "Dynamic queries per event",
        "repeated_mean_time_vs_queries_binned_20_std_errorbars",
        error_mode="std",
        summary_stat="mean",
    )
    _plot_binned_comparison(
        timing_root,
        output_dir,
        "dims",
        "Hits per event",
        "repeated_time_vs_hits_binned_15",
        num_bins=15,
    )
    _plot_binned_comparison(
        timing_root,
        output_dir,
        "query_counts",
        "Dynamic queries per event",
        "repeated_time_vs_queries_binned_15",
        num_bins=15,
    )
    _plot_binned_comparison(
        timing_root,
        output_dir,
        "dims",
        "Hits per event",
        "repeated_time_vs_hits_binned_15_std_errorbars",
        num_bins=15,
        error_mode="std",
    )
    _plot_binned_comparison(
        timing_root,
        output_dir,
        "query_counts",
        "Dynamic queries per event",
        "repeated_time_vs_queries_binned_15_std_errorbars",
        num_bins=15,
        error_mode="std",
    )
    _plot_binned_comparison(
        timing_root,
        output_dir,
        "dims",
        "Hits per event",
        "repeated_mean_time_vs_hits_binned_15_std_errorbars",
        num_bins=15,
        error_mode="std",
        summary_stat="mean",
    )
    _plot_binned_comparison(
        timing_root,
        output_dir,
        "query_counts",
        "Dynamic queries per event",
        "repeated_mean_time_vs_queries_binned_15_std_errorbars",
        num_bins=15,
        error_mode="std",
        summary_stat="mean",
    )
    print(f"Saved comparison plots to {output_dir}")


if __name__ == "__main__":
    main()
