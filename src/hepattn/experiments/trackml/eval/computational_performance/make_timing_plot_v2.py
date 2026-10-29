"""Inference-time-vs-hits plot using InferenceTimer data."""
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 400
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.constrained_layout.use"] = True

PLOTS_ROOT = "/eos/user/t/tmangat/plots"
PREFIX = "/eos/user/t/tmangat/logs/paper_reco/TRK-v8-eta4-lca-3900-0p3-lw100-w256-pippacode-epochs30_20261002-T103231/times/TRK-v8-eta4-lca-3900-0p3-lw100-w256-pippacode-epochs30"

times = np.load(f"{PREFIX}_times.npy").astype(np.float64)
dims = np.load(f"{PREFIX}_dims.npy").astype(np.float64)


def scatter_with_fit(ax, x, y, colour, label):
    ax.scatter(x, y, color=colour, s=14, alpha=0.7, label=label)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, slope * xs + intercept, color=colour, lw=1.5)


fig, ax = plt.subplots(figsize=(7, 4.5))
scatter_with_fit(ax, dims, times, "tab:orange", "lsca")
ax.set_xlabel("Hits per event")
ax.set_ylabel("Inference time [ms]")
ax.grid(zorder=0, alpha=0.25, linestyle="--")
ax.legend(frameon=False)
ax.set_title("Consistent per-event hit count (InferenceTimer)", fontsize=9)

fig.savefig(f"{PLOTS_ROOT}/lsca/inference_time_vs_hits.png")
fig.savefig(f"{PLOTS_ROOT}/lsca/inference_time_vs_hits.pdf")
plt.close(fig)
print("Wrote inference_time_vs_hits.png/.pdf to lsca/")
