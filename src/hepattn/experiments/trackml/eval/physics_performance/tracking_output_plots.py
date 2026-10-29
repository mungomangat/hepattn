"""Tracking model evaluation: paper_reco (lsca) test-set output.

Python-script port of tracking_output_plots.ipynb (kept in sync manually) -- editing/running a
.py file directly is far quicker to iterate on than round-tripping through the notebook.
"""

import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from plot_utils import binned, hist_plot, profile_plot
from track_eval_utils import load_events

plt.rcParams["figure.dpi"] = 200
plt.rcParams["text.usetex"] = False
# disabled due to missing font in texlive on the Nikhef clusters
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.constrained_layout.use"] = True

# ── Plot parameters ──────────────────────────────────────────────────────────

training_colours = {
    "paper_reco": "tab:purple",
}

qty_bins = {
    "pt": np.array([0.6, 0.75, 1.0, 1.5, 2, 3, 4, 6, 10]),
    "eta": np.array([-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]),
    "phi": np.array([-math.pi, -2.36, -1.57, -0.79, 0, 0.79, 1.57, 2.36, math.pi]),
    "vz": np.array([-100, -50, -20, -10, 0, 10, 20, 50, 100]),
}

qty_symbols = {"pt": "p_\\mathrm{T}", "eta": "\\eta", "phi": "\\phi", "vz": "v_z"}
qty_units = {"pt": "[GeV]", "eta": "", "phi": "", "vz": "[mm]"}
out_dir = Path("plots/")
out_dir.mkdir(parents=True, exist_ok=True)

# ── Retrieve tracking model configuration ────────────────────────────────────

tracking_fnames = {
    "paper_reco": "/eos/user/t/tmangat/logs/paper_reco/TRK-v8-eta4-lca-3900-0p3-lw100-w256-epochs30_20260904-T083610/ckpts/epoch=024-val_loss=0.28934__test.h5",
}
load_events_kwargs = {
    # no track regression head in this checkpoint (px/py/pz are NaN) -- regression must stay False
    "paper_reco": {"regression": False, "iou_threshold": 0.0, "track_valid_threshold": 0.5},
}
tracking_config_fname = {
    "paper_reco": "/eos/user/t/tmangat/logs/paper_reco/TRK-v8-eta4-lca-3900-0p3-lw100-w256-epochs30_20260904-T083610/config.yaml",
}
# reconstruct-anything output schema (see track_eval_utils.get_masks): "old" is the
# paper-compatible schema used here; "sparse"/dense are also supported by load_events
# for other checkpoint types.
key_modes = {
    "paper_reco": "old",
}


def data_cfg(fconfig):
    """Return the data config dict, handling both flat and jsonargparse class_path/init_args configs."""
    data = fconfig["data"]
    return data["init_args"] if "init_args" in data else data


def load_tracking_configs():
    tracking_params = ["particle_min_pt", "particle_max_abs_eta"]
    tracking_configs = {}
    for name in tracking_config_fname:
        with Path(tracking_config_fname[name]).open() as f:
            fconfig = yaml.safe_load(f)
            print("name: " + fconfig["name"])
            for i in tracking_params:
                print("> " + i + "\t: ", data_cfg(fconfig)[i])
        tracking_configs[name] = fconfig
    return tracking_configs


# ── Load evaluation files ────────────────────────────────────────────────────


def load_single_model(name, fname, eta_cut, pt_cut, index_list, num_events, load_events_kwargs_for_model, key_mode):
    """Helper function to load a single model's events."""
    particle_targets = ["pt", "eta", "phi"]
    print(f"Loading {name} model with PT > {pt_cut} and |eta| < {eta_cut}\n")
    result = load_events(
        fname=fname,
        eta_cut=eta_cut,
        index_list=index_list,
        pt_cut=pt_cut,
        randomize=num_events,
        particle_targets=particle_targets,
        key_mode=key_mode,
        **load_events_kwargs_for_model,
    )
    print(f"Finished {name}\n")
    return name, result


def load_all(tracking_configs, num_events=100, index_list=None):
    args_list = []
    for name, fname in tracking_fnames.items():
        eta_cut = data_cfg(tracking_configs[name])["particle_max_abs_eta"]
        pt_cut = data_cfg(tracking_configs[name])["particle_min_pt"]
        load_events_kwargs_for_model = load_events_kwargs.get(name, {"regression": False})
        key_mode = key_modes.get(name)
        args_list.append((name, fname, eta_cut, pt_cut, index_list, num_events, load_events_kwargs_for_model, key_mode))

    with Pool(processes=len(tracking_fnames)) as pool:
        results = pool.starmap(load_single_model, args_list)

    return dict(results)


# ── Plot metrics: efficiency and fake rate ───────────────────────────────────


def plot_efficiency_and_fake_rate(tracking_results, particle_targets):
    plot_fr = True
    plot_trainings = tracking_fnames.keys()
    for qty in particle_targets:
        if qty not in {"pt", "eta", "phi"}:
            continue

        axlist = []
        if plot_fr:
            fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(12, 4), constrained_layout=True)
            axlist.extend([ax, ax1])
            ax1.set_xlabel(rf"Particle ${qty_symbols[qty]}^\mathrm{{True}}$ {qty_units[qty]}")
            ax1.set_ylabel("Fake Rate")
        else:
            fig, ax = plt.subplots(ncols=1, figsize=(6, 4), constrained_layout=True)
            axlist.append(ax)

        names = []
        for name, (tracks, parts) in tracking_results.items():
            if name not in plot_trainings:
                continue
            names.append(name)
            """Efficiency plots"""
            reconstructable = parts["reconstructable"]
            # double majority
            bin_count, bin_error = binned(
                parts["eff_dm"][reconstructable],
                parts["particle_" + qty][reconstructable],
                qty_bins[qty],
                underflow=True,
                overflow=True,
                binomial=False,
            )
            profile_plot(bin_count, bin_error, qty_bins[qty], axes=ax, colour=training_colours[name], ls="solid")

            # perfect
            bin_count, bin_error = binned(
                parts["eff_perfect"][reconstructable],
                parts["particle_" + qty][reconstructable],
                qty_bins[qty],
                underflow=True,
                overflow=True,
                binomial=False,
            )
            profile_plot(bin_count, bin_error, qty_bins[qty], axes=ax, colour=training_colours[name], ls="dotted")

            """Fake rate plots"""
            if plot_fr:
                x = f"matched_{qty}" if f"matched_{qty}" in tracks.columns else f"part_{qty}"
                # Unmatched tracks (matched_pid == -1: genuine fakes, and duplicate copies that
                # lost the "kept" slot) have NaN matched_pt/eta/phi. np.digitize sends NaN past
                # the last bin edge, and overflow=True then silently piles every one of them into
                # the last (highest) bin, spiking its fake/duplicate rate regardless of the
                # tracks' real kinematics. Exclude undefined-x tracks instead.
                reconstructable = tracks["reconstructable"] & tracks[x].notna()
                # fake rate
                fakes = (~tracks["eff_dm"]) & (~tracks["duplicate"])
                bin_count, bin_error = binned(
                    fakes[reconstructable],
                    tracks[x][reconstructable],
                    qty_bins[qty],
                    underflow=True,
                    overflow=True,
                    binomial=False,
                )
                profile_plot(bin_count, bin_error, qty_bins[qty], axes=ax1, colour=training_colours[name], ls="solid")
                # duplicate
                bin_count, bin_error = binned(
                    tracks["duplicate"][reconstructable],
                    tracks[x][reconstructable],
                    qty_bins[qty],
                    underflow=True,
                    overflow=True,
                    binomial=False,
                )
                profile_plot(bin_count, bin_error, qty_bins[qty], axes=ax1, colour=training_colours[name], ls="dotted")

        # axis ranges
        # lower bound dropped from 0.8 to fit the "perfect" (strict) efficiency curve, which dips
        # to ~0.66 in some eta/phi bins -- well below the double-majority efficiency plotted alongside it
        ax.set_ylim(0.6, 1.05)
        ax.set_ylabel("Efficiency")
        ax.set_xlabel(rf"Particle ${qty_symbols[qty]}^\mathrm{{True}}$ {qty_units[qty]}")

        for i in axlist:
            i.grid(zorder=0, alpha=0.25, linestyle="--")
            if qty == "pt":
                i.set_xlim([0, 10.5])
                i.set_xticks(np.arange(start=2, stop=11, step=2))
            if qty == "eta":
                i.set_xlim([-4.5, 4.5])
                i.set_xticks(np.arange(start=-4, stop=4.5, step=1))
            if qty == "phi":
                i.set_xlim([-3.5, 3.5])
                i.set_xticks(np.arange(start=-3, stop=3.5, step=1))
            if qty == "vz":
                i.set_xlim([-112, 112])
                i.set_xticks(np.arange(start=-100, stop=110, step=25))

        # custom legends
        legend_elements_0 = [Line2D([0], [0], color=training_colours[training], label=training) for training in names]
        leg1_0 = ax.legend(handles=legend_elements_0, frameon=False, loc="upper left")
        ax.add_artist(leg1_0)

        legend_elements_eff = [Line2D([0], [0], color="black", label="DM"), Line2D([0], [0], color="black", ls="dotted", label="Perfect")]
        leg2_0 = ax.legend(handles=legend_elements_eff, frameon=False, loc="upper right")
        ax.add_artist(leg2_0)
        if plot_fr:
            leg1_1 = ax1.legend(handles=legend_elements_0, frameon=False, loc="upper left")
            ax1.add_artist(leg1_1)
            legend_elements_fake = [Line2D([0], [0], color="black", label="Fake"), Line2D([0], [0], color="black", ls="dotted", label="Duplicate")]
            leg2_1 = ax1.legend(handles=legend_elements_fake, frameon=False, loc="upper right")
            ax1.add_artist(leg2_1)
            # widened from the shared 0.04 default: real (non-overflow-bin) fake rates reach ~0.09
            # in the eta bins around |eta|~1.75-2 -- was silently clipping there before
            axlist[1].set_ylim(0.0, 0.04)
            if qty == "pt":
                axlist[1].set_ylim(0.0, 0.06)
            if qty == "eta":
                axlist[1].set_ylim(0.0, 0.10)

        fig.savefig(out_dir / f"{qty}_eff.pdf")
        plt.close(fig)


# ── Regression residual / response plots (skipped: no checkpoint has a track regression head) ──


def plot_regression_residuals(tracking_results):
    plot_trainings = {"paper_reco"}
    nbins = 55
    qty_res_bins = {
        "pt": np.linspace(-1, 1, nbins),
        "eta": np.linspace(-0.1, 0.1, nbins),
        "phi": np.linspace(-0.1, 0.1, nbins),
        "vz": np.linspace(-15, 15, nbins),
    }
    fig, ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    fig.set_size_inches(10, 4)
    ax = ax.flatten()

    for i, qty in enumerate(["pt", "eta", "phi", "vz"]):
        labels = []
        colours = []
        for name, (tracks, _parts) in tracking_results.items():
            if name in plot_trainings:
                bins = qty_res_bins[qty]
                colour = training_colours[name]
                tracks_qty = tracks["track_" + qty][tracks["eff_dm"] & tracks["reconstructable_parts"]]
                parts_qty = tracks["matched_" + qty][tracks["eff_dm"] & tracks["reconstructable_parts"]]
                res = tracks_qty - parts_qty

                label = hist_plot(xs=res, bins=bins, xrange=(bins[0], bins[-1]), name=name, axes=ax[i], colour=colour)
                labels.append(label)
                colours.append(colour)

        ax[i].grid(zorder=0, alpha=0.25, linestyle="--")
        ax[i].set_xlabel(rf"${qty_symbols[qty]}^\mathrm{{Reco}} - {qty_symbols[qty]}^\mathrm{{True}}$ {qty_units[qty]}")
        ax[i].set_ylabel("Density")

        ticks = None
        if qty in {"eta", "phi"}:
            ticks = np.arange(-0.1, 0.11, 0.05).round(2)
        if qty == "pt":
            ticks = np.arange(-1, 1.1, 0.5).round(2)
        if qty == "vz":
            ticks = np.arange(-15, 16, 5).round(2)

        ax[i].set_xticks(ticks)
        ax[i].set_xticklabels(ticks)
        legend_elements = [Line2D([0], [0], color=colours[j], label=labels[j]) for j in range(len(labels))]
        ax[i].legend(handles=legend_elements, frameon=False, loc="upper left", fontsize=8)

    fig.savefig(out_dir / "regr_residuals.pdf")
    plt.close(fig)


def plot_pt_response(tracking_results):
    plot_trainings = {"paper_reco"}
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    fig.set_size_inches(10, 3)
    qty = "pt"
    for name, (tracks, _parts) in tracking_results.items():
        if name in plot_trainings:
            parts_eff_qty = tracks["matched_" + qty][tracks["eff_dm"]]
            tracks_eff_qty = tracks["track_" + qty][tracks["eff_dm"]]
            n_assigned = tracks["n_pred_hits"][tracks["eff_dm"]]
            response = tracks_eff_qty / parts_eff_qty

            bins = qty_bins[qty]
            ys, ys_err = binned(response, parts_eff_qty, bins, underflow=True, overflow=True, binomial=False)
            profile_plot(ys, ys_err, bins, axes=ax[0], colour=training_colours[name], ls="solid", label=name)

            bins = np.linspace(3, 10, 8)
            ys, ys_err = binned(response, n_assigned, bins, binomial=False)
            profile_plot(ys, ys_err, bins, axes=ax[1], colour=training_colours[name], ls="solid", label=name)

    ax[0].set_xlabel(rf"${qty_symbols[qty]}^\mathrm{{True}}$ {qty_units[qty]}")
    ax[0].set_ylabel(rf"${qty_symbols[qty]}^\mathrm{{Reco}}/{qty_symbols[qty]}^\mathrm{{True}}$")
    ax[0].grid(zorder=0, alpha=0.25, linestyle="--")
    ax[0].legend(frameon=False)
    ax[1].set_xlabel(r"Number of assigned hits")
    ax[1].set_ylabel(rf"${qty_symbols[qty]}^\mathrm{{Reco}}/{qty_symbols[qty]}^\mathrm{{True}}$")
    ax[1].grid(zorder=0, alpha=0.25, linestyle="--")
    ax[1].legend(frameon=False)
    fig.savefig(out_dir / "pt_regr-nhits-response_paper.pdf")
    plt.close(fig)


# ── Efficiency and fake rate numbers ──────────────────────────────────────────


def print_summary(tracking_results, num_events):
    for name, (tracks, parts) in tracking_results.items():
        print(name)
        tgts = parts[parts.reconstructable]
        trks = tracks[tracks.reconstructable]
        high_pt_parts = tgts[tgts.particle_pt > 1.0]
        high_pt_parts_900 = tgts[tgts.particle_pt > 0.9]
        high_pt_eff = high_pt_parts.eff_dm.mean()
        high_pt_eff_900 = high_pt_parts_900.eff_dm.mean()
        high_pt_tracks = trks[trks.matched_pt > 1.0]
        high_pt_tracks_900 = trks[trks.matched_pt > 0.9]
        high_pt_fr = (~high_pt_tracks.eff_dm & ~trks.duplicate).mean()
        high_pt_fr_900 = (~high_pt_tracks_900.eff_dm & ~trks.duplicate).mean()

        integrated_fr = (~trks.eff_dm & ~trks.duplicate).mean()

        print(f"N events: {100 if num_events is None else num_events}, N particles: {len(parts)}, N tracks: {len(tracks)}")
        print(f"DM Integrated efficiency: {tgts.eff_dm.mean():.1%}")
        print(f"DM Efficiency for pT > 1.0 GeV: {high_pt_eff:.1%}")
        print(f"DM Efficiency for pT > 0.9 GeV: {high_pt_eff_900:.1%}")
        print()
        print(f"DM Integrated fake rate: {integrated_fr:.1%}")
        print(f"DM Fake rate for pT > 1.0 GeV: {high_pt_fr:.1%}")
        print(f"DM Fake rate for pT > 0.9 GeV: {high_pt_fr_900:.1%}")
        print()
        print(f"Perfect integrated Efficiency: {tgts.eff_perfect.mean():.1%}")
        print(f"Perfect Efficiency for pT > 1.0 GeV: {high_pt_parts.eff_perfect.mean():.1%}")
        print(f"Perfect Efficiency for pT > 0.9 GeV: {high_pt_parts_900.eff_perfect.mean():.1%}")
        print()
        print(f"Duplicate rate: {tracks.duplicate.mean():.1%}")
        print("\n")


def main():
    tracking_configs = load_tracking_configs()
    particle_targets = data_cfg(next(iter(tracking_configs.values())))["targets"]["particle"]
    print("> particle targets: ", particle_targets)

    num_events = 100
    tracking_results = load_all(tracking_configs, num_events=num_events, index_list=None)

    plot_efficiency_and_fake_rate(tracking_results, particle_targets)

    # Skipped: paper_reco checkpoint has no track regression head.
    plot_regression = False
    if plot_regression:
        plot_regression_residuals(tracking_results)
        plot_pt_response(tracking_results)

    print_summary(tracking_results, num_events)


if __name__ == "__main__":
    main()
