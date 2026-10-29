# physics_performance — efficiency, fake rate, duplicate rate

Covers the paper's physics-performance numbers (DM/perfect-match efficiency,
fake rate, duplicate rate — Table 4 / efficiency-figure style), as distinct from
`../computational_performance/`'s inference latency and GPU memory.

All of these read a `__test.h5` prediction file (the `PredictionWriter`'s output,
`<ckpt_dir>/<ckpt_stem>__test.h5` — see `../../README.md`'s "Paper reproduction
artifacts" section for how to produce one). `load_events`/`load_all_results` etc. work on any such file
regardless of which model produced it; the template dicts in
`run_track_eval_plots.py`/`tracking_output_plots.py` default to a single `lsca`
(`BEST_CKPTS`) entry, reproducible end-to-end from this repo alone.

## Core matching library

`track_eval_utils.py` implements the shared matching pipeline (`matched_kinematics`,
`load_regression`, `check_valid`, `check_reconstructable`,
`recover_out_of_acceptance_matches`, `get_masks`, `process_particles`,
`process_tracks`, `eval_tracks`, `load_event`, `load_events`), including event-cache
support (`_cache_signature`/`_cache_base_path`, used by `reuse_saved_event_cache=True`
in `run_track_eval_plots.py`) and `match_min_hits`/`match_min_pt` filtering on
`process_tracks`. Both driver scripts below import `load_events` from it.

## Driver / template scripts

None of these are CLI tools — each hardcodes its input file paths and plot parameters
(near the top of the file, or under its `if __name__ == "__main__":` block) that you
edit directly before running with no arguments.

| script | what it does |
|---|---|
| `run_track_eval_plots.py` | Richer tracking-eval library: `load_all_results`, `plot_efficiency` (DM/perfect efficiency vs. pt/eta/phi, multi-model overlay), `print_and_save_summary` (writes a `track_eval_threshold_summary.csv`). Edit `tracking_fnames`/`tracking_config_fnames` under `__main__` to point at your `__test.h5`/`config.yaml` pairs. |
| `tracking_output_plots.py` (+ `.ipynb`) | Similar efficiency/fake-rate/duplicate-rate reporting via `main()` (supports comparing multiple models if you add entries, defaults to one), plus optional track-regression residual/pT-response plots (only relevant for checkpoints with a regression head — none of the `BEST_CKPTS` checkpoints have one). Edit the module-level `tracking_config_fname` dict (read by `load_tracking_configs()`). |

## Hit-filter-level evaluation (separate from tracking)

Evaluates the upstream pixel hit-filter model's own hit-level classification
performance (precision/recall of "is this hit on a reconstructable particle"), not the
tracking model — relevant to `../../configs/paper/HIT_FILTER_CKPTS/`.

- `hit_evaluate.py` — `load_event`/`load_events` for hit-filter eval files (reads a
  `threshold` param for the valid-hit discriminant cut), builds per-hit DataFrames.
- `run_hit_eval_plots.py` — template driver (same edit-then-run pattern as above),
  imports `load_events` from `hit_evaluate.py`.
- `filter_output_plots.ipynb` — notebook version of the same hit-filter analysis.

## Standalone utilities

- `plot_event.py` — `plot_trackml_event_reconstruction(inputs, reconstruction, ...)`,
  a single function returning a matplotlib figure of one event's hits/tracks in
  x-y space. Meant to be imported and called with in-memory model outputs, not a
  script to run directly (no `__main__`).
- `plot_kinematics.py` — `plot_trackml_kinematics(dataset)`, histograms a
  `TrackMLDataset`'s pt/eta/hit-count/particle-count distributions directly from the
  raw data — no checkpoint or `__test.h5` needed.
- `plot_utils.py` — shared plotting helpers (`binned`, `profile_plot`, `hist_plot`)
  used by several of the driver scripts above.
