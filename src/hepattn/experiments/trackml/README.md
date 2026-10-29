# TrackML

Setup

```shell
srun --pty  --cpus-per-task 15  --gres gpu:l40s:1 --mem=100G -p GPU bash
apptainer shell --nv --bind /share/rcifdata/maxhart/data/trackml/ hepattn/pixi.sif
cd hepattn && pixi shell
cd hepattn/src/hepattn/experiments/trackml/
```

## Data
You can use the `detectors.csv` file available in `data/trackml/` directory.
This file was obtained by unzipping the file provided on the [Kaggle trackML webpage](https://www.kaggle.com/competitions/trackml-particle-identification/data).
For training, use the codalab data from [this webpage](https://competitions.codalab.org/competitions/20112#participate-get_data).
Sample data is available for testing in the `data/trackml/raw/` directory.

## Hit Filter

```shell
# train
python run_filtering.py fit --config configs/filtering.yaml --trainer.fast_dev_run 10

# test
python run_filtering.py test --config PATH

# evaluate on train/val
python run_filtering.py test --config PATH --data.test_dir /share/rcifdata/maxhart/data/trackml/prepped/train
```

## Tracking

```shell
# train
python run_tracking.py fit --config configs/tracking.yaml --trainer.fast_dev_run 10

# test
python run_tracking.py test --config PATH
```


## Batch Submit

```shell
sbatch /share/rcifdata/svanstroud/hepattn/src/hepattn/experiments/trackml/submit/submit_trackml_filtering.sh
sbatch /share/rcifdata/svanstroud/hepattn/src/hepattn/experiments/trackml/submit/submit_trackml_tracking.sh
```


## Configurations

A central pixel detector with a low pt cut:

```yaml
hit_volume_ids: [7, 8, 9] # pixel barrel and endcaps
particle_min_pt: 0.6
particle_max_abs_eta: 2.5
```

A full pixel detector with a reasonable pt cut for targeting >1GeV particles:

```yaml
hit_volume_ids: [7, 8, 9] # pixel barrel and endcaps
particle_min_pt: 0.9
particle_max_abs_eta: 4.0
```

A more lightweight configuration for testing purposes:

```yaml
hit_volume_ids: [8] # pixel barrel only
particle_min_pt: 0.9
particle_max_abs_eta: 2.5
```

For which the following pre-trained hit filter evaluations can be used:

```yaml
hit_eval_train: /share/rcifdata/svanstroud/hepattn/src/hepattn/experiments/trackml/logs/HC-v3-lite_20250620-T114025/ckpts/epoch=016-val_loss=0.15730_train_eval.h5
hit_eval_val: /share/rcifdata/svanstroud/hepattn/src/hepattn/experiments/trackml/logs/HC-v3-lite_20250620-T114025/ckpts/epoch=016-val_loss=0.15730_val_eval.h5
hit_eval_test: /share/rcifdata/svanstroud/hepattn/src/hepattn/experiments/trackml/logs/HC-v3-lite_20250620-T114025/ckpts/epoch=016-val_loss=0.15730_test_eval.h5
```

## Paper reproduction artifacts

`configs/paper/BEST_CKPTS/` and `configs/paper/HIT_FILTER_CKPTS/` hold the actual
trained checkpoints (via git-lfs) behind arXiv:2606.17631 ("Better Queries, Cheaper
Attention").

### `configs/paper/BEST_CKPTS/` — the 5 tracking checkpoints

| dir | paper label | val_loss | notes |
|---|---|---|---|
| `dqlsca600/` | Pix0.6 DQ+LSCA | 0.30247 | **the** best LSCA/0.6GeV checkpoint |
| `dqlsca6002/` | Pix0.6 DQ+LSCA (rerun) | 0.31587 | second seed, slightly worse, kept for provenance |
| `dqlsca900/` | Pix0.9 DQ+LSCA | 0.35284 | |
| `dqma600/` | Pix0.6 DQ+MA | 0.28455 | masked-attention variant |
| `dqma900/` | Pix0.9 DQ+MA | 0.16017 | |

Each dir's `config.yaml` is the exact config used to produce that checkpoint. `submit/{600,900}/*.sh` are the original cluster submission scripts.

### `configs/paper/HIT_FILTER_CKPTS/` — the 2 upstream hit-filter checkpoints

The tracking checkpoints above reference these via `hit_eval_train/val/test`.

| dir | used by |
|---|---|
| `HF-pix-600MeV-eta4_20260214-T195401/` | `dqlsca600`, `dqlsca6002`, `dqma600` |
| `HF-pix-900MeV-eta4_20260214-T194855/` | `dqlsca900`, `dqma900` |

Each dir has `config.yaml` + `metadata.yaml` (committed) and the trained `.ckpt`
(committed via git-lfs). The large eval output files
(`epoch=070-val_loss=0.20153_test_eval.h5` / `test_eval.h5`, 417MB and 241MB) are
intentionally *not* committed. They're regenerable by rerunning eval against the
committed checkpoint.

### Paths that won't resolve outside the original cluster

These configs are committed byte-identical to the originals (a deliberate choice —
historical record, not a "working out of the box" config). `train_dir`/`val_dir`/
`test_dir`, the Weights & Biases `offline_directory`/`default_root_dir`, and
`hit_eval_train/val/test` all point at `/share/rcif2/pduckett/...` and will not resolve
on any other machine. `ckpt_path` (resume-from-checkpoint) in 4 of 5 `BEST_CKPTS`
configs likewise points at a previous training run's checkpoint on that same cluster.

To actually run one of these configs elsewhere:
- Repoint `train_dir`/`val_dir`/`test_dir` at wherever this project's TrackML data
  lives (see `## Data` above). **`train_dir` is required even for `test`**: the data
  module's `setup()` builds a (discarded) train dataset on both the `"fit"` and
  `"test"` stages, so a `test` run crashes with `ValueError: num_events must be
  greater than 0` if `train_dir` doesn't resolve to a non-empty directory. Easiest
  fix is `--data.train_dir <dir> --data.num_train 10` (any small positive number;
  that split is never actually used during `test`).
- Repoint `hit_eval_train/val/test` at the matching checkpoint in
  `configs/paper/HIT_FILTER_CKPTS/` per the table above.
- Set `ckpt_path: null` unless you specifically want to resume from Pippa's
  intermediate checkpoint (which you don't have).

### `GenericPredictionWriter` — required for the hit-filter configs

The default `PredictionWriter` callback only supports tracking-model output (it
requires `track_valid`/`track_hit_valid` predictions); it raises
`ValueError: Paper-compatible output requires 'preds/final/track_valid' or
'track_valid_prob'.` on a hit-filter checkpoint, which has no such output. This isn't
an environment-specific path problem like the ones above — it fails identically on
any machine, so **both `HIT_FILTER_CKPTS` configs have their `PredictionWriter`
callback changed to `hepattn.callbacks.GenericPredictionWriter`**, the one deliberate
deviation from "committed byte-identical to the originals." `GenericPredictionWriter`
(`src/hepattn/callbacks/generic_prediction_writer.py`) is a restored, task-agnostic
writer — ported from `main`'s pre-"Dynamic Queries" history (commit `3d9aa75`) — that
mirrors whatever `{layer: {task: {key: value}}}` structure a model's outputs actually
have, rather than assuming a tracking schema. It also names its output
`{ckpt_stem}_{split}_eval.h5` (not `{ckpt_stem}__test.h5`), matching the real
pre-existing hit-filter eval files' naming. The `BEST_CKPTS` tracking configs are
unaffected and still use the regular `PredictionWriter`.

### How to run evaluation

Two separate sets of scripts cover the paper's two reproduction axes:

- **`eval/physics_performance/`** — physics performance (DM/perfect-match efficiency
  vs. pt/eta/phi, fake rate, duplicate rate).
- **`eval/computational_performance/`** — inference latency and GPU memory vs.
  hits/queries.

First generate a `__test.h5` prediction file for the checkpoint (see the two-step
`run_filtering.py test` → `run_tracking.py test` process below), then feed it into one
of the `eval/physics_performance/` scripts for physics metrics. For computational
performance, the relevant data instead comes from an `InferenceTimer`-instrumented
`test` run (see `eval/computational_performance/README.md`) — it is not derived from
the `__test.h5` file.

**1. Generate the hit-filter eval file** (pick the 600 MeV or 900 MeV variant matching
the tracking checkpoint you want):

```shell
python run_filtering.py test \
  --config configs/paper/HIT_FILTER_CKPTS/HF-pix-600MeV-eta4_20260214-T195401/config.yaml \
  --ckpt_path configs/paper/HIT_FILTER_CKPTS/HF-pix-600MeV-eta4_20260214-T195401/epoch=070-val_loss=0.20153.ckpt \
  --data.train_dir <path-to-your-trackml-train-data> --data.num_train 10 \
  --data.test_dir <path-to-your-trackml-test-data>
```

This writes `epoch=070-val_loss=0.20153_test_eval.h5` next to that checkpoint (via
`GenericPredictionWriter`, named `<ckpt_stem>_test_eval.h5` — see below).

**2. Run tracking test/eval against a `BEST_CKPTS` checkpoint**, pointing
`--data.hit_eval_test` at the file from step 1:

```shell
python run_tracking.py test \
  --config configs/paper/BEST_CKPTS/dqlsca600/config.yaml \
  --ckpt_path "configs/paper/BEST_CKPTS/dqlsca600/epoch=029-val_loss=0.30247-256289.ckpt" \
  --data.train_dir <path-to-your-trackml-train-data> --data.num_train 10 \
  --data.test_dir <path-to-your-trackml-test-data> \
  --data.hit_eval_test configs/paper/HIT_FILTER_CKPTS/HF-pix-600MeV-eta4_20260214-T195401/epoch=070-val_loss=0.20153_test_eval.h5
```

This writes `epoch=029-val_loss=0.30247-256289__test.h5` next to the tracking
checkpoint.

**3. Run the `eval/physics_performance/` scripts against that file.** Each has hardcoded dicts/paths near the bottom of the file (or a
module-level config dict) that you edit directly before running with no arguments,
e.g. `run_track_eval_plots.py`'s `tracking_fnames`/`tracking_config_fnames` dicts
under its `if __name__ == "__main__":` block. Point those at the `__test.h5`/
`config.yaml` pair from step 2, then run e.g.
`python eval/physics_performance/run_track_eval_plots.py`.
