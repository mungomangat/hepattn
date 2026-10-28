# HIT_FILTER_CKPTS — pixel hit-filter models used upstream of BEST_CKPTS

The two pixel hit-filter checkpoints (600 MeV / 900 MeV truth-pT thresholds) that the
`BEST_CKPTS/*/config.yaml` files reference via `hit_eval_train/val/test`. A tracking
config will not run standalone without one of these.

| dir | used by |
|---|---|
| `HF-pix-600MeV-eta4_20260214-T195401/` | `dqlsca600`, `dqlsca6002`, `dqma600` |
| `HF-pix-900MeV-eta4_20260214-T194855/` | `dqlsca900`, `dqma900` |

Each dir has `config.yaml` + `metadata.yaml` (committed) and the trained `.ckpt`
(committed via git-lfs). **The large eval output files
(`epoch=070-val_loss=0.20153_test_eval.h5` / `test_eval.h5`, 417MB and 241MB) are
intentionally *not* committed** — they're regenerable by rerunning eval against the
committed checkpoint, and adding them would roughly double this directory's LFS
footprint for no new information.
