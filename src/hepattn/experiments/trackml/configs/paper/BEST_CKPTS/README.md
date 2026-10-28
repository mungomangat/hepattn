# BEST_CKPTS — the 5 real checkpoints published in the paper

Originals from Pippa Duckett's own training runs for "Better Queries, Cheaper
Attention: Adapting Transformers for Efficient Sparse Reconstruction" (arXiv:2606.17631).
Committed here so the `pippa-code` branch is a self-contained, citable reference to the
exact weights behind the paper's tables/figures.

| dir | paper label | val_loss | notes |
|---|---|---|---|
| `dqlsca600/` | Pix0.6 DQ+LSCA | 0.30247 | **the** best LSCA/0.6GeV checkpoint |
| `dqlsca6002/` | Pix0.6 DQ+LSCA (rerun) | 0.31587 | second seed, slightly worse, kept for provenance |
| `dqlsca900/` | Pix0.9 DQ+LSCA | 0.35284 | |
| `dqma600/` | Pix0.6 DQ+MA | 0.28455 | masked-attention variant |
| `dqma900/` | Pix0.9 DQ+MA | 0.16017 | |

Each dir's `config.yaml` is the exact config used to produce that checkpoint, not a
template. `submit/{600,900}/*.sh` are the original cluster submission scripts.

## Paths that won't resolve outside the original cluster

These configs are committed byte-identical to the originals (a deliberate choice —
historical record, not a "working out of the box" config). `train_dir`/`val_dir`/
`test_dir`, the Weights & Biases `offline_directory`/`default_root_dir`, and
`hit_eval_train/val/test` all point at `/share/rcif2/pduckett/...` and will not resolve
on any other machine. `ckpt_path` (resume-from-checkpoint) in 4 of 5 configs likewise
points at a previous training run's checkpoint on that same cluster.

To actually run one of these configs elsewhere:
- Repoint `train_dir`/`val_dir`/`test_dir` at wherever this project's TrackML data
  lives (see `/shared/submit-scripts/trackml` data-path notes, or the top-level
  `trackml/README.md`).
- Repoint `hit_eval_train/val/test` at the matching checkpoint in
  `../HIT_FILTER_CKPTS/` (600 MeV variant for `dqlsca600`/`dqlsca6002`/`dqma600`,
  900 MeV variant for `dqlsca900`/`dqma900`) — see that dir's README.
- Set `ckpt_path: null` unless you specifically want to resume from Pippa's
  intermediate checkpoint (which you don't have).
