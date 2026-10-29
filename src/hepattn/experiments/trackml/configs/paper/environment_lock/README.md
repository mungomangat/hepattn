# environment_lock — reference snapshot, not the active lockfile

The exact `pixi.lock`/`pyproject.toml` pair the `BEST_CKPTS`/`HIT_FILTER_CKPTS`
checkpoints were trained and evaluated under. Kept here purely as a point-in-time
reference for exact dependency-version reproducibility — this is **not** used by
`pixi install` and does not replace the repo's own root `pixi.lock`/`pyproject.toml`,
which remains the active environment definition for this branch.
