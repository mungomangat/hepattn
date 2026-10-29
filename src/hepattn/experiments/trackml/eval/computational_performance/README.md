# computational_performance — inference latency / GPU memory

Inference latency and peak GPU memory for `lsca`, as distinct from
`../physics_performance/`'s physics performance (efficiency/fake rate).

Neither is driven by a `__test.h5` prediction file. They read per-event timing and
memory arrays (`*_times.npy`, `*_dims.npy`, `*_query_counts.npy`,
`*_peak_allocated_bytes.npy`) written by `hepattn.callbacks.InferenceTimer` (per-batch
repeats, a warmup phase, and peak GPU memory tracking are already built in) during a
dedicated `test` run with that callback enabled in the trainer config.

| script | reads | produces |
|---|---|---|
| `make_timing_plot_v2.py` | `*_times.npy`/`*_dims.npy` | `inference_time_vs_hits.{png,pdf}` — latency vs. hits per event |
| `make_gpu_mem_comparison.py` | `*_dims.npy`/`*_query_counts.npy`/`*_peak_allocated_bytes.npy` | `gpu_mem_vs_hits.{png,pdf}` and `gpu_mem_vs_queries.{png,pdf}` |

Both are not CLI tools — each hardcodes its input array path (`PREFIX` near the top)
and output directory (`PLOTS_ROOT`) to this project's own
`/eos/user/t/tmangat/logs/...` and `/eos/user/t/tmangat/plots` paths. Edit those
before running elsewhere; run with no arguments once edited.
