# Performance figures

Regenerate all figures with `python docs/performance-views/render.py` (matplotlib and numpy, no GPU).

- `01-benchmark.svg`: the final full check of commit `026c4ea`. Baseline runs 0.2522 and 0.2516 s per update (median
  updates 0.2498 and 0.2495 s), candidate 0.1179 s (median 0.1175 s; windows 11.776 / 11.799 / 11.790 s per 100
  updates). The bars show the median update; the speedup, 2.1247×, is the smaller of the median ratio and the
  total-time ratio (2.1369×). eight 80 GB Hopper GPUs, 2,048 tokens per GPU.
- `02-history.svg`: the eight full checks, each with freshly measured baseline runs on the same host
  ([history.json](history.json)). Points are cumulative configurations, not isolated ablations.
- `03-time-budget.svg`: profiled time per update of each component of the final configuration (torch profiler,
  rank 0) next to its estimated lower bound on this hardware ([time-budget.json](time-budget.json)). The components
  overlap and the profiler adds overhead, so the bars do not sum to the update time; the bounds are estimates, not
  checks.
- `04-validation.svg`: the paired 20,000-update training runs, two seeds per implementation
  ([validation.json](validation.json)): training loss per 100 updates, held-out loss per checkpoint, and the
  same-seed differences against the baseline's seed-to-seed difference (training loss as a 1,000-update rolling
  mean). `render.py` recomputes the criterion counts from the data and checks them against the recorded verdict.
- `../change-views/module-map.svg`: the changed files grouped by area.
