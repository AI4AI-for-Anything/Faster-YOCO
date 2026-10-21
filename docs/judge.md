# The judge

Every speedup in this repository comes from a full check by a frozen judge built with the ResearchBoost-Agent method
before any optimization was judged. The judge itself (benchmark, numerical adapter, frozen reference records and
trajectories) is a separate artifact and is not included; this page describes what it checks and how it was accepted.

## What is compared

- **Reference:** the unmodified upstream tree (branch `upstream`, the `YOCO/` directory of microsoft/unilm at
  `f4f7b27`) with no switch file.
- **Candidate:** a tree plus a switch file (`{"environment": {...}}`, see [`optimizations/`](../optimizations)).
  Switch files may not set variables the judge manages (temporary and compile-cache directories, visible devices,
  `PYTHONPATH`, offline switches, the tiktoken cache and similar).
- **Recipe:** the upstream `scripts/train.sh` arguments with the deviations listed in
  [training.md](training.md#launch): `--arch yoco_3b`, 8 data-parallel ranks on one node, `--zero-sharding os`,
  `--seed 1`, `--save-dir`, `--max-update`. One environment serves both trees (PyTorch 2.3.1 + CUDA 12.1).
- **Frozen inputs:** 132 files are hashed at the start and end of every check: the training shards and their JSON
  lists, the data manifest, the tiktoken cache and the environment files that define the training semantics (fairseq's
  `fairseq_cli/train.py`, trainer, optimizer, criterion, distributed, checkpoint, iterator and logging modules,
  fairscale's OSS, infinibatch's iterators, apex's fused Adam, flash-attn's interface). The tree's source digest is
  also taken at start and end.

## Gates

The gates run in order and stop at the first failure.

0. **Structure.** The entry file and the parts of fairseq's `cli_main`, `main` and `train` outside the per-update
   loop body must have the same syntax tree as the reference, read from the live functions after the tree's modules
   are imported (a vendored or patched copy of fairseq's entry is caught too); otherwise ERROR. The tree's data loader,
   called as the trainer calls it, must yield the frozen first three batches on each of the eight shards; otherwise
   INCORRECT.
1. **Benefit.** Reference → candidate → reference through the real entry point, each in a fresh `torchrun` process
   with cold Triton/Inductor/CUDA caches, on the host and in the session of the check. Each run: 50 warm-up updates,
   then three windows of 100 updates (an update is one step of the shard-local Adam inside fairscale's OSS, timed on
   the host). Speedup = min(mean reference total / candidate total, mean reference median update / candidate
   median update). The candidate must exceed max(1.02, 1 + 2 × the relative gap between the two reference runs);
   otherwise UNAVAILABLE (no benefit). A check is INVALID and rerun if the two reference medians differ by more than
   3%, if any run's windows differ by more than 30%, or if a log shows a recompile limit, skipped CUDA graphs or a
   skipped update (gradient overflow). Timing is never compared with a stored number.
2. **Bitwise.** In deterministic mode, one 8-rank process runs the tree's own loop body for updates 0 and 1 (learning
   rate 0 and 3e-6), then updates 2–59, then records updates 60 and 61 (learning rate ≈ 1.5e-4): 4 updates × 8 ranks ×
   1,202 tensors = **38,464 tensors** (loss, the FP32 gradients Adam consumes, master weights before and after, both
   Adam moments, the BF16 model after the update; large tensors by exact hash, norm and a 4,096-wide sketch).
   **Exact items** must hold or the candidate is INCORRECT: phase, CPU and CUDA RNG fingerprints before and after each
   update, forward and criterion call counts, update counter, scheduled learning rate, Adam hyper-parameters and step
   count, dtypes of master weights, gradients and moments, per-rank and cumulative batch digests, sample size, config
   and trainable-parameter digests, tensor names and shapes, finiteness and the cross-rank pattern. If every tensor
   is bitwise identical, the check passes here (tier N0).
3. **Statistical.** Otherwise paired trajectories decide: 200 updates from scratch in deterministic mode, one process
   per seed. A seed multiplies each initial FP32 weight by (1 + 2⁻⁶ z), z ~ N(0, 1) from a private generator; data
   order, initialization and every other random draw stay those of `--seed 1`. Endpoints are the mean loss per token
   (nats, averaged over the eight ranks) over updates [5, 15), [15, 25), [25, 35), [35, 50), [50, 100) and
   [100, 200). The tolerance is δ = 2σ per endpoint, σ being the reference's own seed-to-seed standard deviation over
   six seeds, frozen before any candidate was seen. With three seeds: PASS if every endpoint's mean paired difference
   is within δ, INCORRECT if any exceeds δ with paired-t p < 0.05, otherwise extend to six seeds.

The 50-update warm-up of the recipe is where a change of training semantics shows most clearly against this spread;
the recorded updates 60–61 and the trajectory windows after update 50 cover the full learning rate.

| Window (updates) | 5–15 | 15–25 | 25–35 | 35–50 | 50–100 | 100–200 |
|---|---:|---:|---:|---:|---:|---:|
| δ (nats) | 0.002084 | 0.001280 | 0.000838 | 0.000923 | 0.008496 | 0.019017 |

Known limit: by updates 100–200 paired runs have decorrelated, so a legitimate candidate has a small chance of a false
INCORRECT in that window. It was kept because the reference's own non-deterministic runs pass.

## Acceptance

Before any candidate was judged, the judge had to give the expected verdict on known cases:

| Case | Expected | Result |
|---|---|---|
| Numerics self-test (12 cases: unmodified records and 11 mutations) | all as expected | 12 / 12 |
| Unmodified tree, quick check | bitwise PASS | PASS, N0, 38,464 / 38,464 bitwise |
| Unmodified tree, full check | no benefit | UNAVAILABLE: 1.0015× against a bar of 1.02 |
| Entry enables TF32 before `cli_main` | ERROR | ERROR at gate 0 |
| Weight decay set to 0 | INCORRECT | INCORRECT (exact item: Adam hyper-parameters) |
| Final RMSNorm weight never updated | INCORRECT | INCORRECT (exact items) |
| Loss gradient of the second half of each sequence dropped | INCORRECT | INCORRECT at gate 3, all six endpoints (22–314 σ) |
| RMSNorm `x * rsqrt(m + eps)` → `x / sqrt(m + eps)` | PASS | PASS at gate 3 (equivalent) |
| Control arm: learning rate × 1.02 | INCORRECT | INCORRECT at gate 3: −11.8, −26.4, −42.6 and −4.3 σ in the four warm-up windows |

The judge was then frozen (hashes of every judge file and frozen input recorded). Later, one bench bug was repaired
with a written reason before any candidate was judged under it: a data-loader probe could hang on its prefetch
process after printing its result; it now exits right after printing. The records, thresholds and δ were unaffected,
and the whole acceptance was rerun under the repaired judge, with the results above.

## Host portability

The judge was first formulated and frozen on one L20Z host. On a second host of the same model, with hardware,
VBIOS, driver and every library binary identical by sha256, the unmodified reference did not reproduce the frozen
records: 5,157 of 38,464 tensors were bitwise identical, the forward losses of the first two updates matched and
the gradients after the all-reduce did not. The same judge code was therefore re-recorded on the second host with
the same procedure (records, thresholds, six reference seeds, δ, acceptance) and frozen again. All full checks in
this repository ran on that second host against its own records. The cause was not isolated; the large BF16
gradient all-reduces used NCCL's NVLS algorithm (reduction inside the NVSwitch) on both hosts, and a chassis-level
difference there is a plausible but unverified explanation.

Bitwise records are therefore specific to a host. On a new host, first check that the unmodified reference
reproduces its own records bit for bit before comparing anything with them.

## Final result

Commit `026c4ea` with [`scripts/configs/faster.env`](../scripts/configs/faster.env):

| Gate | Result |
|---|---|
| 0, structure | pass; loader digests equal; 132 / 132 inputs unchanged at start and end |
| 1, benefit | reference 0.2522 / 0.2516 s per update (medians 0.2498 / 0.2495 s), candidate 0.1179 s (median 0.1175 s); 2.1369× by total, **2.1247×** by median; reference gap 0.22%, bar 1.02 |
| 2, bitwise | 5,123 / 38,464 tensors bitwise; 0 exact-item failures (tier N1) |
| 3, statistical | 3 seeds, all six endpoints within δ: equivalent |
| Verdict | **PASS** |

The full check took 2.53 GPU-hours (8 GPUs). See [the per-endpoint results](../README.md#validation) and the
[full-check history](performance-views/history.json).
