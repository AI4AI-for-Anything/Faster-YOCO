# Training and reproduction

The optimizations are environment switches on the upstream trainer (`yoco/train.py`, which calls fairseq's
`cli_main`). Every `YOCO_*` switch is off unless set; with none set, the code takes the upstream path (by
construction: the switched-off branches call the upstream code; this configuration was not run through the check
separately). The measured configuration is [`scripts/configs/faster.env`](../scripts/configs/faster.env) (identical
values in [`.harness/config.json`](../.harness/config.json) and
[`optimizations/fused_adam_qkvg.json`](../optimizations/fused_adam_qkvg.json)).

## Environment

The checks and the validation ran on 8 NVIDIA H100 GPUs (sm_90, 80 GB, NVLink between all eight), driver
570.133.20, with:

| Package | Version |
|---|---|
| Python | 3.10.14 |
| PyTorch | 2.3.1+cu121 |
| Triton | 2.3.1 |
| NCCL / cuBLAS / cuDNN | 2.20.5 / 12.1.3.1 / 8.9.2.26 (PyTorch wheels) |
| fairseq | [sunyt32/fairseq](https://github.com/sunyt32/fairseq) branch `zero_fs` at `e872403f` (with its Megatron submodule at `b5d1da12`), the fork named in upstream `requirements.txt` |
| apex | [NVIDIA/apex](https://github.com/NVIDIA/apex) at `70018365`, built with `--cpp_ext --cuda_ext` for sm_90 against the CUDA 12.1.1 toolkit |
| flash-attn | 2.6.3 (official prebuilt wheel for torch 2.3, CUDA 12, cxx11abi FALSE, cp310) |
| infinibatch / torchscale | [shumingma/infinibatch](https://github.com/shumingma/infinibatch) at `cf9a5cb9` / [microsoft/torchscale](https://github.com/microsoft/torchscale) at `4d1e0e82` |
| fairscale / numpy / tiktoken | 0.4.0 / 1.23.0 / 0.14.0 (`cl100k_base`) |

PyTorch 2.4.0 with Triton 3.0 was tried first and rejected: Triton 3.0 refuses to compile YOCO's own gated-retention
recurrence kernel. `yoco/models/decoder/ce_reduce.py` loads `libcudart` from the `nvidia-cuda-runtime-cu12` wheel
that comes with PyTorch. One way to build a matching environment (the measured one was built from the same pinned
sources; these commands have not been run as written):

```bash
python3.10 -m venv yoco-env && source yoco-env/bin/activate
pip install "pip==24.0"        # fairseq pins omegaconf<2.1, whose metadata newer pip rejects
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.23.0 cython wheel
pip install --no-build-isolation fairscale==0.4.0 tiktoken sentencepiece ninja boto3 iopath \
  "git+https://github.com/sunyt32/fairseq.git@e872403f976bdeb4555fdf7e45fa3c582afe98df" \
  "git+https://github.com/shumingma/infinibatch.git@cf9a5cb9c7503298ce95f1fee66ad627c647246f" \
  "git+https://github.com/microsoft/torchscale.git@4d1e0e82e5adf86dd424f1463192635b73fc8efc"
git clone https://github.com/NVIDIA/apex && git -C apex checkout 70018365c8add3e46574b897149db5d6dd21ef5c
(cd apex && TORCH_CUDA_ARCH_LIST=9.0 pip install -v --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./)   # CUDA 12.1 toolkit
pip install flash-attn==2.6.3 --no-build-isolation
```

## Data

Upstream trains on its own data. The checks and the validation used a public corpus in the layout of the upstream
README, built by [`scripts/data/build_data.py`](../scripts/data/build_data.py):

- `fineweb_edu` (weight 0.8): [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  revision `87f09149`, `sample/10BT/000_00000.parquet`, 726,000 documents in 73 shards;
- `codeparrot` (weight 0.2): [codeparrot/codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean)
  revision `35a59fb0`, `file-000000000001.json.gz`, 100,000 Python files in 10 shards.

About 975M `cl100k_base` tokens (732M + 243M), about 59,000 updates at 16,384 tokens per update. `json/valid.json`
is an empty list: the fairseq fork opens the validation split's JSON even with `--disable-validation`. No data,
checkpoints or per-file listings are included in this repository.

## Launch

```bash
export DATA_DIR=/path/to/Data
export OUTPUT_DIR=/path/to/output
bash scripts/train_faster_yoco_3b.sh --max-update 20000    # NPROC=8 by default
```

The script sources the switches and runs upstream `scripts/train.sh` with every argument unchanged
(`--tokens-per-sample 2048 --batch-size 1 --update-freq 1 --bf16`, Adam (0.9, 0.95), eps 1e-6, weight decay 0.05,
clip norm 2.0, learning rate 1.5e-4, `polynomial_decay` with 50 warm-up updates and `--total-num-update 300000`,
`--batch-read-ahead 1000`, `--no-save`, ...) except for these recorded deviations:

1. the data path;
2. `--arch yoco_3b` instead of `yoco_base`, which is registered nowhere in the repository (`yoco_3b` is the only
   registered YOCO architecture: dim 3072, 26 layers, 13 gated-retention self-decoder layers and 13 cross-decoder
   layers sharing one KV cache, SwiGLU 8192, 24 heads, 8 KV heads, vocabulary 100,288, 3,445,303,296 parameters);
3. 8 processes on one node (upstream's example uses one); per-GPU load is unchanged, global batch 8 × 2,048 tokens;
4. `--seed 1` written out (fairseq's default) and `--save-dir` (fairseq checks it is writable even with `--no-save`);
5. `--zero-sharding os`: the exact upstream per-GPU setup runs out of memory on 80 GB GPUs in the first optimizer
   step (70.2 GiB allocated when Adam allocates its state). This is the fork's own ZeRO stage-1 option; model, data,
   sequence length, per-GPU batch, precision and optimizer math are unchanged.

The switches print nothing at start-up; record `env | grep '^YOCO_'` with each run. Before trusting a timing, check
the log for `skipping cudagraphs due to`, recompile-limit messages and `gradient overflow`; the check treats each as an
invalid run, and the measured runs had none.

**Checkpoints.** Upstream's recipe passes `--no-save`. Saving full fairseq checkpoints (with consolidation of the
sharded optimizer state) and resuming with the switches on were not tested. The validation runs saved model weights
only, through [`scripts/validate/train_wrapper.py`](../scripts/validate/train_wrapper.py).

## Switches

Rounds refer to the optimization log; the numerical verdict applies to the combined configuration (see
[validation](#validation)).

| Switch | What it does | Files (under `yoco/`) | Adopted | Numerics |
|---|---|---|---|---|
| `YOCO_ZERO1_FAST` | Under `--zero-sharding os`, the gradient norm is computed on the BF16 gradients with fairseq's own routine (exact FP32 upcast); only the rank's own shard is copied to FP32, scaled and stepped (same Adam call); the owner casts its shard to BF16 and sends BF16 instead of upstream's full FP32 broadcast. FP32 copies of shards a rank does not own are no longer kept. | `tasks/zero1_fast.py` | R01 | Alone: loss, Adam moments and BF16 weights bitwise identical in the quick check |
| `YOCO_LAYER_GRAPHS` | The 13 self- and 13 cross-decoder layers are captured once as CUDA graphs, forward and backward (`make_graphed_callables`), before DDP wraps the model; eval, no-grad and other shapes run eagerly. | `models/decoder/layer_graphs.py`, `models/decoder/yoco.py` | R02 | Graph replays not bitwise equal to eager kernels here |
| `YOCO_ZERO1_ALLGATHER`, `YOCO_ZERO1_PARAM_VIEWS` | One in-place `all_gather_into_tensor` over equal per-owner segments replaces 8 broadcasts; the BF16 parameters are views of that buffer, placed before DDP with fairscale OSS's greedy partition (checked against OSS at the first step). | `tasks/zero1_fast.py`, `models/decoder/layer_graphs.py` | R03 | Data movement only |
| `YOCO_COMPILE_NORM` | RMSNorm expression compiled with Inductor. Superseded by `YOCO_TRITON_NORM` when both are set; kept in the measured configuration. | `models/decoder/rms_norm.py` | R04 | Inductor fusion |
| `YOCO_PARAM_ALLREDUCE` | DDP's reducer stays idle; each gradient gets an asynchronous NCCL all-reduce (AVG) as soon as it is accumulated, waited for before clipping. Dividing by 8 (a power of two) equals DDP's pre-division. | `models/decoder/layer_graphs.py` | R05 | Different reduction grouping |
| `YOCO_TRAINER_NOSYNC` | fairseq's `train_step` without per-update host syncs: GPU-side normalizer, the non-finite and cross-rank grad-norm checks read back without blocking (raised at the latest one update later, before that update's optimizer step), pinned batch copy, logging from cached device constants (extended in R10). | `tasks/trainer_nosync.py` | R06, R10 | Same values; abort checks may fire one update later |
| `YOCO_GRAD_REDUCE_OWNER` | Each gradient is reduced (AVG) only to the rank owning its optimizer state (half the traffic of an all-reduce); the clip norm sums the owners' squared norms with one scalar all-reduce. Needs the three switches above. | `models/decoder/layer_graphs.py`, `tasks/zero1_fast.py`, `models/decoder/ce_reduce.py` | R07 | Norm summation order |
| `YOCO_TRITON_NORM` | Triton RMSNorm forward and backward: FP32 statistics, upstream's two BF16 roundings, weight gradient from per-program FP32 partials (no atomics). | `models/decoder/rms_triton.py`, `models/decoder/rms_norm.py` | R09 | Unit test: forward 99.9998% bitwise, backward within BF16 rounding (relative 2.8e-3) |
| `YOCO_OWNER_EMBED_GRAD` | The token-embedding gradient (zero outside this update's tokens; the embedding is untied) is formed on its owner from all ranks' token ids and output-gradient rows with the same `embedding_dense_backward`, × 1/8. | `models/decoder/layer_graphs.py` | R12 | FP32 row sums in another order |
| `YOCO_SUBLN_FUSED` | The weightless per-head RMSNorm (rows of 128) and the following head → token transpose in one Triton kernel each way, 32 rows per program. | `models/decoder/gate_retention.py`, `models/decoder/rms_triton.py` | R13 | Unit test: 99.998–99.9998% bitwise vs the R09 kernel |
| `YOCO_Z1_ONE_PASS` | The owned FP32 gradient is `float(g16) * factor` in one pass (the same numbers as copy-then-multiply); one foreach cast after Adam. | `tasks/zero1_fast.py` | R14 | Same values |
| `YOCO_COMPILE_GR` | Key scaling, gate cumsum, decay factors and the decayed operands around the gated-retention recurrence compiled with Inductor in two pieces; the products stay cuBLAS. | `models/decoder/kernel/gate_recurrent.py` | R15 | Unit test: relative 2.5e-3–3.5e-3 (BF16 rounding; FP32 kept between fused ops) |
| `YOCO_FUSED_CE` | For the default language-model loss: a Triton kernel reads each BF16 logit row once (FP32 running max and sum, loss = lse − x[target]); the backward writes the BF16 gradient directly. | `tasks/fused_ce.py` | R16 | Unit test: loss relative 2.2e-7, logit gradient 99.993% bitwise |
| `YOCO_CE_REDUCE` | Gradients go to their owner through the copy engines: CUDA IPC `cudaMemcpyAsync` pushes on a copy stream as each gradient is ready, a 1-element NCCL all-reduce as GPU-side barrier, then one FP32 sum of the eight slots in rank order on the owner, × 1/8, rounded once to BF16. From R19 the embedding rows and token ids take the same path. | `models/decoder/ce_reduce.py`, `models/decoder/layer_graphs.py` | R18, R19 | FP32 rank-order sum (unit test: bitwise equal to it) |
| `YOCO_FUSED_ADAM` | The shard's `multi_tensor_adam` is replaced by a Triton kernel with apex's Adam expression that also stores the BF16 copy of each updated master weight. | `tasks/fused_adam_cast.py`, `tasks/zero1_fast.py` | R21 | Moments bitwise equal to apex; about 1% of new master weights differ in the last FP32 bit (IEEE division and square root) |
| `YOCO_FUSE_QKVG` | In training, the q/k/v/g projections are one GEMM against the concatenated weights (copied each forward); one input-gradient and one weight-gradient GEMM backward. | `models/decoder/gate_retention.py` | R22 | FP32 accumulation over four products instead of three BF16 adds |
| `YOCO_Z1_SCALE_KERNEL` | The one-pass gradient scaling in a vectorized Triton kernel (same products). Needs `YOCO_Z1_ONE_PASS`. | `tasks/zero1_fast.py` | R25 | Same values |

"Unit test" figures come from standalone comparisons against the upstream expression during the optimization run;
those tests are not included.

Switches present in the code but **off** in the measured configuration:

| Switch | Round | Status |
|---|---|---|
| `YOCO_DDP_BUCKET_VIEW` | R04 | DDP `gradient_as_bucket_view`; part of the R04 configuration, replaced by `YOCO_PARAM_ALLREDUCE` from R05 |
| `YOCO_CE_ALLGATHER` | R18 | Parameter all-gather through the copy engines: slower than NCCL (about 170 vs 340 GB/s per GPU) |
| `YOCO_CE_PHASES` | R20 | Summing gradients during the backward: no gain |
| `YOCO_FUSE_ADDNORM` | R23 | Residual add fused into the following RMSNorm (bitwise equal): no gain |
| `YOCO_NORM_BWD_PPS` | R24 | RMSNorm backward programs per SM (default 4 kept): no gain |
| `YOCO_DIAG_DEFER_REDUCE` | — | Diagnostic only: launch the gradient reductions after the backward |

[`optimizations/`](../optimizations) keeps the switch set of every full-checked configuration (see
[history](#optimization-history)). Later rounds changed code under some of the same switch names (for example R10
extended `YOCO_TRAINER_NOSYNC` and R19 extended `YOCO_CE_REDUCE`), so an older switch set on the final commit is not
the exact code that was checked at that round; that is the commit listed in the history.

## Memory, throughput and start-up

| | Reference | Faster-YOCO |
|---|---:|---:|
| Peak allocated memory per GPU, final full check | 52.1 GiB | 56.4 GiB |
| Peak reserved memory per GPU, final full check | 53.7 GiB | 63.0 GiB |
| Peak allocated memory per GPU (max over ranks), 20,000-update run | 48.3 GiB | 55.4 GiB |
| Tokens per second, all 8 GPUs, final full check | 64,967 / 65,109 | 138,983 |
| Process start to first update, final full check | 74.2 / 74.9 s | 95.4 s |
| Time per update, 20,000-update run (excluding saves) | 0.2497 s | 0.1180 s |
| Wall clock, 20,000-update run | 5,345 s | 2,799 s |

The extra memory is mostly the copy-engine staging and output buffers of `YOCO_CE_REDUCE` (each owner keeps a slot
for every peer's gradients). The longer start-up comes from CUDA-graph capture and compilation with a cold cache.
The wall clocks include about 3 minutes of start-up and about 3 minutes of model saves per run, so the end-to-end
ratio is 1.91×.

## Benchmark protocol

Gate 1 of the [judge](judge.md) gives the speedup: reference → candidate → reference in fresh `torchrun` processes on
one host and in one session, cold compile caches, the real `train.py` entry, 50 warm-up updates and then three
100-update windows per run (below update 1,000, where the data loader stalls to fill its next shuffle block). A step
includes fetching the batch, forward, loss, backward, gradient reduction, clipping, the optimizer step, the parameter
exchange, the learning-rate update and logging bookkeeping; process start, model construction, compilation and
warm-up are excluded. Speedup = min(total-time ratio, median-update ratio) against the mean of the two reference
runs; it must exceed max(1.02, 1 + 2 × the gap between the reference runs).

Single-run `time` screens (the same windows, without the paired reference) were used to choose candidates. Every
speedup in the README and the history figure comes from a full check; the time budget comes from a profile and the
bounds are estimates.

## Validation

### Numerical check

The final configuration passed every gate of the [judge](judge.md#final-result): no exact-item failure, 5,123 of
38,464 recorded tensors bitwise identical (so the bitwise gate did not pass), and the statistical gate with three
seeds (paired mean differences of −0.75σ to +0.03σ against δ = 2σ). Each of the eight full checks in the history
passed the same way.

### Paired training runs

The check covers 200 updates. To see whether the optimized trainer drifts over a longer run, both implementations
were trained for 20,000 updates (328M tokens, less than one pass over either training file) with two seeds each:

| Arm | Code | Seed | Host | Wall (s) | Time / update (s) | Peak allocated (GiB) | Non-finite | Final training loss (bits) | Held-out mix @ 20k (bits) |
|---|---|---:|---|---:|---:|---:|---|---:|---:|
| ref | upstream | 1 | A | 5,345 | 0.2497 | 48.3 | none | 4.8981 | 4.9600 |
| ref_s2 | upstream | 2 | A | 5,321 | 0.2497 | 48.3 | none | 4.7428 | 4.9345 |
| opt | Faster-YOCO | 1 | A | 2,799 | 0.1180 | 55.4 | none | 4.9014 | 4.9647 |
| opt_s2 | Faster-YOCO | 2 | B | 2,804 | 0.1193 | 55.4 | none | 4.7510 | 4.9445 |

- **Recipe:** the measured recipe (same arguments as the timing check) with `--max-update 20000` and `--seed 1` or
  `--seed 2`, production mode (no deterministic algorithms). Within a seed, both implementations see the same data
  order. A and B are two hosts of the same model; `opt_s2` ran on host B in parallel with `ref_s2`, and its time per
  update is not compared.
- **Checkpoints:** a wrapper outside both trees replaces fairseq's checkpoint save with a model-only BF16 state dict
  on rank 0 every 2,000 updates (about 16 s per save, excluded from the time per update).
- **Held-out set:** the first 3,000 documents of fineweb-edu `sample/10BT/001_00000.parquet` (training used
  `000_00000`) and the first 800 of codeparrot-clean `file-000000000002.json.gz` (training used file 1), same dataset
  revisions; 4 fineweb documents whose exact text occurs in the training shards were dropped. Packed exactly like the
  training loader into 800 fineweb samples (1,638,400 target tokens) and 200 code samples (409,600).
- **Evaluation:** identical for every arm: the reference tree's model code in BF16 and eval mode, FP32
  log-softmax cross-entropy of every next token as fairseq's criterion computes it, in bits; mix = 0.8 fineweb + 0.2
  code. Training loss is fairseq's logged loss (bits per token).
- **Criterion** (defined before the seed-2 arms finished, not changed afterwards): Y = ref_s2 − ref, E1 = opt − ref,
  E2 = opt_s2 − ref_s2, S = opt_s2 − opt; a checkpoint is consistent if |E1|, |E2| and |S| are each at most |Y|.
  Training-loss rows use the mean of the 1,000 updates up to each checkpoint.

| Metric | Consistent checkpoints | E1 within | E2 within | S within | Final checkpoint |
|---|---:|---:|---:|---:|:---:|
| Held-out mix | 10 / 10 | 10 | 10 | 10 | ✓ |
| Held-out fineweb-edu | 10 / 10 | 10 | 10 | 10 | ✓ |
| Held-out code | 5 / 10 | 9 | 10 | 5 | ✓ |
| Training loss (1,000-update mean) | 7 / 10 | 10 | 10 | 7 | ✓ |

The held-out mix per checkpoint is in the [README](../README.md#validation); the two metrics with misses:

| Update | Code: Y | E1 | E2 | S | | Training: Y | E1 | E2 | S |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 2,000 | −0.0656 | +0.0316 | −0.0118 | **−0.1089** | | −0.1213 | −0.0065 | +0.0136 | −0.1013 |
| 4,000 | −0.1464 | +0.0154 | +0.0265 | −0.1353 | | −0.1580 | −0.0019 | +0.0068 | −0.1493 |
| 6,000 | −0.1496 | −0.0123 | −0.0069 | −0.1442 | | −0.0828 | −0.0009 | +0.0024 | −0.0795 |
| 8,000 | −0.0528 | +0.0117 | −0.0112 | **−0.0757** | | −0.0744 | −0.0027 | +0.0014 | −0.0703 |
| 10,000 | +0.1078 | +0.0193 | +0.0248 | **+0.1133** | | −0.0762 | +0.0044 | +0.0081 | −0.0724 |
| 12,000 | −0.0322 | +0.0155 | +0.0052 | **−0.0425** | | −0.0869 | +0.0025 | +0.0064 | −0.0830 |
| 14,000 | −0.0160 | **+0.0179** | +0.0054 | **−0.0284** | | +0.0854 | +0.0007 | +0.0056 | **+0.0903** |
| 16,000 | +0.0400 | +0.0139 | −0.0043 | +0.0218 | | +0.1414 | +0.0026 | +0.0040 | **+0.1428** |
| 18,000 | −0.0252 | −0.0202 | +0.0157 | +0.0107 | | +0.1349 | +0.0026 | +0.0058 | **+0.1380** |
| 20,000 | −0.0516 | −0.0002 | +0.0144 | −0.0369 | | −0.1553 | +0.0033 | +0.0081 | −0.1505 |

Bits per token; bold = exceeds |Y|. All data are in
[`performance-views/validation.json`](performance-views/validation.json).

Reading the misses: S − Y = E2 − E1 by definition. Within a seed both implementations see the same data, so S tracks
Y, and |S| > |Y| whenever E2 − E1 has the same sign as Y. All eight misses are in S; at code 14,000, E1 also exceeds
|Y| (0.0179 against 0.0160). The code set has 0.41M tokens and is the noisiest metric. Same-seed training curves correlate above 0.9999 over the 200 per-100-update means; their absolute
differences average 0.0042 bits (seed 1) and 0.0073 bits (seed 2), against 0.13 bits between the reference's two
seeds.

**Limits of this evidence.**

- Faster-YOCO's held-out loss is slightly but consistently higher than the reference's: on the mix, E1 averages
  +0.0024 bits (8 of 10 checkpoints positive, +0.0047 at the end) and E2 +0.0082 bits (10 of 10, +0.0100 at the end).
  This is at most 0.2% of the loss and smaller than |Y| at every checkpoint (by a factor of 2.6 or more), but two
  seeds cannot tell whether a systematic offset of this size exists.
- E2 compares runs on two hosts that are known not to reproduce each other bit for bit
  ([judge](judge.md#host-portability)), so it also contains any host effect.
- One run per seed and arm, 20,000 of the recipe's 300,000 updates, no downstream evaluation.

The scripts are in [`scripts/validate/`](../scripts/validate).

## Optimization history

All rows are full checks of cumulative configurations by the same judge on the same host (data in
[`performance-views/history.json`](performance-views/history.json)). Times are mean seconds per update.

| Round | Commit | Change | Reference (s) | Faster-YOCO (s) | Speedup |
|---|---|---|---:|---:|---:|
| R03 | `da94f9b` | Sharded ZeRO-1 work, layer CUDA graphs, one in-place all-gather (R01–R03) | 0.2508 | 0.1802 | 1.3865× |
| R04 | `00c33a0` | Compiled RMSNorm + DDP bucket views | 0.2511 | 0.1577 | 1.5834× |
| R07 | `c9cbd0e` | Per-parameter reduction to the optimizer-state owner, no host syncs in `train_step` (R05–R07) | 0.2520 | 0.1460 | 1.7123× |
| R09 | `5fc4184` | Triton RMSNorm | 0.2519 | 0.1432 | 1.7447× |
| R17 | `bfd2fd6` | R10–R16 combined: logging without syncs, owner embedding gradient, fused subln, one-pass scaling, compiled retention, Triton loss | 0.2508 | 0.1303 | 1.9199× |
| R18 | `348f510` | Gradient reduction through the copy engines | 0.2512 | 0.1221 | 2.0501× |
| R19 | `7d1be84` | Embedding gradient rows through the copy engines | 0.2514 | 0.1208 | 2.0723× |
| R25 | `026c4ea` | Fused Adam + BF16 cast, Triton gradient scaling, fused q/k/v/g GEMM (R21, R22, R25) | 0.2519 | 0.1179 | 2.1247× |

The speedup column is the smaller of the median and total ratios, so it can be lower than the ratio of the two time
columns. The final commit is the one recorded as `f4f2819` in the optimization log (commit messages were edited
afterwards to drop host identifiers; trees unchanged).

Per round, from single-run screens (seconds per update; each against the best version at the time):

| Round | Change | Screen | Result |
|---|---|---|---|
| R01 | Sharded ZeRO-1 FP32 work, BF16 exchange | screened on the first host only | adopted |
| R02 | CUDA graphs of the decoder layers | 0.1877 | adopted |
| R03 | One in-place all-gather; parameters as views of its buffer | 0.1856 → 0.1800 | adopted |
| R04 | Inductor RMSNorm; DDP bucket views | 0.1621 / 0.1753, together 0.1575 | adopted |
| R05 | Per-parameter asynchronous all-reduce | 0.1563 | adopted |
| R06 | `train_step` without host syncs; NCCL channel and NVLS settings screened | 0.1544; NCCL settings no gain | adopted |
| R07 | Reduce each gradient to its owner | 0.1459 | adopted |
| R08 | Pinned statistics all-reduce, foreach gradient copy | 0.1464 | no gain (kept, values unchanged) |
| R09 | Triton RMSNorm | 0.1433 | adopted |
| R10 | End-of-update logging without host syncs | 0.1393 | adopted |
| R11 | NCCL stream priority and CTA limits for the reductions and the all-gather | 0.1393–0.1749 | no gain, removed |
| R12 | Embedding gradient on its owner | 0.1385 | adopted |
| R13 | Fused per-head subln | 0.1372 | adopted |
| R14 | One-pass gradient scaling | 0.1359 | adopted |
| R15 | Compiled gate-retention elementwise work | 0.1356 vs 0.1385 | adopted |
| R16 | Triton cross-entropy | 0.1357 vs 0.1385 | adopted |
| R17 | R10–R16 combined | 0.1303 | full check |
| R18 | Copy-engine reduction; copy-engine all-gather | 0.1220; all-gather 0.1473 | reduction adopted, all-gather off |
| R19 | Embedding rows through the copy engines | 0.1208 | adopted |
| R20 | Summing gradients during the backward | 0.1212 | no gain, off |
| R21 | Adam fused with the BF16 cast | 0.1191 | adopted |
| R22 | q/k/v/g projections as one GEMM | 0.1204 | adopted in the combination |
| R23 | Residual add fused into RMSNorm | 0.1207 | no gain, off |
| R24 | Fewer RMSNorm backward programs | 0.1208 | no gain, off |
| R25 | Vectorized gradient scaling; combination | 0.1198; combined 0.1179 | adopted, full check |

R02 and R03 were screened on the second host before its judge was frozen; R01 on the first host.

## Why it stopped here

The run targeted 3×: at most 84.0 ms per update against the 251.9 ms reference of the final check. It reached
2.1247× (117.9 ms). The owner stopped it after round 28 and asked for the training validation instead.

![Time budget of the final configuration](performance-views/03-time-budget.svg)

Lower bounds per update and GPU, from the final configuration's profile and measurements on this hardware
(BF16 dense GEMM 750 TFLOP/s for 8192³, HBM copy 3.04 TB/s, NVLink 450 GB/s per direction and GPU):

- **GEMM:** 40.2 TFLOP per GPU and update (6.28 GFLOP per token in the forward, three times that for forward and
  backward over 2,048 tokens, plus about 1.6 TFLOP of attention and retention products) need 53.6 ms at 750 TFLOP/s and 57–59 ms at the rates measured for
  these shapes (551–732 TFLOP/s). Profiled: 65.3 ms.
- **Parameter all-gather:** after the owners' Adam steps every GPU must receive 7/8 of the 6.9 GB of BF16 parameters
  (6.0 GB) before anything reads them. The check observes the BF16 parameters right after each update, and the same
  code path must run in timing and in checking, so the exchange cannot simply be deferred into the next forward.
  13.4 ms at the NVLink ingress bound; NCCL takes 17.35 ms (about 345 GB/s); copy-engine pulls were slower (R18).
- **Everything else:** other kernels 18.0 ms (bound about 8–10), launch gaps 3.6 ms (about 2), the gradient-reduction
  tail 3.6 ms (about 1.5), clip + Adam + casts 5.3 ms exposed (Adam is at the HBM bound; about 1.5 if pipelined with
  the all-gather).

The sum of the bounds is 80–91 ms (2.75–3.1×) and would need every component at its hardware limit at once; GEMM
and the all-gather alone take 67.0–76.4 ms. After round 19 the remaining credible items were estimated to reach
about 108–112 ms (about 2.25–2.3×); of those tried afterwards, the fused q/k/v/g GEMM gave 0.3 ms and the others
gave no gain. FP8, INT8 and FP16 were excluded by the task rules.

### Directions tried and not adopted

| Rounds | Direction | Result |
|---|---|---|
| R06, R11, R26 | NCCL settings: NVLS channels, NVLS off, channel caps, stream priority, CTA limits, protocol, buffer size | No gain; fewer CTAs starve the reductions, more slow the GEMMs; 16 MB buffers ran out of memory |
| R18 | Parameter all-gather through the copy engines | 0.1473 vs 0.1303 s: about 170 GB/s vs about 340 GB/s for NCCL |
| R20 | Summing gradients on side streams during the backward | 0.1212 vs 0.1208 s: the sums compete with the backward for SMs and HBM |
| R23, R24 | Residual add fused into RMSNorm (bitwise equal); fewer norm-backward programs | No measurable gain |
| R26 | All-gather split in two so that the first part overlaps Adam | 0.1178 / 0.1184 vs 0.1179 s: no gain; not included |
| R27 | Clip norm fused into the owner's slot sum | 0.1175 vs 0.1179 s in a screen and a quick check; never full-checked; not included |
| R28 | Parameter exchange overlapped with the next forward (separate forward stream, chunked send/receive in forward order, owner-served embedding rows) | A feasibility test hid about half of the transfer. The first version had a real race (stale embedding rows, found by the quick check and fixed); the next ran out of memory in the row service. Unfinished at the stop; not included |
