#!/usr/bin/env bash
# Faster-YOCO: YOCO-3B pretraining with the upstream scripts/train.sh arguments, the recorded deviations and the
# measured optimization switches.
# Usage:
#   export DATA_DIR=/path/to/Data       # README layout: json/train.json, json/<source>.json, shard/<source>/*.jsonl
#   export OUTPUT_DIR=/path/to/output   # logs; --save-dir (fairseq checks that it is writable; --no-save is kept)
#   bash scripts/train_faster_yoco_3b.sh [extra train.py arguments, e.g. --max-update 20000]
# Deviations from upstream scripts/train.sh (docs/training.md): --arch yoco_3b instead of the unregistered yoco_base,
# 8 processes on one node, --zero-sharding os (the exact upstream setup runs out of memory on 80 GB GPUs),
# --seed 1 written out, --save-dir. NPROC (default 8) sets the GPU count; the 2.1247x figure and every check used 8.
# REFERENCE=1 runs this tree with no switch set (the upstream code path); the timing and validation references used
# the unmodified upstream tree (branch `upstream`, e.g. `git worktree add ../yoco-upstream upstream`).
set -euo pipefail
: "${DATA_DIR:?Set DATA_DIR to the training data root that contains json/train.json}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the output directory}"
NPROC=${NPROC:-8}
[[ "$NPROC" =~ ^[1-9][0-9]*$ ]] || { echo 'NPROC must be a positive integer.' >&2; exit 2; }
[[ -f "$DATA_DIR/json/train.json" ]] || { echo "Missing $DATA_DIR/json/train.json" >&2; exit 2; }
DATA_DIR=$(cd "$DATA_DIR" && pwd)
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ "${REFERENCE:-0}" != 1 ]]; then
  source "$REPO_ROOT/scripts/configs/faster.env"
fi
cd "$REPO_ROOT/yoco"
torchrun --nproc-per-node="$NPROC" --master-addr 127.0.0.1 --master-port "${MASTER_PORT:-29501}" train.py "$DATA_DIR" \
    --save-interval-updates 5000 \
    --no-epoch-checkpoints \
    --arch yoco_3b \
    --criterion cross_entropy \
    --task gpt \
    --tokens-per-sample 2048 \
    --tokenizer-pad-to-multiple 8 \
    --pad-to-max-len \
    --optimizer adam --adam-betas "(0.9, 0.95)" \
    --adam-eps 1e-06 \
    --clip-norm 2.0 \
    --lr 0.00015 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 50 \
    --weight-decay 0.05 \
    --batch-size 1 \
    --model-parallel-size 1 \
    --update-freq 1 \
    --batch-read-ahead 1000 \
    --total-num-update 300000 \
    --log-format simple --log-interval 10 --disable-validation \
    --tiktoken-model cl100k_base \
    --no-save \
    --bf16 \
    --zero-sharding os \
    --seed 1 \
    --save-dir "$OUTPUT_DIR/checkpoints" \
    "$@" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
exit "${PIPESTATUS[0]}"
