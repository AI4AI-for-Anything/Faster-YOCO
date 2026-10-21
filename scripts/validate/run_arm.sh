#!/usr/bin/env bash
# One arm of the paired training validation: the measured recipe (scripts/train_faster_yoco_3b.sh) for MAXU updates
# with model-only checkpoints every SAVEINT updates (train_wrapper.py), in production mode (no deterministic
# algorithms), fresh compile caches per arm.
# usage: run_arm.sh <arm name> <tree> <seed> <switch json | none>
#   <tree>: the checkout whose yoco/train.py runs: this repository for the Faster-YOCO arms, with the switch file
#           optimizations/fused_adam_qkvg.json; the unmodified upstream code for the reference arms
#           (git worktree add ../yoco-upstream upstream), with "none"
# environment: DATA_DIR (training data root, README layout), VAL_DIR (output root; <VAL_DIR>/<arm>/ is created);
#   optional: MAXU (20000), SAVEINT (2000), NPROC (8), MASTER_PORT (random), PYTHON (python), TORCHRUN (torchrun)
set -u
ARM=$1; TREE=$(cd "$2" && pwd); SEED=$3; CFG=$4
: "${DATA_DIR:?Set DATA_DIR}" "${VAL_DIR:?Set VAL_DIR}"
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python}
OUT=$VAL_DIR/$ARM; mkdir -p "$OUT"/tmp/caches/{inductor,triton,cuda,xdg,mpl} "$OUT/ckpt"
export TMPDIR=$OUT/tmp TMP=$OUT/tmp TEMP=$OUT/tmp
export TORCHINDUCTOR_CACHE_DIR=$OUT/tmp/caches/inductor TRITON_CACHE_DIR=$OUT/tmp/caches/triton
export CUDA_CACHE_PATH=$OUT/tmp/caches/cuda XDG_CACHE_HOME=$OUT/tmp/caches/xdg MPLCONFIGDIR=$OUT/tmp/caches/mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled WANDB_DISABLED=true
export PYTHONNOUSERSITE=1
unset PYTHONPATH
for v in $(env | grep -o '^YOCO_[A-Z0-9_]*'); do unset "$v"; done      # only the switch file decides
if [ "$CFG" != none ]; then
    eval "$("$PY" -c "import json,shlex,sys; d=json.load(open(sys.argv[1]))['environment']; print(' '.join('export %s=%s' % (k, shlex.quote(v)) for k, v in d.items()))" "$CFG")"
fi
env | grep '^YOCO_' | sort > "$OUT/switches.txt"
ARGS=("$DATA_DIR"
  --save-interval-updates "${SAVEINT:-2000}" --no-epoch-checkpoints --arch yoco_3b --criterion cross_entropy --task gpt
  --tokens-per-sample 2048 --tokenizer-pad-to-multiple 8 --pad-to-max-len
  --optimizer adam --adam-betas '(0.9, 0.95)' --adam-eps 1e-06 --clip-norm 2.0 --lr 0.00015
  --lr-scheduler polynomial_decay --warmup-updates 50 --weight-decay 0.05 --batch-size 1 --model-parallel-size 1
  --update-freq 1 --batch-read-ahead 1000 --total-num-update 300000
  --log-format simple --log-interval 10 --disable-validation --tiktoken-model cl100k_base --no-save --bf16
  --zero-sharding os --seed "$SEED" --save-dir "$OUT/save" --max-update "${MAXU:-20000}")
echo "$(date '+%F %T') start $ARM tree=$TREE seed=$SEED cfg=$CFG" >> "$OUT/run.log"
git -C "$TREE" log --oneline -1 >> "$OUT/run.log" 2>/dev/null
T0=$(date +%s)
"${TORCHRUN:-torchrun}" --nnodes=1 --nproc-per-node="${NPROC:-8}" --master-addr 127.0.0.1 \
    --master-port "${MASTER_PORT:-$((20000 + RANDOM % 20000))}" \
    "$HERE/train_wrapper.py" "$TREE" "$OUT/ckpt" "${ARGS[@]}" > "$OUT/train.log" 2>&1
RC=$?
echo "$(date '+%F %T') end $ARM rc=$RC wall_s=$(( $(date +%s) - T0 ))" >> "$OUT/run.log"
exit $RC
