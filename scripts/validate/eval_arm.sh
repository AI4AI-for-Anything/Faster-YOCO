#!/usr/bin/env bash
# Held-out loss of every saved checkpoint of one arm, with identical evaluation code for all arms (eval_ckpt.py runs
# the REFERENCE tree's model code); then, unless KEEP_ALL_CKPTS=1, keeps only the arm's last checkpoint.
# usage: eval_arm.sh <arm name>
# environment: VAL_DIR (output root of run_arm.sh), REF_TREE (unmodified upstream checkout);
#   optional: HELDOUT (default $VAL_DIR/heldout/heldout.pt, from build_heldout.py), NPROC (8), MASTER_PORT,
#   TORCHRUN (torchrun), KEEP_ALL_CKPTS
set -u
ARM=$1
: "${VAL_DIR:?Set VAL_DIR}" "${REF_TREE:?Set REF_TREE to the unmodified upstream checkout}"
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REF_TREE=$(cd "$REF_TREE" && pwd)
HELDOUT=${HELDOUT:-$VAL_DIR/heldout/heldout.pt}
OUT=$VAL_DIR/$ARM; mkdir -p "$OUT"/tmp_eval/caches/{inductor,triton,cuda,xdg,mpl}
export TMPDIR=$OUT/tmp_eval TMP=$OUT/tmp_eval TEMP=$OUT/tmp_eval
export TORCHINDUCTOR_CACHE_DIR=$OUT/tmp_eval/caches/inductor TRITON_CACHE_DIR=$OUT/tmp_eval/caches/triton
export CUDA_CACHE_PATH=$OUT/tmp_eval/caches/cuda XDG_CACHE_HOME=$OUT/tmp_eval/caches/xdg MPLCONFIGDIR=$OUT/tmp_eval/caches/mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONNOUSERSITE=1
unset PYTHONPATH
for v in $(env | grep -o '^YOCO_[A-Z0-9_]*'); do unset "$v"; done      # evaluation runs the reference code path
echo "$(date '+%F %T') eval start $ARM" >> "$OUT/run.log"
"${TORCHRUN:-torchrun}" --nnodes=1 --nproc-per-node="${NPROC:-8}" --master-addr 127.0.0.1 \
    --master-port "${MASTER_PORT:-$((20000 + RANDOM % 20000))}" \
    "$HERE/eval_ckpt.py" "$REF_TREE" "$HELDOUT" "$OUT/eval.json" $(ls "$OUT"/ckpt/model_*.pt | sort) > "$OUT/eval.log" 2>&1
RC=$?
echo "$(date '+%F %T') eval end $ARM rc=$RC" >> "$OUT/run.log"
if [ $RC -eq 0 ] && [ "${KEEP_ALL_CKPTS:-0}" != 1 ]; then
    LAST=$(ls "$OUT"/ckpt/model_*.pt | sort | tail -1)
    for f in "$OUT"/ckpt/model_*.pt; do [ "$f" != "$LAST" ] && rm -f "$f"; done
fi
exit $RC
