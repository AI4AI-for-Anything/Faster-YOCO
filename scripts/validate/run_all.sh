#!/usr/bin/env bash
# The four arms of the paired training validation, each followed by its held-out evaluation, then the summary and
# the verdict. Arms run back to back on all GPUs of one host (the recorded run put opt_s2 on a second host of the
# same model in parallel; run it there with ARMS=opt_s2 if you do the same).
# Prerequisites: the held-out set (see README.md in this directory) at $VAL_DIR/heldout/heldout.pt.
# environment: DATA_DIR, VAL_DIR, REF_TREE (unmodified upstream checkout); optional ARMS (default: opt ref ref_s2
#   opt_s2) and everything run_arm.sh / eval_arm.sh accept.
set -u
: "${DATA_DIR:?Set DATA_DIR}" "${VAL_DIR:?Set VAL_DIR}" "${REF_TREE:?Set REF_TREE}"
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
CFG=$REPO/optimizations/fused_adam_qkvg.json
for ARM in ${ARMS:-opt ref ref_s2 opt_s2}; do
    case $ARM in
        opt)    bash "$HERE/run_arm.sh" opt    "$REPO"     1 "$CFG" ;;
        opt_s2) bash "$HERE/run_arm.sh" opt_s2 "$REPO"     2 "$CFG" ;;
        ref)    bash "$HERE/run_arm.sh" ref    "$REF_TREE" 1 none ;;
        ref_s2) bash "$HERE/run_arm.sh" ref_s2 "$REF_TREE" 2 none ;;
        *) echo "unknown arm $ARM" >&2; exit 2 ;;
    esac
    echo "$(date '+%F %T') $ARM training rc=$?"
    bash "$HERE/eval_arm.sh" "$ARM"; echo "$(date '+%F %T') $ARM evaluation rc=$?"
done
"${PYTHON:-python}" "$HERE/analyze.py" "$VAL_DIR"
"${PYTHON:-python}" "$HERE/verdict.py" "$VAL_DIR"
