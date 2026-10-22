# Paired training validation

These scripts rerun the comparison in the [README](../../README.md#validation): Faster-YOCO and the unmodified
upstream code, two seeds each, 20,000 updates of the measured recipe, held-out loss every 2,000 updates. They are the
scripts of the recorded run with machine-specific paths replaced by arguments and environment variables; they have
not been rerun in this form. Each arm needs 8 GPUs (about 47 minutes for Faster-YOCO and 89 minutes for the reference
on 8 Hopper GPUs, plus about 13 minutes of evaluation) and about 7 GB per saved checkpoint.

| File | Role |
|---|---|
| `extract_heldout.py` | First documents of two public files the training corpus does not use, as jsonl |
| `dedup_heldout.py`, `check_disjoint.py` | Drop held-out documents whose exact text occurs in the training shards; verify |
| `build_heldout.py` | Tokenize and pack them exactly like the training loader (`heldout.pt`) |
| `run_arm.sh` | One arm: the recipe with `--max-update 20000` and a seed, via `train_wrapper.py` |
| `train_wrapper.py` | Runs `<tree>/yoco/train.py` unchanged, replacing fairseq's checkpoint save by a model-only BF16 save on rank 0 and logging peak memory; identical for every arm |
| `eval_arm.sh`, `eval_ckpt.py` | Held-out loss of each checkpoint with the reference tree's model code (FP32 cross-entropy, bits) |
| `analyze.py` | `summary.json` and `tables.md`: training loss per 100 updates, held-out losses, wall clock, time per update without saves, peak memory, non-finite values |
| `verdict.py` | The criterion: Y = ref_s2 − ref, E1 = opt − ref, E2 = opt_s2 − ref_s2, S = opt_s2 − opt; consistent if \|E1\|, \|E2\|, \|S\| ≤ \|Y\| |
| `run_all.sh` | All four arms with their evaluations, then `analyze.py` and `verdict.py` |

## Steps

```bash
export DATA_DIR=/path/to/Data                 # training corpus, see scripts/data/build_data.py
export VAL_DIR=/path/to/validation            # outputs: <arm>/train.log, run.log, eval.json, ckpt/
git worktree add ../yoco-upstream upstream    # the unmodified reference
export REF_TREE=$(cd ../yoco-upstream && pwd)

# 1. held-out set (once; the two source files are on Hugging Face at the training revisions)
python - <<'EOF'
from huggingface_hub import hf_hub_download
print(hf_hub_download("HuggingFaceFW/fineweb-edu", "sample/10BT/001_00000.parquet", repo_type="dataset",
                      revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", local_dir="downloads"))
print(hf_hub_download("codeparrot/codeparrot-clean", "file-000000000002.json.gz", repo_type="dataset",
                      revision="35a59fb025bc0a102f7d96eac09d145b896d487b", local_dir="downloads"))
EOF
python scripts/validate/extract_heldout.py downloads/sample/10BT/001_00000.parquet \
    downloads/file-000000000002.json.gz "$VAL_DIR/heldout" 3000 800
python scripts/validate/dedup_heldout.py "$DATA_DIR" "$VAL_DIR/heldout"
python scripts/validate/check_disjoint.py "$DATA_DIR" "$VAL_DIR/heldout"
python scripts/validate/build_heldout.py "$REF_TREE" "$VAL_DIR/heldout" "$VAL_DIR/heldout/heldout.pt" 800 200

# 2. the four arms, evaluations, summary and verdict
bash scripts/validate/run_all.sh
```

The recorded held-out set kept 2,996 fineweb-edu and 800 code documents and packed them into 800 + 200 samples of
2,049 tokens. The recorded `heldout.pt` had sha256 `9543aab2616ee04b347ebe38b160aeb2d2d33ee4baf7220acb774d3e3d52739b`
(the serialized bytes may differ with another PyTorch version even when the token tensors are equal).

Expect differences from the recorded numbers: the runs are not deterministic, and the reference's two seeds differ
by up to 0.16 bits on the held-out mix. Compare against your own reference runs, not against the recorded values.
