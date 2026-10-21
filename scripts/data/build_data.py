"""Build the YOCO pretraining corpus used by the checks and the validation, in the layout of the upstream README,
from two public Hugging Face files.

Data/json/train.json            [{"name": "fineweb_edu", "weight": 0.8}, {"name": "codeparrot", "weight": 0.2}]
Data/json/valid.json            [] (never read by training: --disable-validation; the fairseq fork still opens it)
Data/json/<SRC>.json            absolute paths of that source's shards
Data/shard/<SRC>/NNNNN.jsonl    <= 10,000 lines of {"text": ...}, in source-file row order, empty documents skipped

Sources (pinned revisions; sha256 = Hugging Face LFS oid):
  fineweb_edu: HuggingFaceFW/fineweb-edu @87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 sample/10BT/000_00000.parquet
               b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871 (column "text")
  codeparrot:  codeparrot/codeparrot-clean @35a59fb025bc0a102f7d96eac09d145b896d487b file-000000000001.json.gz
               3726a0239b5cb7d0ef3ea36886c533d0becc7404217763015559edb546d53c94 (field "content")
The recorded corpus had 726,000 fineweb-edu documents in 73 shards and 100,000 codeparrot files in 10 shards.

Download, for example:
  from huggingface_hub import hf_hub_download
  hf_hub_download("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet", repo_type="dataset",
                  revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", local_dir="downloads")
  hf_hub_download("codeparrot/codeparrot-clean", "file-000000000001.json.gz", repo_type="dataset",
                  revision="35a59fb025bc0a102f7d96eac09d145b896d487b", local_dir="downloads")
usage: python build_data.py <000_00000.parquet> <file-000000000001.json.gz> <out_root>
       (out_root/Data is created and must not exist; out_root/data_manifest.json records shard hashes)
Needs pyarrow.
"""
import gzip, hashlib, json, os, sys

import pyarrow.parquet as pq

LINES = 10000
SOURCES = [
    ("fineweb_edu", "b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871", 0.8),
    ("codeparrot", "3726a0239b5cb7d0ef3ea36886c533d0becc7404217763015559edb546d53c94", 0.2),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def docs(name, path):
    if name == "fineweb_edu":
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=8192, columns=["text"]):
            for t in batch.column(0).to_pylist():
                yield t
    else:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["content"]


def main():
    srcs, out = sys.argv[1:3], os.path.abspath(sys.argv[3])
    root = os.path.join(out, "Data")
    assert not os.path.exists(root), root
    os.makedirs(os.path.join(root, "json"))
    manifest = {"sources": {}}
    for (name, want, weight), src in zip(SOURCES, srcs):
        got = sha256(src)
        assert got == want, (src, got)
        sdir = os.path.join(root, "shard", name)
        os.makedirs(sdir)
        paths, n_docs, n_chars, n_empty, buf = [], 0, 0, 0, []

        def flush():
            p = os.path.join(sdir, "%05d.jsonl" % len(paths))
            with open(p, "w", encoding="utf-8") as f:
                f.write("".join(buf))
            paths.append(p)
            buf.clear()

        for text in docs(name, src):
            if not text:
                n_empty += 1
                continue
            buf.append(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n_docs += 1
            n_chars += len(text)
            if len(buf) == LINES:
                flush()
        if buf:
            flush()
        with open(os.path.join(root, "json", name + ".json"), "w") as f:
            json.dump(paths, f, indent=1)
        manifest["sources"][name] = {"file": os.path.basename(src), "sha256": got, "weight": weight, "docs": n_docs,
                                     "skipped_empty": n_empty, "chars": n_chars, "shards": len(paths),
                                     "shard_sha256": {os.path.basename(p): sha256(p) for p in paths}}
        print(name, "docs", n_docs, "shards", len(paths), "chars", n_chars, flush=True)
    with open(os.path.join(root, "json", "train.json"), "w") as f:
        json.dump([{"name": n, "weight": w} for n, _, w in SOURCES], f, indent=1)
    # fairseq_cli/train.py of the fork loads the "valid" split's json even with --disable-validation (it is never
    # iterated then); the README layout has no valid split, so an empty source list is provided
    with open(os.path.join(root, "json", "valid.json"), "w") as f:
        json.dump([], f)
    with open(os.path.join(out, "data_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
