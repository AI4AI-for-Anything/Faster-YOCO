"""The first documents of two public files that the training corpus does not use, as jsonl with a "text" field (the
layout of the training shards, see scripts/data/build_data.py). Needs pyarrow.
  fineweb_edu: HuggingFaceFW/fineweb-edu @87f09149 sample/10BT/001_00000.parquet (training used 000_00000)
  codeparrot:  codeparrot/codeparrot-clean @35a59fb0 file-000000000002.json.gz (training used file-000000000001)
Writes <out dir>/heldout_fineweb_edu.jsonl and <out dir>/heldout_codeparrot.jsonl and prints the sha256 of the inputs
and outputs. The validation used n_fineweb = 3000 and n_codeparrot = 800.
usage: python extract_heldout.py <001_00000.parquet> <file-000000000002.json.gz> <out dir> <n_fineweb> <n_codeparrot>"""
import gzip, hashlib, json, os, sys
import pyarrow.parquet as pq
fw_path, cp_path, out_dir, nf, nc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
os.makedirs(out_dir, exist_ok=True)
out = {}
with open(os.path.join(out_dir, 'heldout_fineweb_edu.jsonl'), 'w') as f:
    n = 0
    for batch in pq.ParquetFile(fw_path).iter_batches(batch_size=1024, columns=['text']):
        for t in batch.column(0).to_pylist():
            if t:
                f.write(json.dumps({'text': t}) + '\n'); n += 1
            if n >= nf:
                break
        if n >= nf:
            break
    out['fineweb_edu'] = n
with open(os.path.join(out_dir, 'heldout_codeparrot.jsonl'), 'w') as f, gzip.open(cp_path, 'rt', encoding='utf-8') as g:
    n = 0
    for line in g:
        t = json.loads(line)['content']
        if t:
            f.write(json.dumps({'text': t}) + '\n'); n += 1
        if n >= nc:
            break
    out['codeparrot'] = n
for path in (fw_path, cp_path, os.path.join(out_dir, 'heldout_fineweb_edu.jsonl'),
             os.path.join(out_dir, 'heldout_codeparrot.jsonl')):
    out[os.path.basename(path)] = hashlib.sha256(open(path, 'rb').read()).hexdigest()
print(json.dumps(out, indent=1))
