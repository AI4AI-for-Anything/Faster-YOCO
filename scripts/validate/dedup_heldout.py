"""Drop held-out documents whose exact text (sha1) occurs in the training shards; rewrite
<heldout dir>/heldout_<source>.jsonl in place (the original is kept as .orig). The validation dropped 4 fineweb-edu
documents and no code document.
usage: python dedup_heldout.py <training data root (contains shard/)> <heldout dir>"""
import glob, hashlib, json, os, sys
data, held = sys.argv[1], sys.argv[2]
train = set()
for f in sorted(glob.glob(os.path.join(data, 'shard', '*', '*.jsonl'))):
    for line in open(f):
        train.add(hashlib.sha1(json.loads(line)['text'].encode()).hexdigest())
for name in ('fineweb_edu', 'codeparrot'):
    p = os.path.join(held, f'heldout_{name}.jsonl')
    if not os.path.exists(p + '.orig'):
        os.rename(p, p + '.orig')
    kept = dropped = 0
    with open(p, 'w') as out:
        for line in open(p + '.orig'):
            if hashlib.sha1(json.loads(line)['text'].encode()).hexdigest() in train:
                dropped += 1
                continue
            out.write(line); kept += 1
    print(name, 'kept', kept, 'dropped', dropped)
