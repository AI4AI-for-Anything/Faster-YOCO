"""Exact-duplicate check of the held-out documents against the training shards (sha1 of the text); after
dedup_heldout.py it should report no duplicates.
usage: python check_disjoint.py <training data root (contains shard/)> <heldout dir>"""
import glob, hashlib, json, os, sys
data, held_dir = sys.argv[1], sys.argv[2]
held = {}
for name in ('fineweb_edu', 'codeparrot'):
    for line in open(os.path.join(held_dir, f'heldout_{name}.jsonl')):
        held[hashlib.sha1(json.loads(line)['text'].encode()).hexdigest()] = name
hits = {}
for f in sorted(glob.glob(os.path.join(data, 'shard', '*', '*.jsonl'))):
    for line in open(f):
        h = hashlib.sha1(json.loads(line)['text'].encode()).hexdigest()
        if h in held:
            hits[held[h]] = hits.get(held[h], 0) + 1
print(json.dumps({'heldout_docs': len(held), 'exact_duplicates_in_training_shards': hits}))
