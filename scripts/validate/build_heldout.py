"""Tokenize and pack the held-out jsonl files exactly as the training loader packs a shard
(<ref tree>/yoco/tasks/data/lm_loader.py: LMLoader._doc_to_ids per document, then BOS-started samples of
tokens_per_sample+1 tokens, overflow truncated; the tokenizer of the GPT task: TiktokenTokenizer cl100k_base, pad
multiple 8). Keeps the first n samples per source; the validation used n_fw = 800 and n_cp = 200.
usage: python build_heldout.py <ref tree> <heldout dir with heldout_<source>.jsonl> <out.pt> <n_fw> <n_cp>"""
import json, os, sys
ref, dl, out, n_fw, n_cp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, os.path.join(ref, 'yoco'))
import torch
from tasks.data.tiktoken_tokenizer import TiktokenTokenizer
from tasks.data.lm_loader import LMLoader
tok = TiktokenTokenizer('cl100k_base', 8)
T = 2048
res = {}
for name, n_want in (('fineweb_edu', n_fw), ('codeparrot', n_cp)):
    ids = []
    for line in open(os.path.join(dl, f'heldout_{name}.jsonl')):
        text = json.loads(line)['text']
        if len(text) == 0:
            continue
        ids.extend(LMLoader._doc_to_ids(text, tok))
    data, doc = [], [tok.bos_id]
    for x in ids:                                      # the packing loop of LMLoader._read_from_files
        if len(doc) + len(x) > T + 1:
            doc.extend(x)
            doc = doc[:T + 1]
            data.append(doc)
            doc = [tok.bos_id]
        else:
            doc.extend(x)
    if len(data) < n_want:
        raise SystemExit(f'{name}: only {len(data)} samples, need {n_want}')
    res[name] = torch.tensor(data[:n_want], dtype=torch.int64)
    print(name, 'samples', len(data), 'kept', n_want, 'tokens', n_want * T, flush=True)
torch.save({'samples': res, 'tokens_per_sample': T, 'n_words': tok.n_words, 'pad_id': tok.pad_id}, out)
