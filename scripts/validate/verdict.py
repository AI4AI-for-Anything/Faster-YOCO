"""Final comparison of the four arms from <val dir>/summary.json (written by analyze.py).
Yardstick Y = ref_s2 - ref_s1. Effects: E1 = opt_s1 - ref_s1, E2 = opt_s2 - ref_s2, S = opt_s2 - opt_s1.
Criterion (defined before the seed-2 arms finished, unchanged): opt is consistent with ref at a checkpoint if |E1|, |E2| and |S|
are each <= |Y| at that checkpoint; reported per checkpoint and at the final checkpoint, for the held-out 0.8/0.2 mix,
each held-out source, and the training loss (mean of the 1000 updates up to the checkpoint).
Writes <val dir>/verdict.md. usage: python verdict.py <val dir>"""
import json, os, sys
V = os.path.abspath(sys.argv[1])
S = json.load(open(os.path.join(V, 'summary.json')))['arms']
need = ['opt', 'ref', 'ref_s2', 'opt_s2']
missing = [a for a in need if a not in S]
if missing:
    raise SystemExit(f'missing arms {missing}')


def train_window(arm, k, w=1000):
    c = S[arm]['train_curve_per100']
    vals = [c[str(j)] for j in range(k - w + 100, k + 1, 100) if str(j) in c]
    return sum(vals) / len(vals) if len(vals) == w // 100 else None


def metric(arm, k, m):
    if m == 'train':
        return train_window(arm, k)
    h = S[arm]['heldout'].get(str(k))
    return None if h is None else h[{'mix': 'mix_bits', 'fineweb': 'fineweb_bits', 'code': 'code_bits'}[m]]


rows, verdict = [], {}
for m in ('mix', 'fineweb', 'code', 'train'):
    rows.append(f'\n### {m} (bits)\n')
    rows.append('| update | opt_s1 | ref_s1 | ref_s2 | opt_s2 | Y=ref_s2-ref_s1 | E1=opt_s1-ref_s1 | E2=opt_s2-ref_s2 | S=opt_s2-opt_s1 | within |Y| |')
    rows.append('|---|---|---|---|---|---|---|---|---|---|')
    ok_all = []
    for k in range(2000, 20001, 2000):
        v = {a: metric(a, k, m) for a in need}
        if any(x is None for x in v.values()):
            continue
        Y = v['ref_s2'] - v['ref']
        E1, E2, Sd = v['opt'] - v['ref'], v['opt_s2'] - v['ref_s2'], v['opt_s2'] - v['opt']
        ok = all(abs(x) <= abs(Y) for x in (E1, E2, Sd))
        ok_all.append((k, ok, abs(E1) <= abs(Y), abs(E2) <= abs(Y), abs(Sd) <= abs(Y)))
        rows.append(f'| {k} | {v["opt"]:.4f} | {v["ref"]:.4f} | {v["ref_s2"]:.4f} | {v["opt_s2"]:.4f} | {Y:+.4f} | {E1:+.4f} | '
                    f'{E2:+.4f} | {Sd:+.4f} | {"yes" if ok else "no (" + ",".join(n for n, f in zip(("E1", "E2", "S"), ok_all[-1][2:]) if not f) + ")"} |')
    verdict[m] = {'checkpoints_within': sum(1 for x in ok_all if x[1]), 'checkpoints': len(ok_all),
                  'final_within': ok_all[-1][1] if ok_all and ok_all[-1][0] == 20000 else None,
                  'E1_within': sum(1 for x in ok_all if x[2]), 'E2_within': sum(1 for x in ok_all if x[3]),
                  'S_within': sum(1 for x in ok_all if x[4])}
text = '\n'.join(rows)
open(os.path.join(V, 'verdict.md'), 'w').write(text + '\n\n' + json.dumps(verdict, indent=1) + '\n')
print(text)
print(json.dumps(verdict, indent=1))
