"""Summarize the validation arms: training-loss means per 100 updates, held-out losses, wall clock, s/update
(excluding the model saves), peak memory, non-finite values; opt vs ref vs ref_s2. Reads <val dir>/<arm>/{train.log,
run.log,eval.json} written by run_arm.sh and eval_arm.sh; writes <val dir>/summary.json and <val dir>/tables.md.
usage: python analyze.py <val dir> [arms...]   (default arms: opt ref ref_s2 opt_s2)"""
import json, math, os, re, sys
from datetime import datetime
V = os.path.abspath(sys.argv[1])
arms = sys.argv[2:] or ['opt', 'ref', 'ref_s2', 'opt_s2']
# found anywhere in a line: another rank's print may precede it on the same line
LINE = re.compile(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \| INFO \| train_inner \| (.*)$')
KV = re.compile(r'(\w+)=([^,]+)')
out = {}
for arm in arms:
    d = os.path.join(V, arm)
    if not os.path.exists(os.path.join(d, 'train.log')):
        continue
    rows, saves, mem, nonfinite = [], [], {}, []
    for line in open(os.path.join(d, 'train.log'), errors='replace'):
        m = LINE.search(line)
        if m:
            t = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S').timestamp()
            kv = dict(KV.findall(m.group(2)))
            loss, gn, n = float(kv['loss']), float(kv['gnorm']), int(kv['num_updates'])
            rows.append((n, loss, gn, t))
            if not (math.isfinite(loss) and math.isfinite(gn)):
                nonfinite.append(n)
        if 'VALIDATION_MEM' in line:                       # ranks' prints may share a line
            for js in re.findall(r'\{[^{}]*\}', line):
                try:
                    r = json.loads(js)
                except ValueError:
                    continue
                if 'rank' in r:
                    mem[r['rank']] = max(mem.get(r['rank'], 0), r['max_allocated_gib'])
        for m2 in re.finditer(r'VALIDATION_CKPT \S+ ([0-9.]+)s', line):
            saves.append(float(m2.group(1)))
        if 'FloatingPointError' in line or 'nan' in line.lower() and 'loss=nan' in line:
            nonfinite.append(line.strip()[:120])
    if not rows:
        continue
    bins = {}
    for n, loss, gn, t in rows:
        bins.setdefault((n - 1) // 100, []).append(loss)
    curve = {str((k + 1) * 100): sum(v) / len(v) for k, v in sorted(bins.items()) if len(v) == 10}
    # s/update between the 1000th and the last logged update, minus the save time
    a = next((r for r in rows if r[0] >= 1000), rows[0])
    b = rows[-1]
    s_per_update = ((b[3] - a[3]) - sum(saves)) / (b[0] - a[0]) if b[0] > a[0] else None
    run = open(os.path.join(d, 'run.log')).read()
    walls = [int(x) for x in re.findall(r'wall_s=(\d+)', run)]
    ev = json.load(open(os.path.join(d, 'eval.json'))) if os.path.exists(os.path.join(d, 'eval.json')) else {}
    held = {str(v['num_updates']): {'mix_bits': v['mix_0.8_0.2_bits'], 'fineweb_bits': v['fineweb_edu']['loss_bits'],
                                    'code_bits': v['codeparrot']['loss_bits']} for v in ev.values()}
    last = [r[1] for r in rows if r[0] > rows[-1][0] - 1000]
    out[arm] = {'updates': rows[-1][0], 'final_train_loss_bits_last1000': sum(last) / len(last),
                'train_curve_per100': curve, 'heldout': held, 'wall_s_total': walls[-1] if walls else None,
                's_per_update_excl_saves': s_per_update, 'save_s_total': sum(saves),
                'peak_alloc_gib_max_rank': max(mem.values()) if mem else None, 'nonfinite': nonfinite[:5]}
cmp = {}
def diffs(x, y, key):
    ks = sorted(set(out[x][key]) & set(out[y][key]), key=int)
    if key == 'heldout':
        return {k: out[x][key][k]['mix_bits'] - out[y][key][k]['mix_bits'] for k in ks}
    return {k: out[x][key][k] - out[y][key][k] for k in ks}
for x, y in (('opt', 'ref'), ('ref_s2', 'ref')):
    if x in out and y in out:
        tr = diffs(x, y, 'train_curve_per100')
        late = [v for k, v in tr.items() if int(k) > 10000]
        cmp[f'{x}-{y}'] = {'heldout_mix_bits': diffs(x, y, 'heldout'),
                          'train_per100_mean_abs_diff_after_10k': sum(map(abs, late)) / len(late) if late else None,
                          'train_per100_mean_diff_after_10k': sum(late) / len(late) if late else None}
json.dump({'arms': out, 'comparisons': cmp}, open(os.path.join(V, 'summary.json'), 'w'), indent=1)
for arm, r in out.items():
    print(arm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ('train_curve_per100', 'heldout')})
    print('   heldout', {k: round(v['mix_bits'], 4) for k, v in r['heldout'].items()})
for k, v in cmp.items():
    print(k, {kk: ({a: round(b, 4) for a, b in vv.items()} if isinstance(vv, dict) else (round(vv, 4) if vv is not None else None)) for kk, vv in v.items()})


# ---- markdown tables at the 2k checkpoints: training loss (mean of the 1000 updates up to the checkpoint, from the
# per-100 means), held-out losses per source and mix; differences opt-ref and ref_s2-ref
def train_window(arm, k, w=1000):
    c = out[arm]['train_curve_per100']
    vals = [c[str(j)] for j in range(k - w + 100, k + 1, 100) if str(j) in c]
    return sum(vals) / len(vals) if len(vals) == w // 100 else None


lines = []
cols = [a for a in ('opt', 'ref', 'ref_s2') if a in out]
lines.append('| update | ' + ' | '.join(f'train {a}' for a in cols) + ' | ' +
             ' | '.join(f'held-out mix {a}' for a in cols) + ' | opt-ref mix | ref_s2-ref mix | opt-ref train | ref_s2-ref train |')
lines.append('|' + '---|' * (1 + 2 * len(cols) + 4))
for k in range(2000, 20001, 2000):
    tr = {a: train_window(a, k) for a in cols}
    hm = {a: out[a]['heldout'].get(str(k), {}).get('mix_bits') for a in cols}
    def f(x):
        return f'{x:.4f}' if x is not None else '-'
    def dd(x, y):
        return f'{x - y:+.4f}' if x is not None and y is not None else '-'
    lines.append(f'| {k} | ' + ' | '.join(f(tr[a]) for a in cols) + ' | ' + ' | '.join(f(hm[a]) for a in cols) + ' | ' +
                 dd(hm.get('opt'), hm.get('ref')) + ' | ' + dd(hm.get('ref_s2'), hm.get('ref')) + ' | ' +
                 dd(tr.get('opt'), tr.get('ref')) + ' | ' + dd(tr.get('ref_s2'), tr.get('ref')) + ' |')
lines.append('')
lines.append('| update | ' + ' | '.join(f'fineweb {a}' for a in cols) + ' | ' + ' | '.join(f'code {a}' for a in cols) +
             ' | opt-ref fineweb | ref_s2-ref fineweb | opt-ref code | ref_s2-ref code |')
lines.append('|' + '---|' * (1 + 2 * len(cols) + 4))
for k in range(2000, 20001, 2000):
    fw = {a: out[a]['heldout'].get(str(k), {}).get('fineweb_bits') for a in cols}
    cd = {a: out[a]['heldout'].get(str(k), {}).get('code_bits') for a in cols}
    def f(x):
        return f'{x:.4f}' if x is not None else '-'
    def dd(x, y):
        return f'{x - y:+.4f}' if x is not None and y is not None else '-'
    lines.append(f'| {k} | ' + ' | '.join(f(fw[a]) for a in cols) + ' | ' + ' | '.join(f(cd[a]) for a in cols) + ' | ' +
                 dd(fw.get('opt'), fw.get('ref')) + ' | ' + dd(fw.get('ref_s2'), fw.get('ref')) + ' | ' +
                 dd(cd.get('opt'), cd.get('ref')) + ' | ' + dd(cd.get('ref_s2'), cd.get('ref')) + ' |')
open(os.path.join(V, 'tables.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
