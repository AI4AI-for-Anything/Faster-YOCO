#!/usr/bin/env python3
"""Render the Faster-YOCO README figures from archived measurements; no GPU work.

Inputs: history.json (full-check results), time-budget.json (final-version profile and lower bounds) and
validation.json (paired 20,000-update training runs).
Outputs: 01-benchmark.svg, 02-history.svg, 03-time-budget.svg, 04-validation.svg and ../change-views/module-map.svg.
Usage: python docs/performance-views/render.py   (needs matplotlib and numpy)
"""
from pathlib import Path
import json
import math
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
RED = '#8C0000'
LIGHT = '#D7BDBF'
GREY = '#A99A9C'
MUTED = '#79696B'
INK = '#292122'
GRID = '#EADFE0'
CARD = '#FAF5F5'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'svg.hashsalt': 'faster-yoco-crimson', 'font.size': 12})


def save(fig, path):
    target = path.with_suffix('.svg')
    fig.savefig(target, facecolor='white', metadata={'Date': None})
    target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
    plt.close(fig)


def frame(ax, axis='y'):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(length=0, labelsize=11.5, colors=MUTED, pad=9)
    ax.grid(axis=axis, color=GRID, linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def load_history():
    data = json.loads((ROOT / 'history.json').read_text())
    for row in data['rows']:
        assert row['speedup'] == min(row['speedup_by_total'], row['speedup_by_median']), row
        assert math.isclose(row['reference_median_s'] / row['candidate_median_s'], row['speedup_by_median'],
                            abs_tol=2e-3), row
        assert math.isclose(row['reference_s'] / row['candidate_s'], row['speedup_by_total'], abs_tol=4e-3), row
    return data


def hero():
    final = load_history()['rows'][-1]
    values = [final['reference_median_s'] * 1e3, final['candidate_median_s'] * 1e3]
    fig = plt.figure(figsize=(9, 4.5), facecolor='white')
    fig.text(.06, .88, 'Faster-YOCO, optimized with AI agents', color=RED, weight='bold', fontsize=19)
    fig.text(.06, .78, 'YOCO-3B pretraining · 8 NVIDIA L20Z GPUs · 2,048 tokens per GPU · BF16, ZeRO-1, no FP8',
             color=MUTED, fontsize=10.5)
    ax = fig.add_axes([.24, .24, .39, .45])
    ax.barh([1, 0], values, color=[LIGHT, RED], height=.24, zorder=3)
    ax.set_yticks([1, 0], ['Reference', 'Faster-YOCO'])
    for y, value in zip([1, 0], values):
        ax.text(value + 6, y, f'{value:.1f} ms', va='center', color=INK, fontsize=11.5, weight='bold')
    ax.set_xlim(0, 335)
    ax.set_ylim(-.6, 1.6)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    frame(ax, 'x')
    ax.tick_params(labelsize=10)
    ax.set_xlabel('Median time per training update (ms) ↓', color=MUTED, labelpad=12, fontsize=11)
    fig.text(.85, .54, f"{final['speedup']:.2f}×", color=RED, fontsize=35, weight='bold', ha='center')
    fig.text(.85, .43, 'update speedup', color=MUTED, fontsize=12, ha='center')
    fig.text(.85, .36, 'full check: PASS', color=MUTED, fontsize=10, ha='center')
    fig.text(.06, .075, 'Reference → candidate → reference in independent processes on one host; reference = mean of '
             'the two reference runs.', color=MUTED, fontsize=9.5)
    fig.text(.06, .03, f"Speedup = the smaller of the median-update ratio ({final['speedup_by_median']:.4f}×) and the "
             f"total-time ratio ({final['speedup_by_total']:.4f}×).", color=MUTED, fontsize=9.5)
    save(fig, ROOT / '01-benchmark')


def history():
    data = load_history()
    rows = data['rows']
    height = 9.0
    top = lambda inches: 1 - inches / height       # figure fraction from inches below the top edge
    fig = plt.figure(figsize=(9, height), facecolor='white')
    fig.text(.09, top(.36), 'Faster-YOCO', fontsize=11, weight='bold', color=RED)
    fig.text(.09, top(.79), 'Agent Optimization History', fontsize=21.5, weight='bold', color=INK)
    fig.text(.94, top(.65), f"{rows[-1]['speedup']:.2f}×", fontsize=29, ha='right', weight='bold', color=RED)
    fig.text(.94, top(.88), 'update speedup', fontsize=10.5, ha='right', color=MUTED)
    fig.text(.09, top(1.31), f"Cumulative full-check speedup · 8 L20Z GPUs · 2,048 tokens per GPU · "
             f"{len(rows)} full checks in {data['final_round']} rounds", fontsize=11, color=MUTED)
    ax = fig.add_axes([.09, top(4.25), .85, 2.67 / height])
    x = list(range(len(rows)))
    values = [row['speedup'] for row in rows]
    ax.plot(x, values, color=RED, linewidth=2.3, zorder=2)
    ax.plot(x, values, linestyle='none', marker='o', markersize=7, color=RED, zorder=3)
    for i, value in enumerate(values):
        ax.annotate(f'{value:.2f}×', (x[i], value), xytext=(0, 11 if i % 2 == 0 else -23),
                    textcoords='offset points', ha='center', fontsize=11,
                    color=RED if i == len(rows) - 1 else INK, weight='bold')
    ax.set_xticks(x, [f"R{row['round']:02d}" for row in rows])
    ax.tick_params(axis='x', labelsize=10)
    ax.set_yticks([1.0, 1.5, 2.0, 2.5])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.1f}×'))
    ax.set_ylim(1.0, 2.5)
    ax.set_xlim(-.6, len(rows) - .4)
    frame(ax)
    for i, row in enumerate(rows):
        bx, by = .09 + (i % 2) * .44, top(5.2 + (i // 2) * .7)
        fig.add_artist(FancyBboxPatch((bx, by - .4 / height), .41, .58 / height,
                       boxstyle='round,pad=0.004,rounding_size=0.007',
                       transform=fig.transFigure, facecolor=CARD, edgecolor='none'))
        fig.text(bx + .014, by - .11 / height, f"R{row['round']:02d}", fontsize=11.5, weight='bold', color=RED,
                 va='center')
        label = textwrap.fill(row['label'], width=33, break_long_words=False, break_on_hyphens=False)
        fig.text(bx + .079, by - .11 / height, label, fontsize=10.3, color=INK, weight='bold', va='center',
                 linespacing=1.25)
    fig.text(.09, top(8.55), 'Each point is a full check by the same frozen judge on the same host; every configuration '
             'passed the statistical gate.\nR17 combines rounds 10–16, R25 rounds 21, 22 and 25. Target was 3×; the run '
             'was stopped after round 28, and rounds\n26–28 are not included. Cumulative configurations, not isolated '
             'ablations.', fontsize=9.3, color=MUTED, linespacing=1.4)
    save(fig, ROOT / '02-history')


def time_budget():
    data = json.loads((ROOT / 'time-budget.json').read_text())
    parts = data['components']
    steps = data['step_ms']
    lo = sum(p['bound_ms'][0] for p in parts)
    hi = sum(p['bound_ms'][1] for p in parts)
    fig = plt.figure(figsize=(9, 6.2), facecolor='white')
    fig.text(.06, .925, 'Why Faster-YOCO stops at 2.12×', color=INK, weight='bold', fontsize=19)
    fig.text(.06, .868, 'Final configuration · profiled time per update and GPU (rank 0) vs estimated lower bound of '
             'each component', color=MUTED, fontsize=10.2)
    ax = fig.add_axes([.33, .33, .6, .47])
    n = len(parts)
    ys = np.arange(n)[::-1]
    for y, p in zip(ys, parts):
        b0, b1 = p['bound_ms']
        ax.barh(y + .18, p['now_ms'], height=.32, color=RED, zorder=3)
        ax.barh(y - .18, b0, height=.32, color='white', edgecolor=RED, linewidth=1.3, hatch='///', zorder=3)
        if b1 > b0:
            ax.plot([b0, b1], [y - .18, y - .18], color=RED, linewidth=1.3, zorder=4)
            ax.plot([b1, b1], [y - .29, y - .07], color=RED, linewidth=1.3, zorder=4)
        ax.text(p['now_ms'] + 1, y + .18, f"{p['now_ms']:.1f}", va='center', fontsize=9.5, color=INK, zorder=5)
        bound = f'{b0:.1f}' if b1 == b0 else f'{b0:.1f}–{b1:.1f}'
        ax.text(b1 + 1, y - .18, bound, va='center', fontsize=9.5, color=MUTED, zorder=5)
    ax.set_yticks(ys, [p['name'] for p in parts])
    ax.set_xlim(0, 80)
    ax.set_xticks([0, 20, 40, 60, 80])
    frame(ax, 'x')
    ax.tick_params(axis='y', labelsize=10.5, colors=INK)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_xlabel('ms per update and GPU', color=MUTED, labelpad=8, fontsize=10.5)
    ax.legend(handles=[Patch(facecolor=RED, label='now (profiled)'),
                       Patch(facecolor='white', edgecolor=RED, hatch='///', label='lower bound')],
              loc='lower right', frameon=False, fontsize=9.5, labelcolor=INK)
    gemm, gather = parts[0]['bound_ms'], parts[4]['bound_ms']
    fig.text(.06, .17, f"Sum of the lower bounds: {lo:.1f}–{hi:.1f} ms per update vs {steps['target_3x']:.1f} ms for 3× "
             f"(measured {steps['faster_yoco']:.1f} ms; reference {steps['reference']:.1f} ms).",
             fontsize=10, color=INK)
    fig.text(.06, .035, f"The GEMM bound ({gemm[0]:.1f} ms at the measured 750 TFLOP/s BF16 peak) and the bf16 parameter "
             f"all-gather\n({gather[0]:.1f} ms at the NVLink bound, needed before the next update reads the parameters) alone "
             f"take {gemm[0] + gather[0]:.1f}–{gemm[1] + gather[1]:.1f} ms.\nThe profiled components overlap in time and "
             "do not sum to the update; the bounds are estimates, not checks.", fontsize=9.3, color=MUTED,
             linespacing=1.4)
    save(fig, ROOT / '03-time-budget')


def load_validation():
    data = json.loads((ROOT / 'validation.json').read_text())
    arms = data['arms']
    ck = [str(k) for k in range(2000, 20001, 2000)]

    def train_window(arm, k, w=1000):
        c = arms[arm]['train_loss_bits_per100']
        return sum(c[(k - w) // 100:k // 100]) / (w // 100)

    def metric(arm, k, m):
        return train_window(arm, int(k)) if m == 'train' else arms[arm]['heldout_bits'][k][m]

    for m in ('mix', 'fineweb', 'code', 'train'):          # recompute the recorded verdict counts
        ok = []
        for k in ck:
            v = {a: metric(a, k, m) for a in arms}
            y = abs(v['ref_s2'] - v['ref'])
            e = [abs(v['opt'] - v['ref']), abs(v['opt_s2'] - v['ref_s2']), abs(v['opt_s2'] - v['opt'])]
            ok.append([all(x <= y for x in e)] + [x <= y for x in e])
        rec = data['verdict'][m]
        got = [sum(o[i] for o in ok) for i in range(4)]
        assert got == [rec['checkpoints_within'], rec['E1_within'], rec['E2_within'], rec['S_within']], (m, got)
        assert ok[-1][0] == rec['final_within'], m
    return data


def validation():
    data = load_validation()
    arms = data['arms']
    x = np.arange(100, 20001, 100)
    tr = {a: np.array(arms[a]['train_loss_bits_per100']) for a in arms}
    ck = np.arange(2000, 20001, 2000)
    held = {a: np.array([arms[a]['heldout_bits'][str(k)]['mix'] for k in ck]) for a in arms}
    style = {'ref': dict(color=GREY, linestyle='-', linewidth=2.6), 'ref_s2': dict(color=GREY, linestyle='--', linewidth=2.6),
             'opt': dict(color=RED, linestyle='-', linewidth=1.1), 'opt_s2': dict(color=RED, linestyle='--', linewidth=1.1)}
    label = {'ref': 'Reference, seed 1', 'ref_s2': 'Reference, seed 2', 'opt': 'Faster-YOCO, seed 1',
             'opt_s2': 'Faster-YOCO, seed 2'}
    order = ['ref', 'ref_s2', 'opt', 'opt_s2']
    fig = plt.figure(figsize=(12, 9.6), facecolor='white')
    fig.text(.06, .955, 'Faster-YOCO vs reference: 2 seeds × 20,000 updates', color=INK, weight='bold',
             fontsize=18)
    fig.text(.06, .922, 'YOCO-3B, upstream recipe, 8 NVIDIA L20Z GPUs · loss in bits per token · held-out: 1.64M '
             'fineweb-edu + 0.41M code tokens, mixed 0.8 / 0.2', color=MUTED, fontsize=10.5)
    axs = [fig.add_axes(r) for r in ([.07, .54, .4, .32], [.57, .54, .4, .32], [.07, .11, .4, .32],
                                     [.57, .11, .4, .32])]

    def panel(ax, title):
        frame(ax, 'both')
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color(GRID)
        ax.tick_params(labelsize=9.5, pad=5)
        ax.set_title(title, loc='left', color=INK, fontsize=11.5, weight='bold', pad=8)

    ax = axs[0]
    m = x >= 500
    for a in order:
        ax.plot(x[m], tr[a][m], label=label[a], zorder=3 if a.startswith('opt') else 2, **style[a])
    ax.set_ylim(4.3, 8.2)
    ax.set_xlim(0, 20000)
    panel(ax, 'Training loss (mean of each 100 updates)')
    ax.set_xlabel('update', color=MUTED, fontsize=10)
    ax.legend(loc='upper right', frameon=False, fontsize=9, labelcolor=INK, ncol=2)
    ins = ax.inset_axes([.43, .38, .54, .34])
    mm = x > 16000
    for a in order:
        ins.plot(x[mm], tr[a][mm], **style[a])
    ins.set_xticks([16000, 18000, 20000], ['16k', '18k', '20k'])
    ins.tick_params(labelsize=8, colors=MUTED, length=0)
    for s in ins.spines.values():
        s.set_color(GRID)
    ins.set_title('updates 16,100–20,000', fontsize=8.5, color=MUTED, pad=3)

    ax = axs[1]
    for a in order:
        st = dict(style[a])
        ax.plot(ck, held[a], marker='o', markersize=6.5 if a.startswith('ref') else 3.5, label=label[a],
                zorder=3 if a.startswith('opt') else 2, **st)
    ax.set_xticks(ck, [f'{k // 1000}k' for k in ck])
    panel(ax, 'Held-out loss at each checkpoint')
    ax.set_xlabel('update', color=MUTED, fontsize=10)
    ax.legend(loc='upper right', frameon=False, fontsize=9, labelcolor=INK, ncol=2)

    def roll(v, n=10):
        return np.convolve(v, np.ones(n) / n, mode='valid')

    xs = x[9:]
    y_tr = roll(tr['ref_s2'] - tr['ref'])
    e1_tr, e2_tr = roll(tr['opt'] - tr['ref']), roll(tr['opt_s2'] - tr['ref_s2'])
    ax = axs[2]
    ax.fill_between(xs, -np.abs(y_tr), np.abs(y_tr), color=GRID, alpha=.6, linewidth=0, zorder=1,
                    label='±|Y|: criterion region')
    ax.axhline(0, color=MUTED, linewidth=.8, zorder=2)
    ax.plot(xs, y_tr, color=GREY, linewidth=1.8, zorder=3, label='Y: reference seed 2 − seed 1')
    ax.plot(xs, e1_tr, color=RED, linewidth=1.6, zorder=4, label='E1: Faster-YOCO − reference, seed 1')
    ax.plot(xs, e2_tr, color=RED, linewidth=1.6, linestyle='--', zorder=4, label='E2: Faster-YOCO − reference, seed 2')
    ax.set_xlim(0, 20000)
    ax.set_ylim(-.44, .27)
    panel(ax, 'Training-loss difference (1,000-update rolling mean)')
    ax.set_xlabel('update', color=MUTED, fontsize=10)
    ax.legend(loc='lower center', frameon=False, fontsize=8.6, labelcolor=INK, ncol=2, columnspacing=1.2)

    ax = axs[3]
    y_h = held['ref_s2'] - held['ref']
    ax.fill_between(ck, -np.abs(y_h), np.abs(y_h), color=GRID, alpha=.6, linewidth=0, zorder=1,
                    label='±|Y|: criterion region')
    ax.axhline(0, color=MUTED, linewidth=.8, zorder=2)
    ax.plot(ck, y_h, color=GREY, linewidth=1.8, marker='o', markersize=5, zorder=3, label='Y')
    ax.plot(ck, held['opt'] - held['ref'], color=RED, linewidth=1.6, marker='o', markersize=4.5, zorder=4, label='E1')
    ax.plot(ck, held['opt_s2'] - held['ref_s2'], color=RED, linewidth=1.6, linestyle='--', marker='s', markersize=4,
            zorder=4, label='E2')
    ax.set_xticks(ck, [f'{k // 1000}k' for k in ck])
    panel(ax, 'Held-out difference at each checkpoint')
    ax.set_xlabel('update', color=MUTED, fontsize=10)
    ax.legend(loc='lower right', frameon=False, fontsize=9, labelcolor=INK, ncol=4)
    v = data['verdict']
    fig.text(.06, .008, f"Consistent at a checkpoint if |E1|, |E2| and |S| ≤ |Y| (S = Faster-YOCO seed 2 − seed 1, not "
             f"drawn). Checkpoints consistent: held-out mix {v['mix']['checkpoints_within']}/10, fineweb "
             f"{v['fineweb']['checkpoints_within']}/10, code {v['code']['checkpoints_within']}/10,\ntraining loss "
             f"(1,000-update mean) {v['train']['checkpoints_within']}/10; all four at the final checkpoint. Seed 2 of "
             "Faster-YOCO ran on a second host of the same model.", fontsize=9.3, color=MUTED, linespacing=1.4)
    save(fig, ROOT / '04-validation')


def modulemap():
    fig = plt.figure(figsize=(9, 6.8), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(.055, .925, 'What changed in Faster-YOCO', fontsize=23, color=INK, weight='bold')
    cards = [
        ('ZeRO-1 optimizer path', 'FP32 work only on the owned shard\nOne in-place bf16 all-gather\n'
         'Adam fused with the bf16 cast',
         'tasks/zero1_fast.py\ntasks/fused_adam_cast.py', '2 files'),
        ('Gradient reduction', 'Reduce to the optimizer-state owner\nCopy engines instead of NCCL kernels\n'
         'Embedding gradient formed on its owner',
         'models/decoder/ce_reduce.py\nmodels/decoder/layer_graphs.py\ntasks/zero1_fast.py (clip norm)', '3 files'),
        ('Launches and host syncs', 'CUDA graphs of the 26 decoder layers\nfairseq train_step and logging\n'
         'without per-update host syncs',
         'models/decoder/layer_graphs.py\nmodels/decoder/yoco.py\ntasks/trainer_nosync.py', '3 files'),
        ('Kernels', 'Triton RMSNorm and fused per-head subln\nCompiled gate retention · Triton loss\n'
         'q/k/v/g projections as one GEMM',
         'models/decoder/: rms_norm.py · rms_triton.py\n    gate_retention.py · kernel/gate_recurrent.py\n'
         'tasks/fused_ce.py', '5 files'),
    ]
    for i, (title, description, paths, count) in enumerate(cards):
        x, y = .055 + (i % 2) * .455, .475 - (i // 2) * .42
        ax.add_patch(FancyBboxPatch((x, y), .425, .37,
                     boxstyle='round,pad=0.009,rounding_size=0.012', facecolor=CARD, edgecolor=GRID))
        ax.plot([x, x + .425], [y + .37, y + .37], color=RED, linewidth=2, solid_capstyle='round')
        ax.text(x + .014, y + .33, f'{i + 1:02d}', fontsize=11, color=RED, weight='bold')
        ax.text(x + .41, y + .33, count, fontsize=10, color=MUTED, ha='right')
        ax.text(x + .014, y + .284, title, fontsize=12.5, color=INK, weight='bold')
        ax.text(x + .014, y + .232, description, fontsize=10.2, color=INK, va='top', linespacing=1.6)
        ax.text(x + .014, y + .03, paths, fontsize=9.2, color=MUTED, linespacing=1.5)
    ax.text(.055, .02, 'Paths under yoco/. 11 files: 7 new, 4 upstream files edited. Every switch is off by default.',
            fontsize=9.5, color=MUTED)
    save(fig, ROOT.parent / 'change-views/module-map')


if __name__ == '__main__':
    hero()
    history()
    time_budget()
    validation()
    modulemap()
