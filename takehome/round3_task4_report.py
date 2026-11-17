import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
O = Path(__file__).resolve().parents[1] / 'results/takehome/round3/task4'
T = [json.load(open(O / f'teacher_{s}.json')) for s in range(200, 205)]
rec = np.mean([t['recorded_successes'] for t in T]) / 5
rep = np.mean([t['reproduced_successes'] for t in T]) / 5
names = list(T[0]['students'])
rows = []
for n in names:
    v = [t['students'][n] for t in T]
    rows.append((n, v[0]['precision'], v[0]['weights'], v[0]['bytes'], np.mean([x['successes'] for x in v]) / 5,
                 np.std([x['successes'] / 5 for x in v], ddof=1), np.mean([x['time_us_cpu_batch1'] for x in v])))
tw = T[0]['teacher_weights']; tt = np.mean([t['teacher_time_us'] for t in T])
base = 61.4
lines = ['| student | precision | weights | bytes | success % (mean of 5, test 1001, SD) | us per decision (CPU, batch 1, indicative) |', '|---|---|---|---|---|---|']
lines += [f'| {n.split("_fp")[0].split("_int")[0]} | {p} | {w:,} | {b:,} | {s:.1f} (SD {sd:.1f}) | {t:.0f} |' for n, p, w, b, s, sd, t in rows]
lines += [f'| teacher (old policy + corrector, two nets) | fp32 | {tw:,} | {tw*4:,} | recorded {rec:.1f}; reproduced {rep:.1f} | {tt:.0f} |']
(O / 'table.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
fig, ax = plt.subplots(figsize=(7, 4.5))
for prec, mk in (('fp32', 'o'), ('fp16', 's'), ('int8', '^')):
    pts = [(w, s) for n, p, w, b, s, sd, t in rows if p == prec]
    ax.plot(*zip(*pts), marker=mk, label=prec + (' (simulated)' if prec != 'fp32' else ''))
ax.axhline(rec, color='k', ls='--', label='teacher (recorded)')
ax.axhline(base, color='gray', ls=':', label='old policy alone (about 61%)')
ax.set_xlabel('number of weights'); ax.set_ylabel('success on 500 test episodes (%)'); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(O / 'success_vs_size.png', dpi=130)
best = max(s for *_, s, sd, t in rows)
print('best student', round(best, 1), 'teacher recorded', round(rec, 1), 'gap', round(rec - best, 1))
