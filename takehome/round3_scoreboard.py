"""Rebuild results/takehome/round3/scoreboard.{md,csv} from finished result files (single writer: run by hand)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv, json  # noqa: E402
import numpy as np  # noqa: E402
from takehome.round3_settings import SETTINGS  # noqa: E402

R3 = Path(__file__).resolve().parents[1] / 'results/takehome/round3'
BASE_MOVE = {s: json.load(open(Path(__file__).resolve().parents[1] / f'results/takehome/goal6/movement/seed_{s}/result.json'))['final']['successes'] for s in range(5)}
BASE_AIM = {0: 365, 1: 278, 2: 315}
N = 500
EXTRA_ROWS = R3 / 'extra_rows.json'     # hand-written rows (3.1 hand wiggle, Task 4 etc.)
HEAD = ['Task', 'Name', 'What it is', 'Base', 'Base result (same seeds)', 'Seed results', 'Mean (%)', 'Spread (pp)',
        'Delta vs base (pp)', 'Verdict', 'Steps to insert', 'Peak force (N)', 'Practice episodes',
        'Hand-written routine', 'Hidden information']


def verdict(vals, base):
    m, b = np.mean(vals), np.mean(base)
    sp = max(np.std(vals, ddof=1) if len(vals) > 1 else 0, np.std(base, ddof=1) if len(base) > 1 else 0)
    if min(vals) < 0.5 * b:
        return 'UNSTABLE', m - b, sp
    if m - b > sp:
        return ('HELPED' if len(vals) >= 5 else 'PROMISING'), m - b, sp
    if m - b < -sp:
        return 'HURT', m - b, sp
    return 'NO REAL DIFFERENCE', m - b, sp


def rows():
    out = []
    for name, (task, desc, base, _, hand) in SETTINGS.items():
        if task in ('0', 'F'):
            continue
        results = {}
        for s in range(5):
            p = R3 / f'task{task}' / name / f'seed_{s}' / 'result.json'
            if p.exists():
                results[s] = json.loads(p.read_text())
        if not results:
            continue
        seeds = sorted(results)
        vals = [100 * results[s]['final']['successes'] / N for s in seeds]
        bsrc = BASE_MOVE if base == 'movement' else BASE_AIM
        bvals = [100 * bsrc[s] / N for s in seeds if s in bsrc]
        v, d, sp = verdict(vals, bvals)
        steps = [results[s]['final']['steps_mean'] for s in seeds if results[s]['final']['steps_mean']]
        force = [results[s]['final']['peak_force_mean'] for s in seeds]
        out.append([task, name, desc, base, f'{np.mean(bvals):.1f} +- {np.std(bvals, ddof=1) if len(bvals) > 1 else 0:.1f}',
                    '; '.join(f's{s} {results[s]["final"]["successes"]}/500' for s in seeds),
                    f'{np.mean(vals):.1f}', f'{sp:.1f}', f'{d:+.1f}', v,
                    f'{np.mean(steps):.0f}' if steps else 'n/a', f'{np.mean(force):.0f}', 200,
                    'yes' if hand else 'no', 'no'])
    if EXTRA_ROWS.exists():
        out += json.loads(EXTRA_ROWS.read_text())
    return out


def main():
    r = rows()
    with open(R3 / 'scoreboard.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(HEAD); w.writerows(r)
    summary = (R3 / 'scoreboard_summary.md').read_text() if (R3 / 'scoreboard_summary.md').exists() else ''
    lines = ['# Round 3 scoreboard', '', summary, '', '| ' + ' | '.join(HEAD) + ' |', '|' + ' --- |' * len(HEAD)]
    lines += ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in r]
    (R3 / 'scoreboard.md').write_text('\n'.join(lines) + '\n')
    print(len(r), 'rows')


if __name__ == '__main__':
    main()
