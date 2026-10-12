"""Make the results figures from results/summary.csv and results/curves/*.csv.

  results/success_vs_clearance.png  success rate vs hole clearance, two lines
                                    (position only, position + force), 95% CI
                                    error bars, individual seeds as faint dots
  results/learning_curves.png       eval success vs env steps, one panel per
                                    clearance, one line per run
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'mpl'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


RESULTS = ROOT / 'results'
# Validated categorical slots 1-2 (blue, orange) from the reference palette.
COLORS = {False: '#2a78d6', True: '#eb6834'}
LABELS = {False: 'position only', True: 'position + wrist force'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def _style(ax):
  ax.grid(axis='y', color=GRID, linewidth=0.8)
  ax.set_axisbelow(True)
  for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
  for side in ('left', 'bottom'):
    ax.spines[side].set_color(MUTED)
  ax.tick_params(colors=MUTED)


def success_vs_clearance(df: pd.DataFrame) -> Path:
  main = df[df['kind'] == 'main'].copy()
  fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
  _style(ax)
  for force in (False, True):
    sub = main[main['force_obs'] == force]
    if sub.empty:
      continue
    rows = []
    for c, g in sub.groupby('clearance_mm'):
      # Seeds are the unit of replication: mean of per-seed success rates,
      # bar = min..max across seeds (per-run Wilson CIs are in summary.csv).
      rates = g['success'].to_numpy()
      rows.append((c, rates.mean(), rates.min(), rates.max(), len(g)))
      ax.scatter([c] * len(g), rates * 100, s=22, color=COLORS[force],
                 alpha=0.35, edgecolors='none', zorder=2)
    rows.sort()
    c, p, lo, hi, nseeds = map(np.array, zip(*rows))
    ax.errorbar(c, p * 100, yerr=[(p - lo) * 100, (hi - p) * 100],
                color=COLORS[force], linewidth=2, marker='o', markersize=8,
                markeredgecolor='white', markeredgewidth=1.5, capsize=4,
                label=f'{LABELS[force]} (n={int(nseeds.max())} seeds)', zorder=3)
    # direct label on the tightest clearance point
    i = int(np.argmin(c))
    ax.annotate(f'{p[i] * 100:.0f}%', (c[i], p[i] * 100), textcoords='offset points',
                xytext=(10, -4 if force else 4), color=INK, fontsize=9)
  ax.set_xscale('log')
  ax.set_xticks([2.0, 1.0, 0.5])
  ax.set_xticklabels(['2.0', '1.0', '0.5'])
  ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
  ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
  ax.invert_xaxis()  # tighter holes to the right
  ax.set_xlim(2.6, 0.38)
  ax.set_ylim(-3, 103)
  ax.set_xlabel('hole clearance per side (mm)  -> tighter', color=INK)
  ax.set_ylabel('insertion success (%)', color=INK)
  ax.set_title('Learned peg insertion vs. hole clearance\n'
               '(mean over seeds of 500-episode evals; bar = min-max across seeds;'
               ' dots = seeds)',
               color=INK, fontsize=10, loc='left')
  ax.legend(frameon=False, loc='lower left', labelcolor=INK)
  fig.tight_layout()
  out = RESULTS / 'success_vs_clearance.png'
  fig.savefig(out)
  plt.close(fig)
  return out


def learning_curves(df: pd.DataFrame) -> Path | None:
  curves_dir = RESULTS / 'curves'
  main = df[df['kind'] == 'main']
  clearances = sorted(main['clearance_mm'].unique(), reverse=True)
  if not clearances or not curves_dir.exists():
    return None
  fig, axes = plt.subplots(1, len(clearances), figsize=(4 * len(clearances), 3.6),
                           dpi=150, sharey=True, squeeze=False)
  for ax, c in zip(axes[0], clearances):
    _style(ax)
    for _, row in main[main['clearance_mm'] == c].iterrows():
      path = curves_dir / f"{row['run']}.csv"
      if not path.exists():
        continue
      cur = pd.read_csv(path)
      ax.plot(cur['step'] / 1e6, cur['eval_success'] * 100, color=COLORS[bool(row['force_obs'])],
              linewidth=2, alpha=0.9)
    ax.set_title(f'{c:g} mm clearance', color=INK, fontsize=10)
    ax.set_xlabel('env steps (millions)', color=INK)
  axes[0][0].set_ylabel('in-training eval success (%)\n(stochastic policy, 128 episodes)',
                        color=INK)
  handles = [plt.Line2D([], [], color=COLORS[f], linewidth=2, label=LABELS[f])
             for f in (False, True)]
  fig.legend(handles=handles, frameon=False, loc='upper center', ncol=2, labelcolor=INK)
  fig.tight_layout(rect=(0, 0, 1, 0.9))
  out = RESULTS / 'learning_curves.png'
  fig.savefig(out)
  plt.close(fig)
  return out


def main():
  df = pd.read_csv(RESULTS / 'summary.csv')
  df['force_obs'] = df['force_obs'].astype(str).str.lower().eq('true')
  print('wrote', success_vs_clearance(df))
  print('wrote', learning_curves(df))


if __name__ == '__main__':
  main()
