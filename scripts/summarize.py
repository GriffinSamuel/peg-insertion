"""Aggregate results/summary.csv per condition -> results/summary_by_condition.csv
and a markdown table (results/results_table.md) used in the README.

Seeds are the unit of replication: mean, min and max of per-seed success rates.
Each seed's success rate comes from 500 evaluation episodes.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'


def best_in_training(run: str) -> float:
  path = RESULTS / 'curves' / f'{run}.csv'
  if not path.exists():
    return float('nan')
  return float(pd.read_csv(path)['eval_success'].max())


def main():
  df = pd.read_csv(RESULTS / 'summary.csv')
  df['force_obs'] = df['force_obs'].astype(str).str.lower().eq('true')
  main_df = df[df['kind'] == 'main'].copy()
  main_df['best_in_training'] = main_df['run'].map(best_in_training)
  rows = []
  for (force, c), g in main_df.groupby(['force_obs', 'clearance_mm']):
    rows.append({
        'sensing': 'position + force' if force else 'position only',
        'clearance_mm': c,
        'seeds': len(g),
        'success_mean': g['success'].mean(),
        'success_min': g['success'].min(),
        'success_max': g['success'].max(),
        'per_seed': ' / '.join(f'{x * 100:.0f}' for x in g.sort_values('seed')['success']),
        'steps_to_success_mean': np.nanmean(g['steps_mean']) if g['steps_mean'].notna().any() else np.nan,
        'peak_force_mean_N': g['peak_force_mean'].mean(),
        'still_inserted_at_end': g['still_inserted_at_end'].mean(),
        'through_wall_episodes': int(g['through_wall_episodes'].sum()),
        'best_in_training_eval_mean': g['best_in_training'].mean(),
        'wall_time_min_mean': g['wall_time_s'].mean() / 60,
        'sps_mean': g['sps'].mean(),
    })
  agg = pd.DataFrame(rows).sort_values(['sensing', 'clearance_mm'], ascending=[False, False])
  agg.to_csv(RESULTS / 'summary_by_condition.csv', index=False)

  lines = [
      '| Sensing | Clearance | Seeds | Success (mean) | Per seed (%) | Steps to insert | Peak force (N) |',
      '|---|---|---|---|---|---|---|',
  ]
  for _, r in agg.iterrows():
    steps = '–' if np.isnan(r['steps_to_success_mean']) else f"{r['steps_to_success_mean']:.0f}"
    lines.append(
        f"| {r['sensing']} | {r['clearance_mm']:g} mm | {r['seeds']} | "
        f"**{r['success_mean'] * 100:.1f}%** | {r['per_seed']} | {steps} | "
        f"{r['peak_force_mean_N']:.0f} |"
    )
  extra = df[df['kind'] != 'main']
  if not extra.empty:
    lines += ['', '| Extra evaluation | Tested at | Success | 95% CI |', '|---|---|---|---|']
    for _, r in extra.sort_values('run').iterrows():
      lines.append(
          f"| {r['run']} | {r['clearance_mm']:g} mm, bias std {r['obs_hole_bias_std'] * 1000:g} mm | "
          f"{r['success'] * 100:.1f}% | {r['ci_lo'] * 100:.1f}–{r['ci_hi'] * 100:.1f}% |"
      )
  (RESULTS / 'results_table.md').write_text('\n'.join(lines) + '\n')
  print('\n'.join(lines))
  print(agg.to_string(), file=sys.stderr)


if __name__ == '__main__':
  main()
