"""Evaluate a trained policy over many episodes and record it in summary.csv.

Runs N full episodes (deterministic policy, training-time domain randomization,
fresh hole position and observation bias every episode) and reads episode
statistics from the final-step state.info.

Usage:
  python scripts/evaluate.py --run pos_2mm_s0
  python scripts/evaluate.py --run pos_2mm_s0 --clearance_mm 0.5 --tag gen0.5mm
  python scripts/evaluate.py --run force_0.5mm_s0 --obs_hole_bias_std 0 --tag nobias
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'osmesa')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from peg_insertion import rollout  # noqa: E402

SUMMARY = ROOT / 'results' / 'summary.csv'
COLUMNS = [
    'run', 'kind', 'clearance_mm', 'trained_clearance_mm', 'force_obs', 'seed',
    'obs_hole_bias_std', 'episodes', 'success', 'ci_lo', 'ci_hi', 'steps_mean',
    'peak_force_mean', 'final_xy_err_mean_mm', 'still_inserted_at_end',
    'through_wall_episodes', 'wall_time_s', 'sps', 'num_envs',
    'num_timesteps',
]


def find_run_dir(run: str) -> Path:
  for d in (ROOT / 'logs' / run, ROOT / 'results' / 'policies' / run):
    if (d / 'params.pkl').exists():
      return d
  raise FileNotFoundError(f'No params.pkl for run {run}')


def load_final_metrics(run: str) -> dict:
  for p in (ROOT / 'logs' / run / 'final_metrics.json',
            ROOT / 'results' / f'final_metrics_{run}.json'):
    if p.exists():
      return json.loads(p.read_text())
  return {}


def upsert_summary(row: dict) -> None:
  SUMMARY.parent.mkdir(parents=True, exist_ok=True)
  df = pd.read_csv(SUMMARY) if SUMMARY.exists() else pd.DataFrame(columns=COLUMNS)
  df = df[df['run'] != row['run']]
  df = pd.concat([df, pd.DataFrame([row])[COLUMNS]], ignore_index=True)
  df = df.sort_values(['kind', 'force_obs', 'clearance_mm', 'seed'],
                      ascending=[False, True, False, True])
  df.to_csv(SUMMARY, index=False)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--run', required=True)
  parser.add_argument('--episodes', type=int, default=500)
  parser.add_argument('--clearance_mm', type=float, default=None,
                      help='test on a different clearance than trained on')
  parser.add_argument('--obs_hole_bias_std', type=float, default=None)
  parser.add_argument('--tag', default='', help='suffix for extra rows')
  parser.add_argument('--seed', type=int, default=123)
  parser.add_argument('--no_summary', action='store_true')
  args = parser.parse_args()
  if (args.clearance_mm is not None or args.obs_hole_bias_std is not None) and not args.tag:
    parser.error('--clearance_mm / --obs_hole_bias_std overrides need a --tag '
                 '(otherwise they would overwrite the main result row)')

  run_dir = find_run_dir(args.run)
  policy, net_cfg = rollout.load_policy(run_dir)
  trained_c = float(net_cfg['clearance_mm'])
  c = trained_c if args.clearance_mm is None else args.clearance_mm
  extra = {k: v for k, v in net_cfg.get('env_overrides', {}).items()
           if k not in ('clearance_mm', 'use_force_obs', 'naconmax')}
  if args.obs_hole_bias_std is not None:
    extra['obs_hole_bias_std'] = args.obs_hole_bias_std
  env = rollout.make_env(c, net_cfg['use_force_obs'], naconmax=20 * 256, **extra)

  t0 = time.time()
  stats = rollout.rollout_stats(env, jax.jit(policy), args.episodes,
                                seed=args.seed, randomize=True, batch=256)
  summary = rollout.summarize(stats)
  hist, edges = np.histogram(stats['xy_err'] * 1000, bins=[0, 0.25, 0.5, 1, 2, 5, 10, 100])
  name = args.run + (f'_{args.tag}' if args.tag else '')
  out = {
      'run': name,
      'policy_run': args.run,
      'clearance_mm': c,
      'trained_clearance_mm': trained_c,
      'use_force_obs': net_cfg['use_force_obs'],
      'obs_hole_bias_std': env._config.obs_hole_bias_std,
      **summary,
      'final_xy_err_hist_mm': {'edges': edges.tolist(), 'counts': hist.tolist()},
      'eval_seconds': round(time.time() - t0, 1),
  }
  path = ROOT / 'results' / f'eval_{name}.json'
  path.write_text(json.dumps(out, indent=2))
  print(json.dumps(out, indent=2))

  if not args.no_summary:
    fm = load_final_metrics(args.run)
    if args.tag.startswith('gen'):
      kind = 'generalisation'
    elif args.tag:
      kind = 'diagnostic'
    else:
      kind = 'main'
    upsert_summary({
        'run': name, 'kind': kind, 'clearance_mm': c,
        'trained_clearance_mm': trained_c,
        'force_obs': bool(net_cfg['use_force_obs']),
        'seed': fm.get('seed', np.nan),
        'obs_hole_bias_std': env._config.obs_hole_bias_std,
        'episodes': summary['episodes'], 'success': summary['success_rate'],
        'ci_lo': summary['ci_lo'], 'ci_hi': summary['ci_hi'],
        'steps_mean': summary['steps_mean'],
        'peak_force_mean': summary['peak_force_mean'],
        'final_xy_err_mean_mm': summary['final_xy_err_mean_mm'],
        'still_inserted_at_end': summary['still_inserted_at_end'],
        'through_wall_episodes': summary['through_wall_episodes'],
        'wall_time_s': fm.get('wall_time_s', np.nan), 'sps': fm.get('sps', np.nan),
        'num_envs': fm.get('num_envs', np.nan),
        'num_timesteps': fm.get('num_timesteps', np.nan),
    })
    print(f'updated {SUMMARY}')


if __name__ == '__main__':
  main()
