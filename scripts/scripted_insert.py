"""Phase 3: hand-written insertion controller (no learning).

Proves the simulation can physically do the task before asking RL to learn it.
The controller cheats on purpose: it reads the *true* hole position.
  1. move the peg tip to 3 cm above the hole entrance (plus an optional offset),
  2. descend at 1 mm per step while holding x, y.

Usage:
  python scripts/scripted_insert.py            # 20 episodes x 3 clearances
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'osmesa')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jp
import numpy as np

from peg_insertion.env import PegInsertion, default_config

ROOT = Path(__file__).resolve().parent.parent
APPROACH_HEIGHT = 0.03
ALIGN_TOL = 0.0005
DESCENT_PER_STEP = 0.001
XY_GAIN = 0.2


def make_env(clearance_mm: float, impl: str | None = None) -> PegInsertion:
  cfg = default_config()
  cfg.clearance_mm = clearance_mm
  cfg.impl = impl or os.environ.get('PEG_IMPL', cfg.impl)
  return PegInsertion(cfg)


def scripted_action(env: PegInsertion, state, descending, offset_xy):
  """Setpoint controller: moves the commanded tip target (not the measured
  tip) toward the goal; the arm's position servos do the tracking.
  Returns (action, descending)."""
  scale = env._config.action_scale
  cmd = state.info['current_pos']
  tip = state.data.site_xpos[env._tip_site]
  entrance = state.data.site_xpos[env._entrance_site]
  goal_xy = entrance[:2] + offset_xy
  approach_z = entrance[2] + APPROACH_HEIGHT
  at_approach = (jp.linalg.norm(goal_xy - tip[:2]) < ALIGN_TOL) & (
      jp.abs(tip[2] - approach_z) < 0.002
  )
  descending = descending | at_approach
  z_step = jp.where(descending, -DESCENT_PER_STEP, approach_z - cmd[2])
  # While descending, slowly integrate the measured xy error so the servo's
  # tracking lag does not push the peg into a wall.
  xy_step = jp.where(descending, XY_GAIN * (goal_xy - tip[:2]), goal_xy - cmd[:2])
  action = jp.concatenate([xy_step, z_step[None]]) / scale
  return jp.clip(action, -1.0, 1.0), descending


def run_scripted(env: PegInsertion, n: int = 20, offset_xy=(0.0, 0.0),
                 seed: int = 0, steps: int | None = None, return_traj=False):
  steps = steps or env._config.episode_length
  offset = jp.array(offset_xy)

  def rollout(key):
    state = env.reset(key)

    def body(carry, _):
      state, descending = carry
      action, descending = scripted_action(env, state, descending, offset)
      state = env.step(state, action)
      return (state, descending), (state.data.qpos, state.data.mocap_pos)

    (state, _), traj = jax.lax.scan(
        body, (state, jp.array(False)), None, length=steps
    )
    return state.info, traj

  keys = jax.random.split(jax.random.PRNGKey(seed), n)
  info, traj = jax.jit(jax.vmap(rollout))(keys)
  info = jax.tree.map(np.array, info)
  out = {
      'success_rate': float(info['succeeded'].mean()),
      'steps_to_success': float(
          info['first_success_step'][info['succeeded'] > 0.5].mean()
      ) if info['succeeded'].sum() > 0 else float('nan'),
      'peak_force_mean': float(info['peak_force'].mean()),
      'final_depth_mean': float(info['depth'].mean()),
      'final_xy_err_mean': float(info['xy_err'].mean()),
  }
  if return_traj:
    out['traj'] = jax.tree.map(np.array, traj)
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--n', type=int, default=20)
  parser.add_argument('--clearances', type=float, nargs='+',
                      default=[2.0, 1.0, 0.5])
  parser.add_argument('--offset_mm', type=float, default=1.5,
                      help='off-centre start for the jamming experiment')
  parser.add_argument('--out', default=str(ROOT / 'results' / 'scripted_baseline.csv'))
  args = parser.parse_args()

  rows = []
  for c in args.clearances:
    env = make_env(c)
    for label, off in (('centred', 0.0), (f'offset_{args.offset_mm}mm', args.offset_mm)):
      t0 = time.time()
      r = run_scripted(env, n=args.n, offset_xy=(off / 1000.0, 0.0))
      row = {'clearance_mm': c, 'condition': label, 'episodes': args.n, **r,
             'wall_time_s': round(time.time() - t0, 1)}
      rows.append(row)
      print(row, flush=True)

  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  with open(args.out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
  print(f'wrote {args.out}')


if __name__ == '__main__':
  main()
