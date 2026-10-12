"""Render rollout videos (osmesa software rendering, works on WSL2).

Each frame shows the close-up camera at the hole with a small overview inset
and a text overlay (clearance, step, success).

Usage:
  python scripts/render.py --run pos_2mm_s0              # trained policy
  python scripts/render.py --scripted --clearance_mm 2.0 # hand-written controller
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'osmesa')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import mediapy  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

WIDTH, HEIGHT = 640, 480
INSET_W, INSET_H = 224, 168
RENDER_EVERY = 2  # ctrl_dt 0.02 s -> 25 fps real time


def collect(env, act_fn, init_carry, episodes: int, seed: int):
  """Runs single-env episodes; returns per-episode dicts of numpy arrays."""
  n_steps = env._config.episode_length

  @jax.jit
  def episode(key):
    state = env.reset(key)

    def body(carry, _):
      state, c = carry
      action, c = act_fn(state, c)
      state = env.step(state, action)
      rec = {
          'qpos': state.data.qpos,
          'mocap_pos': state.data.mocap_pos,
          'succeeded': state.info['succeeded'],
          'force': jp.linalg.norm(state.data.sensordata[env._force_slice]),
      }
      return (state, c), rec

    first = {
        'qpos': state.data.qpos,
        'mocap_pos': state.data.mocap_pos,
        'succeeded': state.info['succeeded'],
        'force': jp.array(0.0),
    }
    _, recs = jax.lax.scan(body, (state, init_carry), None, length=n_steps)
    return jax.tree.map(lambda a, b: jp.concatenate([a[None], b]), first, recs)

  keys = jax.random.split(jax.random.PRNGKey(seed), episodes)
  return [jax.tree.map(np.array, episode(k)) for k in keys]


def _overlay(img: np.ndarray, lines) -> np.ndarray:
  pil = Image.fromarray(img)
  draw = ImageDraw.Draw(pil)
  draw.rectangle([0, 0, 300, 18 * len(lines) + 8], fill=(0, 0, 0))
  for i, (text, color) in enumerate(lines):
    draw.text((8, 4 + 18 * i), text, fill=color)
  return np.array(pil)


def frames_for(env, episodes, title: str):
  m = env.mj_model
  d = mujoco.MjData(m)
  # One renderer only: a second osmesa context in the same process renders
  # garbage, so the overview inset is rendered full size and downscaled.
  renderer = mujoco.Renderer(m, HEIGHT, WIDTH)
  frames = []
  for ep_i, ep in enumerate(episodes):
    n = ep['qpos'].shape[0]
    for t in range(0, n, RENDER_EVERY):
      d.qpos[:] = ep['qpos'][t]
      d.mocap_pos[:] = ep['mocap_pos'][t]
      mujoco.mj_forward(m, d)
      renderer.update_scene(d, camera='closeup')
      img = renderer.render().copy()
      renderer.update_scene(d, camera='overview')
      small = np.array(Image.fromarray(renderer.render()).resize((INSET_W, INSET_H)))
      img[HEIGHT - INSET_H - 6:HEIGHT - 6, WIDTH - INSET_W - 6:WIDTH - 6] = small
      ok = ep['succeeded'][t] > 0.5
      img = _overlay(img, [
          (f'{title}', (255, 255, 255)),
          (f'clearance {env.clearance * 1000:.1f} mm   episode {ep_i + 1}', (255, 255, 255)),
          (f'step {t:3d}   |F| {ep["force"][t]:5.1f} N', (255, 255, 255)),
          ('INSERTED' if ok else 'not inserted', (80, 255, 80) if ok else (255, 180, 80)),
      ])
      frames.append(img)
  return frames


def render_policy(env, policy, out_path: Path, episodes: int = 3, seed: int = 0,
                  title: str = 'learned policy') -> Path:
  def act_fn(state, key):
    key, sub = jax.random.split(key)
    obs = jax.tree.map(lambda x: x[None], state.obs)
    action, _ = policy(obs, sub)
    return action[0], key

  eps = collect(env, act_fn, jax.random.PRNGKey(seed + 7), episodes, seed)
  return _write(env, eps, out_path, title)


def render_scripted(env, out_path: Path, episodes: int = 1, seed: int = 0) -> Path:
  import scripted_insert  # pylint: disable=g-import-not-at-top

  def act_fn(state, descending):
    return scripted_insert.scripted_action(env, state, descending, jp.zeros(2))

  eps = collect(env, act_fn, jp.array(False), episodes, seed)
  return _write(env, eps, out_path, 'scripted controller (true hole pos)')


def _write(env, eps, out_path: Path, title: str) -> Path:
  t0 = time.time()
  frames = frames_for(env, eps, title)
  out_path = Path(out_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fps = 1.0 / (env._config.ctrl_dt * RENDER_EVERY)
  mediapy.write_video(str(out_path), frames, fps=fps)
  print(f'wrote {out_path} ({len(frames)} frames in {time.time() - t0:.1f} s)', flush=True)
  return out_path


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--run', help='run name; loads logs/<run> or results/policies/<run>')
  parser.add_argument('--scripted', action='store_true')
  parser.add_argument('--clearance_mm', type=float, default=None)
  parser.add_argument('--episodes', type=int, default=3)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--out', default=None)
  args = parser.parse_args()

  from peg_insertion import rollout  # pylint: disable=g-import-not-at-top

  if args.scripted:
    c = args.clearance_mm or 2.0
    env = rollout.make_env(c, False)
    out = args.out or ROOT / 'results' / 'videos' / f'scripted_{c:g}mm.mp4'
    render_scripted(env, Path(out), episodes=args.episodes, seed=args.seed)
    return

  run_dir = ROOT / 'logs' / args.run
  if not (run_dir / 'params.pkl').exists():
    run_dir = ROOT / 'results' / 'policies' / args.run
  policy, net_cfg = rollout.load_policy(run_dir)
  c = args.clearance_mm if args.clearance_mm is not None else net_cfg['clearance_mm']
  extra = {k: v for k, v in net_cfg.get('env_overrides', {}).items()
           if k not in ('clearance_mm', 'use_force_obs', 'naconmax')}
  env = rollout.make_env(c, net_cfg['use_force_obs'], **extra)
  out = args.out or ROOT / 'results' / 'videos' / f'{args.run}.mp4'
  sensing = 'position + force' if net_cfg['use_force_obs'] else 'position only'
  render_policy(env, jax.jit(policy), Path(out), episodes=args.episodes,
                seed=args.seed, title=f'{args.run} ({sensing})')


if __name__ == '__main__':
  main()
