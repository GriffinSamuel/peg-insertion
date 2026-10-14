"""Render rollout videos (osmesa software rendering, works on WSL2).

Each 1280x720 frame shows a three-quarter view of the hole from above and in
front, a cutaway inset (same instant, side view with the near wall hidden) so
the peg can be seen inside the cavity, a text overlay and a depth bar.
Visual changes live in a render-only model (scripts/render_scene.py); the
training scene and physics are untouched.

Usage:
  python scripts/render.py --run pos_2mm_s0              # trained policy
  python scripts/render.py --scripted --clearance_mm 2.0 # hand-written controller
"""

import argparse
import functools
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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import render_scene  # noqa: E402

WIDTH, HEIGHT = render_scene.WIDTH, render_scene.HEIGHT
INSET_W, INSET_H = 448, 336
MARGIN = 16
RENDER_EVERY = 2  # ctrl_dt 0.02 s -> 25 fps real time
HOLE_DEPTH_MM = 30.0
# Main view: three-quarter, from the front-right of the robot, 40 deg down.
MAIN_CAM = dict(azimuth=145.0, elevation=-40.0, distance=0.70, lookat_dz=0.005)
# Cutaway: level side view from -y with the near wall (wall_ny) hidden.
CUT_CAM = dict(azimuth=90.0, elevation=-6.0, distance=0.22, lookat_dz=0.0)
_FONT_PATHS = ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',)


@functools.lru_cache(maxsize=None)
def _font(size: int):
  for path in _FONT_PATHS:
    if os.path.exists(path):
      return ImageFont.truetype(path, size)
  return ImageFont.load_default(size=size)


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


def _free_camera(lookat, azimuth, elevation, distance, lookat_dz):
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.lookat[:] = np.asarray(lookat) + np.array([0.0, 0.0, lookat_dz])
  cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
  return cam


def _scene_option(show_near_wall: bool) -> mujoco.MjvOption:
  opt = mujoco.MjvOption()
  opt.sitegroup[:] = 0  # hide debug sites (hole entrance/bottom markers)
  opt.geomgroup[render_scene.NEAR_WALL_GROUP] = int(show_near_wall)
  return opt


def _render_rect(renderer, d, cam, opt, w: int, h: int) -> np.ndarray:
  """Renders into the lower-left w x h of the one osmesa buffer.

  A second osmesa context in the same process renders garbage, so the inset
  reuses the main renderer's context with a smaller viewport.
  """
  renderer.update_scene(d, camera=cam, scene_option=opt)
  rect = mujoco.MjrRect(0, 0, w, h)
  mujoco.mjr_render(rect, renderer.scene, renderer._mjr_context)  # pylint: disable=protected-access
  out = np.empty((h, w, 3), dtype=np.uint8)
  mujoco.mjr_readPixels(out, None, rect, renderer._mjr_context)  # pylint: disable=protected-access
  return np.flipud(out)


def _overlay(img: np.ndarray, lines, depth_mm: float) -> np.ndarray:
  pil = Image.fromarray(img)
  draw = ImageDraw.Draw(pil, 'RGBA')
  font = _font(22)
  box_h = 30 * len(lines) + 14
  draw.rectangle([MARGIN, MARGIN, MARGIN + 520, MARGIN + box_h], fill=(0, 0, 0, 170))
  for i, (text, color) in enumerate(lines):
    draw.text((MARGIN + 12, MARGIN + 8 + 30 * i), text, fill=color, font=font)
  _depth_bar(draw, depth_mm)
  _inset_frame(draw)
  return np.array(pil)


def _depth_bar(draw, depth_mm: float) -> None:
  """Vertical bar: insertion depth vs the 30 mm hole depth."""
  x0, y0, bw, bh = MARGIN + 6, MARGIN + 190, 34, 300
  font = _font(18)
  draw.rectangle([x0 - 6, y0 - 34, x0 + bw + 118, y0 + bh + 40], fill=(0, 0, 0, 170))
  draw.text((x0, y0 - 28), 'insertion depth', fill=(255, 255, 255), font=font)
  draw.rectangle([x0, y0, x0 + bw, y0 + bh], fill=(40, 40, 46), outline=(220, 220, 220))
  frac = float(np.clip(depth_mm / HOLE_DEPTH_MM, 0.0, 1.0))
  if frac > 0:
    draw.rectangle([x0 + 2, y0 + int(bh * (1 - frac)), x0 + bw - 2, y0 + bh - 1],
                   fill=(80, 200, 255))
  for mm in (0, 10, 20, 30):
    y = y0 + int(bh * (1 - mm / HOLE_DEPTH_MM))
    draw.line([x0 + bw, y, x0 + bw + 8, y], fill=(220, 220, 220), width=2)
    draw.text((x0 + bw + 12, y - 10), f'{mm} mm', fill=(220, 220, 220), font=font)
  draw.text((x0, y0 + bh + 10), f'{max(depth_mm, 0.0):4.1f} / 30 mm',
            fill=(255, 255, 255), font=font)


def _inset_frame(draw) -> None:
  x1, y1 = WIDTH - MARGIN, HEIGHT - MARGIN
  x0, y0 = x1 - INSET_W, y1 - INSET_H
  draw.rectangle([x0 - 3, y0 - 3, x1 + 2, y1 + 2], outline=(255, 255, 255), width=3)
  draw.rectangle([x0, y0, x0 + 232, y0 + 28], fill=(0, 0, 0, 170))
  draw.text((x0 + 8, y0 + 4), 'cutaway: near wall hidden', fill=(255, 255, 255), font=_font(17))


def _max_penetration(m, d) -> float:
  """Deepest peg/wall overlap (m) among this frame's contacts, 0 if none."""
  worst = 0.0
  for i in range(d.ncon):
    c = d.contact[i]
    names = {m.geom(c.geom1).name, m.geom(c.geom2).name}
    if 'peg_geom' in names and names & {'wall_px', 'wall_nx', 'wall_py', 'wall_ny'}:
      worst = max(worst, -c.dist)
  return worst


def frames_for(env, episodes, title: str):
  m = render_scene.build_render_model(env)
  d = mujoco.MjData(m)
  renderer = mujoco.Renderer(m, HEIGHT, WIDTH)
  main_opt, cut_opt = _scene_option(True), _scene_option(False)
  tip, entrance = m.site('peg_tip').id, m.site('hole_entrance').id
  frames, worst_pen = [], 0.0
  for ep_i, ep in enumerate(episodes):
    n = ep['qpos'].shape[0]
    for t in range(0, n, RENDER_EVERY):
      d.qpos[:] = ep['qpos'][t]
      d.mocap_pos[:] = ep['mocap_pos'][t]
      mujoco.mj_forward(m, d)
      worst_pen = max(worst_pen, _max_penetration(m, d))
      hole = d.site_xpos[entrance].copy()
      img = _render_rect(renderer, d, _free_camera(hole, **MAIN_CAM), main_opt, WIDTH, HEIGHT)
      inset = _render_rect(renderer, d, _free_camera(hole, **CUT_CAM), cut_opt, INSET_W, INSET_H)
      img = img.copy()
      img[HEIGHT - MARGIN - INSET_H:HEIGHT - MARGIN, WIDTH - MARGIN - INSET_W:WIDTH - MARGIN] = inset
      depth_mm = (hole[2] - d.site_xpos[tip][2]) * 1000.0
      ok = ep['succeeded'][t] > 0.5
      img = _overlay(img, [
          (f'{title}', (255, 255, 255)),
          (f'clearance {env.clearance * 1000:.1f} mm   episode {ep_i + 1}', (255, 255, 255)),
          (f'step {t:3d}   |F| {ep["force"][t]:5.1f} N', (255, 255, 255)),
          ('INSERTED' if ok else 'not inserted', (80, 255, 80) if ok else (255, 180, 80)),
      ], depth_mm)
      frames.append(img)
  print(f'max peg/wall penetration over all frames: {worst_pen * 1000:.3f} mm', flush=True)
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
