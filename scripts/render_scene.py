"""Render-only variant of the scene XML (visuals only, never used for training).

It starts from the exact XML the env compiles (`xml_builder.build_xml`) and
changes only how things look:
  * opaque hole walls in a distinct colour, plus a dark visual-only floor plate
    so the square opening reads clearly (the plate has contype=conaffinity=0
    and no <pair>, so it can never touch anything),
  * the near wall (wall_ny, the one facing the cutaway camera) moved to geom
    group 5 so a second render pass can hide it,
  * a coloured checker table, a plain neutral background and softer lighting,
  * a 1280x720 offscreen buffer.

Geometry, joints and mocap bodies are unchanged, so recorded qpos/mocap_pos
replay exactly; `build_render_model` checks this against the env model.
"""

import mujoco

from peg_insertion import xml_builder

WIDTH, HEIGHT = 1280, 720
NEAR_WALL = 'wall_ny'
NEAR_WALL_GROUP = 5  # hidden in the cutaway pass

WALL_RGBA = '0.93 0.55 0.18 1'  # opaque orange; the peg is blue
FLOOR_RGBA = '0.06 0.06 0.08 1'  # near-black hole floor
_OLD_WALL_RGBA = 'rgba="0.85 0.45 0.2 0.55"'

_OLD_SKYBOX = (
    '<texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.85 0.9 0.95"'
    ' width="800" height="800"/>'
)
_NEW_SKYBOX = (
    '<texture type="skybox" builtin="flat" rgb1="0.80 0.81 0.83" rgb2="0.80 0.81 0.83"'
    ' width="32" height="32"/>'
)
_OLD_TABLE = (
    '<texture type="2d" name="tabletex" builtin="checker" mark="edge" rgb1="0.92 0.92 0.92"\n'
    '      rgb2="0.85 0.85 0.85" markrgb="0.6 0.6 0.6" width="300" height="300"/>\n'
    '    <material name="tablemat" texture="tabletex" texuniform="true" texrepeat="10 10" reflectance="0"/>'
)
_NEW_TABLE = (
    '<texture type="2d" name="tabletex" builtin="checker" rgb1="0.36 0.45 0.50"\n'
    '      rgb2="0.30 0.38 0.43" width="300" height="300"/>\n'
    '    <material name="tablemat" texture="tabletex" texuniform="true" texrepeat="20 20" reflectance="0"/>'
)
_OLD_LIGHTS = (
    '<headlight diffuse=".8 .8 .8" ambient=".3 .3 .3" specular="0.3 0.3 0.3"/>\n'
    '    <global azimuth="-40" elevation="-30" offwidth="640" offheight="480"/>'
)
_NEW_LIGHTS = (
    '<headlight diffuse=".55 .55 .55" ambient=".42 .42 .42" specular="0.1 0.1 0.1"/>\n'
    f'    <global azimuth="-40" elevation="-30" fovy="25" offwidth="{WIDTH}" offheight="{HEIGHT}"/>'
)
_OLD_SUN = '<light pos="0.5 0 1.5" dir="0 0 -1" directional="true"/>'
# Tilted, dimmer key light: shorter, lighter shadows than the overhead sun.
_NEW_SUN = (
    '<light pos="1.2 -0.8 1.6" dir="-0.45 0.3 -0.84" directional="true"'
    ' diffuse="0.22 0.22 0.22" specular="0.1 0.1 0.1"/>'
)


def _replace_once(xml: str, old: str, new: str) -> str:
  if xml.count(old) != 1:
    raise ValueError(f'render_scene: expected exactly one match for {old[:60]!r}')
  return xml.replace(old, new)


def render_xml(env) -> str:
  """The env's scene XML with render-only visual changes applied."""
  xml = xml_builder.build_xml(env.clearance, env._pw, env._pl, env._hd)
  if xml.count(_OLD_WALL_RGBA) != 4:
    raise ValueError('render_scene: wall rgba not found on all four walls')
  xml = xml.replace(_OLD_WALL_RGBA, f'rgba="{WALL_RGBA}"')
  xml = _move_near_wall(xml)
  inner = env._pw / 2 + env.clearance + 0.001  # tucks under the walls: no seam
  floor = (
      f'\n      <geom name="hole_floor_vis" type="box" pos="0 0 0.00025"'
      f' size="{inner:.6f} {inner:.6f} 0.00025" contype="0" conaffinity="0"'
      f' group="1" rgba="{FLOOR_RGBA}"/>'
  )
  xml = _replace_once(xml, '\n      <site name="hole_entrance"', floor + '\n      <site name="hole_entrance"')
  xml = _replace_once(xml, _OLD_SKYBOX, _NEW_SKYBOX)
  xml = _replace_once(xml, _OLD_TABLE, _NEW_TABLE)
  xml = _replace_once(xml, _OLD_LIGHTS, _NEW_LIGHTS)
  return _replace_once(xml, _OLD_SUN, _NEW_SUN)


def _move_near_wall(xml: str) -> str:
  """Puts the near wall in its own geom group so the cutaway pass can hide it."""
  start = xml.index(f'<geom name="{NEAR_WALL}"')
  end = xml.index('/>', start)
  tag = xml[start:end]
  if ' group="1"' not in tag:
    raise ValueError(f'render_scene: {NEAR_WALL} has no group="1"')
  return xml[:start] + tag.replace(' group="1"', f' group="{NEAR_WALL_GROUP}"') + xml[end:]


def build_render_model(env) -> mujoco.MjModel:
  """Compiles the render-only model and checks it replays env states exactly."""
  m = mujoco.MjModel.from_xml_string(render_xml(env), assets=env._model_assets)
  ref = env.mj_model
  same = (m.nq == ref.nq and m.nmocap == ref.nmocap and m.nbody == ref.nbody
          and m.ngeom == ref.ngeom + 1)
  if not same:
    raise ValueError('render model does not match the env model structure')
  for name in ('peg_geom', 'wall_px', 'wall_nx', 'wall_py', 'wall_ny', 'table'):
    a, b = m.geom(name), ref.geom(name)
    if not (abs(a.size - b.size).max() < 1e-9 and abs(a.pos - b.pos).max() < 1e-9):
      raise ValueError(f'render model geometry differs for {name}')
  return m
