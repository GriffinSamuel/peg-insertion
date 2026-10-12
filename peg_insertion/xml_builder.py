"""Builds the MuJoCo scene XML for the peg-insertion task.

The scene is Playground's Franka Panda scene (same solver options) with:
  * a square peg rigidly attached to the hand (the "already grasped" setup),
  * a table, and a square hole made of four box walls whose gap is
    peg_width + 2 * clearance,
  * a wrist force/torque sensor between the peg and the hand.

Every new geom has contype=conaffinity=0; collisions happen only through the
explicit <pair> elements, so the peg never touches the fingertip pads (which
would pollute the force sensor).
"""

from pathlib import Path
import re
from typing import Dict

ASSETS_DIR = Path(__file__).parent / 'assets'

PEG_WIDTH = 0.020
PEG_LEN = 0.060
HOLE_DEPTH = 0.030
WALL_THICKNESS = 0.020
HOLE_CENTER = (0.5, 0.0)
# The hand's TCP ("gripper" site) sits 0.100 m along hand +z. The peg base is
# 1 cm inside the fingers, so the tip protrudes peg_len - 0.01 past the TCP.
TCP_Z_IN_HAND = 0.100
PEG_INSET = 0.010
PANDA_XML = 'mjx_panda.xml'

_PAIR = (
    '<pair geom1="peg_geom" geom2="{g}" condim="3" solref="0.01 1"'
    ' friction="0.6 0.6 0.005 0.0001 0.0001"/>'
)


def peg_body_xml(peg_width: float = PEG_WIDTH, peg_len: float = PEG_LEN) -> str:
  """Peg body, written into the hand body of the Panda XML."""
  z = TCP_Z_IN_HAND + peg_len / 2 - PEG_INSET
  hw, hl = peg_width / 2, peg_len / 2
  return f"""
                      <body name="peg" pos="0 0 {z:.6f}">
                        <geom name="peg_geom" type="box" size="{hw:.6f} {hw:.6f} {hl:.6f}" mass="0.05"
                         rgba="0.2 0.4 0.9 1" contype="0" conaffinity="0" group="1"/>
                        <site name="peg_tip" pos="0 0 {hl:.6f}" size="0.002"/>
                        <site name="peg_base" pos="0 0 {-hl:.6f}" size="0.002"/>
                        <site name="ft_site" pos="0 0 {-hl:.6f}" size="0.002"/>
                      </body>
"""


def add_gravity_compensation(panda_xml: str) -> str:
  """Turn on MuJoCo gravity compensation for every arm/hand body.

  Menagerie's Panda servos are soft (kp 300-1000), so without compensation the
  arm sags ~2 cm under gravity. A real Franka's controller compensates gravity
  too. The peg itself is NOT compensated, so the wrist sensor still feels it.
  """
  return re.sub(
      r'<body name="(link\d|hand|left_finger|right_finger)"',
      r'<body gravcomp="1" name="\1"',
      panda_xml,
  )


def patch_panda_xml(panda_xml: str, peg_width: float, peg_len: float) -> str:
  """Add gravity compensation and insert the peg body before the gripper site."""
  panda_xml = add_gravity_compensation(panda_xml)
  marker = '<site name="gripper"'
  idx = panda_xml.find(marker)
  if idx < 0:
    raise ValueError('Could not find the gripper site in mjx_panda.xml')
  line_start = panda_xml.rfind('\n', 0, idx) + 1
  return (
      panda_xml[:line_start]
      + peg_body_xml(peg_width, peg_len).strip('\n')
      + '\n'
      + panda_xml[line_start:]
  )


def hole_walls_xml(clearance_m: float, peg_width: float, hole_depth: float) -> str:
  """Four box walls around a square gap of width peg_width + 2*clearance."""
  hw = peg_width + 2 * clearance_m  # inner hole width
  t = WALL_THICKNESS / 2
  off = hw / 2 + t  # wall centre distance from the hole axis
  hz = hole_depth / 2
  rgba = 'rgba="0.85 0.45 0.2 0.55"'
  common = f'contype="0" conaffinity="0" group="1" {rgba}'
  # x-walls span the full outer width (covering the corners), y-walls the inner.
  return f"""
      <geom name="wall_px" type="box" pos="{off:.7f} 0 {hz:.6f}" size="{t:.6f} {hw / 2 + 2 * t:.7f} {hz:.6f}" {common}/>
      <geom name="wall_nx" type="box" pos="{-off:.7f} 0 {hz:.6f}" size="{t:.6f} {hw / 2 + 2 * t:.7f} {hz:.6f}" {common}/>
      <geom name="wall_py" type="box" pos="0 {off:.7f} {hz:.6f}" size="{hw / 2:.7f} {t:.6f} {hz:.6f}" {common}/>
      <geom name="wall_ny" type="box" pos="0 {-off:.7f} {hz:.6f}" size="{hw / 2:.7f} {t:.6f} {hz:.6f}" {common}/>"""


def build_xml(
    clearance_m: float,
    peg_width: float = PEG_WIDTH,
    peg_len: float = PEG_LEN,
    hole_depth: float = HOLE_DEPTH,
) -> str:
  """Returns the full scene XML (it includes the peg-patched mjx_panda.xml)."""
  pairs = '\n    '.join(
      _PAIR.format(g=g) for g in ('wall_px', 'wall_nx', 'wall_py', 'wall_ny', 'table')
  )
  g = peg_width / 2  # finger opening that just holds the peg
  hx, hy = HOLE_CENTER
  return f"""<mujoco model="panda peg insertion">
  <include file="{PANDA_XML}"/>

  <statistic center="0.4 0 0.2" extent="0.8"/>

  <option timestep="0.005" iterations="5" ls_iterations="8" integrator="implicitfast">
    <flag eulerdamp="disable"/>
  </option>

  <custom>
    <numeric data="20" name="max_contact_points"/>
  </custom>

  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".3 .3 .3" specular="0.3 0.3 0.3"/>
    <global azimuth="-40" elevation="-30" offwidth="640" offheight="480"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.85 0.9 0.95" width="800" height="800"/>
    <texture type="2d" name="tabletex" builtin="checker" mark="edge" rgb1="0.92 0.92 0.92"
      rgb2="0.85 0.85 0.85" markrgb="0.6 0.6 0.6" width="300" height="300"/>
    <material name="tablemat" texture="tabletex" texuniform="true" texrepeat="10 10" reflectance="0"/>
  </asset>

  <worldbody>
    <light pos="0.5 0 1.5" dir="0 0 -1" directional="true"/>
    <camera name="overview" pos="0.95 -0.5 0.45" xyaxes="0.707 0.707 0 -0.212 0.212 0.707"/>
    <camera name="closeup" pos="{hx} -0.30 0.14" xyaxes="1 0 0 0 0.11 0.3"/>
    <geom name="table" type="box" size="0.5 0.5 0.02" pos="0.4 0 -0.02" material="tablemat"
     contype="0" conaffinity="0" group="1"/>
    <body name="hole_block" mocap="true" pos="{hx} {hy} 0">{hole_walls_xml(clearance_m, peg_width, hole_depth)}
      <site name="hole_entrance" pos="0 0 {hole_depth:.6f}" size="0.002" rgba="0 1 0 1"/>
      <site name="hole_bottom" pos="0 0 0" size="0.002" rgba="1 0 0 1"/>
    </body>
  </worldbody>

  <contact>
    {pairs}
  </contact>

  <sensor>
    <force name="peg_force" site="ft_site"/>
    <torque name="peg_torque" site="ft_site"/>
    <framepos name="tip_pos" objtype="site" objname="peg_tip"/>
    <framelinvel name="tip_vel" objtype="site" objname="peg_tip"/>
    <framepos name="hole_pos" objtype="site" objname="hole_entrance"/>
  </sensor>

  <keyframe>
    <key name="home" qpos="0 0.3 0 -1.57079 0 2.0 -0.7853 {g} {g}"
      ctrl="0 0.3 0 -1.57079 0 2.0 -0.7853 {g}"/>
  </keyframe>
</mujoco>
"""


def get_assets(
    peg_width: float = PEG_WIDTH, peg_len: float = PEG_LEN
) -> Dict[str, bytes]:
  """Playground's Panda assets with the peg patched into mjx_panda.xml."""
  from mujoco_playground._src.manipulation.franka_emika_panda import panda

  assets = dict(panda.get_assets())
  panda_xml = assets[PANDA_XML].decode('utf-8')
  assets[PANDA_XML] = patch_panda_xml(panda_xml, peg_width, peg_len).encode('utf-8')
  return assets


def write_scene(clearance_mm: float, **kwargs) -> Path:
  """Writes the generated scene XML to peg_insertion/assets for inspection."""
  ASSETS_DIR.mkdir(parents=True, exist_ok=True)
  path = ASSETS_DIR / f'scene_c{clearance_mm:.1f}.xml'
  path.write_text(build_xml(clearance_mm / 1000.0, **kwargs))
  return path
