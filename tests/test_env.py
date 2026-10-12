"""Environment tests: geometry, shapes, stability, randomization, force sensor."""

import functools

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest

import peg_insertion  # noqa: F401  (registers the envs)
from peg_insertion import xml_builder
from peg_insertion.env import PegInsertion, default_config, domain_randomize
from mujoco_playground import wrapper

CLEARANCES_MM = (2.0, 1.0, 0.5)


def _mj_model(clearance_mm):
  xml = xml_builder.build_xml(clearance_mm / 1000.0)
  return mujoco.MjModel.from_xml_string(xml, assets=xml_builder.get_assets())


@pytest.fixture(scope='module')
def env():
  return PegInsertion()


@pytest.fixture(scope='module')
def force_env():
  cfg = default_config()
  cfg.use_force_obs = True
  return PegInsertion(cfg)


@pytest.mark.parametrize('clearance_mm', CLEARANCES_MM)
def test_hole_width_matches_clearance(clearance_mm):
  m = _mj_model(clearance_mm)
  px = m.geom_pos[m.geom('wall_px').id][0] - m.geom_size[m.geom('wall_px').id][0]
  nx = m.geom_pos[m.geom('wall_nx').id][0] + m.geom_size[m.geom('wall_nx').id][0]
  py = m.geom_pos[m.geom('wall_py').id][1] - m.geom_size[m.geom('wall_py').id][1]
  ny = m.geom_pos[m.geom('wall_ny').id][1] + m.geom_size[m.geom('wall_ny').id][1]
  expected = xml_builder.PEG_WIDTH + 2 * clearance_mm / 1000.0
  assert abs((px - nx) - expected) < 1e-6
  assert abs((py - ny) - expected) < 1e-6


def test_peg_attached_below_hand():
  m = _mj_model(2.0)
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, m.keyframe('home').id)
  mujoco.mj_forward(m, d)
  tip = d.site_xpos[m.site('peg_tip').id]
  tcp = d.site_xpos[m.site('gripper').id]
  hand = d.xpos[m.body('hand').id]
  assert tip[2] < hand[2]
  expected = xml_builder.PEG_LEN - xml_builder.PEG_INSET
  assert abs(np.linalg.norm(tip - tcp) - expected) < 1e-4


def test_fixed_orientation_is_down_and_axis_aligned(env):
  rot = env._hand_rot_check
  # hand z points down; hand x is along a world axis (peg faces match walls)
  assert np.allclose(rot[:, 2], [0, 0, -1], atol=1e-3)
  assert np.max(np.abs(rot[:2, 0])) > 1 - 1e-3


def test_reset_step_shapes(env, force_env):
  for e, size in ((env, 10), (force_env, 16)):
    state = jax.jit(e.reset)(jax.random.PRNGKey(0))
    assert state.obs['state'].shape == (size,)
    state = jax.jit(e.step)(state, jp.zeros(3))
    assert state.obs['state'].shape == (size,)
    assert np.isfinite(float(state.reward))


def test_random_steps_stable(env):
  step = jax.jit(env.step)
  state = jax.jit(env.reset)(jax.random.PRNGKey(1))
  rng = jax.random.PRNGKey(2)
  for _ in range(50):
    rng, key = jax.random.split(rng)
    state = step(state, jax.random.uniform(key, (3,), minval=-1, maxval=1))
  assert not np.any(np.isnan(np.array(state.data.qpos)))


def test_auto_reset_resamples_hole(env):
  wrapped = wrapper.wrap_for_brax_training(
      env, episode_length=3, action_repeat=1,
      randomization_fn=functools.partial(
          domain_randomize, rng=jax.random.split(jax.random.PRNGKey(0), 2)
      ),
      full_reset=True,
  )
  reset = jax.jit(wrapped.reset)
  step = jax.jit(wrapped.step)
  state = reset(jax.random.split(jax.random.PRNGKey(3), 2))
  holes = [np.array(state.data.mocap_pos)]
  for _ in range(7):  # episodes end at steps 3 and 6 -> two auto-resets
    state = step(state, jp.zeros((2, 3)))
    holes.append(np.array(state.data.mocap_pos))
  # Within an episode the hole is fixed; each auto-reset re-samples it.
  assert np.allclose(holes[0], holes[2])
  assert not np.allclose(holes[2], holes[3])
  assert not np.allclose(holes[3], holes[6])


def test_force_sensor_free_and_pressed():
  m = _mj_model(2.0)
  d = mujoco.MjData(m)
  e = PegInsertion()
  f_adr = m.sensor_adr[m.sensor('peg_force').id]

  def settle(tip_target, n=200):
    q = e._solve_ik_np(np.array(tip_target) - e._tip_from_ik, e._home_ref_q)
    d.qpos[:7] = q
    d.qpos[7:9] = e._grip
    d.qvel[:] = 0
    d.ctrl[:7] = q
    d.ctrl[7] = e._grip
    mujoco.mj_forward(m, d)
    for _ in range(n):
      mujoco.mj_step(m, d)
    return np.linalg.norm(d.sensordata[f_adr:f_adr + 3])

  # Free space, 10 cm above the table away from the hole.
  free = settle([0.35, 0.15, 0.10])
  assert free < 1.0
  # Tip commanded 5 mm below the table top, away from the hole walls.
  d.qpos[:] = 0
  settle([0.35, 0.15, 0.02], n=100)
  pressed = settle([0.35, 0.15, -0.005])
  assert pressed > 2.0
