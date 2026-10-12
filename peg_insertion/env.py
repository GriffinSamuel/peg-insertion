"""PegInsertion: a Franka Panda inserts a held square peg into a square hole.

Control scheme (copied from Playground's PandaPickCubeCartesian): the policy
outputs a 3-D Cartesian step (dx, dy, dz) for the peg tip; analytic inverse
kinematics turns the new tip target into joint position targets for the arm's
position servos. The gripper orientation is held fixed (pointing straight down).

The hole position is randomized every episode, and the policy sees the hole
through a noisy, per-episode *biased* position estimate (like a camera that is
slightly miscalibrated). At tight clearances that bias is larger than the gap,
which is what makes the task hard without touch. The optional force
observation gives the policy a wrist force/torque sensor.
"""

import functools
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as reward_util
from mujoco_playground._src.manipulation.franka_emika_panda import panda_kinematics

from peg_insertion import xml_builder

_ARM_DOF = 7
_PEG_WEIGHT_N = 0.05 * 9.81


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=200,
      action_repeat=1,
      action_scale=0.002,  # metres per step per axis (fine control for insertion)
      clearance_mm=2.0,
      use_force_obs=False,
      peg_width=xml_builder.PEG_WIDTH,
      peg_len=xml_builder.PEG_LEN,
      hole_depth=xml_builder.HOLE_DEPTH,
      hole_xy_noise=0.01,  # metres: random hole offset per episode (uniform +-)
      # Per-episode constant bias on the observed hole xy position. With a
      # 0.75 mm std, |bias| > 0.5 mm in about half of episodes per axis, so
      # position-only control should struggle at 0.5 mm clearance while force
      # feedback can still find the hole.
      obs_hole_bias_std=0.00075,
      obs_noise_std=0.0002,  # metres: per-step white noise on positions
      start_height=0.05,  # tip starts this far above the hole entrance
      start_xy_noise=0.01,  # and this far off-centre (uniform +-)
      max_servo_error=0.003,  # anti-windup: target never leads the tip by more (m)
      force_clip=20.0,  # N, for normalising the force observation
      torque_clip=2.0,  # N*m, for normalising the torque observation
      reward_config=config_dict.create(
          scales=config_dict.create(
              reach=1.0,
              align=1.0,
              insert=2.0,
              success=10.0,
              action_rate=-0.05,
              force_pen=-0.01,
              no_soln=-0.01,
          )
      ),
      impl='jax',
      naconmax=20 * 1024,  # max_contact_points * num_envs; scale with num_envs
      njmax=128,
  )


def force_config() -> config_dict.ConfigDict:
  cfg = default_config()
  cfg.use_force_obs = True
  return cfg


def _down_rotation(fk_rot: np.ndarray) -> np.ndarray:
  """Rotation with hand z pointing straight down and hand x snapped to the
  nearest world axis, so the square peg's faces line up with the hole walls."""
  x = np.array(fk_rot[:, 0], dtype=np.float64)
  yaw = np.arctan2(x[1], x[0])
  yaw = np.round(yaw / (np.pi / 2)) * (np.pi / 2)
  x_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
  z_axis = np.array([0.0, 0.0, -1.0])
  y_axis = np.cross(z_axis, x_axis)
  return np.stack([x_axis, y_axis, z_axis], axis=1)


class PegInsertion(mjx_env.MjxEnv):
  """Insert a held square peg into a square hole with a given clearance."""

  def __init__(
      self,
      config: Optional[config_dict.ConfigDict] = None,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config or default_config(), config_overrides)
    cfg = self._config
    self._clearance = float(cfg.clearance_mm) / 1000.0
    self._pw = float(cfg.peg_width)
    self._pl = float(cfg.peg_len)
    self._hd = float(cfg.hole_depth)

    xml = xml_builder.build_xml(self._clearance, self._pw, self._pl, self._hd)
    self._xml_path = (
        xml_builder.ASSETS_DIR / f'scene_c{float(cfg.clearance_mm):.1f}.xml'
    ).as_posix()
    self._model_assets = xml_builder.get_assets(self._pw, self._pl)
    mj_model = mujoco.MjModel.from_xml_string(xml, assets=self._model_assets)
    mj_model.opt.timestep = cfg.sim_dt
    self._mj_model = mj_model
    self._mjx_model = mjx_env.put_model(mj_model, impl=cfg.impl)
    self._post_init()

  # ---------------------------------------------------------------- setup
  def _post_init(self) -> None:
    m = self._mj_model
    self._tip_site = m.site('peg_tip').id
    self._entrance_site = m.site('hole_entrance').id
    self._gripper_site = m.site('gripper').id
    self._hand_body = m.body('hand').id
    self._peg_body = m.body('peg').id
    self._hole_mocap = int(m.body('hole_block').mocapid[0])

    def sensor_slice(name):
      sid = m.sensor(name).id
      adr = int(m.sensor_adr[sid])
      return slice(adr, adr + int(m.sensor_dim[sid]))

    self._force_slice = sensor_slice('peg_force')
    self._torque_slice = sensor_slice('peg_torque')
    self._tip_vel_slice = sensor_slice('tip_vel')

    key = m.keyframe('home')
    self._home_q = np.array(key.qpos[:_ARM_DOF])
    self._q7 = float(self._home_q[6])
    self._grip = self._pw / 2
    self._lowers, self._uppers = (jp.array(x) for x in m.actuator_ctrlrange.T)
    self._hole_home = np.array(m.body('hole_block').pos)

    # Fixed gripper orientation: straight down, faces aligned with the hole.
    fk = np.array(panda_kinematics.compute_franka_fk(jp.array(self._home_q)))
    self._tip_rot = _down_rotation(fk[:3, :3])

    # Measure the constant offset from the IK frame to the peg tip by solving
    # IK for a reference pose and running forward kinematics in MuJoCo.
    entrance_home = self._hole_home + np.array([0.0, 0.0, self._hd])
    ref_ik_pos = entrance_home + np.array([0.0, 0.0, 0.10])
    q_ref = self._solve_ik_np(ref_ik_pos, self._home_q)
    d = mujoco.MjData(m)
    d.qpos[:_ARM_DOF] = q_ref
    d.qpos[_ARM_DOF:] = self._grip
    mujoco.mj_forward(m, d)
    self._tip_from_ik = np.array(d.site_xpos[self._tip_site]) - ref_ik_pos
    self._hand_rot_check = np.array(d.xmat[self._hand_body]).reshape(3, 3)
    self._home_ref_q = q_ref

  def _solve_ik_np(self, ik_pos: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
    tmat = np.eye(4)
    tmat[:3, :3] = self._tip_rot
    tmat[:3, 3] = ik_pos
    q = panda_kinematics.compute_franka_ik(
        jp.array(tmat), self._q7, jp.array(q_seed)
    )
    q = np.array(q)
    if np.any(np.isnan(q)):
      raise ValueError(f'IK failed for reference pose {ik_pos}')
    return q

  def _ik(self, tip_target: jax.Array, q_seed: jax.Array):
    """Joint targets that put the peg tip at tip_target (fixed orientation)."""
    tmat = jp.eye(4)
    tmat = tmat.at[:3, :3].set(jp.array(self._tip_rot))
    tmat = tmat.at[:3, 3].set(tip_target - jp.array(self._tip_from_ik))
    q = panda_kinematics.compute_franka_ik(tmat, jp.array(self._q7), q_seed)
    no_soln = jp.any(jp.isnan(q))
    return jp.where(no_soln, q_seed, q), no_soln

  # ---------------------------------------------------------------- reset
  def reset(self, rng: jax.Array) -> mjx_env.State:
    cfg = self._config
    rng, k_hole, k_bias, k_start, k_noise = jax.random.split(rng, 5)
    hole_xy = jp.array(self._hole_home[:2]) + jax.random.uniform(
        k_hole, (2,), minval=-cfg.hole_xy_noise, maxval=cfg.hole_xy_noise
    )
    hole_pos = jp.concatenate([hole_xy, jp.zeros(1)])
    entrance = hole_pos + jp.array([0.0, 0.0, self._hd])
    bias_xy = cfg.obs_hole_bias_std * jax.random.normal(k_bias, (2,))
    bias = jp.concatenate([bias_xy, jp.zeros(1)])

    start_xy = jax.random.uniform(
        k_start, (2,), minval=-cfg.start_xy_noise, maxval=cfg.start_xy_noise
    )
    start_tip = entrance + jp.concatenate(
        [start_xy, jp.array([cfg.start_height])]
    )
    q_arm, _ = self._ik(start_tip, jp.array(self._home_ref_q))
    qpos = jp.concatenate([q_arm, jp.full(2, self._grip)])
    ctrl = jp.concatenate([q_arm, jp.full(1, self._grip)])

    data = mjx_env.make_data(
        self._mj_model,
        qpos=qpos,
        qvel=jp.zeros(self._mj_model.nv),
        ctrl=ctrl,
        mocap_pos=hole_pos[None, :],
        mocap_quat=jp.array([[1.0, 0.0, 0.0, 0.0]]),
        impl=self._mjx_model.impl.value,
        naconmax=cfg.naconmax,
        njmax=cfg.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    info = {
        'rng': rng,
        'current_pos': start_tip,  # commanded tip target
        'prev_action': jp.zeros(3),
        'bias': bias,
        'hole_pos': hole_pos,
        'step_count': jp.array(0, dtype=jp.int32),
        'succeeded': jp.array(0.0),
        'peak_force': jp.array(0.0),
        'first_success_step': jp.array(-1.0),
        'depth': jp.array(0.0),
        'xy_err': jp.linalg.norm(start_tip[:2] - hole_xy),
    }
    metrics = {
        'success': jp.array(0.0),
        'insert_reward': jp.array(0.0),
        'reach_reward': jp.array(0.0),
        'force_pen': jp.array(0.0),
    }
    obs = self._get_obs(data, info, jp.zeros(3), k_noise)
    return mjx_env.State(data, obs, jp.array(0.0), jp.array(0.0), metrics, info)

  # ----------------------------------------------------------------- step
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    cfg = self._config
    info = dict(state.info)
    rng, k_noise = jax.random.split(info['rng'])
    action = jp.clip(action, -1.0, 1.0)
    data = state.data

    entrance = info['hole_pos'] + jp.array([0.0, 0.0, self._hd])
    tip_now = data.site_xpos[self._tip_site]
    target = info['current_pos'] + action * cfg.action_scale
    lo = entrance + jp.array([-0.05, -0.05, -(self._hd + 0.005)])
    hi = entrance + jp.array([0.05, 0.05, 0.10])
    target = jp.clip(target, lo, hi)
    # Anti-windup: a blocked peg can never build up more than max_servo_error
    # of target lead, which keeps contact forces bounded.
    target = jp.clip(
        target, tip_now - cfg.max_servo_error, tip_now + cfg.max_servo_error
    )

    q_arm, no_soln = self._ik(target, data.ctrl[:_ARM_DOF])
    target = jp.where(no_soln, info['current_pos'], target)
    ctrl = jp.concatenate([q_arm, jp.full(1, self._grip)])
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)
    data = mjx_env.step(self.mjx_model, data, ctrl, self.n_substeps)

    # --- task geometry from the true (privileged) state
    tip = data.site_xpos[self._tip_site]
    entrance = data.site_xpos[self._entrance_site]
    rel = tip - entrance
    xy_err = jp.linalg.norm(rel[:2])
    depth = jp.maximum(0.0, -rel[2])
    in_hole = xy_err < (self._clearance + self._pw / 2)
    force = data.sensordata[self._force_slice]
    force_mag = jp.linalg.norm(force)

    reached_depth = (depth >= 0.8 * self._hd) & in_hole
    newly = reached_depth & (info['succeeded'] < 0.5)
    succeeded = jp.maximum(info['succeeded'], reached_depth.astype(float))

    terms = self._reward_terms(rel, xy_err, depth, in_hole, newly, action,
                               info['prev_action'], force_mag, no_soln)
    scales = cfg.reward_config.scales
    pos_sum = sum(v for v in scales.values() if v > 0)
    reward = sum(scales[k] * terms[k] for k in terms) / pos_sum

    out = jp.any(jp.abs(rel[:2]) > 0.07) | (rel[2] > 0.13)
    nan = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = (out | nan).astype(float)
    reward = jp.where(nan, 0.0, reward)

    step_count = info['step_count'] + 1
    info.update(
        rng=rng,
        current_pos=target,
        prev_action=action,
        step_count=step_count,
        succeeded=succeeded,
        peak_force=jp.maximum(info['peak_force'], force_mag),
        first_success_step=jp.where(
            newly, step_count.astype(float), info['first_success_step']
        ),
        depth=depth,
        xy_err=xy_err,
    )
    # Merge (don't replace): wrappers add their own keys, e.g. 'reward'.
    metrics = {
        **state.metrics,
        'success': newly.astype(float),
        'insert_reward': terms['insert'],
        'reach_reward': terms['reach'],
        'force_pen': terms['force_pen'],
    }
    obs = self._get_obs(data, info, action, k_noise)
    return state.replace(
        data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info
    )

  def _reward_terms(self, rel, xy_err, depth, in_hole, newly, action,
                    prev_action, force_mag, no_soln) -> Dict[str, jax.Array]:
    reach = reward_util.tolerance(
        jp.linalg.norm(rel), bounds=(0.0, 0.002), margin=0.08, sigmoid='linear'
    )
    near_entrance = rel[2] < 0.01  # within 1 cm above the entrance (or inside)
    align = reward_util.tolerance(
        xy_err, bounds=(0.0, self._clearance * 0.5), margin=0.02
    ) * near_entrance
    insert = jp.clip(depth / self._hd, 0.0, 1.0) * in_hole
    return {
        'reach': reach,
        'align': align,
        'insert': insert,
        'success': newly.astype(float),
        'action_rate': jp.sum(jp.square(action - prev_action)),
        'force_pen': jp.clip(force_mag - 15.0, 0.0, 20.0) / 20.0,
        'no_soln': no_soln.astype(float),
    }

  # ------------------------------------------------------------------ obs
  def _get_obs(self, data: mjx.Data, info: Dict[str, Any], action: jax.Array,
               rng: jax.Array) -> Dict[str, jax.Array]:
    cfg = self._config
    tip = data.site_xpos[self._tip_site]
    entrance = data.site_xpos[self._entrance_site]
    noise = cfg.obs_noise_std * jax.random.normal(rng, (3,))
    rel = tip - entrance + info['bias'] + noise
    tip_vel = data.sensordata[self._tip_vel_slice]
    depth = jp.maximum(0.0, -rel[2:3])
    parts = [rel, tip_vel, action, depth]
    if cfg.use_force_obs:
      force = data.sensordata[self._force_slice] / cfg.force_clip
      torque = data.sensordata[self._torque_slice] / cfg.torque_clip
      parts += [jp.clip(force, -1.0, 1.0), jp.clip(torque, -1.0, 1.0)]
    return {'state': jp.concatenate(parts)}

  # ----------------------------------------------------------- properties
  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return 3

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  @property
  def clearance(self) -> float:
    return self._clearance


@functools.lru_cache(maxsize=1)
def _peg_body_id() -> int:
  xml = xml_builder.build_xml(0.002)
  mj_model = mujoco.MjModel.from_xml_string(xml, assets=xml_builder.get_assets())
  return int(mj_model.body('peg').id)


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Per-env randomization of contact friction, peg mass and arm servo gains."""
  peg_body = _peg_body_id()

  @jax.vmap
  def rand(key):
    k_fric, k_mass, k_gain = jax.random.split(key, 3)
    fric_scale = jax.random.uniform(k_fric, (), minval=0.7, maxval=1.3)
    pair_friction = model.pair_friction.at[:, :2].multiply(fric_scale)
    mass_scale = jax.random.uniform(k_mass, (), minval=0.8, maxval=1.2)
    body_mass = model.body_mass.at[peg_body].multiply(mass_scale)
    gain_scale = jax.random.uniform(
        k_gain, (_ARM_DOF,), minval=0.9, maxval=1.1
    )
    # Scale kp and -kp together, or the position servo becomes biased.
    gainprm = model.actuator_gainprm.at[:_ARM_DOF, 0].multiply(gain_scale)
    biasprm = model.actuator_biasprm.at[:_ARM_DOF, 1].multiply(gain_scale)
    return pair_friction, body_mass, gainprm, biasprm

  pair_friction, body_mass, gainprm, biasprm = rand(rng)
  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      'pair_friction': 0,
      'body_mass': 0,
      'actuator_gainprm': 0,
      'actuator_biasprm': 0,
  })
  model = model.tree_replace({
      'pair_friction': pair_friction,
      'body_mass': body_mass,
      'actuator_gainprm': gainprm,
      'actuator_biasprm': biasprm,
  })
  return model, in_axes
