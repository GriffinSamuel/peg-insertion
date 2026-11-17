"""Synthetic contracts for Phase F touch helpers; no simulator construction."""
from takehome import followup_runtime  # configure caches before numerical imports

import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

from takehome.followup_touch import TOUCH_STEP_CAP, TouchEnv, spiral_proposal
from takehome.followup_touch_diagnosis import describe, pressure_point


def test_spiral_stays_bounded_and_pauses_without_activity():
    center = jnp.array([0.0002, -0.0001])
    guess = center.copy()
    theta = jnp.array(0.0)
    cap = 0.001
    for _ in range(40):
        updated, next_theta = spiral_proposal(guess, center, theta, True, cap)
        assert np.max(np.abs(np.asarray(updated - guess))) <= TOUCH_STEP_CAP + 1e-8
        assert np.max(np.abs(np.asarray(updated))) <= cap + 1e-8
        assert float(next_theta) > float(theta)
        guess, theta = updated, next_theta
    paused, paused_theta = spiral_proposal(guess, center, theta, False, cap)
    np.testing.assert_array_equal(np.asarray(paused), np.asarray(guess))
    assert float(paused_theta) == float(theta)


def _touch_fixture(*, contact=True, stalled_history=True, near_surface=True,
                   terminated=False, truncated=False):
    env = object.__new__(TouchEnv)
    env.base_obs_size = 16
    env.aim = SimpleNamespace(guess_cap=0.003)
    raw = jnp.zeros((16,))
    raw = raw.at[2].set(0.0005 if near_surface else 0.0015)
    if stalled_history:
        history = jnp.zeros((20, 16))
    else:
        # The earlier five readings descend by 1 mm relative to the later five.
        history = jnp.zeros((20, 16)).at[10:15, 2].set(0.002)
    state = SimpleNamespace(
        obs={'state': raw},
        info={
            'contact_seen': jnp.asarray(contact),
            'reading_history': history,
            'step_count': jnp.asarray(10),
            'learner_terminated': jnp.asarray(terminated),
            'learner_truncated': jnp.asarray(truncated),
            'touch_started': jnp.asarray(False),
            'touch_center': jnp.zeros((2,)),
            'touch_theta': jnp.asarray(0.0),
            'aim_guess': jnp.array([0.0001, -0.0002]),
            # Deliberately no hidden 'bias' key.
        },
    )
    return env, state


def test_touch_gate_activates_on_observable_conditions_without_hidden_bias():
    env, state = _touch_fixture()
    guess, _, center, active = env.touch_update(state)
    assert bool(active)
    np.testing.assert_array_equal(np.asarray(center), np.asarray(state.info['aim_guess']))
    assert np.isfinite(np.asarray(guess)).all()


def test_touch_gate_requires_contact_stall_surface_and_unended_state():
    cases = [
        {'contact': False},
        {'stalled_history': False},
        {'near_surface': False},
        {'terminated': True},
        {'truncated': True},
    ]
    for options in cases:
        env, state = _touch_fixture(**options)
        _, _, _, active = env.touch_update(state)
        assert not bool(active), options


def test_pressure_point_recovers_known_contact_and_is_sign_and_rotation_invariant():
    force_world = np.array([3.0, -2.0, 10.0])
    known_r = np.array([0.012, -0.007, -0.060])
    torque_world = np.cross(known_r, force_world)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    point, usable = pressure_point(force_world, torque_world, np.eye(3))
    rotated_point, rotated_usable = pressure_point(
        force_world @ rotation, torque_world @ rotation, rotation
    )
    flipped_point, flipped_usable = pressure_point(-force_world, -torque_world, np.eye(3))
    assert bool(usable) and bool(rotated_usable) and bool(flipped_usable)
    np.testing.assert_allclose(point, known_r[:2], atol=1e-12)
    np.testing.assert_allclose(rotated_point, point, atol=1e-12)
    np.testing.assert_allclose(flipped_point, point, atol=1e-12)


def test_saved_diagnosis_pairs_sensor_obs_t_with_geometry_true_rel_t_minus_1():
    # Each incoming-state observation is paired with the prior transition's
    # saved geometry. The only partial-overlap sample is true_rel[9], aligned
    # with sensor observation obs[10] and reported as incoming step 10.
    obs = np.zeros((12, 1, 16), dtype=np.float64)
    true_rel = np.zeros((12, 1, 3), dtype=np.float64)
    valid = np.ones((12, 1), dtype=bool)
    true_rel[9, 0] = [0.010, -0.006, 0.0]

    force = np.array([3.0, -2.0, 10.0])
    r = np.array([0.010, -0.006, -0.060])
    torque = np.cross(r, force)
    # obs sensor channels are scaled by 20 for force and 2 for torque.
    for t in range(6, 11):
        obs[t, 0, 10:13] = force / 20.0
        obs[t, 0, 13:16] = torque / 2.0

    result = describe({'obs': obs, 'true_rel': true_rel, 'valid': valid}, np.eye(3))
    assert result['partial_rest_episode_count'] == 1
    row = result['rows'][0]
    assert row['incoming_control_step'] == 10
    np.testing.assert_allclose(row['true_xy_mm'], [10.0, -6.0], atol=1e-10)
    np.testing.assert_allclose(row['force_n'], force, atol=1e-10)
    np.testing.assert_allclose(row['estimated_pressure_xy_mm'], [10.0, -6.0], atol=1e-10)
