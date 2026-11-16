"""Predeclared camera formula checks and test-only integration probes."""
from types import SimpleNamespace
from takehome import runtime  # noqa: F401
import jax
import jax.numpy as jp
import numpy as np
import pytest
from takehome.camera import CameraEnv, CameraConfig, CameraExperimentRunner, PROBLEMS, cap_multiplier
from takehome.variants import VariantEnv
from takehome.correction import AimConfig, policy_digest


def fake_base_observation(self, data, info, action, rng):
    del action, rng
    rel = data.site_xpos[0] - data.site_xpos[1] + info['bias']
    return {'state': jp.concatenate([rel, jp.zeros(6), jp.maximum(0., -rel[2:3]), jp.zeros(6)])}


def fixture_env(monkeypatch, problem):
    monkeypatch.setattr(VariantEnv, '_get_obs', fake_base_observation)
    env = object.__new__(CameraEnv)
    env.camera_problem = problem
    env._tip_site, env._entrance_site = 0, 1
    return env


@pytest.mark.parametrize('problem', ('normal', *PROBLEMS))
def test_camera_formula_changes_reading_and_recorded_error_only(monkeypatch, problem):
    env = fixture_env(monkeypatch, problem)
    rel = jp.array([.004, -.002, -.003])
    bias = jp.array([-.001, .0005, 0.]) * cap_multiplier(problem)
    info = {'bias': bias, 'step_count': jp.array(100)}
    data = SimpleNamespace(site_xpos=jp.stack([rel, jp.zeros(3)]))
    original_geometry = np.asarray(data.site_xpos).copy()
    key = jax.random.PRNGKey(15)
    observed = env._get_obs(data, info, jp.zeros(3), key)['state']
    expected = np.asarray(rel + bias).copy()
    if problem == 'one_sided':
        expected[:2] = np.asarray(rel[:2] + jp.abs(bias[:2]))
    elif problem == 'drift':
        expected[:2] += np.array([.0005, -.0005])
    elif problem == 'jitter':
        expected[:2] += np.asarray(.0006 * jax.random.normal(jax.random.fold_in(key, 7001), (2,)))
    elif problem == 'rotation':
        angle = np.pi / 18
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        expected[:2] = rotation @ np.asarray(rel[:2]) + np.asarray(bias[:2])
    elif problem == 'height':
        expected[2] += .002
    np.testing.assert_allclose(observed[:3], expected, atol=1e-9)
    np.testing.assert_allclose(info['bias'], expected - np.asarray(rel), atol=1e-9)
    np.testing.assert_array_equal(info['camera_base_bias'], bias)
    np.testing.assert_array_equal(data.site_xpos, original_geometry)
    assert float(observed[9]) == pytest.approx(max(0., -expected[2]), abs=1e-9)
    assert observed.shape == (16,)


def test_delay_uses_three_step_old_geometry_and_does_not_accumulate_bias(monkeypatch):
    env = fixture_env(monkeypatch, 'delay')
    bias = jp.array([.001, -.002, 0.])
    info = {'bias': bias, 'step_count': jp.array(0)}
    for step in range(8):
        rel = jp.array([step*.001, 0., -.002*step])
        data = SimpleNamespace(site_xpos=jp.stack([rel, jp.zeros(3)]))
        info['step_count'] = jp.array(step)
        obs = env._get_obs(data, info, jp.zeros(3), jax.random.PRNGKey(step))['state']
        delayed = max(step-3, 0)
        expected = jp.array([delayed*.001, 0., -.002*delayed]) + bias
        np.testing.assert_allclose(obs[:3], expected, atol=1e-9)
        np.testing.assert_array_equal(info['camera_base_bias'], bias)


def test_oracle_corrects_available_xy_error_with_cap_but_not_height():
    env = object.__new__(CameraEnv)
    env.oracle = True
    env.aim = AimConfig(.0001, .003)
    state = SimpleNamespace(info={'bias': jp.array([.005, -.002, .002])})
    np.testing.assert_allclose(env.next_guess(state, jp.zeros(2)), [.003, -.002])


@pytest.mark.parametrize('problem', ('delay', 'height'))
def test_camera_integration_is_finite_frozen_and_test_only(problem):
    runner = CameraExperimentRunner(CameraConfig(camera_problem=problem, outputs=3))
    initial_hash = policy_digest(runner.env.frozen_policy)
    summary, _, trace = runner.episodes.test(runner.learner.initialize(999), episodes=4, seed=123)
    assert summary['episodes'] == 4 and summary['test_only'] and not summary['learned_from']
    assert all(np.isfinite(value).all() for value in trace.values())
    assert trace['obs'].shape == (200, 4, 18)
    np.testing.assert_array_equal(trace['action'], np.zeros_like(trace['action']))
    assert policy_digest(runner.env.frozen_policy) == initial_hash
