"""Synthetic control tests for observable follow-up mechanics; no physics setup."""
from takehome import followup_runtime  # noqa: F401 -- initialize paths before JAX imports

from types import SimpleNamespace

import jax.numpy as jp
import numpy as np

from takehome.followup_config import FollowupConfig
from takehome.followup_env import FollowupEnv
from takehome.followup_learner import FollowupLearner


def _config(**changes):
    defaults = dict(policy="force_0.5mm_s0", outputs=2, total_episodes=200,
                    fill_episodes=40, drift_gate=False, pullback=False,
                    fast_contact=False, constant_fill=False, within_grid=False,
                    memory=False, success_labels=False, step_cap=0.0001,
                    movement_cap=0.0002, guess_cap=0.003)
    defaults.update(changes)
    return FollowupConfig(**defaults)


def _env(**changes):
    """Bypass the physics constructor; these methods need only config/aim/grid."""
    env = object.__new__(FollowupEnv)
    env.followup_config = _config(**changes)
    env.oracle = False
    env.aim = SimpleNamespace(step_cap=env.followup_config.step_cap,
                              guess_cap=env.followup_config.guess_cap)
    env.base_obs_size = 16
    from takehome.followup_env import offset_grid
    env.grid = offset_grid()
    return env


def _state(*, guess=(0.001, -0.001), force=(0.0, 0.0, 0.0), z=0.0005,
           history_z=None, contact=False, step_count=12, grid_target=40,
           constant_fill_active=False, bias=(0.0, 0.0)):
    raw = jp.zeros((16,)).at[2].set(z).at[10:13].set(jp.asarray(force))
    if history_z is None:
        history_z = np.linspace(0.002, 0.001, 20, dtype=np.float32)
    history = jp.zeros((20, 16)).at[:, 2].set(jp.asarray(history_z))
    info = {
        'aim_guess': jp.asarray(guess, dtype=jp.float32),
        'contact_seen': jp.asarray(contact),
        'reading_history': history,
        'step_count': jp.asarray(step_count),
        'grid_target': jp.asarray(grid_target, dtype=jp.int32),
        'constant_fill_active': jp.asarray(constant_fill_active),
        'bias': jp.asarray(bias, dtype=jp.float32),
    }
    return SimpleNamespace(obs={'state': raw}, info=info)


def test_drift_gate_requires_contact_and_stall_but_allows_updates_then():
    env = _env(drift_gate=True)
    action = jp.asarray([1.0, -1.0])
    start = jp.asarray([0.001, -0.001])
    # A clear observed descent (> 0.2 mm across the history windows) is not
    # classified as stalled, so the contact gate holds the previous guess.
    descending = np.linspace(0.004, 0.0, 20, dtype=np.float32)
    no_contact = _state(guess=start, force=(0, 0, 0), history_z=descending)
    np.testing.assert_allclose(env.next_guess(no_contact, action), [0, 0], atol=1e-8)
    contacted_descending = _state(guess=start, force=(0.3, 0, 0),
                                  history_z=descending)
    np.testing.assert_allclose(env.next_guess(contacted_descending, action), start,
                               atol=1e-8)
    # Flat camera height is stalled; with observed contact this permits the
    # proposed correction.
    flat = np.full(20, 0.001, dtype=np.float32)
    contacted_stalled = _state(guess=start, force=(0.3, 0, 0), history_z=flat)
    np.testing.assert_allclose(env.next_guess(contacted_stalled, action),
                               start + jp.asarray([0.0001, -0.0001]), atol=1e-8)


def test_fast_contact_doubles_nominal_nudge_only_after_contact():
    env = _env(fast_contact=True)
    action = jp.asarray([1.0, 0.0])
    start = (0.0, 0.0)
    no_contact = _state(guess=start, force=(0, 0, 0))
    contact_now = _state(guess=start, force=(0.3, 0, 0))
    np.testing.assert_allclose(env.next_guess(no_contact, action), [0.0001, 0], atol=1e-8)
    np.testing.assert_allclose(env.next_guess(contact_now, action), [0.0002, 0], atol=1e-8)


def test_pullback_shrinks_existing_guess_by_one_percent_with_zero_action():
    env = _env(pullback=True)
    guess = jp.asarray([0.001, -0.002])
    got = env.next_guess(_state(guess=guess), jp.zeros(2))
    np.testing.assert_allclose(got, guess * 0.99, atol=1e-8)


def test_constant_fill_active_preserves_guess_over_gate_and_pullback():
    env = _env(drift_gate=True, pullback=True, constant_fill=True)
    guess = jp.asarray([0.001, -0.001])
    state = _state(guess=guess, force=(0.3, 0, 0), constant_fill_active=True,
                   history_z=np.full(20, 0.001, dtype=np.float32))
    np.testing.assert_allclose(env.next_guess(state, jp.asarray([1.0, 1.0])),
                               guess, atol=1e-8)


def test_within_grid_proposal_uses_height_and_history_not_force_and_is_capped():
    env = _env(within_grid=True)
    a = _state(guess=(0.0, 0.0), force=(0, 0, 0), z=0.0005,
               history_z=np.full(20, 0.001, dtype=np.float32), grid_target=80)
    b = _state(guess=(0.0, 0.0), force=(2.0, -1.0, 0.5), z=0.0005,
               history_z=np.full(20, 0.001, dtype=np.float32), grid_target=80)
    action = jp.asarray([0.0, 0.0])
    proposal_a = env.next_guess(a, action)
    proposal_b = env.next_guess(b, action)
    np.testing.assert_allclose(proposal_a, proposal_b, atol=1e-8)
    assert float(jp.max(jp.abs(proposal_a - a.info['aim_guess']))) <= 0.00020001


def test_nonoracle_control_outputs_ignore_hidden_bias():
    controls = [
        _env(drift_gate=True), _env(pullback=True), _env(fast_contact=True),
        _env(constant_fill=True, drift_gate=True, pullback=True), _env(within_grid=True),
    ]
    state_a = _state(guess=(0.0002, -0.0001), force=(0.3, 0, 0),
                     history_z=np.full(20, 0.001, dtype=np.float32), grid_target=80,
                     constant_fill_active=True, bias=(0.0, 0.0))
    state_b = _state(guess=(0.0002, -0.0001), force=(0.3, 0, 0),
                     history_z=np.full(20, 0.001, dtype=np.float32), grid_target=80,
                     constant_fill_active=True, bias=(9.0, -8.0))
    action = jp.asarray([0.5, -0.5])
    for active in (False,True):
        state_a.info['constant_fill_active']=jp.asarray(active)
        state_b.info['constant_fill_active']=jp.asarray(active)
        for env in controls:
            np.testing.assert_array_equal(env.next_guess(state_a, action),
                                          env.next_guess(state_b, action))


def test_memory_augmentation_appends_raw_guess_mean_then_change():
    env = _env(memory=True)
    raw = jp.arange(16, dtype=jp.float32)
    guess = jp.asarray([0.25, -0.5])
    history = jp.arange(20 * 16, dtype=jp.float32).reshape((20, 16))
    augmented = env.augment(raw, guess, history)
    assert augmented.shape == (16 + 2 + 16 + 16,)
    np.testing.assert_array_equal(augmented[:16], raw)
    np.testing.assert_array_equal(augmented[16:18], guess)
    np.testing.assert_allclose(augmented[18:34], jp.mean(history, axis=0))
    np.testing.assert_array_equal(augmented[34:50], history[-1] - history[0])


class _FakeActor:
    def __init__(self, raw_output):
        self.raw_output = jp.asarray(raw_output, dtype=jp.float32)

    def apply(self, params, encoded):
        return jp.broadcast_to(self.raw_output, encoded.shape[:-1] + self.raw_output.shape)


def test_success_label_actor_predicts_absolute_guess_and_preserves_movement_tail():
    learner = object.__new__(FollowupLearner)
    learner.config = _config(outputs=5, success_labels=True, step_cap=0.0002,
                              guess_cap=0.003)
    learner.base_size = 16
    learner.encode = lambda obs: obs
    raw_output = jp.asarray([0.1, -0.1, 0.2, -0.3, 0.4])
    learner.actor_net = _FakeActor(raw_output)
    learner_guess = jp.asarray([0.0001, -0.0001])
    obs = jp.zeros((1, 20)).at[:, 16:18].set(learner_guess)
    result = learner.act(object(), obs)
    # The actor's first pair is an absolute guess normalized by 3 mm;
    # the controller turns its error into a bounded nudge.
    absolute_prediction = raw_output[:2] * 0.003
    expected_nudge = jp.clip((absolute_prediction - learner_guess) / 0.0002, -1, 1)
    np.testing.assert_allclose(result[0, :2], expected_nudge, atol=1e-7)
    np.testing.assert_array_equal(result[0, 2:], raw_output[2:])
