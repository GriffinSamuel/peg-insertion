"""Correction bounds, terminal flags, and in-memory policy immutability."""
from pathlib import Path

from takehome import runtime  # noqa: F401
import jax
import jax.numpy as jp
import numpy as np
from peg_insertion import rollout
from takehome.correction import (
    AimConfig, AimCorrectedEnv, advance_guess, corrected_observation,
    learner_outcome, policy_digest, zero_nudge,
)

POLICY = Path("results/policies/force_0.5mm_s2")


def test_guess_caps_and_observation_correction():
    cfg = AimConfig()
    guess = jp.zeros(2)
    for _ in range(100):
        guess = advance_guess(guess, jp.array([100.0, -100.0]), cfg)
    np.testing.assert_allclose(guess, [cfg.guess_cap, -cfg.guess_cap])
    raw = jp.arange(16, dtype=jp.float32)
    corrected = corrected_observation(raw, jp.array([0.002, -0.001]))["state"]
    np.testing.assert_allclose(corrected[:2], [-0.002, 1.001])
    np.testing.assert_array_equal(corrected[2:], raw[2:])


def test_first_success_and_timeout_are_distinct():
    def outcome(ended=False, old_success=False, success=False, failure=False, limit=False):
        return learner_outcome(*(jp.array(v) for v in
                                 (ended, old_success, success, failure, limit)))
    assert tuple(map(float, outcome())) == (0.0, 0.0, 0.0)
    assert tuple(map(float, outcome(success=True))) == (1.0, 1.0, 0.0)
    assert tuple(map(float, outcome(ended=True, old_success=True, success=True))) == (0.0, 0.0, 0.0)
    assert tuple(map(float, outcome(limit=True))) == (0.0, 0.0, 1.0)
    assert tuple(map(float, outcome(failure=True))) == (0.0, 1.0, 0.0)
    assert tuple(map(float, outcome(success=True, limit=True))) == (1.0, 1.0, 0.0)


def test_reset_and_frozen_policy_after_test_rollout():
    env = AimCorrectedEnv(POLICY)
    before = policy_digest(env.frozen_policy)
    before_file = (POLICY / "params.pkl").read_bytes()
    wrapped = rollout._wrap(env, 4, jax.random.PRNGKey(20), randomize=True)
    reset = jax.jit(wrapped.reset)
    state = reset(jax.random.split(jax.random.PRNGKey(21), 4))
    np.testing.assert_array_equal(state.info["aim_guess"], np.zeros((4, 2)))
    assert state.obs["state"].shape == (4, 18)
    # Test-only episodes; no replay or learner exists.
    stats = rollout.rollout_stats(env, zero_nudge, 4, seed=123, randomize=True)
    assert stats["succeeded"].shape == (4,)
    assert all(np.isfinite(v).all() for v in stats.values())
    assert policy_digest(env.frozen_policy) == before
    assert (POLICY / "params.pkl").read_bytes() == before_file
    fresh = reset(jax.random.split(jax.random.PRNGKey(22), 4))
    np.testing.assert_array_equal(fresh.info["aim_guess"], np.zeros((4, 2)))
