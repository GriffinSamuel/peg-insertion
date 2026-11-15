"""Main learner contracts; synthetic checks never collect practice episodes."""
from dataclasses import replace
from pathlib import Path

from takehome import runtime  # noqa: F401
from brax.training.acme import running_statistics
import jax
import jax.numpy as jp
import numpy as np
import pytest

from takehome.budget import EpisodeBudget
from takehome.config import TrainConfig
from takehome.correction import AimCorrectedEnv
from takehome.learner import Learner, draw_indices
from takehome.replay import Replay, REQUIRED_KEYS, TRAIN_KEYS


@pytest.fixture(scope="module")
def env():
    return AimCorrectedEnv(Path("results/policies/force_0.5mm_s2"))


@pytest.mark.parametrize("outputs", [2, 3, 5])
def test_all_output_dimensions_start_exactly_zero(env, outputs):
    learner = Learner(TrainConfig(outputs=outputs), env)
    state = learner.initialize(999)
    observations = jax.random.normal(jax.random.PRNGKey(22), (11, env.base_obs_size + 2))
    actions = learner.act(state.actor, observations)
    assert actions.shape == (11, outputs)
    np.testing.assert_array_equal(actions, np.zeros((11, outputs)))


def test_actual_compiled_sampler_has_exact_mix_and_empty_source_fallbacks():
    sampler = jax.jit(lambda key, fill, total: draw_indices(key, 128, fill, total))
    for seed in range(8):
        key = jax.random.PRNGKey(seed)
        mixed = np.asarray(sampler(key, 17, 30))
        assert np.sum(mixed < 17) == 64
        assert np.sum(mixed >= 17) == 64
        assert mixed.min() >= 0 and mixed.max() < 30
        for fill, total in ((0, 7), (7, 7)):
            single = np.asarray(sampler(key, fill, total))
            assert single.shape == (128,)
            assert single.min() >= 0 and single.max() < 7


def test_checkpoint_normalization_and_physical_action_rescaling(env):
    cfg = TrainConfig()
    learner = Learner(cfg, env)
    changed_units = Learner(replace(cfg, step_cap=cfg.step_cap * 2), env)
    state = learner.initialize(998)
    obs = jax.random.normal(jax.random.PRNGKey(32), (4, env.base_obs_size + 2)) * 0.001
    expected = running_statistics.normalize({"state": obs[:, :env.base_obs_size]},
                                            learner.normalizer)["state"]
    encoded = learner.encode(obs)
    np.testing.assert_array_equal(encoded[:, :env.base_obs_size], expected)
    np.testing.assert_allclose(encoded[:, -2:], obs[:, -2:] / cfg.guess_cap)
    normalized_action = jp.full((4, 2), 0.4)
    # Same normalized control, different physical units: critic input/rating agrees.
    q1 = learner.q_values(state.critics, obs, normalized_action * cfg.step_cap)
    q2 = changed_units.q_values(state.critics, obs, normalized_action * cfg.step_cap * 2)
    np.testing.assert_allclose(q1, q2)


def test_replay_rejects_test_data_and_excludes_true_bias_from_training(tmp_path):
    budget = EpisodeBudget(tmp_path / "budget.json")
    ids = budget.reserve(1, "fill")
    trajectory = {key: np.zeros((2, 1), dtype=np.float32) for key in REQUIRED_KEYS}
    trajectory.update(valid=np.ones((2, 1), bool), episode_id=np.zeros((2, 1), int),
                      true_bias=np.full((2, 1, 2), 12345.0))
    replay = Replay(budget)
    with pytest.raises(ValueError, match="Test data"):
        replay.add(trajectory, ids, "fill", test_only=True)
    assert replay.count == 0
    replay.add(trajectory, ids, "fill", test_only=False)
    assert set(replay.training_arrays()) == set(TRAIN_KEYS)
    assert "true_bias" in replay.arrays and "true_bias" not in replay.training_arrays()
    with pytest.raises(ValueError, match="already inserted"):
        replay.add(trajectory, ids, "fill", test_only=False)
