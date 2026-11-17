"""Synthetic contracts for follow-up decider, coordination, and manifests."""
from takehome import followup_runtime  # noqa: F401 -- initialize caches before JAX

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest

from peg_insertion.rollout import load_policy
from takehome.followup_config import FollowupConfig
from takehome.followup_c_manifest import settings as phase_c_settings
from takehome.followup_d_manifest import settings as phase_d_settings
from takehome.followup_coordination import FollowupCoordinatedLearner
from takehome.followup_learner import FollowupLearner


@pytest.fixture(scope="module")
def no_physics_env():
    policy, _ = load_policy(Path("results/policies/force_0.5mm_s2"))
    return SimpleNamespace(base_obs_size=16, learner_obs_size=18, frozen_policy=policy)


@pytest.mark.parametrize("outputs", [2, 3])
def test_final_decider_is_zero_initialized_bounded_and_scales_candidates(no_physics_env, outputs):
    config = FollowupConfig(outputs=outputs, final_decider=True, movement_cap=0.0002)
    learner = FollowupLearner(config, no_physics_env)
    initial = learner.initialize(17)
    observations = jp.zeros((4, 18))
    action = learner.choose(initial.actor, initial.critics, observations)
    np.testing.assert_array_equal(action, np.zeros((4, outputs)))

    actor_action = jp.asarray([0.8, -0.6] if outputs == 2 else [0.8, -0.6, 0.2])
    learner.act = lambda params, obs: jp.broadcast_to(actor_action, (obs.shape[0], outputs))
    seen = {}

    def score_options(critics, obs, physical_action, bias=None):
        seen["physical_action"] = physical_action
        seen["obs"] = obs
        option_count = physical_action.shape[-2]
        scores = jp.arange(option_count, dtype=jp.float32)
        return jp.broadcast_to(scores[None, None, :], (1, physical_action.shape[0], option_count))

    learner.q_values = score_options
    chosen = learner.choose(initial.actor, initial.critics, observations)
    options = jp.stack([
        jp.zeros_like(actor_action), actor_action, actor_action * 0.5,
        jp.clip(actor_action * 1.5, -1.0, 1.0),
    ])
    np.testing.assert_allclose(seen["physical_action"],
                               jp.broadcast_to(options[None, :, :] * learner.action_scale,
                                               (4, 4, outputs)))
    np.testing.assert_allclose(chosen, jp.broadcast_to(options[-1], (4, outputs)))
    assert np.all(np.abs(np.asarray(chosen)) <= 1.0)
    assert seen["obs"].shape == (4, 4, 18)


def _synthetic_batch():
    key = jax.random.PRNGKey(23)
    obs_key, next_key, action_key = jax.random.split(key, 3)
    obs = jax.random.normal(obs_key, (128, 18))
    next_obs = jax.random.normal(next_key, (128, 18))
    action = jax.random.uniform(action_key, (128, 5), minval=-0.8, maxval=0.8)
    return {
        "obs": obs,
        "next_obs": next_obs,
        "action": action,
        "reward": jp.full((128,), 0.2),
        "terminated": jp.zeros((128,)),
    }


def _assert_tree_bytes_equal(left, right):
    left_leaves, left_def = jax.tree_util.tree_flatten(left)
    right_leaves, right_def = jax.tree_util.tree_flatten(right)
    assert left_def == right_def
    assert len(left_leaves) == len(right_leaves)
    for old, new in zip(left_leaves, right_leaves):
        old_array, new_array = np.asarray(old), np.asarray(new)
        assert old_array.shape == new_array.shape and old_array.dtype == new_array.dtype
        assert old_array.tobytes() == new_array.tobytes()


@pytest.mark.parametrize("mode", ["aim_then_move", "move_then_aim"])
def test_followup_sequential_pair_freezes_inactive_full_state_and_updates_both(no_physics_env, mode):
    config = FollowupConfig(outputs=5, coordination=mode, camera_condition="fixed")
    learner = FollowupCoordinatedLearner(config, no_physics_env)
    initial = learner.initialize(31)
    initial = initial.replace(aim=initial.aim.replace(start_guess=jp.asarray([0.0004, -0.0003])))
    observations = jp.zeros((3, 18))
    np.testing.assert_array_equal(learner.act(initial.actor, observations), np.zeros((3, 5)))
    np.testing.assert_array_equal(initial.start_guess, np.asarray([0.0004, -0.0003],dtype=np.float32))

    data = _synthetic_batch()
    first, first_metrics = learner.train_block(initial, data, 64, 128, 4, practice_end=120)
    assert bool(first_metrics["finite"])
    if mode == "aim_then_move":
        assert int(first.aim.updates) == 4 and int(first.movement.updates) == 0
        _assert_tree_bytes_equal(initial.movement, first.movement)
        assert int(first.aim.updates) > int(initial.aim.updates)
    else:
        assert int(first.movement.updates) == 4 and int(first.aim.updates) == 0
        _assert_tree_bytes_equal(initial.aim, first.aim)
        assert int(first.movement.updates) > int(initial.movement.updates)

    second, second_metrics = learner.train_block(first, data, 64, 128, 4, practice_end=130)
    assert bool(second_metrics["finite"])
    assert int(second.aim.updates) == 4
    assert int(second.movement.updates) == 4
    np.testing.assert_array_equal(second.start_guess, initial.start_guess)
    for leaf in jax.tree_util.tree_leaves(second):
        assert np.all(np.isfinite(np.asarray(leaf)))


def test_all_planned_phase_c_and_d_manifest_configs_construct_and_are_single_switches():
    c_rows = [row for row in phase_c_settings() if row["status"] == "planned"]
    d_rows = [row for row in phase_d_settings() if row["status"] == "planned"]
    assert len(c_rows) == 20
    assert len(d_rows) == 18

    for row in c_rows:
        switches = {key: value for key, value in row["overrides"].items() if key != "outputs"}
        assert len(switches) == 1, row["name"]
        config = FollowupConfig(**row["overrides"])
        assert config.outputs == 3
        assert config.total_episodes == 200

    for row in d_rows:
        switches = {key: value for key, value in row["overrides"].items()
                    if key not in ("outputs", "camera_condition")}
        assert len(switches) == 1, row["name"]
        config = FollowupConfig(**row["overrides"])
        assert config.outputs == 2
        assert config.camera_condition == "fixed"
        assert config.total_episodes == 200
