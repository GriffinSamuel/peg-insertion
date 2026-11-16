"""Synthetic contracts for follow-up controls, labels, and launch guards."""

from dataclasses import dataclass
from types import SimpleNamespace

from takehome import followup_runtime  # noqa: F401
import jax.numpy as jp
import numpy as np
import pytest

from takehome.followup_config import FollowupConfig
from takehome.followup_env import contact_seen, offset_grid, stalled
from takehome.followup_learner import draw_indices as followup_draw_indices
from takehome.followup_learner import success_targets
from takehome.followup_runner import FollowupRunner
from takehome.learner import draw_indices as original_draw_indices


def test_stall_detection_averages_noisy_flat_window_but_sees_descent():
    flat = np.zeros((10, 16), dtype=np.float32)
    flat[:5, 2] = [0.0040, 0.0042, 0.0039, 0.0041, 0.0038]
    flat[5:, 2] = [0.0041, 0.0039, 0.0040, 0.0042, 0.0038]
    descending = flat.copy()
    descending[5:, 2] -= 0.0004

    assert bool(stalled(jp.asarray(flat)))
    assert not bool(stalled(jp.asarray(descending)))


def test_contact_detection_uses_normalized_force_threshold():
    raw = np.zeros((3, 16), dtype=np.float32)
    raw[0, 10] = 0.249
    raw[1, 10] = 0.250
    raw[2, 10:13] = [0.15, 0.20, 0.0]

    np.testing.assert_array_equal(np.asarray(contact_seen(jp.asarray(raw))), [False, True, True])


def _synthetic_replay():
    capacity = count = 5
    next_obs = np.zeros((capacity, 18), dtype=np.float32)
    next_obs[2, :2] = [0.00125, -0.00075]
    return SimpleNamespace(
        capacity=capacity,
        count=count,
        episode_ids={10, 11},
        arrays={
            "episode_id": np.array([10, 10, 10, 11, 11]),
            "step": np.array([2, 0, 1, 0, 1]),
            "success": np.array([1, 0, 1, 0, 0]),
            "next_obs": next_obs,
            "true_bias": np.arange(capacity * 2, dtype=np.float32).reshape(capacity, 2),
            "source": np.ones(capacity, dtype=np.int32),  # own online practice
        },
    )


def test_success_targets_use_first_success_next_observation_not_hidden_bias():
    replay = _synthetic_replay()
    labels, valid = success_targets(replay)
    expected = np.array([0.00125, -0.00075], dtype=np.float32)
    np.testing.assert_allclose(labels[:3], np.broadcast_to(expected, (3, 2)))
    np.testing.assert_array_equal(valid, [1, 1, 1, 0, 0])
    np.testing.assert_array_equal(labels[3:], np.zeros((2, 2)))

    replay.arrays["true_bias"][:] = -999.0
    changed_labels, changed_valid = success_targets(replay)
    np.testing.assert_array_equal(changed_labels, labels)
    np.testing.assert_array_equal(changed_valid, valid)


@pytest.mark.parametrize("switch", [
    "drift_gate", "pullback", "memory", "success_labels", "fast_contact",
    "constant_fill", "heatmap", "within_grid",
])
def test_movement_only_output_rejects_aim_switches(switch):
    with pytest.raises(ValueError, match="require an aim output"):
        FollowupConfig(outputs=3, **{switch: True})


def test_heatmap_requires_constant_offset_fill():
    with pytest.raises(ValueError, match="requires its constant-offset fill"):
        FollowupConfig(outputs=2, heatmap=True)


def test_supervised_hidden_labels_are_forbidden():
    with pytest.raises(ValueError, match="hidden labels"):
        FollowupConfig(outputs=2, supervised_reference=True)


@pytest.mark.parametrize("seed,test_seed", [(200, 123), (0, 1001), (200, 1001)])
def test_fresh_seed_pair_is_blocked_before_runner_initialization(seed, test_seed, tmp_path, monkeypatch):
    monkeypatch.setattr("takehome.followup_runner.OUT", tmp_path)
    # Bypass __init__: it is the only method that checks for a GPU and builds an env.
    runner = object.__new__(FollowupRunner)
    runner.config = FollowupConfig(test_seed=test_seed)
    output = tmp_path / "must-not-be-created"

    with pytest.raises(ValueError, match="Fresh seeds forbidden"):
        runner.run(seed, output)
    assert not output.exists()


@dataclass
class _State:
    start_guess: object

    def replace(self, **changes):
        return _State(**{**self.__dict__, **changes})


def _replay_for_heatmap(cells, successes):
    grid = np.asarray(offset_grid())
    guesses = np.asarray([grid[cell] for cell in cells], dtype=np.float32)
    n = len(cells)
    return SimpleNamespace(
        count=n,
        episode_ids=set(range(n)),
        arrays={
            "episode_id": np.arange(n),
            "step": np.zeros(n, dtype=np.int32),
            "guess": guesses,
            "success": np.asarray(successes, dtype=np.float32),
            "source": np.zeros(n, dtype=np.int32),  # fill-up practice only
        },
    )


def test_heatmap_defaults_to_zero_then_uses_best_practice_cell_and_nearest_tie():
    runner = object.__new__(FollowupRunner)
    runner.config = FollowupConfig(outputs=2, heatmap=True, constant_fill=True)
    initial = runner.after_collect(_State(np.array([0.001, -0.001])),
                                   _replay_for_heatmap([], []))
    np.testing.assert_array_equal(np.asarray(initial.start_guess), np.zeros(2))

    grid = np.asarray(offset_grid())
    far_cell = int(np.flatnonzero(np.all(np.isclose(grid, [0.001, 0.001]), axis=1))[0])
    best = runner.after_collect(_State(np.zeros(2)),
                                _replay_for_heatmap([far_cell], [1]))
    np.testing.assert_array_equal(np.asarray(best.start_guess), grid[far_cell])

    near_cell = int(np.flatnonzero(np.all(np.isclose(grid, [-0.0005, 0.0]), axis=1))[0])
    farther_cell = int(np.flatnonzero(np.all(np.isclose(grid, [0.001, 0.0]), axis=1))[0])
    tied = runner.after_collect(_State(np.zeros(2)),
                                _replay_for_heatmap([near_cell, farther_cell], [1, 1]))
    np.testing.assert_array_equal(np.asarray(tied.start_guess), grid[near_cell])


def test_followup_reuses_the_original_exact_mix_sampler():
    # The exact 64/64 behavior is already covered by test_learner.py.
    assert followup_draw_indices is original_draw_indices
