"""Synthetic unit contracts for the fixed ceiling verification helper."""

import json

import numpy as np
import pytest

from takehome.evaluation_checks import verify_ceiling


class FakeRunner:
    def __init__(self, summary=None):
        self.summary = summary or {
            "episodes": 500,
            "test_seed": 789,
            "test_only": True,
            "learned_from": False,
            "successes": 2,
            "success_rate": 0.8,
            "steps_mean": 12.0,
            "steps_median": 12.0,
        }
        self.calls = []

    def test(self, state, *, episodes, seed, output):
        self.calls.append({"state": state, "episodes": episodes,
                           "seed": seed, "output": output})
        stats = {
            "succeeded": np.array([1.0, 0.0, 1.0]),
            "max_lateral_when_deep": np.array([0.001, 0.002, 0.003]),
        }
        trace = {"executed_action": np.array([
            [[0.1, -0.2], [0.0, 0.0], [0.3, -0.4]],
        ])}
        return dict(self.summary), stats, trace


def test_aim_only_more_than_five_points_above_ceiling_raises(tmp_path):
    with pytest.raises(RuntimeError, match="exceeds ceiling by more than five points"):
        verify_ceiling(FakeRunner(), object(), {"success_rate": 0.76},
                       outputs=2, ceiling=0.70, output=tmp_path / "unused.json")


def test_movement_exceeding_ceiling_runs_fixed_500_seed789_and_saves_flags(tmp_path):
    runner = FakeRunner()
    output = tmp_path / "movement_ceiling.json"
    result = verify_ceiling(runner, "state", {"success_rate": 0.8},
                            outputs=3, ceiling=0.7, output=output)

    assert result == {"path": str(output), "reused": False, "new_test_episodes": 500}
    assert len(runner.calls) == 1
    assert runner.calls[0]["state"] == "state"
    assert runner.calls[0]["episodes"] == 500
    assert runner.calls[0]["seed"] == 789
    assert runner.calls[0]["output"] == output.with_suffix(".npz")
    saved = json.loads(output.read_text())
    assert saved["episodes"] == 500
    assert saved["test_seed"] == 789
    assert saved["test_only"] is True
    assert saved["learned_from"] is False
    assert saved["used_for_selection"] is False


def test_existing_exact_model_verification_is_reused_without_new_test(tmp_path):
    existing = tmp_path / "existing.json"
    existing.write_text(json.dumps({
        "episodes": 500,
        "test_seed": 789,
        "test_only": True,
        "learned_from": False,
        "used_for_selection": False,
    }))
    runner = FakeRunner()
    result = verify_ceiling(runner, object(), {"success_rate": 0.9},
                            outputs=5, ceiling=0.8,
                            output=tmp_path / "new.json", existing=existing)

    assert result == {"path": str(existing), "reused": True, "new_test_episodes": 0}
    assert runner.calls == []


def test_below_ceiling_does_not_run_a_test(tmp_path):
    runner = FakeRunner()
    result = verify_ceiling(runner, object(), {"success_rate": 0.69},
                            outputs=5, ceiling=0.7,
                            output=tmp_path / "unused.json")
    assert result is None
    assert runner.calls == []
