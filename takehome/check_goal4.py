"""Run contract tests and the fixed 10-episode code trial in one process."""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome.runtime import ROOT  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from takehome.config import TrainConfig  # noqa: E402
from takehome.train import ExperimentRunner, write_json  # noqa: E402


def main():
    t0 = time.perf_counter()
    status = pytest.main(["-q", "-p", "no:cacheprovider",
        "--basetemp=results/takehome/goal4_test_tmp",
        "takehome/test_budget.py", "takehome/test_correction.py", "takehome/test_learner.py"])
    if status:
        raise SystemExit(status)
    cfg = TrainConfig(total_episodes=10, fill_episodes=0, code_test=True)
    result = ExperimentRunner(cfg).run(999, ROOT / "results/takehome/goal4_trial/seed_999")
    assert result["practice_episodes"] == 10
    assert result["gradient_updates"] > 0
    assert result["frozen_policy_unchanged"]
    assert not result["test_data_learned_from"]
    assert not result["setting_counted"]
    with np.load(ROOT / "results/takehome/goal4_trial/seed_999/test_0010_traces.npz") as trace:
        assert bool(trace["test_only"]) and not bool(trace["learned_from"])
        assert trace["guess"].shape == (200, 8, 2)
        assert trace["true_bias"].shape == (200, 8, 2)
        assert trace["q"].shape == trace["force"].shape == (200, 8)
    write_json(ROOT / "results/takehome/goal4_checks.json", {
        "tests_passed": True, "ten_episode_trial_passed": True,
        "budget_guard_passed": True, "symmetric_sampling_passed": True,
        "single_source_sampling_passed": True, "normalization_and_action_scale_passed": True,
        "outputs_2_3_5_start_zero": True, "test_data_guard_passed": True,
        "trace_metadata_passed": True, "practice_episodes": 10,
        "settings_tried": 0, "seconds": time.perf_counter() - t0})


if __name__ == "__main__":
    main()
