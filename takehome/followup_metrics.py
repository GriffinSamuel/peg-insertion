"""Small saved-result metrics for the follow-up Phase A diagnostics.

This module reads completed result metadata and test traces only. It does not
load experiment code, run simulations, fit predictors, or write files.
"""

import json
from pathlib import Path

import numpy as np


TRACE_FIELDS = (
    "guess",
    "next_guess",
    "true_bias",
    "success",
    "valid",
    "test_only",
    "learned_from",
    "test_seed",
    "episode_success",
)
EXPECTED_PRACTICE_EPISODES = 200
EXPECTED_TEST_EPISODES = 500
AIM_OUTPUT_COUNTS = (2, 5)


def _scalar(trace, name):
    value = np.asarray(trace[name])
    if value.shape != ():
        raise ValueError(f"trace {name} must be a scalar")
    return value.item()


def _mean_or_none(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    if not np.isfinite(values).all():
        raise ValueError("metric inputs must be finite")
    return float(values.mean())


def _final_stat(final, key, *, allow_none=False):
    raw_value = final.get(key)
    if raw_value is None and allow_none:
        return None
    value = float(raw_value)
    if not np.isfinite(value):
        raise ValueError(f"final summary {key} must be finite")
    return value


def summarize_trace(trace, final, *, outputs):
    """Summarize an in-memory test trace; supports small synthetic fixtures.

    Trace arrays use time first and episode second. Guess and camera-bias values
    are stored in metres and reported here as XY L2 errors in millimetres.
    """
    missing = [key for key in TRACE_FIELDS if key not in trace]
    if missing:
        raise ValueError(f"trace is missing required fields: {missing}")
    if _scalar(trace, "test_only") is not True:
        raise ValueError("trace must be marked test_only=true")
    if _scalar(trace, "learned_from") is not False:
        raise ValueError("trace must be marked learned_from=false")

    success = np.asarray(trace["success"], dtype=bool)
    valid = np.asarray(trace["valid"], dtype=bool)
    guess = np.asarray(trace["guess"], dtype=np.float64)
    next_guess = np.asarray(trace["next_guess"], dtype=np.float64)
    true_bias = np.asarray(trace["true_bias"], dtype=np.float64)
    if success.ndim != 2 or valid.shape != success.shape:
        raise ValueError("success and valid must have matching [time, episode] shapes")
    time_steps, episodes = success.shape
    if time_steps < 1 or episodes < 1:
        raise ValueError("trace must contain at least one timestep and episode")
    xy_shape = (time_steps, episodes)
    for name, array in (("guess", guess), ("next_guess", next_guess), ("true_bias", true_bias)):
        if array.ndim != 3 or array.shape[:2] != xy_shape or array.shape[2] < 2:
            raise ValueError(f"{name} must have shape [time, episode, at least 2]")
        if not np.isfinite(array[..., :2]).all():
            raise ValueError(f"{name} XY values must be finite")
    if int(final.get("episodes", -1)) != episodes:
        raise ValueError("trace episode count does not match final summary")
    final_successes = int(final.get("successes", -1))
    if final_successes < 0 or final_successes > episodes:
        raise ValueError("final summary has an invalid success count")

    result = {
        "test_episodes": int(episodes),
        "successes": final_successes,
        "success_percent": float(100.0 * final_successes / episodes) if episodes else None,
        "steps_mean": _final_stat(final, "steps_mean", allow_none=final_successes == 0),
        "peak_force_mean": _final_stat(final, "peak_force_mean"),
        "test_seed": int(_scalar(trace, "test_seed")),
        "test_only": True,
        "learned_from": False,
        "outputs": int(outputs),
        "aim_metrics_applicable": int(outputs) in AIM_OUTPUT_COUNTS,
        "hidden_true_bias_used_for_analysis_only": True,
    }

    valid_success = success & valid
    first_success_t = np.where(
        valid_success.any(axis=0), valid_success.argmax(axis=0), -1
    )
    success_ids = np.flatnonzero(first_success_t >= 0)
    valid_success_event_count = int(valid_success.sum())
    result.update({
        "valid_success_event_count": valid_success_event_count,
        "valid_success_episode_count": int(success_ids.size),
        "valid_success_event_sum_matches_final": valid_success_event_count == final_successes,
        "valid_success_episode_count_matches_final": int(success_ids.size) == final_successes,
        "success_count_matches_final": int(success_ids.size) == final_successes,
    })

    episode_success = np.asarray(trace["episode_success"])
    if episode_success.shape != (episodes,):
        raise ValueError("episode_success must have one entry per test episode")
    episode_success_count = int(np.count_nonzero(episode_success > 0.5))
    result["episode_success_vector_count"] = episode_success_count
    result["episode_success_vector_matches_final"] = episode_success_count == final_successes

    if int(outputs) in AIM_OUTPUT_COUNTS:
        initial_error = np.linalg.norm(guess[0, :, :2] - true_bias[0, :, :2], axis=-1) * 1000.0
        # End-of-horizon value follows the recorded final step, including the
        # full test horizon after any success or termination.
        end_error = np.linalg.norm(next_guess[-1, :, :2] - true_bias[-1, :, :2], axis=-1) * 1000.0
        if success_ids.size:
            success_t = first_success_t[success_ids]
            success_error = np.linalg.norm(
                next_guess[success_t, success_ids, :2]
                - true_bias[success_t, success_ids, :2], axis=-1
            ) * 1000.0
            paired_initial = initial_error[success_ids]
            paired_reduction = paired_initial - success_error
        else:
            success_error = np.asarray([], dtype=np.float64)
            paired_initial = np.asarray([], dtype=np.float64)
            paired_reduction = np.asarray([], dtype=np.float64)
        result.update({
            "initial_guess_error_mean_mm": _mean_or_none(initial_error),
            "guess_error_at_first_valid_success_mean_mm": _mean_or_none(success_error),
            "paired_initial_guess_error_mean_mm": _mean_or_none(paired_initial),
            "paired_guess_error_reduction_mean_mm": _mean_or_none(paired_reduction),
            "end200_guess_error_mean_mm": _mean_or_none(end_error),
            "success_episode_count_for_paired_metrics": int(success_ids.size),
        })
    else:
        result.update({
            "initial_guess_error_mean_mm": None,
            "guess_error_at_first_valid_success_mean_mm": None,
            "paired_initial_guess_error_mean_mm": None,
            "paired_guess_error_reduction_mean_mm": None,
            "end200_guess_error_mean_mm": None,
            "success_episode_count_for_paired_metrics": 0,
        })
    return result


def summarize_seed(run_dir):
    """Read one completed seed's result, test trace, and counted budget."""
    run_dir = Path(run_dir)
    result_path = run_dir / "result.json"
    trace_path = run_dir / "test_0200_traces.npz"
    budget_path = run_dir / "budget.json"
    for path in (result_path, trace_path, budget_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    result_doc = json.loads(result_path.read_text(encoding="utf-8"))
    config = result_doc.get("config", {})
    final = result_doc.get("final", {})
    if result_doc.get("code_test") is True or config.get("code_test") is True:
        raise ValueError("code_test runs are not valid for this summary")
    if int(result_doc.get("practice_episodes", -1)) != EXPECTED_PRACTICE_EPISODES:
        raise ValueError("result must record exactly 200 practice episodes")
    if int(final.get("episodes", -1)) != EXPECTED_TEST_EPISODES:
        raise ValueError("final result must contain exactly 500 test episodes")

    test_seed = int(config.get("test_seed", 123))
    if int(final.get("test_seed", -1)) != test_seed:
        raise ValueError("final test seed does not match config")
    if final.get("test_only") is not True:
        raise ValueError("final result must be marked test_only=true")
    if final.get("learned_from") is not False:
        raise ValueError("final result must be marked learned_from=false")
    if result_doc.get("test_data_learned_from") is not False:
        raise ValueError("result must record test_data_learned_from=false")

    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    entries = budget.get("episodes", [])
    if int(budget.get("limit", -1)) != EXPECTED_PRACTICE_EPISODES:
        raise ValueError("practice budget limit must be 200")
    if len(entries) != EXPECTED_PRACTICE_EPISODES:
        raise ValueError("practice budget ledger must contain exactly 200 episodes")
    episode_ids = [int(entry["id"]) for entry in entries]
    if sorted(episode_ids) != list(range(EXPECTED_PRACTICE_EPISODES)):
        raise ValueError("practice budget ledger must contain IDs 0 through 199 once")

    with np.load(trace_path, allow_pickle=False) as archive:
        missing = [key for key in TRACE_FIELDS if key not in archive.files]
        if missing:
            raise ValueError(f"test trace is missing required fields: {missing}")
        trace = {key: archive[key] for key in TRACE_FIELDS}
    if np.asarray(trace["success"]).shape != (200, EXPECTED_TEST_EPISODES):
        raise ValueError("final test trace must have shape [200, 500]")
    if bool(trace["test_only"].item()) is not True:
        raise ValueError("test trace must be marked test_only=true")
    if bool(trace["learned_from"].item()) is not False:
        raise ValueError("test trace must be marked learned_from=false")
    if int(trace["test_seed"].item()) != test_seed:
        raise ValueError("test trace seed does not match config")

    summary = summarize_trace(trace, final, outputs=int(config["outputs"]))
    summary.update({
        "seed": int(result_doc["seed"]),
        "practice_episodes": len(entries),
        "practice_budget_limit": int(budget["limit"]),
        "result_path": result_path.as_posix(),
        "trace_path": trace_path.as_posix(),
        "budget_path": budget_path.as_posix(),
        "success_count_matches_final": summary["valid_success_episode_count_matches_final"],
    })
    return summary


def _synthetic_checks():
    """In-memory contract checks; no files or experiment code are used."""
    time_steps, episodes = 4, 3
    success = np.zeros((time_steps, episodes), dtype=bool)
    valid = np.ones((time_steps, episodes), dtype=bool)
    success[1, 0] = True
    success[3, 0] = True  # Confirms the first success is selected.
    success[0, 1] = True  # Invalid success is skipped.
    success[2, 1] = True
    valid[0, 1] = False
    guess = np.zeros((time_steps, episodes, 2), dtype=np.float64)
    next_guess = np.zeros_like(guess)
    true_bias = np.zeros_like(guess)
    true_bias[:, :, 0] = np.array([0.001, 0.002, 0.003])[None, :]
    guess[1, 0, 0] = 0.01  # Must not be used for applied success-time error.
    next_guess[1, 0, 0] = 0.001  # First valid success: zero error.
    next_guess[3, 0, 0] = 0.02  # Later success is ignored for first-success metric.
    next_guess[2, 1, 0] = 0.001  # First success error is 1 mm.
    trace = {
        "guess": guess,
        "next_guess": next_guess,
        "true_bias": true_bias,
        "success": success,
        "valid": valid,
        "test_only": np.asarray(True),
        "learned_from": np.asarray(False),
        "test_seed": np.asarray(456),
        "episode_success": np.asarray([1.0, 1.0, 0.0]),
    }
    final = {"episodes": episodes, "successes": 2, "steps_mean": 3.0, "peak_force_mean": 4.0}
    out = summarize_trace(trace, final, outputs=2)
    assert out["success_episode_count_for_paired_metrics"] == 2
    assert abs(out["guess_error_at_first_valid_success_mean_mm"] - 0.5) < 1e-9
    assert abs(out["paired_initial_guess_error_mean_mm"] - 1.5) < 1e-9
    assert abs(out["paired_guess_error_reduction_mean_mm"] - 1.0) < 1e-9
    assert out["valid_success_event_count"] == 3
    assert out["valid_success_event_sum_matches_final"] is False
    assert out["success_count_matches_final"] is True

    no_success = dict(trace)
    no_success["success"] = np.zeros_like(success)
    no_success["episode_success"] = np.zeros(episodes)
    none = summarize_trace(
        no_success, {**final, "successes": 0, "steps_mean": None}, outputs=5
    )
    assert none["steps_mean"] is None
    assert none["peak_force_mean"] == 4.0
    assert none["guess_error_at_first_valid_success_mean_mm"] is None
    assert none["paired_initial_guess_error_mean_mm"] is None
    assert none["paired_guess_error_reduction_mean_mm"] is None

    bad = dict(trace)
    bad["learned_from"] = np.asarray(True)
    try:
        summarize_trace(bad, final, outputs=2)
    except ValueError:
        pass
    else:
        raise AssertionError("learned_from=true trace was not rejected")
    bad = dict(trace)
    bad["test_only"] = np.asarray(False)
    try:
        summarize_trace(bad, final, outputs=2)
    except ValueError:
        pass
    else:
        raise AssertionError("test_only=false trace was not rejected")


if __name__ == "__main__":
    _synthetic_checks()
    print("synthetic checks passed")
