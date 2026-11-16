"""Synthetic verdict, saved-metric, and scoreboard contracts for the follow-up."""

# Must run before importing project modules that may initialize JAX caches.
from takehome import followup_runtime  # noqa: F401

import csv
from io import StringIO
import statistics

import numpy as np
import pytest

from takehome.followup_collect import verdict
from takehome.followup_metrics import _synthetic_checks, summarize_trace
from takehome.followup_scoreboard import (
    CSV_FIELDS,
    FIELDS,
    render_csv,
    render_markdown,
    validate_records,
)


def _rows(values, seeds=None):
    if seeds is None:
        seeds = list(range(len(values)))
    return [
        {"seed": seed, "success_percent": value}
        for seed, value in zip(seeds, values)
    ]


def test_seed_matched_collapse_precedes_an_average_gain():
    base = _rows([100.0, 20.0, 20.0, 20.0, 20.0], [0, 1, 2, 3, 4])
    # Reverse the candidate order: seed 0 still has to be compared to base seed 0.
    candidate = _rows([100.0, 100.0, 100.0, 100.0, 49.0], [4, 3, 2, 1, 0])
    candidate_values = [row["success_percent"] for row in candidate]
    base_values = [row["success_percent"] for row in base]
    assert statistics.mean(candidate_values) - statistics.mean(base_values) > max(
        statistics.stdev(candidate_values), statistics.stdev(base_values)
    )
    assert verdict(candidate, base, expected=5, failed=False) == "UNSTABLE"


@pytest.mark.parametrize(
    ("seed_count", "expected_verdict"),
    [(3, "PROMISING"), (5, "HELPED")],
)
def test_gain_verdict_depends_on_three_or_five_completed_seeds(seed_count, expected_verdict):
    base = _rows([25.0] * seed_count)
    candidate = _rows([40.0] * seed_count)
    assert verdict(candidate, base, expected=seed_count, failed=False) == expected_verdict


@pytest.mark.parametrize(
    ("candidate_values", "base_values"),
    [
        # Candidate spread is the larger threshold.
        ([70.0, 100.0, 70.0], [70.0, 70.0, 70.0]),
        # Base spread is the larger threshold.
        ([70.0, 70.0, 70.0], [30.0, 90.0, 30.0]),
    ],
)
def test_verdict_uses_the_larger_sample_spread(candidate_values, base_values):
    candidate = _rows(candidate_values)
    base = _rows(base_values)
    assert verdict(candidate, base, expected=3, failed=False) == "NO REAL DIFFERENCE"


def test_exact_equality_is_no_real_difference():
    values = [40.0, 60.0, 80.0]
    assert verdict(_rows(values), _rows(values), expected=3, failed=False) == "NO REAL DIFFERENCE"


def test_failed_or_missing_seed_is_did_not_run():
    complete = _rows([55.0, 60.0, 65.0])
    base = _rows([50.0, 55.0, 60.0])
    assert verdict(complete, base, expected=3, failed=True) == "DID NOT RUN"
    missing_seed = _rows([55.0, 65.0], [0, 2])
    assert verdict(missing_seed, base, expected=3, failed=False) == "DID NOT RUN"


def test_followup_metrics_synthetic_checks_through_pytest_wrapper():
    _synthetic_checks()


def test_null_success_summary_keeps_null_steps_and_success_metrics():
    time_steps, episodes = 3, 2
    trace = {
        "guess": np.zeros((time_steps, episodes, 2)),
        "next_guess": np.zeros((time_steps, episodes, 2)),
        "true_bias": np.zeros((time_steps, episodes, 2)),
        "success": np.zeros((time_steps, episodes), dtype=bool),
        "valid": np.ones((time_steps, episodes), dtype=bool),
        "test_only": np.asarray(True),
        "learned_from": np.asarray(False),
        "test_seed": np.asarray(456),
        "episode_success": np.zeros(episodes),
    }
    final = {
        "episodes": episodes,
        "successes": 0,
        "steps_mean": None,
        "peak_force_mean": 4.0,
    }
    result = summarize_trace(trace, final, outputs=2)
    assert result["steps_mean"] is None
    assert result["peak_force_mean"] == 4.0
    assert result["successes"] == 0
    assert result["success_count_matches_final"] is True
    assert result["guess_error_at_first_valid_success_mean_mm"] is None
    assert result["paired_initial_guess_error_mean_mm"] is None
    assert result["paired_guess_error_reduction_mean_mm"] is None


def _scoreboard_record(name="synthetic_setting"):
    return {
        "phase": "C",
        "name": name,
        "description": "Synthetic record for renderer checks",
        "base": "movement",
        "base_result": "s0 300/500 (60.0%)",
        "seeds": [
            {"seed": 0, "successes": 301, "episodes": 500},
            {"seed": 1, "successes": 322, "episodes": 500},
        ],
        "mean_percent": 62.3,
        "spread_pp": 1.2,
        "delta_pp": 2.3,
        "verdict": "PROMISING",
        "aim_start_mm": None,
        "aim_success_mm": None,
        "steps_to_insert": 150.0,
        "peak_force_n": 90.0,
        "practice_episodes": 400,
        "hand_search": False,
        "hidden_information": False,
        "camera_condition": "normal",
        "reason": None,
        "base_mean_percent": 60.0,
        "base_spread_pp": 0.5,
    }


def test_scoreboard_rendering_includes_all_fixed_fields_and_seed_counts_in_csv_and_markdown():
    record = _scoreboard_record()
    summary, checked = validate_records({"summary": "Synthetic summary", "records": [record]})
    csv_text = render_csv(checked)
    csv_rows = list(csv.DictReader(StringIO(csv_text)))
    assert next(csv.reader(StringIO(csv_text))) == list(CSV_FIELDS)
    assert len(csv_rows) == 1
    assert all(field in csv_rows[0] for field in FIELDS)
    assert "0: 301/500 (60.2%)" in csv_rows[0]["seeds"]
    assert "1: 322/500 (64.4%)" in csv_rows[0]["seeds"]

    markdown = render_markdown(summary, checked)
    markdown_lines = markdown.splitlines()
    header_index = next(i for i, line in enumerate(markdown_lines) if line.startswith("| Phase |"))
    header_cells = [cell.strip() for cell in markdown_lines[header_index].strip("|").split("|")]
    data_cells = [cell.strip() for cell in markdown_lines[header_index + 2].strip("|").split("|")]
    assert header_cells == [
        "Phase", "Name", "Description", "Base", "Base result", "Seed results",
        "Mean (%)", "Spread (pp)", "Delta (pp)", "Verdict", "Aim start (mm)",
        "Aim at success (mm)", "Steps to insert", "Peak force (N)",
        "Practice episodes", "Hand search", "Hidden information", "Camera condition",
        "Reason", "Base mean (%)", "Base spread (pp)",
    ]
    assert len(header_cells) == len(CSV_FIELDS)
    assert len(data_cells) == len(CSV_FIELDS)
    assert "0: 301/500 (60.2%)" in data_cells[5]
    assert "1: 322/500 (64.4%)" in data_cells[5]
    assert "Hand search" in header_cells and "Hidden information" in header_cells


def test_scoreboard_rejects_duplicate_names_and_bad_verdicts():
    first = _scoreboard_record("duplicate")
    duplicate = _scoreboard_record("duplicate")
    with pytest.raises(ValueError, match="duplicate record name"):
        validate_records({"summary": "Synthetic", "records": [first, duplicate]})

    invalid = _scoreboard_record("bad_verdict")
    invalid["verdict"] = "MAYBE"
    with pytest.raises(ValueError, match="invalid verdict"):
        validate_records({"summary": "Synthetic", "records": [invalid]})
