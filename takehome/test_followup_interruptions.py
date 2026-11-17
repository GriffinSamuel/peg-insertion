"""Synthetic accounting checks for interrupted, uncounted test requests."""
from takehome import followup_runtime  # initialize isolated roots before imports

import json
from pathlib import Path

from takehome.followup_accounting import _count_interrupted_tests


def _write_record(directory: Path, name: str, **overrides):
    record = {
        "requested_episodes": 256,
        "completed_episodes": None,
        "completed_count_known": False,
        "test_only": True,
        "learned_from": False,
    }
    record.update(overrides)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(record), encoding="utf-8")


def test_valid_interrupted_test_is_reported_as_request_not_completed_count(tmp_path):
    out = tmp_path / "followup"
    _write_record(out / "interruptions", "request.json")
    errors = []

    result = _count_interrupted_tests(out, errors)

    assert errors == []
    assert result["request_count"] == 1
    assert result["requested_episodes"] == 256
    assert result["included_in_completed_full_test_total"] is False
    assert result["files"][0]["valid"] is True
    assert result["files"][0]["completed_episodes"] is None


def test_invalid_interruption_provenance_is_rejected(tmp_path):
    out = tmp_path / "followup"
    interruptions = out / "interruptions"
    _write_record(interruptions, "wrong_test_only.json", test_only=False)
    _write_record(interruptions, "wrong_learning.json", learned_from=True)
    _write_record(interruptions, "known_count.json", completed_count_known=True)
    errors = []

    result = _count_interrupted_tests(out, errors)

    assert result["request_count"] == 3
    assert result["requested_episodes"] == 0
    assert all(not row["valid"] for row in result["files"])
    assert len(errors) == 3
    assert all("invalid interrupted-test provenance" in error for error in errors)


def test_no_interruption_records_reports_zero_requests(tmp_path):
    result = _count_interrupted_tests(tmp_path / "empty-followup", [])

    assert result["files"] == []
    assert result["request_count"] == 0
    assert result["requested_episodes"] == 0
    assert result["included_in_completed_full_test_total"] is False
