"""Synthetic ledger checks; these tests collect no simulator episodes."""
import pytest
from takehome.budget import EpisodeBudget


def test_refuses_201_and_persists_reservations(tmp_path):
    path = tmp_path / "ledger.json"
    counter = EpisodeBudget(path)
    assert counter.reserve(40, "fill") == list(range(40))
    assert len(counter.reserve(160, "online")) == 160
    with pytest.raises(RuntimeError, match="budget exceeded"):
        counter.reserve(1, "online")
    resumed = EpisodeBudget(path)
    assert resumed.used == 200
    with pytest.raises(RuntimeError):
        resumed.reserve(1, "fill")
    assert resumed.used == 200


def test_test_data_and_unreserved_data_are_rejected(tmp_path):
    counter = EpisodeBudget(tmp_path / "ledger.json", limit=10)
    ids = counter.reserve(2, "fill")
    counter.require_practice(ids, "fill", test_only=False)
    for source, test_only, bad_ids in (
        ("test", True, ids), ("fill", True, ids),
        ("online", False, ids), ("fill", False, [3]),
    ):
        with pytest.raises(ValueError):
            counter.require_practice(bad_ids, source, test_only=test_only)
    with pytest.raises(ValueError):
        counter.reserve(1, "test")
    with pytest.raises(ValueError):
        counter.reserve(-1, "fill")
    assert counter.used == 2
