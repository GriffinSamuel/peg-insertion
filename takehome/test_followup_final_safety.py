"""Synthetic safety checks for the direct Phase G supervisor and aggregator."""
from takehome import followup_runtime  # noqa: F401 -- cache setup before JAX imports

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from takehome import followup_collect, followup_final_suite


class _FakeProcess:
    def __init__(self, shard, code, stdout):
        self.shard = shard
        self.pid = 10_000 + shard
        self.code = code
        self.terminated = False
        if shard == 0 and code == 1:
            stdout.write("Tripwire: synthetic invalid-number failure\n")
            stdout.flush()
        elif shard == 0 and code == 2:
            stdout.write("ordinary synthetic worker error\n")
            stdout.flush()

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        if self.code is None:
            self.code = -15

    def wait(self, timeout=None):
        return self.code


def _run_with_fake_workers(tmp_path, monkeypatch, failure_code):
    out = tmp_path / "followup"
    (out / "logs").mkdir(parents=True)
    (out / "G").mkdir()
    registry = {
        "status": "active",
        "completed_phases": ["A", "B", "B_mixed", "C", "D", "E", "F"],
    }
    (out / "registry.json").write_text(json.dumps(registry))
    (out / "G" / "choice.json").write_text(json.dumps({"synthetic": True}))
    monkeypatch.setattr(followup_final_suite, "OUT", out)
    monkeypatch.setattr(followup_final_suite, "ROOT", tmp_path)
    monkeypatch.setattr(followup_final_suite, "configurations", lambda choice: [("chosen", object())])
    monkeypatch.setattr(followup_final_suite.time, "sleep", lambda seconds: None)
    processes = []

    def fake_popen(command, *, cwd, stdin, stdout, stderr):
        shard = int(command[-1])
        code = failure_code if shard == 0 else None
        process = _FakeProcess(shard, code, stdout)
        processes.append(process)
        return process

    monkeypatch.setattr(followup_final_suite.subprocess, "Popen", fake_popen)
    with pytest.raises(RuntimeError):
        followup_final_suite.run("synthetic_g")
    saved_registry = json.loads((out / "registry.json").read_text())
    suite = json.loads((out / "synthetic_g_suite.json").read_text())
    return out, processes, saved_registry, suite


def test_supervisor_persists_tripwire_halt_and_terminates_other_workers(tmp_path, monkeypatch):
    out, processes, registry, suite = _run_with_fake_workers(tmp_path, monkeypatch, failure_code=1)
    assert len(processes) == 4
    assert all(process.terminated for process in processes[1:])
    assert registry["status"] == "halted_tripwire"
    assert registry["halted_at"] == "G"
    assert suite["tripwire"] is True
    assert suite["failure"] and "Tripwire:" in suite["failure"]
    assert (out / "logs" / "synthetic_g_shard_0.log").is_file()


def test_supervisor_does_not_label_ordinary_worker_error_as_tripwire(tmp_path, monkeypatch):
    _, processes, registry, suite = _run_with_fake_workers(tmp_path, monkeypatch, failure_code=2)
    assert len(processes) == 4
    assert all(process.terminated for process in processes[1:])
    assert registry["status"] == "active"
    assert "halted_at" not in registry
    assert suite["tripwire"] is False
    assert suite["failure"] and "Tripwire:" not in suite["failure"]


def _synthetic_method_rows(tmp_path, outcomes, *, floor_successes=1, ceiling_successes=1):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"conditions": {
        "normal": {"successes": floor_successes},
        "no_camera_bias": {"successes": ceiling_successes},
    }}))
    rows = []
    for index, seed in enumerate(range(200, 205)):
        run_dir = tmp_path / f"seed_{seed}"
        run_dir.mkdir(exist_ok=True)
        result_path = run_dir / "result.json"
        result_path.write_text(json.dumps({"config": {"baseline_path": "baseline.json"}}))
        trace_path = run_dir / "test_0200_traces.npz"
        np.savez_compressed(trace_path, episode_success=np.asarray(outcomes[index], dtype=np.float32))
        successes = int(np.count_nonzero(outcomes[index]))
        rows.append({"seed": seed, "successes": successes,
                     "trace_path": str(trace_path), "result_path": str(result_path)})
    return rows


def _call_completed_check(tmp_path, monkeypatch, outcomes, *, floor_successes=1,
                          ceiling_successes=1):
    out = tmp_path / "g_outputs"
    for seed in range(200, 205):
        run_dir = out / "G" / "chosen" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text("{}")
    rows = _synthetic_method_rows(tmp_path, outcomes, floor_successes=floor_successes,
                                  ceiling_successes=ceiling_successes)
    monkeypatch.setattr(followup_final_suite, "OUT", out)
    monkeypatch.setattr(followup_final_suite, "FRESH_SEEDS", tuple(range(200, 205)))
    monkeypatch.setattr(followup_final_suite, "_load_method_seed",
                        lambda method, config, seed: rows[seed - 200])
    monkeypatch.setattr(followup_collect, "ROOT", tmp_path)
    checked = set()
    return checked


def test_g_completed_method_rejects_identical_five_seed_vectors(tmp_path, monkeypatch):
    vector = np.zeros(500, dtype=np.float32)
    vector[[3, 27, 401]] = 1.0
    rows = [vector.copy() for _ in range(5)]
    checked = _call_completed_check(tmp_path, monkeypatch, rows)
    with pytest.raises(RuntimeError, match="Tripwire: every learning seed"):
        followup_final_suite.check_completed_methods([("chosen", SimpleNamespace())], checked)
    assert checked == set()


def test_g_completed_method_allows_distinct_vectors_with_equal_success_counts(tmp_path, monkeypatch):
    rows = []
    for i in range(5):
        vector = np.zeros(500, dtype=np.float32)
        vector[[i, i + 100, i + 300]] = 1.0
        rows.append(vector)
    checked = _call_completed_check(tmp_path, monkeypatch, rows)
    followup_final_suite.check_completed_methods([("chosen", SimpleNamespace())], checked)
    assert checked == {"chosen"}


def test_g_completed_method_preserves_zero_floor_and_zero_ceiling_exemption(tmp_path, monkeypatch):
    vector = np.zeros(500, dtype=np.float32)
    rows = [vector.copy() for _ in range(5)]
    checked = _call_completed_check(tmp_path, monkeypatch, rows,
                                    floor_successes=0, ceiling_successes=0)
    followup_final_suite.check_completed_methods([("chosen", SimpleNamespace())], checked)
    assert checked == {"chosen"}
