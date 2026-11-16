"""Audit Goal 6 run artifacts against the predeclared manifest.

This script validates provenance and file-level checks only. It does not rank,
interpret, or select methods.
"""

import argparse
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results/takehome/goal6_manifest.json"
DEFAULT_OUTPUT = ROOT / "results/takehome/goal6_audit.json"
EXPECTED_SEEDS = [0, 1, 2, 3, 4]
EXPECTED_OUTPUTS = {"aim": 2, "movement": 3, "combined": 5,
                    "supervised_cheating_reference": 2}


def _path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _inside_root(path):
    return path.is_relative_to(ROOT)


def _finite_json(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_json(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_json(v) for v in value)
    return True


def _npz_finite(path):
    """Return (ok, error) after checking every numeric array in an archive."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                value = archive[key]
                if (np.issubdtype(value.dtype, np.number)
                        or np.issubdtype(value.dtype, np.bool_)):
                    if not np.isfinite(value).all():
                        return False, f"non-finite values in {key}"
        return True, None
    except Exception as exc:  # malformed or unreadable evidence is a failed check
        return False, f"cannot read NPZ: {type(exc).__name__}: {exc}"


def _fail(failures, method, path, check, detail):
    failures.append({"method": method, "path": str(path),
                     "check": check, "detail": str(detail)})


def _load_json(path, failures, method, check):
    if not _inside_root(path):
        _fail(failures, method, path, check, "path escapes project root")
        return None
    if not path.is_file():
        _fail(failures, method, path, check, "file is missing")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail(failures, method, path, check,
              f"invalid JSON: {type(exc).__name__}: {exc}")
        return None


def _check_budget(result_path, result, method, failures):
    budget_path = result_path.parent / "budget.json"
    budget = _load_json(budget_path, failures, method, "practice budget ledger")
    if budget is None:
        return None
    entries = budget.get("episodes", [])
    ids = [entry.get("id") for entry in entries]
    sources = [entry.get("source") for entry in entries]
    checks = {
        "ledger_limit_200": budget.get("limit") == 200,
        "exactly_200_budget_entries": len(entries) == 200 and ids == list(range(200)),
        "exactly_40_fill_160_online": sources.count("fill") == 40 and sources.count("online") == 160,
        "result_practice_episodes_200": result.get("practice_episodes") == 200,
        "result_fill_episodes_40": result.get("fill_episodes") == 40,
        "result_learner_episodes_160": result.get("learner_episodes") == 160,
    }
    for name, passed in checks.items():
        if not passed:
            _fail(failures, method, budget_path, name,
                  {"limit": budget.get("limit"), "entry_count": len(entries),
                   "fill_count": sources.count("fill"),
                   "online_count": sources.count("online")})
    return {"path": str(budget_path.relative_to(ROOT)), "checks": checks,
            "fill_count": sources.count("fill"),
            "online_count": sources.count("online"), "entry_count": len(entries)}


def _check_arrays(result_path, method, old_goal5, result, failures):
    run_dir = result_path.parent
    practice_paths = sorted(run_dir.glob("practice_[0-9][0-9][0-9][0-9].npz"))
    expected_practice_names = [f"practice_{n:04d}.npz" for n in range(10, 201, 10)]
    actual_practice_names = [p.name for p in practice_paths]
    if actual_practice_names != expected_practice_names:
        _fail(failures, method, run_dir, "practice trace files",
              {"expected": expected_practice_names, "actual": actual_practice_names})
    bad_practice = []
    for path in practice_paths:
        ok, error = _npz_finite(path)
        if not ok:
            bad_practice.append({"path": str(path.relative_to(ROOT)), "error": error})
    if bad_practice:
        _fail(failures, method, run_dir, "finite learner traces", bad_practice)

    episode_path = run_dir / "test_0200_episodes.npz"
    episode_ok, episode_error = _npz_finite(episode_path)
    if not episode_ok:
        _fail(failures, method, episode_path, "finite final test arrays", episode_error)
    else:
        try:
            with np.load(episode_path, allow_pickle=False) as archive:
                succeeded = np.asarray(archive["succeeded"])
            if succeeded.size != 500:
                _fail(failures, method, episode_path, "final test array size",
                      {"expected": 500, "actual": int(succeeded.size)})
            final = result.get("final", {})
            if int(succeeded.sum()) != final.get("successes"):
                _fail(failures, method, episode_path, "final success count matches JSON",
                      {"array": int(succeeded.sum()), "json": final.get("successes")})
        except Exception as exc:
            _fail(failures, method, episode_path, "final success array",
                  f"cannot read succeeded array: {type(exc).__name__}: {exc}")

    trace_path = run_dir / "test_0200_traces.npz"
    trace_ok, trace_error = _npz_finite(trace_path)
    if not trace_ok:
        _fail(failures, method, trace_path, "finite final trace", trace_error)
    else:
        try:
            with np.load(trace_path, allow_pickle=False) as archive:
                labels_ok = (
                    bool(np.asarray(archive["test_only"]).item())
                    and not bool(np.asarray(archive["learned_from"]).item())
                    and int(np.asarray(archive["test_seed"]).item()) == 123
                )
            if not labels_ok:
                _fail(failures, method, trace_path, "final trace test labels",
                      "expected test_only=true, learned_from=false, test_seed=123")
        except Exception as exc:
            _fail(failures, method, trace_path, "final trace test labels",
                  f"missing or invalid trace metadata: {type(exc).__name__}: {exc}")

    trace_paths = sorted(run_dir.glob("test_[0-9][0-9][0-9][0-9]_traces.npz"))
    if not old_goal5:
        extras = [p.name for p in trace_paths if p.name != "test_0200_traces.npz"]
        if extras:
            _fail(failures, method, run_dir, "no intermediate trace NPZ in new run", extras)
    return {
        "practice_trace_count": len(practice_paths),
        "practice_trace_files": actual_practice_names,
        "final_episode_array": str(episode_path.relative_to(ROOT)),
        "final_trace": str(trace_path.relative_to(ROOT)),
        "trace_files": [p.name for p in trace_paths],
        "old_goal5_trace_exception": old_goal5,
    }


def _check_run(method_spec, result_value, method, failures):
    result_path = result_value["path"]
    result = result_value["result"]
    config = result.get("config") or {}
    seed = result.get("seed")
    practice = result.get("practice_episodes", 0)
    learning = practice > 0
    checks = {}

    if learning:
        checks.update(_check_budget(result_path, result, method, failures) or {})
        final = result.get("final") or {}
        final_checks = {
            "final_test_episodes_500": final.get("episodes") == 500,
            "final_test_seed_123": final.get("test_seed") == 123,
            "final_test_only": final.get("test_only") is True,
            "final_not_learned_from": final.get("learned_from") is False,
            "result_test_data_not_learned_from": result.get("test_data_learned_from") is False,
            "frozen_policy_unchanged": result.get("frozen_policy_unchanged") is True,
            "no_through_wall_episodes": final.get("through_wall_episodes") == 0,
            "finite_result_numbers": _finite_json(result),
            "finite_losses": _finite_json(result.get("losses", [])),
        }
        for name, passed in final_checks.items():
            checks[name] = passed
            if not passed:
                _fail(failures, method, result_path, name,
                      final if name.startswith("final") or name == "no_through_wall_episodes" else result.get(name))
        expected_outputs = EXPECTED_OUTPUTS.get(method)
        if expected_outputs is not None and config.get("outputs") != expected_outputs:
            checks["output_count_matches_method"] = False
            _fail(failures, method, result_path, "output_count_matches_method",
                  {"expected": expected_outputs, "actual": config.get("outputs")})
        else:
            checks["output_count_matches_method"] = True
        old_goal5 = "benchmark" in result_path.parts and "goal6" not in result_path.parts
        array_evidence = _check_arrays(result_path, method, old_goal5, result, failures)
        checks["arrays"] = array_evidence
    else:
        checks["no_learning"] = practice == 0
        if practice != 0:
            _fail(failures, method, result_path, "no-learning method classification", practice)
        checks["frozen_policy_unchanged"] = result.get("frozen_policy_unchanged", True) is True
        if not checks["frozen_policy_unchanged"]:
            _fail(failures, method, result_path, "frozen_policy_unchanged", result.get("frozen_policy_unchanged"))

    config_cheats = bool(config.get("supervised_reference", False))
    manifest_cheats = method_spec.get("cheats_on_information")
    result_cheats = result.get("cheats_on_information", config_cheats)
    cheat_checks = {
        "manifest_cheat_flag_matches_config": manifest_cheats is config_cheats,
        "result_cheat_flag_matches_config": result_cheats is config_cheats,
        "cheating_label_explicit": (not config_cheats or
                                     "cheat" in str(method_spec.get("label", "")).lower()),
    }
    checks.update(cheat_checks)
    for name, passed in cheat_checks.items():
        if not passed:
            _fail(failures, method, result_path, name,
                  {"manifest": manifest_cheats, "result": result_cheats,
                   "config": config_cheats, "label": method_spec.get("label")})

    return {
        "path": str(result_path.relative_to(ROOT)),
        "seed": seed,
        "practice_episodes": practice,
        "final_test_episodes": (result.get("final") or {}).get("episodes"),
        "final_test_seed": (result.get("final") or {}).get("test_seed"),
        "successes": (result.get("final") or {}).get("successes"),
        "success_rate": (result.get("final") or {}).get("success_rate"),
        "checks": checks,
    }


def _audit_zero_contract(manifest, failures):
    value = manifest.get("zero_contract_test")
    if isinstance(value, dict):
        value = value.get("path") or value.get("result")
    if not isinstance(value, str) or not value:
        _fail(failures, "combined", DEFAULT_MANIFEST, "explicit zero-contract evidence",
              "manifest must name a separate zero_contract_test exit/result JSON")
        return {"passed": False, "path": None}
    path = _path(value)
    data = _load_json(path, failures, "combined", "explicit zero-contract evidence")
    if data is None:
        return {"passed": False, "path": value}
    command = data.get("command", [])
    command_text = " ".join(str(x) for x in command) if isinstance(command, list) else str(command)
    explicit_contract = (data.get("zero_contract_passed") is True
                         or data.get("combined_zero_init_passed") is True
                         or "test_goal6.py" in command_text)
    passed = data.get("exit_code") == 0 and explicit_contract
    if not passed:
        _fail(failures, "combined", path, "zero-contract test passed separately",
              {"exit_code": data.get("exit_code"), "command": command,
               "explicit_contract_evidence": explicit_contract})
    return {"path": str(path.relative_to(ROOT)), "exit_code": data.get("exit_code"),
            "command": command, "explicit_contract_evidence": explicit_contract,
            "passed": passed}


def audit(manifest_path):
    failures = []
    manifest = _load_json(_path(manifest_path), failures, "manifest", "manifest JSON")
    if manifest is None:
        return {"passed": False, "failures": failures, "methods": []}
    methods = manifest.get("methods")
    if not isinstance(methods, list):
        _fail(failures, "manifest", manifest_path, "methods list", "missing or invalid methods list")
        methods = []

    baseline_evidence = None
    if manifest.get("baseline"):
        baseline_path = _path(manifest["baseline"])
        baseline = _load_json(baseline_path, failures, "baseline", "baseline JSON")
        if baseline is not None:
            conditions = baseline.get("conditions", {})
            expected = {"normal", "no_camera_bias"}
            baseline_checks = {}
            for condition in sorted(expected):
                value = conditions.get(condition, {})
                ok = (value.get("episodes") == 500 and value.get("test_seed") == 123
                      and value.get("test_only") is True and value.get("learned_from") is False)
                baseline_checks[condition] = ok
                if not ok:
                    _fail(failures, "baseline", baseline_path, f"{condition} baseline contract", value)
            baseline_evidence = {"path": str(baseline_path.relative_to(ROOT)),
                                 "checks": baseline_checks}

    method_rows = []
    seen_names = set()
    for spec in methods:
        name = str(spec.get("name", ""))
        if not name or name in seen_names:
            _fail(failures, name or "manifest", manifest_path, "unique method name", name)
            continue
        seen_names.add(name)
        results = spec.get("results", [])
        if not isinstance(results, list):
            _fail(failures, name, manifest_path, "result path list", "results must be a list")
            results = []
        loaded = []
        seen_paths = set()
        for raw_path in results:
            path = _path(raw_path)
            if path in seen_paths:
                _fail(failures, name, path, "unique result path", "duplicate result path")
                continue
            seen_paths.add(path)
            result = _load_json(path, failures, name, "result JSON")
            if result is not None:
                loaded.append({"path": path, "result": result})
        practices = [item["result"].get("practice_episodes", 0) for item in loaded]
        no_learning = bool(loaded) and all(value == 0 for value in practices)
        seeds = [item["result"].get("seed") for item in loaded]
        if no_learning:
            seed_check = len(loaded) == 1 and seeds[0] in (None, 0)
        else:
            seed_check = len(loaded) == 5 and sorted(seeds) == EXPECTED_SEEDS
        if not seed_check:
            _fail(failures, name, manifest_path, "practice seed set", {
                "expected": "one no-learning reference" if no_learning else EXPECTED_SEEDS,
                "actual_seeds": seeds, "result_count": len(loaded)})
        run_rows = [_check_run(spec, item, name, failures) for item in loaded]

        outcomes = []
        if not no_learning and len(loaded) == 5:
            for item in loaded:
                array_path = item["path"].parent / "test_0200_episodes.npz"
                try:
                    with np.load(array_path, allow_pickle=False) as archive:
                        outcomes.append(np.asarray(archive["succeeded"]))
                except Exception:
                    # _check_arrays already recorded the file/array error.
                    outcomes.append(None)
            arrays_valid = all(value is not None for value in outcomes)
            all_identical = arrays_valid and all(np.array_equal(outcomes[0], value)
                                                 for value in outcomes[1:])
            if arrays_valid and all_identical:
                _fail(failures, name, manifest_path,
                      "final outcome arrays differ across five seeds",
                      {"seeds": seeds, "all_identical": True})
        else:
            arrays_valid = None
            all_identical = None
        method_rows.append({
            "name": name, "label": spec.get("label"),
            "cheats_on_information": spec.get("cheats_on_information"),
            "no_learning": no_learning,
            "seed_check_passed": seed_check,
            "runs": run_rows,
            "final_outcome_arrays_present": arrays_valid,
            "final_outcome_arrays_all_identical": all_identical,
        })

    zero_contract = _audit_zero_contract(manifest, failures)
    return {
        "manifest": str(_path(manifest_path).relative_to(ROOT)),
        "baseline": baseline_evidence,
        "zero_contract_test": zero_contract,
        "method_count": len(method_rows),
        "methods": method_rows,
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="Goal 6 manifest (default: results/takehome/goal6_manifest.json)")
    args = parser.parse_args()
    report = audit(args.manifest)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = DEFAULT_OUTPUT.with_suffix(DEFAULT_OUTPUT.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(DEFAULT_OUTPUT)
    print(json.dumps({"audit": str(DEFAULT_OUTPUT.relative_to(ROOT)),
                      "passed": report["passed"],
                      "failure_count": report["failure_count"]}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
