"""Audit predeclared Goal 7 experiment results without loading model state."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def inside_results(path):
    return path.is_relative_to(ROOT / "results" / "takehome")


def display_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def fail(failures, setting, path, check, detail):
    failures.append({"setting": setting, "path": str(path),
                     "check": check, "detail": str(detail)})


def load_json(path, failures, setting, check):
    if not path.is_relative_to(ROOT):
        fail(failures, setting, path, check, "path escapes project root")
        return None
    if not path.is_file():
        fail(failures, setting, path, check, "file is missing")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(failures, setting, path, check,
             f"invalid JSON: {type(exc).__name__}: {exc}")
        return None


def npz_finite(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                arr = archive[key]
                if (np.issubdtype(arr.dtype, np.number)
                        or np.issubdtype(arr.dtype, np.bool_)):
                    if not np.isfinite(arr).all():
                        return False, f"non-finite values in {key}"
        return True, None
    except Exception as exc:
        return False, f"cannot read NPZ: {type(exc).__name__}: {exc}"


def expected_result_paths(spec, seeds, output):
    """Allow explicit legacy result paths as a seed->path map or seed-order list."""
    explicit = spec.get("results")
    if isinstance(explicit, dict):
        return {seed: resolve_path(explicit[str(seed)] if str(seed) in explicit
                                   else explicit[seed])
                for seed in seeds
                if str(seed) in explicit or seed in explicit}
    if isinstance(explicit, list):
        if len(explicit) != len(seeds):
            return {}
        return {seed: resolve_path(path) for seed, path in zip(seeds, explicit)}
    return {seed: output / f"seed_{seed}" / "result.json" for seed in seeds}


def audit_run(spec, seed, result_path, result, failures, baseline_cache):
    name = spec["name"]
    checks = {}
    cfg = result.get("config") or {}
    final = result.get("final") or {}
    run_dir = result_path.parent

    def check(key, condition, path=result_path, detail=None):
        checks[key] = bool(condition)
        if not condition:
            fail(failures, name, path, key, detail if detail is not None else "check failed")

    check("seed_matches_path", result.get("seed") == seed,
          detail={"expected": seed, "actual": result.get("seed")})
    check("config_matches_overrides",
          all(cfg.get(key) == value for key, value in spec.get("overrides", {}).items()),
          detail={"overrides": spec.get("overrides"), "config": cfg})
    check("exactly_200_practice", result.get("practice_episodes") == 200
          and cfg.get("total_episodes", 200) == 200)
    fill = cfg.get("fill_episodes")
    check("fill_matches_config", isinstance(fill, int) and result.get("fill_episodes") == fill)
    check("online_is_budget_remainder", isinstance(fill, int)
          and result.get("learner_episodes") == 200 - fill)
    check("finite_result_and_losses", finite(result) and finite(result.get("losses", [])))
    check("frozen_policy_unchanged", result.get("frozen_policy_unchanged") is True)
    check('no_through_wall_episodes', final.get('through_wall_episodes') == 0)
    if cfg.get('sibling_fill'):
        sibling_checks = result.get('frozen_siblings_unchanged', {})
        check('frozen_siblings_unchanged', len(sibling_checks) == 3 and all(sibling_checks.values()))
    check("test_data_not_learned_from", result.get("test_data_learned_from") is False)
    test_seed = cfg.get("test_seed", 123)
    check("final_test_contract", final.get("episodes") == 500
          and final.get("test_seed") == test_seed
          and final.get("test_only") is True and final.get("learned_from") is False,
          detail={"expected_episodes": 500, "expected_seed": test_seed, "final": final})

    budget_path = run_dir / "budget.json"
    budget = load_json(budget_path, failures, name, "budget ledger")
    budget_evidence = None
    if budget is not None:
        entries = budget.get("episodes", [])
        ids = [entry.get("id") for entry in entries]
        sources = [entry.get("source") for entry in entries]
        fill_count = sources.count("fill")
        online_count = sources.count("online")
        ledger_ok = (budget.get("limit") == 200 and len(entries) == 200
                     and ids == list(range(200)) and isinstance(fill, int)
                     and fill_count == fill and online_count == 200 - fill
                     and fill_count + online_count == len(entries))
        check("budget_ledger_exact", ledger_ok, budget_path,
              {"limit": budget.get("limit"), "entries": len(entries),
               "fill": fill_count, "online": online_count, "config_fill": fill})
        budget_evidence = {"path": str(budget_path.relative_to(ROOT)),
                           "entries": len(entries), "fill": fill_count,
                           "online": online_count}

    model_path = run_dir / "model_0200.pkl"
    replay_path = run_dir / "replay.npz"
    check("final_model_exists", model_path.is_file(), model_path)
    check("replay_exists", replay_path.is_file(), replay_path)
    output_limit_violations = []
    for pattern in ("model_*.pkl", "practice_*.npz"):
        for path in run_dir.glob(pattern):
            try:
                step = int(path.stem.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                output_limit_violations.append(path.name)
            else:
                if step > 200:
                    output_limit_violations.append(path.name)
    check("no_practice_outputs_above_budget", not output_limit_violations,
          run_dir, output_limit_violations)

    curve = result.get("curve")
    if not isinstance(curve, list) or not curve:
        fail(failures, name, result_path, "learning curve", "missing or empty curve")
        curve = []
    point0 = curve[0] if curve else {}
    baseline_path = resolve_path(cfg.get("baseline_path", "results/takehome/baseline.json"))
    if baseline_path not in baseline_cache:
        baseline_cache[baseline_path] = load_json(baseline_path, failures, name, "baseline JSON")
    baseline = baseline_cache.get(baseline_path) or {}
    baseline_normal = baseline.get("conditions", {}).get("normal", {})
    check("point0_baseline_contract", point0.get("practice_episodes") == 0
          and point0.get("label") == "original_policy"
          and point0.get("episodes") == 500
          and point0.get("test_seed", test_seed) == test_seed
          and point0.get("test_only") is True and point0.get("learned_from") is False
          and point0.get("successes") == baseline_normal.get("successes")
          and point0.get("success_rate") == baseline_normal.get("success_rate"),
          detail={"point0": point0, "baseline_normal": baseline_normal})
    check("final_curve_point_200", bool(curve)
          and curve[-1].get("practice_episodes") == 200
          and curve[-1].get("episodes") == 500
          and curve[-1].get("test_seed") == test_seed
          and curve[-1].get("test_only") is True
          and curve[-1].get("learned_from") is False
          and curve[-1].get("successes") == final.get("successes")
          and curve[-1].get("success_rate") == final.get("success_rate"),
          detail=curve[-1] if curve else None)
    intermediate = curve[1:-1]
    check("intermediate_curve_tests", all(
        point.get("episodes") == 256 and point.get("test_seed") == 123
        and point.get("test_only") is True and point.get("learned_from") is False
        for point in intermediate), detail=intermediate)
    check("finite_curve", finite(curve), detail="curve contains non-finite values")

    trace_path = run_dir / "test_0200_traces.npz"
    trace_ok, trace_error = npz_finite(trace_path)
    check("finite_final_trace", trace_ok, trace_path, trace_error)
    if trace_ok:
        try:
            with np.load(trace_path, allow_pickle=False) as archive:
                flags_ok = (bool(np.asarray(archive["test_only"]).item())
                            and not bool(np.asarray(archive["learned_from"]).item())
                            and int(np.asarray(archive["test_seed"]).item()) == test_seed)
                episode_success = np.asarray(archive["episode_success"])
                successes = int(episode_success.sum())
            check("final_trace_episode_count", episode_success.size == 500, trace_path,
                  {"expected": 500, "actual": int(episode_success.size)})
            check("final_trace_flags", flags_ok, trace_path,
                  {"test_seed": test_seed, "test_only": flags_ok})
            check("trace_successes_match_json", successes == final.get("successes"),
                  trace_path, {"array": successes, "json": final.get("successes")})
        except Exception as exc:
            check("final_trace_metadata_readable", False, trace_path,
                  f"{type(exc).__name__}: {exc}")

    traces = sorted(run_dir.glob("test_*_traces.npz"))
    over_budget_test_traces = []
    for path in traces:
        try:
            episodes = int(path.name.split("_")[1])
        except (IndexError, ValueError):
            over_budget_test_traces.append(path.name)
        else:
            if episodes > 200:
                over_budget_test_traces.append(path.name)
    check("no_test_trace_outputs_above_budget", not over_budget_test_traces,
          run_dir, over_budget_test_traces)
    if spec.get("results"):
        intermediate_trace_check = "explicit_result_paths_allow_legacy_outputs"
        check(intermediate_trace_check, True)
    else:
        extras = [path.name for path in traces if path.name != "test_0200_traces.npz"]
        check("no_intermediate_test_traces", not extras, run_dir, extras)

    return {"path": str(result_path.relative_to(ROOT)), "seed": seed,
            "practice_episodes": result.get("practice_episodes"),
            "fill_episodes": result.get("fill_episodes"),
            "final_test_episodes": final.get("episodes"),
            "final_test_seed": final.get("test_seed"),
            "successes": final.get("successes"), "checks": checks,
            "budget": budget_evidence,
            "artifact_paths": {"model": display_path(model_path),
                               "replay": display_path(replay_path),
                               "final_trace": display_path(trace_path)}}


def audit(manifest_path):
    failures = []
    manifest_path = resolve_path(manifest_path)
    manifest = load_json(manifest_path, failures, "manifest", "manifest JSON")
    if manifest is None:
        return {"passed": False, "failures": failures, "settings": []}
    specs = manifest.get("settings")
    if not isinstance(specs, list):
        fail(failures, "manifest", manifest_path, "settings list", "missing settings list")
        specs = []
    rows = []
    seen = set()
    baseline_cache = {}
    for spec in specs:
        if not isinstance(spec, dict):
            fail(failures, "manifest", manifest_path, "setting entry", "entry must be object")
            continue
        name = str(spec.get("name", ""))
        if not name or name in seen:
            fail(failures, name or "manifest", manifest_path, "unique setting name", name)
            continue
        seen.add(name)
        raw_seeds = spec.get("seeds")
        if not isinstance(raw_seeds, list) or not raw_seeds:
            fail(failures, name, manifest_path, "declared seeds", "seeds must be nonempty list")
            seeds = []
        else:
            seeds = raw_seeds
            if len(set(seeds)) != len(seeds):
                fail(failures, name, manifest_path, "unique declared seeds", seeds)
        output = resolve_path(spec.get("output", ""))
        if not inside_results(output):
            fail(failures, name, output, "output location", "output must be under results/takehome")
        paths = expected_result_paths(spec, seeds, output)
        if set(paths) != set(seeds):
            fail(failures, name, manifest_path, "result path coverage",
                 {"seeds": seeds, "paths": sorted(paths)})
        runs = []
        outcomes = []
        for seed in seeds:
            path = paths.get(seed)
            if path is None:
                continue
            if not inside_results(path):
                fail(failures, name, path, "result location", "result must be under results/takehome")
                continue
            result = load_json(path, failures, name, "result JSON")
            if result is None:
                continue
            runs.append(audit_run(spec, seed, path, result, failures, baseline_cache))
            trace = path.parent / "test_0200_traces.npz"
            try:
                with np.load(trace, allow_pickle=False) as archive:
                    outcomes.append(np.asarray(archive["episode_success"]))
            except Exception:
                outcomes.append(None)
        if len(runs) != len(seeds):
            fail(failures, name, output, "all declared seed results present",
                 {"declared": seeds, "loaded": [row["seed"] for row in runs]})
        valid_outcomes = len(outcomes) == len(seeds) and all(x is not None for x in outcomes)
        identical = valid_outcomes and len(outcomes) > 1 and all(
            np.array_equal(outcomes[0], value) for value in outcomes[1:])
        baseline_policy_exemption = False
        if runs and valid_outcomes:
            cfg = (load_json(resolve_path(paths[seeds[0]]), [], name, "result") or {}).get("config", {})
            baseline_path = resolve_path(cfg.get("baseline_path", "results/takehome/baseline.json"))
            baseline = baseline_cache.get(baseline_path) or {}
            conditions = baseline.get("conditions", {})
            normal = conditions.get("normal", {}).get("success_rate")
            ceiling = conditions.get("no_camera_bias", {}).get("success_rate")
            baseline_policy_exemption = (cfg.get("policy") == baseline.get("policy")
                                         and normal == 0 and ceiling == 0)
        if identical and not baseline_policy_exemption:
            fail(failures, name, manifest_path,
                 "final success arrays differ across seeds", {"seeds": seeds})
        rows.append({"name": name, "label": spec.get("label"),
                     "overrides": spec.get("overrides", {}), "seeds": seeds,
                     "output": display_path(output), "runs": runs,
                     "final_success_arrays_present": valid_outcomes,
                     "final_success_arrays_all_identical": identical,
                     "baseline_floor_ceiling_zero_exemption": baseline_policy_exemption})
    return {"manifest": display_path(manifest_path),
            "setting_count": len(rows), "settings": rows,
            "failure_count": len(failures), "failures": failures,
            "passed": not failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = resolve_path(args.output)
    if not inside_results(output_path):
        parser.error("--output must be within results/takehome")
    report = audit(args.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps({"audit": str(output_path.relative_to(ROOT)),
                      "passed": report["passed"],
                      "failure_count": report["failure_count"]}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
