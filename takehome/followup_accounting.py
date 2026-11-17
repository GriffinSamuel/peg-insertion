"""Count follow-up practice ledgers and explicitly saved test summaries.

This collector reads JSON only. It never opens NPZ files or experiment modules.
The only output written by the CLI is results/takehome/followup/accounting.json.
"""

from pathlib import Path
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Initialize the isolated follow-up output root before other project imports.
from takehome import followup_runtime  # noqa: E402,F401


OUT = followup_runtime.OUT
TEST_SUMMARY_RE = re.compile(r"^test_\d{4}\.json$")
EXTRA_CEILING_RE = re.compile(r"^extra_ceiling_\d{4}\.json$")
REFERENCE_CONDITIONS = ("normal", "no_camera_bias")


def setting_counts(settings, old_settings):
    """Keep conservative attempted slots separate from distinct configurations."""
    counted=[e for e in settings if isinstance(e,dict) and e.get('counted') is True]
    repeats=[]
    for entry in counted:
        original=entry.get('repeats_original')
        if not original:continue
        mode=entry['name'].removeprefix('E_normal_')
        expected=('goal6/combined' if mode=='joint' else 'goal7/'+mode)
        if (mode not in ('joint','aim_then_move','move_then_aim')
                or entry['name']!='E_normal_'+mode or entry['phase']!='E'
                or entry['condition']!='normal' or original!=expected
                or entry['overrides']!=dict(outputs=5,camera_condition='normal',coordination=mode)):
            raise ValueError('Only audited unchanged normal Phase E controls can be counted as repeats')
        repeats.append(entry['name'])
    return dict(new_counted_settings_from_registry=len(counted),
        old_study_settings_from_protocol=old_settings,
        whole_project_settings=old_settings+len(counted),
        repeated_original_setting_count=len(repeats),repeated_original_settings=repeats,
        new_distinct_settings=len(counted)-len(repeats),
        whole_project_distinct_settings=old_settings+len(counted)-len(repeats),
        scheduling_cap_includes_repeated_controls=True)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_cache_path(relative_path):
    return "cache" in Path(relative_path).parts


def _int_value(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_test_summary(data, *, label):
    errors = []
    if data.get("test_only") is not True:
        errors.append("test_only must be true")
    if data.get("learned_from") is not False:
        errors.append("learned_from must be false")
    episodes = data.get("episodes")
    if not _int_value(episodes) or episodes <= 0:
        errors.append("episodes must be a positive integer")
        episodes = None
    successes = data.get("successes")
    if successes is not None and (not _int_value(successes) or successes < 0 or
                                  (episodes is not None and successes > episodes)):
        errors.append("successes must be an integer between zero and episodes")
    return episodes, successes, errors


def _test_file_row(path, out, source):
    relative = path.relative_to(out).as_posix()
    try:
        data = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": relative, "source": source, "valid": False,
                "counted_full_test_episodes": 0,
                "validation_errors": [f"could not read JSON: {exc}"]}
    episodes, successes, errors = _validate_test_summary(data, label=relative)
    return {
        "path": relative,
        "source": source,
        "valid": not errors,
        "episodes": episodes,
        "successes": successes,
        "test_seed": data.get("test_seed"),
        "practice_episodes": data.get("practice_episodes"),
        "counted_full_test_episodes": episodes if not errors else 0,
        "validation_errors": errors,
    }


def _reference_row(path, out):
    relative = path.relative_to(out).as_posix()
    try:
        data = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": relative, "source": "references", "valid": False,
                "counted_full_test_episodes": 0,
                "validation_errors": [f"could not read JSON: {exc}"]}
    errors = []
    if data.get("practice_episodes") != 0:
        errors.append("reference practice_episodes must be zero")
    if data.get("test_only") is not True:
        errors.append("reference test_only must be true")
    if data.get("learned_from") is not False:
        errors.append("reference learned_from must be false")
    conditions = data.get("conditions")
    if not isinstance(conditions, dict):
        conditions = {}
        errors.append("conditions must be an object")
    condition_counts = {}
    for name in REFERENCE_CONDITIONS:
        condition = conditions.get(name)
        if not isinstance(condition, dict):
            errors.append(f"conditions.{name} is missing or invalid")
            continue
        count = condition.get("episodes")
        if not _int_value(count) or count <= 0:
            errors.append(f"conditions.{name}.episodes must be a positive integer")
        else:
            condition_counts[name] = count
        if condition.get("test_only") is not True:
            errors.append(f"conditions.{name}.test_only must be true")
        if condition.get("learned_from") is not False:
            errors.append(f"conditions.{name}.learned_from must be false")
    total = sum(condition_counts.values()) if len(condition_counts) == len(REFERENCE_CONDITIONS) else None
    valid = not errors
    return {
        "path": relative,
        "source": "references",
        "valid": valid,
        "practice_episodes": data.get("practice_episodes"),
        "test_seed": data.get("test_seed"),
        "condition_episode_counts": condition_counts,
        "episodes": total,
        "counted_full_test_episodes": total if valid else 0,
        "validation_errors": errors,
    }


def _count_budget_ledgers(out, errors):
    rows = []
    for path in sorted(out.rglob("budget.json")):
        relative = path.relative_to(out)
        if _is_cache_path(relative):
            continue
        row_errors = []
        try:
            data = _read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{relative.as_posix()}: could not read ledger JSON: {exc}")
            rows.append({"path": relative.as_posix(), "valid": False,
                         "reserved_episode_count": 0,
                         "source_counts": {"fill": 0, "online": 0, "unknown": 0},
                         "validation_errors": [f"could not read JSON: {exc}"]})
            continue
        entries = data.get("episodes", [])
        if not isinstance(entries, list):
            entries = []
            row_errors.append("episodes must be a list")
        reserved = len(entries)  # Count reserved entries, including failed/interrupted runs.

        config = {}
        config_path = path.parent / "config.json"
        result_path = path.parent / "result.json"
        try:
            if config_path.is_file():
                config_doc = _read_json(config_path)
                config = config_doc.get("config", config_doc) if isinstance(config_doc, dict) else {}
            elif result_path.is_file():
                result_doc = _read_json(result_path)
                config = result_doc.get("config", {})
                if "code_test" not in config and "code_test" in result_doc:
                    config = {**config, "code_test": result_doc["code_test"]}
        except (OSError, json.JSONDecodeError) as exc:
            row_errors.append(f"could not read run config: {exc}")

        limit = data.get("limit")
        configured_code_test = config.get("code_test")
        inferred_code_test = configured_code_test if isinstance(configured_code_test, bool) else limit == 10
        expected_limit = 10 if inferred_code_test else 200
        if limit != expected_limit:
            row_errors.append(f"limit must be {expected_limit} for this run type")
        if _int_value(limit) and reserved > limit:
            row_errors.append(f"reserved episode count {reserved} exceeds ledger limit {limit}")
        if configured_code_test is True and limit != 10:
            row_errors.append("code_test ledger must have limit 10")
        if configured_code_test is False and limit != 200:
            row_errors.append("normal ledger must have limit 200")

        ids = []
        source_counts = {"fill": 0, "online": 0, "unknown": 0}
        malformed_entries = 0
        for entry in entries:
            if not isinstance(entry, dict) or not _int_value(entry.get("id")):
                malformed_entries += 1
                continue
            ids.append(entry["id"])
            source = entry.get("source")
            if source in ("fill", "online"):
                source_counts[source] += 1
            else:
                source_counts["unknown"] += 1
        if malformed_entries:
            row_errors.append(f"{malformed_entries} ledger entries lack an integer ID")
        if len(set(ids)) != len(entries):
            row_errors.append("episode IDs must be unique")
        if sorted(ids) != list(range(len(entries))):
            row_errors.append("episode IDs must be contiguous from zero")
        if source_counts["unknown"]:
            row_errors.append("every source must be fill or online")

        row = {
            "path": relative.as_posix(),
            "run_directory": path.parent.relative_to(out).as_posix(),
            "run_type": "code_test" if inferred_code_test else "normal",
            "configured_code_test": configured_code_test,
            "limit": limit,
            "reserved_episode_count": reserved,
            "source_counts": source_counts,
            "ids_unique_contiguous_from_zero": not any(
                e in row_errors for e in ("episode IDs must be unique", "episode IDs must be contiguous from zero")
            ),
            "result_json_present": result_path.is_file(),
            "progress_json_present": (path.parent / "progress.json").is_file(),
            "valid": not row_errors,
            "validation_errors": row_errors,
        }
        if row_errors:
            errors.extend(f"{relative.as_posix()}: {message}" for message in row_errors)
        rows.append(row)
    return rows


def _test_summary_paths(out):
    candidates = []
    for path in out.rglob("*.json"):
        relative = path.relative_to(out)
        if _is_cache_path(relative) or "references" in relative.parts:
            continue
        if TEST_SUMMARY_RE.fullmatch(path.name):
            candidates.append((path, "test_NNNN"))
        elif EXTRA_CEILING_RE.fullmatch(path.name):
            candidates.append((path, "extra_ceiling_NNNN"))
    zero_check = out / "B" / "checks" / "zero.json"
    if zero_check.is_file():
        candidates.append((zero_check, "B/checks/zero.json"))
    return sorted(candidates, key=lambda item: item[0].as_posix())


def _count_full_tests(out, errors):
    rows = []
    for path, source in _test_summary_paths(out):
        row = _test_file_row(path, out, source)
        rows.append(row)
        errors.extend(f"{row['path']}: {message}" for message in row["validation_errors"])

    excluded_reference_files = []
    references = out / "references"
    if references.is_dir():
        for path in sorted(references.glob("*.json")):
            if path.name == "old_learning.json":
                excluded_reference_files.append({"path": path.relative_to(out).as_posix(),
                                                 "reason": "reused old-study summary, excluded by protocol"})
                continue
            try:
                data = _read_json(path)
            except (OSError, json.JSONDecodeError) as exc:
                excluded_reference_files.append({"path": path.relative_to(out).as_posix(),
                                                 "reason": f"not a readable reference summary: {exc}"})
                continue
            conditions = data.get("conditions")
            if data.get("practice_episodes") != 0 or not isinstance(conditions, dict) or not all(
                    name in conditions for name in REFERENCE_CONDITIONS):
                excluded_reference_files.append({"path": path.relative_to(out).as_posix(),
                                                 "reason": "does not meet practice-zero, two-condition reference rule"})
                continue
            row = _reference_row(path, out)
            rows.append(row)
            errors.extend(f"{row['path']}: {message}" for message in row["validation_errors"])

    source_counts = {}
    for row in rows:
        source = row["source"]
        counts = source_counts.setdefault(source, {"file_count": 0, "valid_file_count": 0,
                                                   "full_test_episodes": 0})
        counts["file_count"] += 1
        if row["valid"]:
            counts["valid_file_count"] += 1
            counts["full_test_episodes"] += row["counted_full_test_episodes"]
    for source in ("test_NNNN", "extra_ceiling_NNNN", "B/checks/zero.json", "references"):
        source_counts.setdefault(source, {"file_count": 0, "valid_file_count": 0,
                                          "full_test_episodes": 0})
    total = sum(row["counted_full_test_episodes"] for row in rows)
    return {
        "files": rows,
        "source_counts": source_counts,
        "candidate_file_count": len(rows),
        "valid_file_count": sum(bool(row["valid"]) for row in rows),
        "full_test_episodes": total,
        "excluded_reference_files": excluded_reference_files,
    }


def _cache_caveat(out):
    paths = []
    for path in sorted(out.rglob("*.json")):
        relative = path.relative_to(out)
        if _is_cache_path(relative) and (
                TEST_SUMMARY_RE.fullmatch(path.name) or EXTRA_CEILING_RE.fullmatch(path.name)):
            paths.append(relative.as_posix())
    return {
        "candidate_summary_paths_seen": paths,
        "candidate_summary_path_count": len(paths),
        "included_in_full_test_totals": False,
        "episode_counted": 0,
        "caveat": "All cache directories are excluded. Any simulation test summaries stored there are listed as excluded candidates and are not included in these episode totals.",
    }


def _count_reset_audits(out, errors):
    rows = []
    for path in sorted(out.glob("validation/*/audit.json")):
        relative_path = path.relative_to(out)
        if _is_cache_path(relative_path):
            continue
        relative = relative_path.as_posix()
        row_errors = []
        try:
            data = _read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            data = {}
            row_errors.append(f"could not read audit JSON: {exc}")
        count = data.get("reset_only_probes") if isinstance(data, dict) else None
        if not _int_value(count) or count < 0:
            row_errors.append("reset_only_probes must be a non-negative integer")
            count = 0
        row = {"path": relative, "valid": not row_errors,
               "reset_only_probes": count, "validation_errors": row_errors}
        rows.append(row)
        errors.extend(f"{relative}: {message}" for message in row_errors)
    return rows


def _count_interrupted_tests(out, errors):
    rows=[]
    for path in sorted((out/'interruptions').glob('*.json')):
        row=_read_json(path)
        count=row.get('requested_episodes')
        valid=(_int_value(count) and count>0 and row.get('test_only') is True
            and row.get('learned_from') is False and row.get('completed_count_known') is False
            and row.get('completed_episodes') is None)
        if not valid:
            errors.append(f'{path.relative_to(out)}: invalid interrupted-test provenance')
        rows.append(dict(path=path.relative_to(out).as_posix(),valid=valid,**row))
    return dict(files=rows,request_count=len(rows),
        requested_episodes=sum(r['requested_episodes'] for r in rows if r['valid']),
        included_in_completed_full_test_total=False,
        note='These requests returned no completed count. Requested episodes are shown separately, not invented as completed tests or successes.')


def build_accounting(out=OUT):
    """Build counters from a follow-up root directory without writing files."""
    out = Path(out)
    errors = []
    protocol_path = out / "protocol.json"
    registry_path = out / "registry.json"
    protocol = _read_json(protocol_path) if protocol_path.is_file() else {}
    registry = _read_json(registry_path) if registry_path.is_file() else {}
    if not protocol:
        errors.append(f"missing required {protocol_path.relative_to(out.parent).as_posix()}")
    if not registry:
        errors.append(f"missing required {registry_path.relative_to(out.parent).as_posix()}")

    budgets = _count_budget_ledgers(out, errors)
    tests = _count_full_tests(out, errors)
    reset_audits = _count_reset_audits(out, errors)
    interruptions = _count_interrupted_tests(out, errors)
    settings = registry.get("settings", [])
    if not isinstance(settings, list):
        settings = []
        errors.append("registry settings must be a list")
    new_practice = sum(row["reserved_episode_count"] for row in budgets)
    old_settings = int(protocol.get("original_settings", 0))
    settings_summary=setting_counts(settings,old_settings)
    old_practice = int(protocol.get("old_practice_episodes", 0))
    old_tests = int(protocol.get("old_full_test_episodes", 0))
    old_reset_probes = int(protocol.get("old_reset_only_probes", 0))
    new_reset_probes = sum(row["reset_only_probes"] for row in reset_audits if row["valid"])

    return {
        "status": "valid" if not errors else "invalid",
        "protocol_path": protocol_path.relative_to(out.parent).as_posix(),
        "registry_path": registry_path.relative_to(out.parent).as_posix(),
        "budget_ledgers": {
            "files": budgets,
            "ledger_count": len(budgets),
            "reserved_practice_episodes": new_practice,
            "source_episode_counts": {
                source: sum(row["source_counts"][source] for row in budgets)
                for source in ("fill", "online", "unknown")
            },
            "valid_ledger_count": sum(bool(row["valid"]) for row in budgets),
        },
        "full_tests": tests,
        "interrupted_test_requests": interruptions,
        "reset_only_probe_audits": {
            "files": reset_audits,
            "audit_file_count": len(reset_audits),
            "valid_audit_file_count": sum(bool(row["valid"]) for row in reset_audits),
            "followup_reset_only_probes": new_reset_probes,
        },
        "settings": settings_summary,
        "whole_project_episode_totals": {
            "old_study_practice_episodes_from_protocol": old_practice,
            "followup_reserved_practice_episodes": new_practice,
            "whole_project_practice_episodes": old_practice + new_practice,
            "old_study_full_test_episodes_from_protocol": old_tests,
            "followup_full_test_episodes": tests["full_test_episodes"],
            "whole_project_full_test_episodes": old_tests + tests["full_test_episodes"],
            "old_study_reset_only_probes_from_protocol": old_reset_probes,
            "followup_reset_only_probes": new_reset_probes,
            "whole_project_reset_only_probes": old_reset_probes + new_reset_probes,
        },
        "excluded_cache_test_caveat": _cache_caveat(out),
        "validation_errors": errors,
    }


def write_accounting(out=OUT):
    """Write only accounting.json under the supplied follow-up root."""
    out = Path(out)
    report = build_accounting(out)
    target = out / "accounting.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    from takehome.followup_scoreboard import _atomic_write
    _atomic_write(target,json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def main():
    report = write_accounting(OUT)
    print(json.dumps({
        "path": (OUT / "accounting.json").as_posix(),
        "status": report["status"],
        "settings": report["settings"]["whole_project_settings"],
        "practice_episodes": report["whole_project_episode_totals"]["whole_project_practice_episodes"],
        "full_test_episodes": report["whole_project_episode_totals"]["whole_project_full_test_episodes"],
        "validation_error_count": len(report["validation_errors"]),
    }, sort_keys=True), flush=True)
    return 0 if report["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
