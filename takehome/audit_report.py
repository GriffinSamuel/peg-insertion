#!/usr/bin/env python3
"""Mechanically audit the generated take-home report and its provenance."""

import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "takehome"
REPORT = ROOT / "TAKEHOME_REPORT.md"
PROVENANCE = RESULTS / "report_provenance.json"
FIGURE_INVENTORY = RESULTS / "figure_rebuild_inventory.json"
CONFIRMATION = RESULTS / "goal7" / "confirmation" / "confirmation_summary.json"
DECISIONS = ROOT / "TAKEHOME_DECISIONS.md"
sys.path.insert(0, str(ROOT))

import takehome.report_tables as tables  # noqa: E402


def _failure(failures, check, detail):
    failures.append({"check": check, "detail": str(detail)})


def _load(path, failures, check):
    if not path.is_file():
        _failure(failures, check, f"missing file: {path.relative_to(ROOT)}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _failure(failures, check,
                 f"cannot parse {path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")
        return None


def _local_target(target):
    target = target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1:target.index(">")]
    else:
        target = target.split()[0] if target else ""
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or target.startswith("#"):
        return None
    path = unquote(parsed.path)
    if not path:
        return None
    candidate = (ROOT / path).resolve()
    return candidate


def _markdown_links(report):
    # Covers inline links and images used by the generated report.
    pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    return [(match.group(1), match.group(0).startswith("!"))
            for match in pattern.finditer(report)]


def _markdown_row(row):
    return "| " + " | ".join(tables.md_cell(cell) for cell in row) + " |"


def _table_rows(failures):
    data = {}
    for key in tables.SOURCES:
        try:
            data[key] = tables.read_source(key)
        except Exception as exc:
            _failure(failures, "report_table_sources",
                     f"{key}: {type(exc).__name__}: {exc}")
    if len(data) != len(tables.SOURCES):
        return {}
    output = {
        "goal6": tables.goal6_table(data["goal6"]),
        "goal7": tables.goal7_table(data["goal7"]),
        "confirmation": tables.confirmation_table(data["confirmation"]),
        "transfer": tables.transfer_table(data["transfer"]),
        "camera": tables.camera_table(data["camera"]),
        "calibration": tables.calibration_table(data["calibration"]),
    }
    _, output["failures"] = tables.failures_table(data["failures"])
    return output


def _check_table_rows(report, failures):
    rows_by_section = _table_rows(failures)
    if not rows_by_section:
        return {"sections": 0, "rows_expected": 0, "rows_found": 0}
    report_lines = set(report.splitlines())
    expected = found = 0
    section_results = {}
    for section, rows in rows_by_section.items():
        missing = []
        for row in rows:
            rendered = _markdown_row(row)
            expected += 1
            if rendered in report_lines:
                found += 1
            else:
                missing.append(rendered)
        section_results[section] = {"expected_rows": len(rows),
                                    "found_rows": len(rows) - len(missing),
                                    "missing_rows": missing}
        if missing:
            _failure(failures, f"report_table_rows:{section}",
                     f"{len(missing)} generated row(s) absent from report")
    return {"sections": len(rows_by_section), "rows_expected": expected,
            "rows_found": found, "section_results": section_results}


def _check_confirmation(report, confirmation, failures):
    if not confirmation:
        return {"checked": False}
    methods = confirmation.get("methods", [])
    selected = [method for method in methods if method.get("final_method") is True]
    if len(selected) != 1:
        _failure(failures, "confirmation_final_method",
                 f"expected one final_method row, found {len(selected)}")
        return {"checked": False}
    method = selected[0]
    expected_headline = f"**{100 * method['mean']:.2f}%**"
    if expected_headline not in report:
        _failure(failures, "confirmation_headline", f"missing final headline {expected_headline}")
    expected_spread = f"{100 * method['std']:.2f} percentage points"
    if expected_spread not in report:
        _failure(failures, "confirmation_spread", f"missing final sample spread {expected_spread}")
    seed_results = {row["seed"]: row for row in method.get("seed_results", [])}
    expected_seeds = {100, 101, 102, 103, 104}
    if set(seed_results) != expected_seeds:
        _failure(failures, "confirmation_seed_set",
                 f"expected seeds 100-104, found {sorted(seed_results)}")
    seed_cells = [tables.result_cell(seed_results[s]["successes"],
                                     seed_results[s]["test_episodes"],
                                     seed_results[s]["success_rate"])
                  for s in sorted(expected_seeds) if s in seed_results]
    missing_cells = [cell for cell in seed_cells if cell not in report]
    if missing_cells:
        _failure(failures, "confirmation_seed_values",
                 f"missing final-method confirmation cells: {missing_cells}")
    total_settings = confirmation.get("total_settings_tried")
    method_settings = method.get("settings_tried_for_selection")
    if not isinstance(total_settings, int) or method_settings != total_settings:
        _failure(failures, "confirmation_settings_counts",
                 f"total={total_settings}, final-method={method_settings}")
    elif f"{total_settings}-setting search" not in report:
        _failure(failures, "confirmation_headline_settings",
                 f"missing {total_settings}-setting search wording")
    return {"checked": True, "final_method": method.get("name"),
            "mean": method.get("mean"), "std": method.get("std"),
            "seeds": sorted(seed_results), "total_settings_tried": total_settings,
            "final_method_settings_tried": method_settings}


def _check_provenance(report_bytes, provenance, failures):
    if not provenance:
        return {"checked": False, "sources_checked": 0}
    sources = provenance.get("result_sources")
    if not isinstance(sources, dict):
        _failure(failures, "provenance_source_hashes", "result_sources must be a path-to-SHA256 object")
        sources = {}
    checked = 0
    for relative, expected in sources.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            _failure(failures, "provenance_source_hashes", f"source path escapes project: {relative}")
            continue
        if not path.is_file():
            _failure(failures, "provenance_source_hashes", f"missing source file: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        checked += 1
        if actual != expected:
            _failure(failures, "provenance_source_hashes",
                     f"SHA256 mismatch for {relative}: expected {expected}, got {actual}")
    actual_report_hash = hashlib.sha256(report_bytes).hexdigest()
    expected_report_hash = provenance.get("report_sha256")
    if actual_report_hash != expected_report_hash:
        _failure(failures, "provenance_report_hash",
                 f"expected {expected_report_hash}, got {actual_report_hash}")
    return {"checked": True, "sources_listed": len(sources),
            "sources_checked": checked, "report_sha256": actual_report_hash,
            "expected_report_sha256": expected_report_hash}


def audit():
    failures = []
    checks = {}
    if not REPORT.is_file():
        _failure(failures, "report_exists", f"missing file: {REPORT.relative_to(ROOT)}")
        report_bytes = b""
        report = ""
    else:
        report_bytes = REPORT.read_bytes()
        report = report_bytes.decode("utf-8")
    checks["report_exists"] = REPORT.is_file()

    checks["no_em_dash"] = "\u2014" not in report
    if not checks["no_em_dash"]:
        _failure(failures, "no_em_dash", "report contains an em dash")

    links = _markdown_links(report)
    missing_links = []
    embedded_pngs = []
    for target, is_image in links:
        candidate = _local_target(target)
        if candidate is None:
            continue
        if is_image and candidate.suffix.lower() == ".png":
            embedded_pngs.append((target, candidate))
        if not candidate.exists():
            missing_links.append(target)
    checks["local_markdown_targets"] = {"count": sum(_local_target(t) is not None for t, _ in links),
                                        "missing": missing_links}
    if missing_links:
        _failure(failures, "local_markdown_targets", f"missing targets: {missing_links}")

    checks["report_table_rows"] = _check_table_rows(report, failures)
    confirmation = _load(CONFIRMATION, failures, "confirmation_summary")
    checks["confirmation_consistency"] = _check_confirmation(report, confirmation, failures)

    decision_lines = [line for line in DECISIONS.read_text(encoding="utf-8").splitlines()
                      if line.lstrip().startswith("- ")] if DECISIONS.is_file() else []
    report_decisions = report.splitlines()
    transformed = [line.replace("\u2014", "; ") for line in decision_lines]
    missing_decisions = [line for line in transformed if line not in report_decisions]
    checks["decision_bullets"] = {"current_count": len(decision_lines),
                                  "plan_gap_count": sum("PLAN GAP" in line for line in decision_lines),
                                  "missing_count": len(missing_decisions),
                                  "missing": missing_decisions}
    if missing_decisions:
        _failure(failures, "decision_bullets", f"{len(missing_decisions)} current decision bullet(s) absent")

    provenance = _load(PROVENANCE, failures, "report_provenance")
    if provenance:
        if provenance.get("decision_bullets") != len(decision_lines):
            _failure(failures, "provenance_decision_count",
                     f"expected {len(decision_lines)}, provenance says {provenance.get('decision_bullets')}")
        gap_count = sum("PLAN GAP" in line for line in decision_lines)
        if provenance.get("plan_gap_entries") != gap_count:
            _failure(failures, "provenance_plan_gap_count",
                     f"expected {gap_count}, provenance says {provenance.get('plan_gap_entries')}")
        checks["provenance"] = _check_provenance(report_bytes, provenance, failures)
    else:
        checks["provenance"] = {"checked": False}

    figure_inventory = _load(FIGURE_INVENTORY, failures, "figure_rebuild_inventory")
    if figure_inventory:
        artifact_by_output = {}
        for artifact in figure_inventory.get("artifacts", []):
            command = artifact.get("command")
            script = artifact.get("script")
            for output in artifact.get("outputs", []):
                artifact_by_output[output] = {"command": command, "script": script}
        missing_commands = []
        missing_scripts = []
        embedded_results = []
        for target, candidate in embedded_pngs:
            normalized = target[2:] if target.startswith("./") else target
            info = artifact_by_output.get(normalized)
            entry = {"target": target, "exists": candidate.is_file(),
                     "command": info.get("command") if info else None,
                     "script": info.get("script") if info else None}
            embedded_results.append(entry)
            if not info or not info.get("command"):
                missing_commands.append(target)
            if not info or not info.get("script") or not (ROOT / info["script"]).is_file():
                missing_scripts.append(target)
        if missing_commands:
            _failure(failures, "embedded_png_rebuild_commands",
                     f"no inventory command for: {missing_commands}")
        if missing_scripts:
            _failure(failures, "embedded_png_source_scripts",
                     f"missing source script for: {missing_scripts}")
        checks["embedded_png_rebuilds"] = {"count": len(embedded_pngs),
                                           "items": embedded_results,
                                           "missing_commands": missing_commands,
                                           "missing_scripts": missing_scripts}
    else:
        checks["embedded_png_rebuilds"] = {"checked": False}

    return {"passed": not failures, "failure_count": len(failures),
            "failures": failures, "checks": checks,
            "scope_note": "Mechanical checks do not validate prose interpretation or causal claims."}


def main():
    result = audit()
    output = RESULTS / "report_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"audit": str(output.relative_to(ROOT)),
                      "passed": result["passed"],
                      "failure_count": result["failure_count"]}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
