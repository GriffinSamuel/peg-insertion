#!/usr/bin/env python3
"""Build factual Markdown tables from completed Goal 6–12 JSON summaries."""

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "takehome" / "report_tables.md"
SOURCES = {
    "goal6": "goal6/goal6_aggregate.json",
    "goal7": "goal7_summary.json",
    "confirmation": "goal7/confirmation/confirmation_summary.json",
    "transfer": "goal10/transfer_summary.json",
    "camera": "goal11/camera_summary.json",
    "calibration": "goal8/summary.json",
    "failures": "goal9/failures.json",
}


def read_source(key):
    path = ROOT / "results" / "takehome" / SOURCES[key]
    if not path.is_file():
        raise FileNotFoundError(f"Required report table source is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{where} must be a finite number")
    return float(value)


def count(value, where):
    if type(value) is not int or value < 0:
        raise ValueError(f"{where} must be a non-negative integer")
    return value


def pct(value):
    return f"{100.0 * finite(value, 'rate'):.1f}%"


def result_cell(successes, episodes, rate=None):
    successes = count(successes, "successes")
    episodes = count(episodes, "test episodes")
    if episodes < 1 or successes > episodes:
        raise ValueError("success count must fit a positive test count")
    rate = successes / episodes if rate is None else finite(rate, "success rate")
    if not math.isclose(rate, successes / episodes, rel_tol=0, abs_tol=1e-10):
        raise ValueError("success rate does not match raw successes/test count")
    return f"{successes}/{episodes} ({pct(rate)})"


def mean_sd(mean, sd):
    return f"{pct(mean)} ± {pct(sd)}"


def ref_label(name, label, flag=None):
    if name == "supervised_cheating_reference":
        return "Supervised reference (cheats: true labels)"
    if name == "oracle_cheating":
        return "Oracle (cheats: true error)"
    return str(label)


def md_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(md_cell(value) for value in row) + " |"
                 for row in rows)
    return "\n".join(lines)


def source_line(key):
    return f"Source JSON: [{SOURCES[key]}]({SOURCES[key]})\n"


def goal6_table(data):
    if data.get("test_episodes_per_result") != 500:
        raise ValueError("Goal 6 aggregate must report 500 tests per result")
    rows = []
    for method in data["methods"]:
        n = count(method["seed_count"], "Goal 6 seed_count")
        if n != 5:
            raise ValueError("Goal 6 methods must include all five seeds")
        stat = method["final_success_rate"]
        if set(stat["per_seed"]) != {str(i) for i in range(5)}:
            raise ValueError(f"Goal 6 method {method['name']} lacks seed results 0–4")
        label = ref_label(method["name"], method["label"], method.get("cheats_on_information"))
        mean = finite(stat["mean"], "Goal 6 mean")
        sd = finite(stat["sample_std_ddof_1"], "Goal 6 sample SD")
        seed_cells = []
        for seed in range(5):
            rate = finite(stat["per_seed"][str(seed)], "Goal 6 per-seed rate")
            successes_f = rate * 500
            successes = round(successes_f)
            if not math.isclose(successes_f, successes, rel_tol=0, abs_tol=1e-8):
                raise ValueError("Goal 6 rate cannot be converted to an exact raw count")
            seed_cells.append(result_cell(successes, 500, rate))
        rows.append([label, *seed_cells, "5", mean_sd(mean, sd)])
    return rows


def goal7_table(data):
    rows = []
    for setting in data["settings"]:
        name, label = setting["name"], setting["label"]
        ref = ref_label(name, label, setting.get("cheats_on_information"))
        mean = finite(setting["mean"], "Goal 7 setting mean")
        sd = finite(setting["std"], "Goal 7 setting SD")
        results = setting["results"]
        if not results:
            raise ValueError(f"Goal 7 setting {name} has no seed rows")
        by_seed = {}
        force_values = []
        for result in results:
            seed = count(result["seed"], "Goal 7 practice seed")
            if seed not in range(5) or seed in by_seed:
                raise ValueError(f"Goal 7 setting {name} has invalid or duplicate seed {seed}")
            episodes = count(result["episodes"], "Goal 7 test episodes")
            by_seed[seed] = result_cell(result["successes"], episodes, result["success_rate"])
            force_values.append(f"s{seed}:{finite(result['peak_force_mean'], 'peak force'):.4g} N")
        seed_cells = [by_seed.get(seed, "not run") for seed in range(5)]
        rows.append([ref, *seed_cells, str(len(results)), mean_sd(mean, sd),
                     str(setting.get("difference_status", "not supplied")),
                     "; ".join(force_values)])
    return rows


def confirmation_table(data):
    rows = []
    for method in data["methods"]:
        ref = ref_label(method["name"], method["label"], method.get("cheats_on_information"))
        if method["final_method"]:
            ref += " [final method]"
        ref += f"; {count(method['settings_tried_for_selection'], 'settings tried')} settings tried"
        by_seed = {run["seed"]: run for run in method["seed_results"]}
        if set(by_seed) != {100, 101, 102, 103, 104}:
            raise ValueError(f"Confirmation method {method['name']} lacks seeds 100–104")
        seed_cells = [result_cell(by_seed[seed]["successes"], by_seed[seed]["test_episodes"],
                                  by_seed[seed]["success_rate"])
                      for seed in range(100, 105)]
        rows.append([ref, *seed_cells, str(len(by_seed)), mean_sd(method["mean"], method["std"])])
    return rows


def transfer_table(data):
    rows = []
    for method in data["policies"]:
        name = method["policy"]
        label = ref_label(name, name, method.get("cheats_on_information"))
        by_seed = {run["seed"]: run for run in method["seeds"]}
        if set(by_seed) != {0, 1, 2}:
            raise ValueError(f"Transfer policy {name} lacks seeds 0–2")
        seed_cells = [result_cell(by_seed[seed]["successes"], by_seed[seed]["test_episodes"],
                                  by_seed[seed]["success_rate"])
                      for seed in range(3)]
        rows.append([label, *seed_cells, str(len(by_seed)),
                     mean_sd(method["mean_success_rate"], method["sample_std_success_rate"]),
                     pct(method["floor_success_rate"]), pct(method["ceiling_success_rate"])])
    return rows


def camera_table(data):
    rows = []
    if data.get("all_conditions_included") is not True:
        raise ValueError("Goal 11 camera summary must include all conditions")
    for condition in data["conditions"]:
        problem = condition["problem"]
        refs = []
        for key, label, flag in (("floor", "Policy floor", False),
                                 ("ceiling", "No-bias ceiling", False),
                                 ("oracle_cheating", "Oracle reference", True)):
            value = condition[key]
            refs.append(f"{ref_label(key, label, flag)}: " +
                        result_cell(value["successes"], value["episodes"], value["success_rate"]))
        transfer = {run["seed"]: run for run in condition["test_only_by_seed"]}
        retrained = {run["seed"]: run for run in condition["retrained_by_seed"]}
        if set(transfer) != {0, 1, 2} or set(retrained) != {0, 1, 2}:
            raise ValueError(f"Camera condition {problem} lacks transfer or retrain seeds 0–2")
        transfer_cells = [result_cell(transfer[s]["successes"], transfer[s]["episodes"],
                                      transfer[s]["success_rate"]) for s in range(3)]
        retrain_cells = [result_cell(retrained[s]["successes"], retrained[s]["episodes"],
                                     retrained[s]["success_rate"]) for s in range(3)]
        rows.append([problem, *refs, *transfer_cells,
                     mean_sd(condition["test_only"]["mean"], condition["test_only"]["sample_sd_ddof1"]),
                     *retrain_cells,
                     mean_sd(condition["retrained"]["mean"], condition["retrained"]["sample_sd_ddof1"])])
    return rows


def calibration_table(data):
    require_seeds = {100, 101, 102, 103, 104}
    rows = []
    final = [r for r in data["per_seed"] if r.get("practice_episodes") == 200]
    if len(final) != 5 or {r.get("seed") for r in final} != require_seeds:
        raise ValueError("Goal 8 summary must contain final practice-200 rows for seeds 100–104")
    pooled = [r for r in data["pooled"] if r.get("practice_episodes") == 200]
    if len(pooled) != 1 or pooled[0].get("seed_count") != 5:
        raise ValueError("Goal 8 summary must contain one pooled practice-200 row for five seeds")
    for record in sorted(final, key=lambda r: r["seed"]):
        episodes = count(record["test_episodes"], "Goal 8 test episodes")
        failed = count(record["failed_episodes"], "Goal 8 failed episodes")
        successes = episodes - failed
        rows.append([f"Seed {record['seed']}", result_cell(successes, episodes),
                     count(record["rows"], "valid trace steps"),
                     pct(record["mean_predicted"]), pct(record["observed"]),
                     f"{finite(record['expected_calibration_error'], 'ECE'):.5g}",
                     f"{finite(record['brier_score'], 'Brier'):.5g}",
                     f"{finite(record['signed_bias'], 'signed bias'):.5g}"])
    pool = pooled[0]
    pooled_episodes = sum(count(r["test_episodes"], "Goal 8 test episodes") for r in final)
    pooled_successes = sum(count(r["test_episodes"], "Goal 8 test episodes") -
                           count(r["failed_episodes"], "Goal 8 failed episodes") for r in final)
    rows.append(["Pooled (5 seeds)", result_cell(pooled_successes, pooled_episodes),
                 count(pool["rows"], "pooled valid trace steps"), pct(pool["mean_predicted"]),
                 pct(pool["observed"]), f"{finite(pool['expected_calibration_error'], 'pooled ECE'):.5g}",
                 f"{finite(pool['brier_score'], 'pooled Brier'):.5g}",
                 f"{finite(pool['signed_bias'], 'pooled signed bias'):.5g}"])
    return rows


def failures_table(data):
    names = list(data["order"])
    rows = []
    before = data["before"]
    rows.append(["Before", "unlearned repeat", result_cell(before["successes"], before["episodes"]),
                 *(before["counts"][name] for name in names)])
    records = data["after_per_seed"]
    if {r.get("practice_seed") for r in records} != {100, 101, 102, 103, 104}:
        raise ValueError("Goal 9 failure summary must contain practice seeds 100–104")
    for record in sorted(records, key=lambda r: r["practice_seed"]):
        rows.append(["After", f"seed {record['practice_seed']}",
                     result_cell(record["successes"], record["episodes"]),
                     *(record["counts"][name] for name in names)])
    pooled = data["after_pooled"]
    rows.append(["After pooled", "seeds 100–104", result_cell(pooled["successes"], pooled["episodes"]),
                 *(pooled["counts"][name] for name in names)])
    return names, rows


def main():
    # Read all required inputs before writing so missing sources never produce
    # a partial table document.
    data = {key: read_source(key) for key in SOURCES}
    sections = ["# Report tables", ""]
    sections.extend(["## Goal 6 final success rates", "", source_line("goal6"),
                     table(["Method / reference label", *[f"Seed {s}" for s in range(5)],
                            "Seed count", "Mean ± sample SD"], goal6_table(data["goal6"])), ""])
    sections.extend(["## Goal 7 settings", "", source_line("goal7"),
                     table(["Setting / reference label", *[f"Seed {s}" for s in range(5)],
                            "Seed count", "Mean ± sample SD", "Difference status",
                            "Per-seed peak force means"], goal7_table(data["goal7"])), ""])
    sections.extend(["## Goal 7 confirmation", "", source_line("confirmation"),
                     table(["Method / reference label", *[f"Seed {s}" for s in range(100, 105)],
                            "Seed count", "Mean ± sample SD"], confirmation_table(data["confirmation"])), ""])
    sections.extend(["## Goal 10 transfer", "", source_line("transfer"),
                     table(["Policy / reference label", *[f"Seed {s}" for s in range(3)],
                            "Seed count", "Mean ± sample SD", "Floor", "Ceiling"], transfer_table(data["transfer"])), ""])
    sections.extend(["## Goal 11 camera conditions", "", source_line("camera"),
                     table(["Condition", "Floor", "Ceiling", "Oracle (cheats: true error)",
                            "Transfer seed 0", "Transfer seed 1", "Transfer seed 2", "Transfer mean ± sample SD",
                            "Retrain seed 0", "Retrain seed 1", "Retrain seed 2",
                            "Retrain mean ± sample SD"], camera_table(data["camera"])), ""])
    sections.extend(["## Goal 8 final calibration", "", source_line("calibration"),
                     "Calibration statistics below are valid-step-weighted. The episode outcome column is episode-level.", "",
                     table(["Practice seed / group", "Episode successes / tests (rate)", "Valid trace steps",
                            "Mean predicted episode-success probability", "Observed valid-step success frequency",
                            "ECE", "Brier score", "Signed bias"], calibration_table(data["calibration"])), ""])
    failure_names, failure_rows = failures_table(data["failures"])
    sections.extend(["## Goal 9 failure counts", "", source_line("failures"),
                     table(["Group", "Practice seed", "Successes / tests (rate)", *failure_names], failure_rows), ""])
    OUT.write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
