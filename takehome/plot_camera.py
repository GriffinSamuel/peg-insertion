"""Validate and summarize Goal 11 camera-condition results.

Expected input is results/takehome/goal11/<problem>/result.json as written by
run_camera.py, with trained/seed_<n>/result.json files alongside it. This
script only reads experiment results and writes a CSV, JSON summary, and PNG.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RESULTS_TAKEHOME = ROOT / "results" / "takehome"
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = str(RESULTS_TAKEHOME / "mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROBLEMS = (
    "bias_2x", "bias_4x", "one_sided", "drift", "jitter", "rotation",
    "delay", "height",
)
SEEDS = (0, 1, 2)
TEST_SEED = 123
TEST_EPISODES = 500
PRACTICE_EPISODES = 200


def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing required result file: {path}")
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc


def validate_measurement(data, where, *, expected_seed=TEST_SEED):
    if not isinstance(data, dict):
        raise ValueError(f"{where}: measurement must be an object")
    if data.get("episodes") != TEST_EPISODES:
        raise ValueError(f"{where}: expected exactly {TEST_EPISODES} test episodes")
    if data.get("test_seed") != expected_seed:
        raise ValueError(f"{where}: expected test seed {expected_seed}")
    if data.get("test_only") is not True or data.get("learned_from") is not False:
        raise ValueError(f"{where}: result must be a test-only measurement not learned from")
    successes = data.get("successes")
    rate = data.get("success_rate")
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise ValueError(f"{where}: missing integer raw success count")
    if not 0 <= successes <= TEST_EPISODES:
        raise ValueError(f"{where}: success count outside [0, {TEST_EPISODES}]")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ValueError(f"{where}: missing numeric success rate")
    if not math.isfinite(float(rate)) or not 0 <= float(rate) <= 1:
        raise ValueError(f"{where}: non-finite or out-of-range success rate")
    if not math.isclose(float(rate), successes / TEST_EPISODES, abs_tol=1e-12):
        raise ValueError(f"{where}: rate does not agree with successes/episodes")
    return {"successes": successes, "episodes": TEST_EPISODES,
            "success_rate": float(rate), "test_seed": expected_seed}


def sample_summary(measurements, where):
    if len(measurements) != len(SEEDS):
        raise ValueError(f"{where}: expected one result for each seed {SEEDS}")
    rates = np.asarray([m["success_rate"] for m in measurements], dtype=float)
    if not np.all(np.isfinite(rates)):
        raise ValueError(f"{where}: non-finite rate")
    return {
        "seeds": [m["seed"] for m in measurements],
        "raw_successes": [m["successes"] for m in measurements],
        "rates": rates.tolist(),
        "mean": float(np.mean(rates)),
        "sample_sd_ddof1": float(np.std(rates, ddof=1)),
    }


def read_condition(problem, condition_dir, goal1_ceiling):
    result_path = condition_dir / "result.json"
    result = load_json(result_path)
    declaration = result.get("declaration", {})
    if declaration.get("problem") != problem:
        raise ValueError(f"{result_path}: declaration problem does not match {problem}")
    if declaration.get("test_seed") != TEST_SEED:
        raise ValueError(f"{result_path}: declaration must use test seed {TEST_SEED}")

    floor = validate_measurement(result.get("floor"), f"{problem} floor")
    oracle = validate_measurement(result.get("oracle"), f"{problem} oracle")
    ceiling = validate_measurement(result.get("ceiling"), f"{problem} Goal 1 ceiling")
    if ceiling["successes"] != goal1_ceiling["successes"]:
        raise ValueError(f"{problem}: ceiling differs from Goal 1 no-camera-bias reference")
    if result.get("practice_episodes_added") != 600:
        raise ValueError(f"{problem}: result must account for three 200-episode retrains")

    transfer_entries = result.get("transfer")
    if not isinstance(transfer_entries, list) or len(transfer_entries) != 3:
        raise ValueError(f"{problem}: expected exactly three test-only transfer results")
    transfer_by_seed = {}
    for entry in transfer_entries:
        seed = entry.get("seed")
        if seed not in SEEDS or seed in transfer_by_seed:
            raise ValueError(f"{problem}: invalid or duplicate transfer seed {seed}")
        if entry.get("practice_episodes_added") != 0:
            raise ValueError(f"{problem} transfer seed {seed}: must add zero practice episodes")
        transfer_by_seed[seed] = validate_measurement(entry, f"{problem} transfer seed {seed}")
        transfer_by_seed[seed]["seed"] = seed
    if set(transfer_by_seed) != set(SEEDS):
        raise ValueError(f"{problem}: transfer results must include exactly seeds {SEEDS}")
    transfer = [transfer_by_seed[s] for s in SEEDS]

    trained_entries = result.get("retrained")
    if not isinstance(trained_entries, list) or len(trained_entries) != 3:
        raise ValueError(f"{problem}: expected exactly three retrained results")
    trained_by_seed = {}
    for entry in trained_entries:
        seed = entry.get("seed")
        if seed not in SEEDS or seed in trained_by_seed:
            raise ValueError(f"{problem}: invalid or duplicate retrain seed {seed}")
        run_path = condition_dir / "trained" / f"seed_{seed}" / "result.json"
        underlying = load_json(run_path)
        if underlying.get("seed") != seed:
            raise ValueError(f"{run_path}: underlying result seed mismatch")
        if underlying.get("practice_episodes") != PRACTICE_EPISODES:
            raise ValueError(f"{run_path}: expected exactly 200 learned-from episodes")
        if underlying.get("final", {}).get("practice_episodes") != PRACTICE_EPISODES:
            raise ValueError(f"{run_path}: final result does not record 200 practice episodes")
        if entry.get("result_path"):
            expected_suffix = f"goal11/{problem}/trained/seed_{seed}/result.json"
            if not str(entry["result_path"]).replace("\\", "/").endswith(expected_suffix):
                raise ValueError(f"{result_path}: retrained result path mismatch for seed {seed}")
        measured = validate_measurement(underlying.get("final"), f"{problem} retrain seed {seed}")
        measured["seed"] = seed
        measured["practice_episodes"] = PRACTICE_EPISODES
        trained_by_seed[seed] = measured
    if set(trained_by_seed) != set(SEEDS):
        raise ValueError(f"{problem}: retrained results must include exactly seeds {SEEDS}")
    retrained = [trained_by_seed[s] for s in SEEDS]

    return {
        "problem": problem,
        "formula": declaration.get("formula"),
        "floor": floor,
        "test_only": sample_summary(transfer, f"{problem} test-only"),
        "test_only_by_seed": transfer,
        "retrained": sample_summary(retrained, f"{problem} retrained"),
        "retrained_by_seed": retrained,
        "ceiling": ceiling,
        "oracle_cheating": oracle,
    }


def make_chart(rows, output):
    labels = [row["problem"].replace("_", " ") for row in rows]
    centers = np.arange(len(rows), dtype=float)
    width = 0.15
    columns = [
        ("floor", "Floor", "success_rate", "sd0"),
        ("test_only", "Test-only", "mean", "sample_sd_ddof1"),
        ("retrained", "Retrained", "mean", "sample_sd_ddof1"),
        ("ceiling", "Goal 1 ceiling", "success_rate", "sd0"),
        ("oracle_cheating", "Oracle (cheats: true error)", "success_rate", "sd0"),
    ]
    fig, ax = plt.subplots(figsize=(16, 8))
    bar_handles = []
    for index, (key, label, value_key, sd_key) in enumerate(columns):
        values = [row[key][value_key] for row in rows]
        errors = [row[key].get(sd_key, 0.0) for row in rows]
        bars = ax.bar(centers + (index - 2) * width, values, width, yerr=errors,
                      capsize=2, label=label)
        bar_handles.append(bars[0])
    seed_markers = ("o", "s", "^")
    seed_colors = ("#202020", "#555555", "#888888")
    for index, key in ((1, "test_only"), (2, "retrained")):
        xbase = centers + (index - 2) * width
        for seed_index, (marker, color) in enumerate(zip(seed_markers, seed_colors)):
            xoffset = (seed_index - 1) * width * 0.22
            values = [row[f"{key}_by_seed"][seed_index]["success_rate"] for row in rows]
            ax.scatter(xbase + xoffset, values, marker=marker, s=22, color=color,
                       edgecolors="white", linewidths=0.45, zorder=4,
                       label=f"Practice seed {seed_index}" if index == 1 else None)
    ax.set_xticks(centers, labels, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success rate (500 test episodes, seed 123)")
    ax.set_title("Goal 11 camera conditions: references, transfer, and retraining")
    ax.grid(axis="y", alpha=0.25)
    from matplotlib.lines import Line2D
    seed_handles = [
        Line2D([], [], marker=marker, linestyle="None", markersize=6,
               markerfacecolor=color, markeredgecolor="white", label=f"Practice seed {seed}")
        for seed, (marker, color) in enumerate(zip(seed_markers, seed_colors))
    ]
    ax.legend(handles=bar_handles + seed_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.24), ncols=4, frameon=False)
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/takehome/goal11"),
                        help="directory containing one result.json per camera condition")
    parser.add_argument("--baseline", type=Path, default=Path("results/takehome/baseline.json"),
                        help="Goal 1 baseline result containing the no-camera-bias ceiling")
    parser.add_argument("--output-dir", type=Path, default=Path("results/takehome/goal11"),
                        help="directory for camera_summary.csv/json and camera_comparison.png")
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    ceiling = validate_measurement(
        baseline.get("conditions", {}).get("no_camera_bias"), "Goal 1 ceiling")
    if baseline.get("test_seed") != TEST_SEED:
        raise ValueError("Goal 1 baseline must use test seed 123")

    found = {p.name for p in args.input.iterdir()
             if p.is_dir() and (p / "result.json").is_file()}
    expected = set(PROBLEMS)
    if found != expected:
        raise ValueError(f"Expected exactly the eight conditions {PROBLEMS}; found {sorted(found)}")
    rows = [read_condition(problem, args.input / problem, ceiling) for problem in PROBLEMS]
    if len(rows) != 8:
        raise ValueError(f"Expected eight output rows, got {len(rows)}")

    output_resolved = args.output_dir.resolve()
    try:
        output_resolved.relative_to(RESULTS_TAKEHOME.resolve())
    except ValueError as exc:
        raise ValueError("--output-dir must resolve inside results/takehome") from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "camera_summary.json"
    csv_path = args.output_dir / "camera_summary.csv"
    chart_path = args.output_dir / "camera_comparison.png"
    summary = {
        "conditions": rows,
        "condition_count": len(rows),
        "test_seed": TEST_SEED,
        "test_episodes_per_measurement": TEST_EPISODES,
        "practice_episodes_per_retrain_seed": PRACTICE_EPISODES,
        "retrain_seeds": list(SEEDS),
        "transfer_seeds": list(SEEDS),
        "all_conditions_included": True,
        "comparison_columns": ["floor", "test_only", "retrained", "ceiling", "oracle_cheating"],
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    columns = ["problem", "formula", "floor_successes", "floor_rate",
               "test_only_seed_0_successes", "test_only_seed_0_rate",
               "test_only_seed_1_successes", "test_only_seed_1_rate",
               "test_only_seed_2_successes", "test_only_seed_2_rate",
               "test_only_mean", "test_only_sample_sd_ddof1",
               "retrained_seed_0_successes", "retrained_seed_0_rate",
               "retrained_seed_1_successes", "retrained_seed_1_rate",
               "retrained_seed_2_successes", "retrained_seed_2_rate",
               "retrained_mean", "retrained_sample_sd_ddof1",
               "ceiling_successes", "ceiling_rate", "oracle_successes", "oracle_rate"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {
                "problem": row["problem"], "formula": row["formula"],
                "floor_successes": row["floor"]["successes"],
                "floor_rate": row["floor"]["success_rate"],
                "test_only_mean": row["test_only"]["mean"],
                "test_only_sample_sd_ddof1": row["test_only"]["sample_sd_ddof1"],
                "retrained_mean": row["retrained"]["mean"],
                "retrained_sample_sd_ddof1": row["retrained"]["sample_sd_ddof1"],
                "ceiling_successes": row["ceiling"]["successes"],
                "ceiling_rate": row["ceiling"]["success_rate"],
                "oracle_successes": row["oracle_cheating"]["successes"],
                "oracle_rate": row["oracle_cheating"]["success_rate"],
            }
            for seed, measurement in enumerate(row["test_only_by_seed"]):
                flat[f"test_only_seed_{seed}_successes"] = measurement["successes"]
                flat[f"test_only_seed_{seed}_rate"] = measurement["success_rate"]
            for seed, measurement in enumerate(row["retrained_by_seed"]):
                flat[f"retrained_seed_{seed}_successes"] = measurement["successes"]
                flat[f"retrained_seed_{seed}_rate"] = measurement["success_rate"]
            writer.writerow(flat)
    make_chart(rows, chart_path)
    print(json.dumps({"conditions": 8, "csv": str(csv_path), "json": str(json_path),
                      "chart": str(chart_path)}, indent=2))


if __name__ == "__main__":
    main()
