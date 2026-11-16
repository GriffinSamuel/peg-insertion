#!/usr/bin/env python3
"""Create Goal 6 comparison and learning-curve figures from saved JSON results."""

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "takehome"
DEFAULT_MANIFEST = RESULTS_ROOT / "goal6_manifest.json"
DEFAULT_OUTPUT = RESULTS_ROOT / "goal6"
EXPECTED_SEEDS = tuple(range(5))
EXPECTED_TEST_SEED = 123
EXPECTED_PRACTICE_EPISODES = 200
SUPERVISED_REFERENCE_LABEL = "Supervised reference (cheats: true labels)"


class InputError(Exception):
    """An input file exists but has an invalid format or value."""


def require(condition, message):
    if not condition:
        raise InputError(message)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"Invalid JSON in {path}: {exc}") from exc


def resolve_manifest_path(value, manifest_path):
    path = Path(value)
    if path.is_absolute():
        return path
    # Manifest paths are relative to the manifest directory. Also accept paths
    # rooted at the project when that is the path that actually exists.
    relative_to_manifest = manifest_path.parent / path
    relative_to_project = ROOT / path
    if relative_to_manifest.exists() or not relative_to_project.exists():
        return relative_to_manifest
    return relative_to_project


def await_inputs(manifest_path):
    """Return validated inputs, or a list of missing files while work is running."""
    if not manifest_path.is_file():
        return None, [str(manifest_path)]

    manifest = read_json(manifest_path)
    baseline_value = manifest.get("baseline")
    methods = manifest.get("methods")
    if not baseline_value or not isinstance(methods, list):
        return None, ["manifest requires baseline and methods list"]

    baseline_path = resolve_manifest_path(baseline_value, manifest_path)
    missing = []
    if not baseline_path.is_file():
        missing.append(str(baseline_path))

    if len(methods) != 4:
        return None, missing + [f"manifest has {len(methods)} methods; waiting for four learned methods"]

    normalized_methods = []
    for method in methods:
        if not isinstance(method, dict):
            raise InputError("Each method entry must be an object")
        name = method.get("name")
        label = method.get("label")
        cheats = method.get("cheats_on_information")
        results = method.get("results")
        if not isinstance(name, str) or not name:
            raise InputError("Each method needs a nonempty name")
        if not isinstance(label, str) or not label:
            raise InputError(f"Method {name} needs a nonempty label")
        if not isinstance(cheats, bool):
            raise InputError(f"Method {name} needs boolean cheats_on_information")
        if not isinstance(results, list) or len(results) != len(EXPECTED_SEEDS):
            return None, missing + [f"method {name} is waiting for five result paths"]
        result_paths = [resolve_manifest_path(value, manifest_path) for value in results]
        missing.extend(str(path) for path in result_paths if not path.is_file())
        normalized_methods.append({
            "name": name,
            "label": SUPERVISED_REFERENCE_LABEL if cheats else label,
            "manifest_label": label,
            "cheats_on_information": cheats,
            "result_paths": result_paths,
        })

    if missing:
        return None, missing

    baseline = read_json(baseline_path)
    if baseline.get("test_seed") != EXPECTED_TEST_SEED:
        raise InputError(f"Baseline must use test seed {EXPECTED_TEST_SEED}")
    conditions = baseline.get("conditions", {})
    floor = conditions.get("normal", {})
    ceiling = conditions.get("no_camera_bias", {})
    for name, condition in (("normal", floor), ("no_camera_bias", ceiling)):
        if condition.get("test_seed") != EXPECTED_TEST_SEED:
            raise InputError(f"Baseline condition {name} must use test seed {EXPECTED_TEST_SEED}")
        if condition.get("episodes") != 500:
            raise InputError(f"Baseline condition {name} must contain 500 tests")
        if not isinstance(condition.get("success_rate"), (int, float)):
            raise InputError(f"Baseline condition {name} lacks success_rate")

    rows = []
    by_method = {}
    for method in normalized_methods:
        runs = {}
        for result_path in method["result_paths"]:
            result = read_json(result_path)
            seed = result.get("seed")
            if seed not in EXPECTED_SEEDS:
                raise InputError(f"Result {result_path} has unexpected seed {seed!r}")
            if seed in runs:
                raise InputError(f"Method {method['name']} repeats seed {seed}")
            if result.get("practice_episodes") != EXPECTED_PRACTICE_EPISODES:
                raise InputError(f"Result {result_path} must use 200 practice episodes")
            final = result.get("final")
            if not isinstance(final, dict):
                raise InputError(f"Result {result_path} lacks final summary")
            test_episodes = final.get("test_episodes", final.get("episodes"))
            if test_episodes != 500 or final.get("test_seed") != EXPECTED_TEST_SEED:
                raise InputError(f"Result {result_path} must use 500 test episodes at seed 123")
            if final.get("test_only") is not True or final.get("learned_from") is not False:
                raise InputError(f"Result {result_path} lacks test-only provenance")
            successes = final.get("successes")
            success_rate = final.get("success_rate")
            if not isinstance(successes, int) or not isinstance(success_rate, (int, float)):
                raise InputError(f"Result {result_path} lacks final successes or success_rate")
            if not 0 <= successes <= test_episodes or not 0.0 <= success_rate <= 1.0:
                raise InputError(f"Result {result_path} has out-of-range test success values")

            curve = result.get("curve")
            if not isinstance(curve, list) or not curve:
                raise InputError(f"Result {result_path} lacks JSON learning-curve points")
            x_values = []
            curve_rates = []
            for point in curve:
                x = point.get("practice_episodes")
                rate = point.get("success_rate")
                if not isinstance(x, int) or not 0 <= x <= EXPECTED_PRACTICE_EPISODES:
                    raise InputError(f"Result {result_path} has invalid curve episode count {x!r}")
                if not isinstance(rate, (int, float)) or not 0.0 <= rate <= 1.0:
                    raise InputError(f"Result {result_path} has invalid curve success_rate")
                x_values.append(x)
                curve_rates.append(float(rate))
            if x_values != sorted(set(x_values)):
                raise InputError(f"Result {result_path} curve episode counts are not unique and sorted")
            if curve_rates[-1] != float(success_rate):
                raise InputError(f"Result {result_path} final and last curve success rates differ")

            row = {
                "method_name": method["name"],
                "label": method["label"],
                "cheats_on_information": method["cheats_on_information"],
                "seed": seed,
                "practice_episodes": result["practice_episodes"],
                "test_episodes": test_episodes,
                "test_seed": final["test_seed"],
                "successes": successes,
                "success_rate": float(success_rate),
                "result_path": str(result_path.relative_to(ROOT)),
            }
            rows.append(row)
            runs[seed] = {"row": row, "x": x_values, "curve": curve_rates}

        if set(runs) != set(EXPECTED_SEEDS):
            raise InputError(f"Method {method['name']} must contain practice seeds 0 through 4")
        by_method[method["name"]] = {**method, "runs": runs}

    if len(rows) != 20:
        raise InputError(f"Expected 20 learned results, found {len(rows)}")
    cheating = [m for m in normalized_methods if m["cheats_on_information"]]
    if len(cheating) != 1:
        raise InputError("Expected exactly one supervised reference with cheats_on_information=true")

    for method in normalized_methods:
        curves = [by_method[method["name"]]["runs"][seed] for seed in EXPECTED_SEEDS]
        reference_x = curves[0]["x"]
        if any(item["x"] != reference_x for item in curves[1:]):
            raise InputError(f"Method {method['name']} has mismatched learning-curve episode counts")

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "baseline_path": baseline_path,
        "baseline": baseline,
        "floor": floor,
        "ceiling": ceiling,
        "methods": normalized_methods,
        "by_method": by_method,
        "rows": sorted(rows, key=lambda row: (row["method_name"], row["seed"])),
    }, []


def aggregate_method(method, runs):
    rates = [runs[seed]["row"]["success_rate"] for seed in EXPECTED_SEEDS]
    x_values = runs[EXPECTED_SEEDS[0]]["x"]
    curve_mean = []
    curve_sd = []
    for index in range(len(x_values)):
        rates_at_point = [runs[seed]["curve"][index] for seed in EXPECTED_SEEDS]
        curve_mean.append(statistics.mean(rates_at_point))
        curve_sd.append(statistics.stdev(rates_at_point))
    return {
        "name": method["name"],
        "label": method["label"],
        "cheats_on_information": method["cheats_on_information"],
        "seed_count": len(rates),
        "final_success_rate": {
            "mean": statistics.mean(rates),
            "sample_std_ddof_1": statistics.stdev(rates),
            "min": min(rates),
            "max": max(rates),
            "per_seed": {str(seed): runs[seed]["row"]["success_rate"] for seed in EXPECTED_SEEDS},
        },
        "learning_curve": {
            "practice_episodes": x_values,
            "mean_success_rate": curve_mean,
            "sample_std_ddof_1": curve_sd,
        },
    }


def write_outputs(data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    mpl_config = RESULTS_ROOT / "mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = output_dir / "goal6_per_seed.csv"
    aggregate_path = output_dir / "goal6_aggregate.json"
    comparison_path = output_dir / "comparison.png"
    curves_path = output_dir / "learning_curves.png"

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["method_name", "label", "cheats_on_information", "seed",
                  "practice_episodes", "test_episodes", "test_seed", "successes",
                  "success_rate", "result_path"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data["rows"])

    method_aggregates = []
    for method in data["methods"]:
        method_aggregates.append(aggregate_method(method, data["by_method"][method["name"]]["runs"]))
    aggregate = {
        "test_seed": EXPECTED_TEST_SEED,
        "test_episodes_per_result": 500,
        "baseline": {
            "floor": {
                "label": "Original policy (floor)",
                "success_rate": data["floor"]["success_rate"],
                "successes": data["floor"].get("successes"),
                "test_episodes": data["floor"]["episodes"],
            },
            "ceiling": {
                "label": "No-camera-bias test (ceiling)",
                "success_rate": data["ceiling"]["success_rate"],
                "successes": data["ceiling"].get("successes"),
                "test_episodes": data["ceiling"]["episodes"],
            },
        },
        "methods": method_aggregates,
        "paths": {
            "manifest": str(data["manifest_path"].relative_to(ROOT)),
            "baseline": str(data["baseline_path"].relative_to(ROOT)),
            "csv": str(csv_path.relative_to(ROOT)),
            "comparison_plot": str(comparison_path.relative_to(ROOT)),
            "learning_curves_plot": str(curves_path.relative_to(ROOT)),
        },
    }

    # Final comparison: show all five seed values and their mean with sample SD.
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.get_cmap("tab10").colors
    x_positions = list(range(len(data["methods"])))
    offsets = [(-0.22 + 0.11 * i) for i in range(len(EXPECTED_SEEDS))]
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        values = [data["by_method"][method["name"]]["runs"][seed]["row"]["success_rate"] * 100.0
                  for method in data["methods"]]
        ax.scatter([x + offsets[seed_index] for x in x_positions], values,
                   color=colors[seed_index], s=36, alpha=0.82,
                   label=f"Seed {seed}", zorder=3)
    for x, method, summary in zip(
            x_positions, data["methods"], method_aggregates):
        stats = summary["final_success_rate"]
        ax.errorbar(x, stats["mean"] * 100.0,
                    yerr=stats["sample_std_ddof_1"] * 100.0,
                    fmt="D", color="black", capsize=5, markersize=7,
                    linewidth=1.8, zorder=4)
    ax.axhline(float(data["floor"]["success_rate"]) * 100.0,
               color="tab:red", linestyle="--", linewidth=1.5,
               label="Original policy (floor)")
    ax.axhline(float(data["ceiling"]["success_rate"]) * 100.0,
               color="tab:green", linestyle=":", linewidth=1.8,
               label="No-camera-bias test (ceiling)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([method["label"] for method in data["methods"]], rotation=18, ha="right")
    ax.set_xlim(-0.55, len(x_positions) - 0.45)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Final test success rate (%)")
    ax.set_title("Goal 6 final results (black diamond: mean ± sample SD)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=160)
    plt.close(fig)

    # Curves come only from saved JSON summaries, never intermediate trace files.
    fig, axes = plt.subplots(len(data["methods"]), 1,
                             figsize=(10, 3.1 * len(data["methods"])),
                             sharex=True, sharey=True)
    if len(data["methods"]) == 1:
        axes = [axes]
    for ax, method in zip(axes, data["methods"]):
        runs = data["by_method"][method["name"]]["runs"]
        x_values = runs[EXPECTED_SEEDS[0]]["x"]
        for seed_index, seed in enumerate(EXPECTED_SEEDS):
            rates = runs[seed]["curve"]
            ax.plot(x_values, [r * 100.0 for r in rates], marker="o",
                    linewidth=1.0, alpha=0.62, color=colors[seed_index],
                    label=f"Seed {seed}")
        mean_rates = [
            statistics.mean(runs[seed]["curve"][i] for seed in EXPECTED_SEEDS) * 100.0
            for i in range(len(x_values))
        ]
        ax.plot(x_values, mean_rates, color="black", marker="o", linewidth=2.6,
                label="Pooled mean (five seeds)")
        ax.axhline(float(data["floor"]["success_rate"]) * 100.0,
                   color="tab:red", linestyle="--", linewidth=1.1,
                   label="Original policy (floor)")
        ax.axhline(float(data["ceiling"]["success_rate"]) * 100.0,
                   color="tab:green", linestyle=":", linewidth=1.2,
                   label="No-camera-bias test (ceiling)")
        ax.set_title(method["label"])
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.23)
    axes[-1].set_xlabel("Total practice episodes (fill-up included)")
    for ax in axes:
        ax.set_ylabel("Test success (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(curves_path, dpi=160)
    plt.close(fig)

    aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="Goal 6 manifest path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="directory for CSV, JSON and PNG outputs")
    args = parser.parse_args(argv)
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output

    try:
        try:
            output_dir.resolve().relative_to(RESULTS_ROOT.resolve())
        except ValueError as exc:
            raise InputError("Output directory must be inside results/takehome/") from exc
        data, missing = await_inputs(manifest_path)
        if data is None:
            print("Goal 6 plot inputs are not complete yet; no output files written:",
                  file=sys.stderr)
            for path in missing:
                print(f"  {path}", file=sys.stderr)
            return 0
        write_outputs(data, output_dir)
    except (InputError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Goal 6 plotting failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
