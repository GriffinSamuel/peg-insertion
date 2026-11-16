#!/usr/bin/env python3
"""Summarize Goal 7 confirmation runs using saved test results only."""

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "takehome"
DEFAULT_MANIFEST = RESULTS_ROOT / "goal7_confirmation_manifest.json"
DEFAULT_CHOICE = RESULTS_ROOT / "goal7_choice.json"
DEFAULT_BASELINE = RESULTS_ROOT / "confirmation_baseline.json"
DEFAULT_OUTPUT = RESULTS_ROOT / "goal7" / "confirmation"
EXPECTED_SEEDS = tuple(range(100, 105))
EXPECTED_TEST_SEED = 456
EXPECTED_DEV_TEST_SEED = 123
EXPECTED_FINAL_PRACTICE = 200
EXPECTED_FINAL_TESTS = 500
EXPECTED_INTERMEDIATE_TESTS = 256
PLAIN_METHODS = ("aim", "movement", "combined")


class InputError(Exception):
    """An input is missing or inconsistent with the confirmation manifest."""


def require(condition, message):
    if not condition:
        raise InputError(message)


def finite_number(value, name, context, low=None, high=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{context}: {name} must be numeric")
    value = float(value)
    require(math.isfinite(value), f"{context}: {name} must be finite")
    if low is not None:
        require(value >= low, f"{context}: {name} is below {low}")
    if high is not None:
        require(value <= high, f"{context}: {name} is above {high}")
    return value


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"Invalid JSON in {path}: {exc}") from exc


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_test_row(row, expected_seed, expected_episodes, context):
    require(isinstance(row, dict), f"{context}: test row must be an object")
    require(row.get("test_only") is True and row.get("learned_from") is False,
            f"{context}: tests must be test-only")
    require(row.get("test_seed") == expected_seed,
            f"{context}: expected test seed {expected_seed}")
    episodes = row.get("episodes", row.get("test_episodes"))
    require(episodes == expected_episodes,
            f"{context}: expected {expected_episodes} test episodes")
    successes = row.get("successes")
    require(isinstance(successes, int) and not isinstance(successes, bool)
            and 0 <= successes <= expected_episodes,
            f"{context}: invalid success count")
    rate = finite_number(row.get("success_rate"), "success_rate", context, 0.0, 1.0)
    require(math.isclose(rate, successes / expected_episodes, rel_tol=0, abs_tol=1e-12),
            f"{context}: success rate does not match success count")
    return {"episodes": episodes, "successes": successes, "success_rate": rate,
            "test_seed": expected_seed}


def load_baseline(path):
    require(path.is_file(), f"Missing confirmation baseline: {path}")
    baseline = read_json(path)
    require(baseline.get("test_seed") == EXPECTED_TEST_SEED,
            "Confirmation baseline must use test seed 456")
    require(baseline.get("episodes_per_condition") == EXPECTED_FINAL_TESTS,
            "Confirmation baseline must use 500 tests per condition")
    conditions = baseline.get("conditions")
    require(isinstance(conditions, dict), "Confirmation baseline lacks conditions")
    rates = {}
    for key in ("normal", "no_camera_bias"):
        row = conditions.get(key)
        require(isinstance(row, dict), f"Confirmation baseline lacks {key} condition")
        checked = validate_test_row(row, EXPECTED_TEST_SEED, EXPECTED_FINAL_TESTS,
                                    f"confirmation baseline {key}")
        rates[key] = checked["success_rate"]
    return baseline, rates


def check_curve(result, baseline_rates, context):
    curve = result.get("curve")
    require(isinstance(curve, list) and len(curve) >= 2,
            f"{context}: missing learning-curve test rows")
    practices = []
    points = []
    for index, point in enumerate(curve):
        require(isinstance(point, dict), f"{context}: curve[{index}] must be an object")
        practice = point.get("practice_episodes")
        require(isinstance(practice, int) and not isinstance(practice, bool)
                and 0 <= practice <= EXPECTED_FINAL_PRACTICE,
                f"{context}: invalid practice count at curve[{index}]")
        if practice == 0:
            checked = validate_test_row(point, EXPECTED_TEST_SEED, EXPECTED_FINAL_TESTS,
                                        f"{context} original-policy reference")
            require(math.isclose(checked["success_rate"], baseline_rates["normal"],
                                 rel_tol=0, abs_tol=1e-12),
                    f"{context}: practice-zero point differs from confirmation baseline")
        elif practice == EXPECTED_FINAL_PRACTICE:
            checked = validate_test_row(point, EXPECTED_TEST_SEED, EXPECTED_FINAL_TESTS,
                                        f"{context} final practice-200 test")
        else:
            checked = validate_test_row(point, EXPECTED_DEV_TEST_SEED,
                                        EXPECTED_INTERMEDIATE_TESTS,
                                        f"{context} intermediate practice-{practice} test")
        practices.append(practice)
        points.append({"practice_episodes": practice, **checked})
    require(practices == sorted(set(practices)),
            f"{context}: practice points must be unique and increasing")
    require(practices[0] == 0 and practices[-1] == EXPECTED_FINAL_PRACTICE,
            f"{context}: curve must include the original policy and practice-200 final")
    return points


def load_inputs(manifest_path, choice_path, baseline_path):
    require(manifest_path.is_file(), f"Missing confirmation manifest: {manifest_path}")
    require(choice_path.is_file(), f"Missing Goal 7 choice: {choice_path}")
    choice = read_json(choice_path)
    selected = choice.get("selected")
    require(isinstance(selected, dict), "Goal 7 choice lacks selected method")
    selected_name = selected.get("name")
    settings_tried = choice.get("total_settings_tried")
    require(isinstance(settings_tried, int) and settings_tried > 0,
            "Goal 7 choice lacks total_settings_tried")
    manifest = read_json(manifest_path)
    settings = manifest.get("settings")
    require(isinstance(settings, list), "Confirmation manifest lacks settings list")
    expected_names = set(PLAIN_METHODS)
    if selected_name not in PLAIN_METHODS:
        expected_names.add(selected_name)
    require(len(settings) == len(expected_names),
            "Manifest must contain three plain methods and any separate tuned method")

    baseline, baseline_rates = load_baseline(baseline_path)
    methods = []
    seen = set()
    for setting in settings:
        require(isinstance(setting, dict), "Each confirmation setting must be an object")
        name = setting.get("name")
        require(name in expected_names, f"Unexpected confirmation method {name!r}")
        require(name not in seen, f"Duplicate confirmation method {name}")
        seen.add(name)
        require(setting.get("seeds") == list(EXPECTED_SEEDS),
                f"{name}: expected exact practice seeds 100–104")
        final_method = setting.get("final_method")
        require(isinstance(final_method, bool), f"{name}: missing final_method flag")
        require(final_method is (name == selected_name),
                f"{name}: final_method does not match written choice")
        expected_setting_count = settings_tried if final_method else 1
        require(setting.get("settings_tried_for_selection") == expected_setting_count,
                f"{name}: incorrect settings_tried_for_selection")
        label = setting.get("label")
        require(isinstance(label, str) and label, f"{name}: missing plain-language label")
        output_value = setting.get("output")
        require(isinstance(output_value, str) and output_value,
                f"{name}: missing output directory")
        output = resolve_path(output_value)
        require(output.resolve().is_relative_to(RESULTS_ROOT.resolve()),
                f"{name}: results directory is outside results/takehome")
        runs = []
        curves = []
        for seed in EXPECTED_SEEDS:
            result_path = output / f"seed_{seed}" / "result.json"
            require(result_path.is_file(), f"Missing {name} seed {seed}: {result_path}")
            result = read_json(result_path)
            context = f"{name} practice seed {seed}"
            require(result.get("seed") == seed, f"{context}: result seed mismatch")
            require(result.get("practice_episodes") == EXPECTED_FINAL_PRACTICE,
                    f"{context}: expected 200 practice episodes")
            require(result.get("test_data_learned_from") is False,
                    f"{context}: test data must not be learned from")
            require(result.get("config", {}).get("test_seed") == EXPECTED_TEST_SEED,
                    f"{context}: config test seed must be 456")
            points = check_curve(result, baseline_rates, context)
            final = result.get("final")
            require(isinstance(final, dict), f"{context}: missing final row")
            final_checked = validate_test_row(final, EXPECTED_TEST_SEED, EXPECTED_FINAL_TESTS,
                                              f"{context} final")
            require(final.get("practice_episodes") == EXPECTED_FINAL_PRACTICE,
                    f"{context}: final row must be at 200 practice episodes")
            require(final_checked["successes"] == points[-1]["successes"]
                    and final_checked["success_rate"] == points[-1]["success_rate"],
                    f"{context}: final row differs from last curve point")
            runs.append({
                "seed": seed,
                "practice_episodes": EXPECTED_FINAL_PRACTICE,
                "test_episodes": EXPECTED_FINAL_TESTS,
                "test_seed": EXPECTED_TEST_SEED,
                "successes": final_checked["successes"],
                "success_rate": final_checked["success_rate"],
                "result_path": str(result_path.relative_to(ROOT)),
            })
            curves.append({"seed": seed, "points": points})
        rates = [run["success_rate"] for run in runs]
        methods.append({
            "name": name,
            "label": label,
            "final_method": final_method,
            "settings_tried_for_selection": expected_setting_count,
            "seed_results": runs,
            "mean": statistics.mean(rates),
            "std": statistics.stdev(rates),
            "curves": curves,
        })
    require(seen == expected_names, "Confirmation manifest does not cover expected methods")
    require(sum(method["final_method"] for method in methods) == 1,
            "Exactly one confirmation method must be the final selection")
    require(baseline_rates["normal"] <= baseline_rates["no_camera_bias"],
            "Confirmation baseline floor exceeds ceiling")
    return {
        "methods": methods,
        "floor": baseline_rates["normal"],
        "ceiling": baseline_rates["no_camera_bias"],
        "baseline_path": str(baseline_path.relative_to(ROOT)),
        "choice_path": str(choice_path.relative_to(ROOT)),
        "selected_name": selected_name,
        "total_settings_tried": settings_tried,
    }


def write_csv(data, path):
    fields = ["name", "label", "final_method", "settings_tried_for_selection", "seed",
              "practice_episodes", "test_episodes", "test_seed", "successes",
              "success_rate", "mean", "std", "floor", "ceiling", "result_path"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for method in data["methods"]:
            for row in method["seed_results"]:
                writer.writerow({
                    "name": method["name"], "label": method["label"],
                    "final_method": method["final_method"],
                    "settings_tried_for_selection": method["settings_tried_for_selection"],
                    **row, "mean": method["mean"], "std": method["std"],
                    "floor": data["floor"], "ceiling": data["ceiling"],
                })


def write_summary(data, path):
    summary = {key: value for key, value in data.items() if key != "methods"}
    summary["methods"] = [{key: value for key, value in method.items() if key != "curves"}
                          for method in data["methods"]]
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_final_chart(data, path, plt):
    methods = data["methods"]
    fig, ax = plt.subplots(figsize=(max(9, 2.3 * len(methods)), 6.5))
    centers = list(range(len(methods)))
    offsets = (-0.20, -0.10, 0.0, 0.10, 0.20)
    for index, method in enumerate(methods):
        center = centers[index]
        for row, offset in zip(method["seed_results"], offsets):
            ax.scatter(center + offset, row["success_rate"] * 100, s=42,
                       color=f"C{row['seed'] % 10}", alpha=0.82, zorder=3,
                       label=f"Seed {row['seed']}" if index == 0 else None)
        ax.errorbar(center + 0.31, method["mean"] * 100,
                    yerr=method["std"] * 100, fmt="D", color="black",
                    capsize=4, markersize=6, zorder=4,
                    label="Mean ± sample SD" if index == 0 else None)
    ax.axhline(data["floor"] * 100, color="dimgray", linestyle="--",
               label="Original-policy floor")
    ax.axhline(data["ceiling"] * 100, color="darkgreen", linestyle=":",
               label="No-camera-bias ceiling")
    labels = [f"{m['label']}\n{m['settings_tried_for_selection']} settings"
              for m in methods]
    ax.set_xticks(centers, labels)
    ax.set_ylabel("Final test success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Goal 7 confirmation: 200 practice episodes, 500 tests at seed 456")
    ax.grid(axis="y", alpha=0.25)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=4, frameon=True)
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)


def write_learning_curves(data, path, plt):
    methods = data["methods"]
    fig, axes = plt.subplots(len(methods), 1,
                             figsize=(10, max(4.0, 3.3 * len(methods))),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, method in zip(axes, methods):
        for seed_index, curve in enumerate(method["curves"]):
            points = curve["points"]
            ax.plot([p["practice_episodes"] for p in points],
                    [p["success_rate"] * 100 for p in points],
                    color=f"C{curve['seed'] % 10}", alpha=0.72,
                    linewidth=1.25, marker="o", markersize=3.2,
                    label=f"Seed {curve['seed']}" if ax is axes[0] else None)
        practice_points = [curve["points"] for curve in method["curves"]]
        x_values = [point["practice_episodes"] for point in practice_points[0]]
        means = [statistics.mean([curve[i]["success_rate"] for curve in practice_points]) * 100
                 for i in range(len(x_values))]
        ax.plot(x_values, means, color="black", linewidth=2.5, marker="o",
                markersize=4.5, label="Mean" if ax is axes[0] else None)
        ax.set_title(f"{method['label']} ({method['settings_tried_for_selection']} settings)")
        ax.set_ylabel("Success rate (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.23)
    axes[-1].set_xlabel("Total practice episodes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=6, frameon=True)
    fig.suptitle("Confirmation learning curves: development checkpoints use 256 tests at seed 123; final uses 500 at seed 456",
                 y=0.995)
    fig.subplots_adjust(top=0.94, bottom=0.10, hspace=0.48)
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)


def write_outputs(data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "mplconfig").mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(RESULTS_ROOT / "mplconfig")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    write_csv(data, output_dir / "confirmation.csv")
    write_summary(data, output_dir / "confirmation_summary.json")
    write_final_chart(data, output_dir / "confirmation.png", plt)
    write_learning_curves(data, output_dir / "confirmation_learning_curves.png", plt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--choice", type=Path, default=DEFAULT_CHOICE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    manifest_path = resolve_path(args.manifest)
    choice_path = resolve_path(args.choice)
    baseline_path = resolve_path(args.baseline)
    output_dir = resolve_path(args.output)
    try:
        require(output_dir.resolve().is_relative_to(RESULTS_ROOT.resolve()),
                "Output directory must be inside results/takehome")
        data = load_inputs(manifest_path, choice_path, baseline_path)
        write_outputs(data, output_dir)
    except (InputError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Goal 7 confirmation plotting failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
