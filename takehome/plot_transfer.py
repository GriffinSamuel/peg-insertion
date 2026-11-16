#!/usr/bin/env python3
"""Summarize completed Goal 10 policy-transfer runs and their baselines."""

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
DEFAULT_MANIFEST = RESULTS_ROOT / "goal10_manifest.json"
DEFAULT_OUTPUT = RESULTS_ROOT / "goal10"
POLICIES = ("force_0.5mm_s4", "pos_0.5mm_s1", "force_0.5mm_s1")
SEEDS = (0, 1, 2)
PRACTICE_EPISODES = 200
TEST_SEED = 123
TEST_EPISODES = 500


class InputError(Exception):
    """An input artifact is incomplete or does not match the declared run."""


def require(condition, message):
    if not condition:
        raise InputError(message)


def number(value, name, context, low=None, high=None):
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


def validate_baseline(path, policy):
    require(path.is_file(), f"Missing policy baseline: {path}")
    baseline = read_json(path)
    require(baseline.get("policy") == policy, f"Baseline {path} is for the wrong policy")
    require(baseline.get("test_seed") == TEST_SEED,
            f"Baseline {path} must use test seed {TEST_SEED}")
    require(baseline.get("episodes_per_condition") == TEST_EPISODES,
            f"Baseline {path} must contain {TEST_EPISODES} tests per condition")
    require(baseline.get("practice_episodes") == 0,
            f"Baseline {path} must use zero practice episodes")
    conditions = baseline.get("conditions")
    require(isinstance(conditions, dict), f"Baseline {path} lacks conditions")
    rates = {}
    for condition_name in ("normal", "no_camera_bias"):
        condition = conditions.get(condition_name)
        require(isinstance(condition, dict),
                f"Baseline {path} lacks condition {condition_name}")
        require(condition.get("test_only") is True
                and condition.get("learned_from") is False,
                f"Baseline {path} condition {condition_name} is not test-only")
        require(condition.get("test_seed") == TEST_SEED,
                f"Baseline {path} condition {condition_name} has the wrong test seed")
        require(condition.get("episodes") == TEST_EPISODES,
                f"Baseline {path} condition {condition_name} has the wrong test count")
        rates[condition_name] = number(condition.get("success_rate"), "success_rate",
                                       f"{path}:{condition_name}", 0.0, 1.0)
    return rates


def load_inputs(manifest_path):
    require(manifest_path.is_file(), f"Missing Goal 10 manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    settings = manifest.get("settings")
    require(isinstance(settings, list) and len(settings) == len(POLICIES),
            "Manifest must contain exactly three policy settings")
    by_policy = {}
    for setting in settings:
        require(isinstance(setting, dict), "Each manifest setting must be an object")
        name = setting.get("name")
        require(name in POLICIES, f"Unexpected Goal 10 policy setting: {name!r}")
        require(name not in by_policy, f"Duplicate policy setting: {name}")
        require(setting.get("seeds") == list(SEEDS),
                f"Setting {name} must contain seeds 0, 1, and 2")
        overrides = setting.get("overrides")
        require(isinstance(overrides, dict) and overrides.get("policy") == name,
                f"Setting {name} must use that policy in its overrides")
        require(overrides.get("test_seed") == TEST_SEED,
                f"Setting {name} must use test seed {TEST_SEED}")
        output_value = setting.get("output")
        require(isinstance(output_value, str) and output_value,
                f"Setting {name} lacks output directory")
        output_dir = resolve_path(output_value)
        require(output_dir.resolve().is_relative_to(RESULTS_ROOT.resolve()),
                f"Setting {name} output is outside results/takehome")
        baseline_value = overrides.get("baseline_path")
        require(isinstance(baseline_value, str) and baseline_value,
                f"Setting {name} lacks its policy-specific baseline_path")
        baseline_path = resolve_path(baseline_value)
        baseline_rates = validate_baseline(baseline_path, name)

        runs = {}
        for seed in SEEDS:
            result_path = output_dir / f"seed_{seed}" / "result.json"
            require(result_path.is_file(), f"Missing final result for {name} seed {seed}: {result_path}")
            result = read_json(result_path)
            context = f"{name} seed {seed} ({result_path})"
            require(result.get("seed") == seed, f"{context}: wrong seed")
            require(result.get("practice_episodes") == PRACTICE_EPISODES,
                    f"{context}: expected {PRACTICE_EPISODES} practice episodes")
            require(result.get("test_data_learned_from") is False,
                    f"{context}: test data must not be learned from")
            config = result.get("config")
            require(isinstance(config, dict) and config.get("policy") == name,
                    f"{context}: config policy mismatch")
            final = result.get("final")
            require(isinstance(final, dict), f"{context}: lacks final test summary")
            require(final.get("practice_episodes") == PRACTICE_EPISODES,
                    f"{context}: final row is not at 200 practice episodes")
            require(final.get("episodes") == TEST_EPISODES,
                    f"{context}: final row must contain {TEST_EPISODES} tests")
            require(final.get("test_seed") == TEST_SEED,
                    f"{context}: final row must use test seed {TEST_SEED}")
            require(final.get("test_only") is True and final.get("learned_from") is False,
                    f"{context}: final row must be test-only")
            successes = final.get("successes")
            require(isinstance(successes, int) and not isinstance(successes, bool)
                    and 0 <= successes <= TEST_EPISODES,
                    f"{context}: invalid success count")
            rate = number(final.get("success_rate"), "success_rate", context, 0.0, 1.0)
            require(math.isclose(rate, successes / TEST_EPISODES, rel_tol=0, abs_tol=1e-12),
                    f"{context}: success rate does not match success count")
            runs[seed] = {
                "seed": seed,
                "practice_episodes": PRACTICE_EPISODES,
                "test_episodes": TEST_EPISODES,
                "test_seed": TEST_SEED,
                "successes": successes,
                "success_rate": rate,
                "result_path": str(result_path.relative_to(ROOT)),
            }
        by_policy[name] = {"runs": runs, "baseline": baseline_rates,
                           "baseline_path": str(baseline_path.relative_to(ROOT)),
                           "output": output_dir}

    require(set(by_policy) == set(POLICIES), "Manifest does not include all three required policies")
    return manifest, by_policy


def aggregate(by_policy):
    policies = []
    for policy in POLICIES:
        item = by_policy[policy]
        rows = [item["runs"][seed] for seed in SEEDS]
        rates = [row["success_rate"] for row in rows]
        policies.append({
            "policy": policy,
            "practice_episodes": PRACTICE_EPISODES,
            "test_episodes": TEST_EPISODES,
            "test_seed": TEST_SEED,
            "seeds": rows,
            "mean_success_rate": statistics.mean(rates),
            "sample_std_success_rate": statistics.stdev(rates),
            "floor_success_rate": item["baseline"]["normal"],
            "ceiling_success_rate": item["baseline"]["no_camera_bias"],
            "baseline_path": item["baseline_path"],
        })
    return {"policies": policies}


def write_csv(by_policy, output_path):
    fields = ["policy", "seed", "practice_episodes", "test_episodes", "test_seed",
              "successes", "success_rate", "mean_success_rate",
              "sample_std_success_rate", "floor_success_rate", "ceiling_success_rate",
              "result_path", "baseline_path"]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        summary = aggregate(by_policy)
        for method in summary["policies"]:
            for row in method["seeds"]:
                writer.writerow({
                    "policy": method["policy"], **row,
                    "mean_success_rate": method["mean_success_rate"],
                    "sample_std_success_rate": method["sample_std_success_rate"],
                    "floor_success_rate": method["floor_success_rate"],
                    "ceiling_success_rate": method["ceiling_success_rate"],
                    "baseline_path": method["baseline_path"],
                })


def write_chart(summary, output_path):
    mpl_config = RESULTS_ROOT / "mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5))
    centers = list(range(len(POLICIES)))
    offsets = (-0.20, -0.10, 0.0, 0.10, 0.20)
    for index, method in enumerate(summary["policies"]):
        center = centers[index]
        rates = [row["success_rate"] for row in method["seeds"]]
        mean = method["mean_success_rate"]
        sd = method["sample_std_success_rate"]
        for seed, rate, offset in zip(SEEDS, rates, offsets[:len(SEEDS)]):
            ax.scatter(center + offset, rate * 100, s=45, alpha=0.85,
                       color=f"C{seed}", zorder=3,
                       label=f"Seed {seed}" if index == 0 else None)
        ax.errorbar(center + 0.32, mean * 100, yerr=sd * 100, fmt="D",
                    color="black", capsize=4, markersize=6, zorder=4,
                    label="Mean ± sample SD" if index == 0 else None)
        ax.scatter(center + 0.43, method["floor_success_rate"] * 100,
                   marker="v", color="dimgray", s=52,
                   label="Original-policy floor" if index == 0 else None)
        ax.scatter(center + 0.52, method["ceiling_success_rate"] * 100,
                   marker="^", color="darkgreen", s=52,
                   label="No-camera-bias ceiling" if index == 0 else None)

    ax.set_xticks(centers, POLICIES)
    ax.set_ylabel("Test success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Goal 10 transfer results at 200 practice episodes")
    ax.grid(axis="y", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, frameon=True)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(output_path, dpi=165, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    try:
        require(output_dir.resolve().is_relative_to(RESULTS_ROOT.resolve()),
                "Output directory must be inside results/takehome")
        _, by_policy = load_inputs(manifest_path)
        summary = aggregate(by_policy)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(by_policy, output_dir / "transfer.csv")
        (output_dir / "transfer_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        write_chart(summary, output_dir / "transfer.png")
    except (InputError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Goal 10 transfer plotting failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
