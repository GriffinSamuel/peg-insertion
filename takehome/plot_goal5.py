#!/usr/bin/env python3
"""Plot the designated Goal 5 runs and summarize their final test results."""

import csv
import json
import os
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "takehome"
BENCHMARK_DIR = RESULTS_DIR / "benchmark"
MANIFEST_PATH = RESULTS_DIR / "benchmark_manifest.json"
BASELINE_PATH = RESULTS_DIR / "baseline.json"
PLOT_PATH = RESULTS_DIR / "goal5_learning_curves.png"
CSV_PATH = RESULTS_DIR / "goal5_per_seed.csv"
AGGREGATE_PATH = RESULTS_DIR / "goal5_aggregate.json"
EXPECTED_SEEDS = tuple(range(5))
EXPECTED_TEST_SEED = 123
EXPECTED_PRACTICE_EPISODES = 200
EXPECTED_FILL_EPISODES = 40
EXPECTED_LEARNER_EPISODES = 160


class InputError(Exception):
    """Raised when an input file is missing or does not match the run design."""


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"Invalid JSON in {path}: {exc}") from exc


def require(condition, message):
    if not condition:
        raise InputError(message)


def designated_result_paths(manifest):
    designations = manifest.get("headline_designations")
    require(isinstance(designations, dict), "Manifest lacks headline_designations mapping")
    expected_keys = {str(seed) for seed in EXPECTED_SEEDS}
    require(set(designations) == expected_keys,
            f"Expected headline seeds {sorted(expected_keys)}, found {sorted(designations)}")

    paths = {}
    for seed in EXPECTED_SEEDS:
        designation = designations[str(seed)]
        parts = Path(designation).parts
        require(len(parts) == 2 and parts[1] == f"seed_{seed}",
                f"Invalid designation for seed {seed}: {designation!r}")
        path = BENCHMARK_DIR / parts[0] / parts[1] / "result.json"
        paths[seed] = path

    require(len(paths) == 5 and set(paths) == set(EXPECTED_SEEDS),
            "Headline results must designate each seed 0 through 4 exactly once")
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise InputError("Missing designated headline result files:\n  " + "\n  ".join(missing))
    return paths


def validate_baseline(baseline):
    require(baseline.get("test_seed") == EXPECTED_TEST_SEED,
            f"Baseline test_seed must be {EXPECTED_TEST_SEED}")
    conditions = baseline.get("conditions", {})
    for name in ("normal", "no_camera_bias"):
        condition = conditions.get(name, {})
        require(condition.get("episodes") == 500,
                f"Baseline condition {name} must have 500 test episodes")
        require(condition.get("test_seed") == EXPECTED_TEST_SEED,
                f"Baseline condition {name} must use test seed {EXPECTED_TEST_SEED}")
        require(condition.get("test_only") is True and condition.get("learned_from") is False,
                f"Baseline condition {name} must be test-only and not learned from")
        require(isinstance(condition.get("success_rate"), (float, int)),
                f"Baseline condition {name} lacks success_rate")
    return conditions["normal"], conditions["no_camera_bias"]


def validate_curve(seed, result, baseline_normal):
    require(result.get("seed") == seed, f"Result seed mismatch: expected {seed}")
    require(result.get("practice_episodes") == EXPECTED_PRACTICE_EPISODES,
            f"Seed {seed}: expected exactly 200 practice episodes")
    require(result.get("fill_episodes") == EXPECTED_FILL_EPISODES,
            f"Seed {seed}: expected exactly 40 fill-up episodes")
    require(result.get("learner_episodes") == EXPECTED_LEARNER_EPISODES,
            f"Seed {seed}: expected exactly 160 learner episodes")

    final = result.get("final")
    require(isinstance(final, dict), f"Seed {seed}: missing final test summary")
    require(final.get("episodes") == 500,
            f"Seed {seed}: final test must contain 500 episodes")
    require(final.get("test_seed") == EXPECTED_TEST_SEED,
            f"Seed {seed}: final test must use seed {EXPECTED_TEST_SEED}")
    require(final.get("test_only") is True and final.get("learned_from") is False,
            f"Seed {seed}: final test must be separate and not learned from")
    require(isinstance(final.get("successes"), int), f"Seed {seed}: final successes missing")
    require(isinstance(final.get("success_rate"), (float, int)),
            f"Seed {seed}: final success_rate missing")

    curve = result.get("curve")
    require(isinstance(curve, list), f"Seed {seed}: missing learning curve")
    expected_x = [0, *range(EXPECTED_FILL_EPISODES, EXPECTED_PRACTICE_EPISODES + 1, 20)]
    actual_x = [point.get("practice_episodes") for point in curve]
    require(actual_x == expected_x,
            f"Seed {seed}: learning curve x values must be {expected_x}, found {actual_x}")

    first = curve[0]
    require(first.get("label") == "original_policy" and first.get("episodes") == 500,
            f"Seed {seed}: point 0 must be the 500-episode original-policy baseline")
    require(first.get("test_seed") == EXPECTED_TEST_SEED,
            f"Seed {seed}: point 0 baseline must use test seed {EXPECTED_TEST_SEED}")
    require(first.get("success_rate") == baseline_normal.get("success_rate"),
            f"Seed {seed}: point 0 does not match the original-policy baseline")

    for point in curve[1:-1]:
        practice_at = point.get("practice_episodes")
        require(point.get("episodes") == 256,
                f"Seed {seed}, practice {practice_at}: intermediate test must contain 256 episodes")
        require(point.get("test_seed") == EXPECTED_TEST_SEED,
                f"Seed {seed}: intermediate tests must use test seed {EXPECTED_TEST_SEED}")
        require(point.get("test_only") is True and point.get("learned_from") is False,
                f"Seed {seed}: intermediate tests must be separate and not learned from")

    last = curve[-1]
    require(last.get("practice_episodes") == EXPECTED_PRACTICE_EPISODES
            and last.get("episodes") == 500
            and last.get("test_seed") == EXPECTED_TEST_SEED,
            f"Seed {seed}: final curve point must be 500 tests at practice 200, seed 123")
    require(last.get("success_rate") == final.get("success_rate"),
            f"Seed {seed}: final result and final curve point disagree")
    return curve


def main():
    try:
        manifest = read_json(MANIFEST_PATH)
        baseline = read_json(BASELINE_PATH)
        result_paths = designated_result_paths(manifest)
        baseline_normal, baseline_ceiling = validate_baseline(baseline)

        runs = {}
        curves = {}
        for seed, path in result_paths.items():
            result = read_json(path)
            curves[seed] = validate_curve(seed, result, baseline_normal)
            runs[seed] = result

        # Matplotlib creates its user configuration directory on import. Keep it in
        # the allowed results directory and set it before importing matplotlib.
        mpl_config = RESULTS_DIR / "mplconfig"
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        x_values = [point["practice_episodes"] for point in curves[0]]
        rates_by_seed = {
            seed: [float(point["success_rate"]) * 100.0 for point in curves[seed]]
            for seed in EXPECTED_SEEDS
        }
        mean_curve = [
            statistics.mean(rates_by_seed[seed][index] for seed in EXPECTED_SEEDS)
            for index in range(len(x_values))
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.get_cmap("tab10").colors
        for index, seed in enumerate(EXPECTED_SEEDS):
            ax.plot(x_values, rates_by_seed[seed], marker="o", linewidth=1.2,
                    alpha=0.72, color=colors[index], label=f"Seed {seed}")
        ax.plot(x_values, mean_curve, marker="o", linewidth=3.0,
                color="black", label="Mean (five seeds)")
        ax.axhline(float(baseline_normal["success_rate"]) * 100.0,
                   color="tab:red", linestyle="--", linewidth=1.6,
                   label="Original-policy floor")
        ax.axhline(float(baseline_ceiling["success_rate"]) * 100.0,
                   color="tab:green", linestyle=":", linewidth=1.8,
                   label="No-bias ceiling")
        ax.set_xlim(0, EXPECTED_PRACTICE_EPISODES)
        ax.set_ylim(0, 100)
        ax.set_xticks(x_values)
        ax.set_xlabel("Total practice episodes (fill-up included)")
        ax.set_ylabel("Test success (%)")
        ax.set_title("Goal 5 learning curves", pad=30)
        ax.text(0.5, 1.01,
                "Checkpoint tests: 256 episodes, seed 123; final: 500; point 0 baseline: 500",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=160)
        plt.close(fig)

        with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, lineterminator="\n", fieldnames=[
                "seed", "result_path", "practice_episodes", "fill_episodes",
                "learner_episodes", "final_test_episodes", "final_test_seed",
                "final_successes", "final_success_rate", "seconds_excluding_imports",
            ])
            writer.writeheader()
            for seed in EXPECTED_SEEDS:
                result = runs[seed]
                final = result["final"]
                writer.writerow({
                    "seed": seed,
                    "result_path": str(result_paths[seed].relative_to(ROOT)),
                    "practice_episodes": result["practice_episodes"],
                    "fill_episodes": result["fill_episodes"],
                    "learner_episodes": result["learner_episodes"],
                    "final_test_episodes": final["episodes"],
                    "final_test_seed": final["test_seed"],
                    "final_successes": final["successes"],
                    "final_success_rate": final["success_rate"],
                    "seconds_excluding_imports": result["seconds_excluding_imports"],
                })

        final_rates = [float(runs[seed]["final"]["success_rate"]) for seed in EXPECTED_SEEDS]
        aggregate = {
            "final_success_rate": {
                "mean": statistics.mean(final_rates),
                "sample_std_ddof_1": statistics.stdev(final_rates),
                "min": min(final_rates),
                "max": max(final_rates),
                "per_seed": {str(seed): final_rates[seed] for seed in EXPECTED_SEEDS},
            },
            "paths": {
                "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
                "baseline": str(BASELINE_PATH.relative_to(ROOT)),
                "headline_results": {
                    str(seed): str(result_paths[seed].relative_to(ROOT))
                    for seed in EXPECTED_SEEDS
                },
                "plot": str(PLOT_PATH.relative_to(ROOT)),
                "per_seed_csv": str(CSV_PATH.relative_to(ROOT)),
                "aggregate": str(AGGREGATE_PATH.relative_to(ROOT)),
            },
        }
        AGGREGATE_PATH.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    except (InputError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Goal 5 plotting failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
