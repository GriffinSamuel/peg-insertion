#!/usr/bin/env python3
"""Plot saved Goal 8 test-only calibration summaries and export event rows."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "takehome"
DEFAULT_INPUT = RESULTS_ROOT / "goal8" / "summary.json"
DEFAULT_OUTPUT = RESULTS_ROOT / "goal8"
EXPECTED_SEED_COUNT = 5
EXPECTED_FINAL_PRACTICE = 200


class InputError(Exception):
    """An input file exists but has an invalid or incomplete schema."""


def require(condition, message):
    if not condition:
        raise InputError(message)


def finite_number(value, name, context):
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{context}: {name} must be numeric")
    number = float(value)
    require(math.isfinite(number), f"{context}: {name} must be finite")
    return number


def require_rate(value, name, context):
    number = finite_number(value, name, context)
    require(0.0 <= number <= 1.0, f"{context}: {name} must be between 0 and 1")
    return number


def load_summary(path):
    if not path.is_file():
        print(f"Goal 8 summary is not available yet: {path}; no files written.",
              file=sys.stderr)
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"Invalid JSON in {path}: {exc}") from exc

    seeds = summary.get("seeds")
    require(isinstance(seeds, list) and len(seeds) == EXPECTED_SEED_COUNT,
            f"summary must list exactly {EXPECTED_SEED_COUNT} seeds")
    require(all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds),
            "summary seeds must be integers")
    require(len(set(seeds)) == EXPECTED_SEED_COUNT, "summary seed list contains duplicates")
    per_seed = summary.get("per_seed")
    pooled = summary.get("pooled")
    require(isinstance(per_seed, list) and per_seed, "summary.per_seed must be a nonempty list")
    require(isinstance(pooled, list) and pooled, "summary.pooled must be a nonempty list")

    by_seed = {seed: {} for seed in seeds}
    for index, record in enumerate(per_seed):
        context = f"per_seed[{index}]"
        require(isinstance(record, dict), f"{context} must be an object")
        seed = record.get("seed")
        require(seed in by_seed, f"{context} contains unexpected seed {seed!r}")
        practice = record.get("practice_episodes")
        require(isinstance(practice, int) and not isinstance(practice, bool)
                and 0 <= practice <= EXPECTED_FINAL_PRACTICE,
                f"{context}.practice_episodes is invalid")
        require(practice not in by_seed[seed], f"duplicate seed/checkpoint: {seed}/{practice}")
        require(isinstance(record.get("test_seed"), int), f"{context} lacks test_seed")
        require(isinstance(record.get("trace_path"), str) and record["trace_path"],
                f"{context} lacks test trace provenance")
        episodes = record.get("test_episodes")
        require(isinstance(episodes, int) and episodes > 0, f"{context} lacks test_episodes")
        require(record.get("predictor_uses_raw_q") is False,
                f"{context} must state that the probability predictor does not use raw Q")
        require(record.get("predictor_uses_true_bias") is False,
                f"{context} must state that the probability predictor does not use true bias")
        require_rate(record.get("mean_predicted"), "mean_predicted", context)
        require_rate(record.get("observed"), "observed", context)
        for metric in ("expected_calibration_error", "brier_score", "signed_bias"):
            value = finite_number(record.get(metric), metric, context)
            if metric != "signed_bias":
                require(0.0 <= value <= 1.0, f"{context}.{metric} must be between 0 and 1")
        bins = record.get("bins")
        raw_bins = record.get("raw_q_bins")
        require(isinstance(bins, list), f"{context}.bins must be a list")
        require(isinstance(raw_bins, list), f"{context}.raw_q_bins must be a list")
        for b, item in enumerate(bins):
            require_rate(item.get("predicted"), "predicted", f"{context}.bins[{b}]")
            require_rate(item.get("observed"), "observed", f"{context}.bins[{b}]")
            require(isinstance(item.get("count"), int) and item["count"] > 0,
                    f"{context}.bins[{b}].count must be positive")
        for b, item in enumerate(raw_bins):
            finite_number(item.get("raw_q"), "raw_q", f"{context}.raw_q_bins[{b}]")
            require_rate(item.get("observed"), "observed", f"{context}.raw_q_bins[{b}]")
            require(isinstance(item.get("count"), int) and item["count"] > 0,
                    f"{context}.raw_q_bins[{b}].count must be positive")
        moments = record.get("confident_failed_moments")
        drops = record.get("failed_confidence_drops")
        require(isinstance(moments, list), f"{context}.confident_failed_moments must be a list")
        require(isinstance(drops, list), f"{context}.failed_confidence_drops must be a list")
        by_seed[seed][practice] = record

    for seed in seeds:
        require(EXPECTED_FINAL_PRACTICE in by_seed[seed],
                f"seed {seed} has no final practice-{EXPECTED_FINAL_PRACTICE} analysis")

    pooled_by_practice = {}
    for index, record in enumerate(pooled):
        context = f"pooled[{index}]"
        require(isinstance(record, dict), f"{context} must be an object")
        practice = record.get("practice_episodes")
        require(isinstance(practice, int) and 0 <= practice <= EXPECTED_FINAL_PRACTICE,
                f"{context}.practice_episodes is invalid")
        require(practice not in pooled_by_practice, f"duplicate pooled checkpoint {practice}")
        require(record.get("seed_count") == EXPECTED_SEED_COUNT,
                f"{context}.seed_count must equal {EXPECTED_SEED_COUNT}")
        require_rate(record.get("mean_predicted"), "mean_predicted", context)
        require_rate(record.get("observed"), "observed", context)
        for metric in ("expected_calibration_error", "brier_score", "signed_bias"):
            value = finite_number(record.get(metric), metric, context)
            if metric != "signed_bias":
                require(0.0 <= value <= 1.0, f"{context}.{metric} must be between 0 and 1")
        for b, item in enumerate(record.get("bins", [])):
            require_rate(item.get("predicted"), "predicted", f"{context}.bins[{b}]")
            require_rate(item.get("observed"), "observed", f"{context}.bins[{b}]")
            require(isinstance(item.get("count"), int) and item["count"] > 0,
                    f"{context}.bins[{b}].count must be positive")
        for b, item in enumerate(record.get("raw_q_bins", [])):
            finite_number(item.get("raw_q"), "raw_q", f"{context}.raw_q_bins[{b}]")
            require_rate(item.get("observed"), "observed", f"{context}.raw_q_bins[{b}]")
            require(isinstance(item.get("count"), int) and item["count"] > 0,
                    f"{context}.raw_q_bins[{b}].count must be positive")
        pooled_by_practice[practice] = record
    require(EXPECTED_FINAL_PRACTICE in pooled_by_practice,
            f"pooled summaries lack practice-{EXPECTED_FINAL_PRACTICE} analysis")

    return {"summary": summary, "seeds": seeds, "by_seed": by_seed,
            "pooled_by_practice": pooled_by_practice}


def write_event_csvs(data, output_dir):
    confident_path = output_dir / "confident_failed.csv"
    drops_path = output_dir / "confidence_drops.csv"
    confident_fields = ["seed", "practice_episodes", "test_seed", "test_episodes",
                        "episode_id", "step", "predicted", "raw_q", "guess_m",
                        "true_bias_m", "force_n", "trace_path"]
    with confident_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=confident_fields, lineterminator="\n")
        writer.writeheader()
        for seed in data["seeds"]:
            for practice, record in sorted(data["by_seed"][seed].items()):
                for moment in record["confident_failed_moments"]:
                    writer.writerow({
                        "seed": seed,
                        "practice_episodes": practice,
                        "test_seed": record["test_seed"],
                        "test_episodes": record["test_episodes"],
                        "episode_id": moment.get("episode_id"),
                        "step": moment.get("step"),
                        "predicted": moment.get("predicted"),
                        "raw_q": moment.get("raw_q"),
                        "guess_m": json.dumps(moment.get("guess_m"), ensure_ascii=False),
                        "true_bias_m": json.dumps(moment.get("true_bias_m"), ensure_ascii=False),
                        "force_n": moment.get("force_n"),
                        "trace_path": record["trace_path"],
                    })

    drop_fields = ["seed", "practice_episodes", "test_seed", "test_episodes",
                   "episode_id", "step", "drop", "trace_path"]
    with drops_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=drop_fields, lineterminator="\n")
        writer.writeheader()
        for seed in data["seeds"]:
            for practice, record in sorted(data["by_seed"][seed].items()):
                for event in record["failed_confidence_drops"]:
                    writer.writerow({
                        "seed": seed,
                        "practice_episodes": practice,
                        "test_seed": record["test_seed"],
                        "test_episodes": record["test_episodes"],
                        "episode_id": event.get("episode_id"),
                        "step": "null" if event.get("step") is None else event.get("step"),
                        "drop": "null" if event.get("drop") is None else event.get("drop"),
                        "trace_path": record["trace_path"],
                    })


def make_final_reliability(data, path):
    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(8, 7))
    for index, seed in enumerate(data["seeds"]):
        record = data["by_seed"][seed][EXPECTED_FINAL_PRACTICE]
        bins = sorted(record["bins"], key=lambda row: row["bin"])
        ax.plot([row["predicted"] for row in bins],
                [row["observed"] for row in bins],
                marker="o", linewidth=1.25, markersize=4.5,
                color=colors[index % len(colors)], alpha=0.78,
                label=f"Seed {seed}")
    pooled = data["pooled_by_practice"][EXPECTED_FINAL_PRACTICE]
    bins = sorted(pooled["bins"], key=lambda row: row["bin"])
    ax.plot([row["predicted"] for row in bins],
            [row["observed"] for row in bins],
            marker="o", linewidth=2.8, markersize=6,
            color="black", label="Pooled (five seeds)")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2,
            label="Ideal calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted episode-success probability")
    ax.set_ylabel("Observed episode success rate")
    ax.set_title("Episode-success calibration at 200 practice episodes")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=4, frameon=True)
    fig.subplots_adjust(bottom=0.20)
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)


def make_final_raw_q(data, path):
    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(8, 7))
    for index, seed in enumerate(data["seeds"]):
        record = data["by_seed"][seed][EXPECTED_FINAL_PRACTICE]
        bins = sorted(record["raw_q_bins"], key=lambda row: row["bin"])
        ax.plot([row["raw_q"] for row in bins],
                [row["observed"] for row in bins],
                marker="o", linewidth=1.25, markersize=4.5,
                color=colors[index % len(colors)], alpha=0.78,
                label=f"Seed {seed}")
    pooled = data["pooled_by_practice"][EXPECTED_FINAL_PRACTICE]
    bins = sorted(pooled["raw_q_bins"], key=lambda row: row["bin"])
    ax.plot([row["raw_q"] for row in bins],
            [row["observed"] for row in bins],
            marker="o", linewidth=2.8, markersize=6,
            color="black", label="Pooled (five seeds)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean raw Q rating (discounted return; not a probability)")
    ax.set_ylabel("Observed episode success rate")
    ax.set_title("Raw Q ratings against episode outcomes at 200 practice episodes")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, frameon=True)
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)


def make_training_plot(data, path):
    colors = plt.get_cmap("tab10").colors
    metrics = [
        ("expected_calibration_error", "Expected calibration error"),
        ("brier_score", "Brier score"),
        ("signed_bias", "Signed prediction bias"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 11), sharex=True)
    for ax, (key, title) in zip(axes, metrics):
        for index, seed in enumerate(data["seeds"]):
            records = sorted(data["by_seed"][seed].values(),
                             key=lambda row: row["practice_episodes"])
            ax.plot([row["practice_episodes"] for row in records],
                    [row[key] for row in records],
                    marker="o", linewidth=1.15, markersize=3.8,
                    color=colors[index % len(colors)], alpha=0.78,
                    label=f"Seed {seed}")
        pooled = sorted(data["pooled_by_practice"].values(),
                        key=lambda row: row["practice_episodes"])
        ax.plot([row["practice_episodes"] for row in pooled],
                [row[key] for row in pooled],
                marker="o", linewidth=2.7, markersize=5.5,
                color="black", label="Pooled (five seeds)")
        if key == "signed_bias":
            ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.23)
    axes[-1].set_xlabel("Total practice episodes used")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=4, frameon=True)
    fig.suptitle("Calibration metrics across training checkpoints", y=0.99)
    fig.subplots_adjust(top=0.94, bottom=0.11, hspace=0.35)
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)


def write_figures(data, output_dir):
    mpl_config = RESULTS_ROOT / "mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config)
    import matplotlib

    matplotlib.use("Agg")
    global plt
    import matplotlib.pyplot as plt

    make_final_reliability(data, output_dir / "reliability_final.png")
    make_final_raw_q(data, output_dir / "raw_q_final.png")
    make_training_plot(data, output_dir / "calibration_over_training.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Goal 8 test-only analysis summary JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="directory for Goal 8 plots and event CSVs")
    args = parser.parse_args(argv)
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    try:
        try:
            output_dir.resolve().relative_to((RESULTS_ROOT / "goal8").resolve())
        except ValueError as exc:
            raise InputError("Output directory must be inside results/takehome/goal8/") from exc
        data = load_summary(input_path)
        if data is None:
            return 0
        output_dir.mkdir(parents=True, exist_ok=True)
        write_event_csvs(data, output_dir)
        write_figures(data, output_dir)
    except (InputError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Goal 8 calibration plotting failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
