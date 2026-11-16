#!/usr/bin/env python3
"""Plot saved Phase A aim diagnostics without loading binary traces."""

import csv
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Set the follow-up cache locations before importing NumPy or Matplotlib.
from takehome import followup_runtime  # noqa: E402,F401

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


INPUT = ROOT / "results" / "takehome" / "followup" / "A" / "diagnosis.json"
OUTPUT_DIR = INPUT.parent
SEEDS = (100, 101, 102, 103, 104)
WINDOWS = (1, 20, 40)
FIRST_CONTACT_MOMENT = "first_contact_reading"
AIM_METRICS = (
    ("guess_error_start_mm", "Start\nall episodes"),
    ("guess_error_first_contact_mm", "First\ncontact"),
    ("guess_error_success_mm", "At\nsuccess"),
    ("guess_error_end200_mm", "End\nstep 200"),
    ("paired_start_error_success_episodes_mm",
     "Start\nsuccessful only\npaired"),
)
SEED_COLORS = ("#2166ac", "#b2182b", "#1b7837", "#762a83", "#e08214")
SEED_MARKERS = ("o", "s", "^", "D", "v")


def require_finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


def load_and_validate():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing Phase A JSON summary: {INPUT}")
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    metrics = data.get("aim_metrics")
    if not isinstance(metrics, list) or len(metrics) != len(SEEDS):
        raise ValueError("aim_metrics must contain exactly five seed records")
    by_seed = {row.get("seed"): row for row in metrics}
    if len(by_seed) != len(metrics) or set(by_seed) != set(SEEDS):
        raise ValueError(f"aim_metrics must contain exactly seeds {SEEDS}")
    for seed in SEEDS:
        for key, _ in AIM_METRICS:
            require_finite(by_seed[seed].get(key), f"seed {seed} {key}")

    probes = data.get("prediction_probes")
    expected = {(seed, window, moment)
                for seed in SEEDS for window in WINDOWS
                for moment in (FIRST_CONTACT_MOMENT, "control_step125_before_action")}
    if not isinstance(probes, list) or len(probes) != 30:
        raise ValueError("prediction_probes must contain exactly 30 raw records")
    actual = [(row.get("seed"), row.get("window_steps"), row.get("moment"))
              for row in probes]
    if len(set(actual)) != 30 or set(actual) != expected:
        raise ValueError("prediction_probes must contain each seed/window/moment combination once")
    for row in probes:
        for key in ("prediction_mean_error_mm", "zero_guess_mean_error_mm"):
            require_finite(row.get(key), f"prediction probe {row.get('seed')}/{row.get('window_steps')} {key}")
    first_contact = [row for row in probes if row.get("moment") == FIRST_CONTACT_MOMENT]
    zero_guess_by_seed = {}
    for seed in SEEDS:
        seed_rows = [row for row in first_contact if row["seed"] == seed]
        if {row["window_steps"] for row in seed_rows} != set(WINDOWS):
            raise ValueError(f"first-contact rows missing window for seed {seed}")
        baseline_values = {float(row["zero_guess_mean_error_mm"]) for row in seed_rows}
        if len(baseline_values) != 1:
            raise ValueError(f"seed {seed} zero-guess baseline differs across predictor windows")
        zero_guess_by_seed[seed] = baseline_values.pop()
    return data, by_seed, first_contact, zero_guess_by_seed


def write_csvs(data, by_seed):
    metrics_path = OUTPUT_DIR / "diagnosis_table.csv"
    metric_fields = ["seed", "test_episodes", "test_seed", "successes", "contacts",
                     "guess_error_start_mm", "guess_error_first_contact_mm",
                     "guess_error_success_mm", "guess_error_end200_mm",
                     "paired_start_error_success_episodes_mm"]
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields, lineterminator="\n")
        writer.writeheader()
        for seed in SEEDS:
            source = by_seed[seed]
            writer.writerow({key: source.get(key) for key in metric_fields})

    prediction_path = OUTPUT_DIR / "prediction_table.csv"
    probes = data["prediction_probes"]
    fields = []
    for row in probes:
        for key in row:
            if key not in fields:
                fields.append(key)
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted(probes, key=lambda r: (r["seed"], r["window_steps"], r["moment"])):
            writer.writerow(row)
    return metrics_path, prediction_path


def add_seed_legend(ax):
    handles = [Line2D([], [], linestyle="None", marker=marker, markersize=6,
                      markerfacecolor=color, markeredgecolor="white",
                      label=f"Practice seed {seed}")
               for seed, marker, color in zip(SEEDS, SEED_MARKERS, SEED_COLORS)]
    handles.append(Line2D([], [], linestyle="None", marker="D", markersize=6,
                          markerfacecolor="black", markeredgecolor="black",
                          label="Across-seed mean ± sample SD"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.38),
              ncols=3, frameon=False)


def draw_group(ax, x, values, *, width=0.3):
    values = np.asarray(values, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("each chart group must contain five finite seed values")
    mean = float(values.mean())
    sample_sd = float(values.std(ddof=1))
    ax.errorbar(x, mean, yerr=sample_sd, fmt="D", color="black", capsize=4,
                markersize=6, zorder=4)
    offsets = np.linspace(-width, width, len(SEEDS))
    for seed_index, (seed, value) in enumerate(zip(SEEDS, values)):
        ax.scatter(x + offsets[seed_index], value, s=42,
                   marker=SEED_MARKERS[seed_index], color=SEED_COLORS[seed_index],
                   edgecolors="white", linewidths=0.5, zorder=5)
    return mean, sample_sd


def write_chart(by_seed, first_contact, zero_guess_by_seed):
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 7.5))

    left_x = np.arange(len(AIM_METRICS), dtype=float)
    for index, (key, _) in enumerate(AIM_METRICS):
        draw_group(left, left_x[index], [by_seed[seed][key] for seed in SEEDS])
    left.set_xticks(left_x, [label for _, label in AIM_METRICS], rotation=0, ha="center")
    left.set_ylabel("Mean L2 guess error (mm)")
    left.set_title("Aim guess error by event")
    left.grid(axis="y", alpha=0.25)

    right_labels = ["Zero guess\n(first-contact episodes)", "Window 1", "Window 20", "Window 40"]
    right_x = np.arange(len(right_labels), dtype=float)
    draw_group(right, right_x[0], [zero_guess_by_seed[seed] for seed in SEEDS])
    for index, window in enumerate(WINDOWS, start=1):
        values = []
        for seed in SEEDS:
            match = next(row for row in first_contact
                         if row["seed"] == seed and row["window_steps"] == window)
            values.append(match["prediction_mean_error_mm"])
        draw_group(right, right_x[index], values)
    right.set_xticks(right_x, right_labels)
    right.set_ylabel("Mean first-contact bias-prediction error (mm)")
    right.set_title("Prediction error by reading window")
    right.grid(axis="y", alpha=0.25)

    for ax in (left, right):
        add_seed_legend(ax)
    fig.suptitle("Phase A diagnostics, confirmation practice seeds 100–104")
    fig.subplots_adjust(bottom=0.40, top=0.88, wspace=0.32)
    fig.savefig(OUTPUT_DIR / "diagnosis.png", dpi=180)
    plt.close(fig)


def main():
    data, by_seed, first_contact, zero_guess_by_seed = load_and_validate()
    write_csvs(data, by_seed)
    write_chart(by_seed, first_contact, zero_guess_by_seed)


if __name__ == "__main__":
    main()
