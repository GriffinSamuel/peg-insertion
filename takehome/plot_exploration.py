#!/usr/bin/env python3
"""Plot every Goal 7 exploration result from a prepared JSON summary."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "takehome"
DEFAULT_INPUT = RESULTS_ROOT / "goal7_summary.json"
DEFAULT_OUTPUT = RESULTS_ROOT / "goal7"


class InputError(Exception):
    """An input file exists but contains invalid or inconsistent data."""


def require(condition, message):
    if not condition:
        raise InputError(message)


def finite_number(value, field, location):
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{location}: {field} must be numeric")
    value = float(value)
    require(math.isfinite(value), f"{location}: {field} must be finite")
    return value


def normalized_text(value):
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def category_for(setting):
    name = normalized_text(setting.get("name"))
    label = normalized_text(setting.get("label"))
    round_name = normalized_text(setting.get("round"))
    combined = f"{name} {label} {round_name}"
    if "confirm" in round_name:
        return "CONFIRMATION"
    if setting.get('changes_from_main') == 0 or "main" in name or "goal5" in name or round_name in {"main", "goal5", "goal 5"}:
        return "MAIN"
    if any(word in combined for word in ("reused", "reuse", "plain", "comparison")):
        return "REUSED / PLAIN COMPARISON"
    if round_name:
        return f"TUNED {round_name.upper()}"
    return "TUNED"


def display_label(setting):
    category = category_for(setting)
    label = str(setting["label"])
    base = setting.get("base")
    base_text = f" [base: {base}]" if base not in (None, "") else ""
    return f"[{category}] {label}{base_text}"


def serialize_cell(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    if value is None:
        return ""
    return value


def load_summary(input_path):
    if not input_path.is_file():
        print(f"Goal 7 summary is not available yet: {input_path}; no files written.",
              file=sys.stderr)
        return None
    try:
        summary = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"Invalid JSON in {input_path}: {exc}") from exc

    settings = summary.get("settings")
    require(isinstance(settings, list), "summary.settings must be a list")
    require(settings, "summary.settings must contain settings")
    floor = finite_number(summary.get("floor"), "floor", "summary")
    ceiling = finite_number(summary.get("ceiling"), "ceiling", "summary")
    require(0.0 <= floor <= 1.0 and 0.0 <= ceiling <= 1.0,
            "floor and ceiling must be success rates from 0 through 1")
    total_settings = summary.get("total_settings_tried")
    require(isinstance(total_settings, int) and not isinstance(total_settings, bool)
            and total_settings >= 0, "total_settings_tried must be a nonnegative integer")

    all_rows = []
    seen_pairs = set()
    validated = []
    for setting_index, setting in enumerate(settings):
        location = f"settings[{setting_index}]"
        require(isinstance(setting, dict), f"{location} must be an object")
        name = setting.get("name")
        label = setting.get("label")
        require(isinstance(name, str) and name, f"{location}.name must be nonempty")
        require(isinstance(label, str) and label, f"{location}.label must be nonempty")
        require("round" in setting and "base" in setting and "changes_from_main" in setting,
                f"{location} lacks round/base/changes_from_main metadata")
        require(isinstance(setting.get("clear_gain_over_base"), bool),
                f"{location}.clear_gain_over_base must be boolean")
        mean = finite_number(setting.get("mean"), "mean", location)
        spread = finite_number(setting.get("std"), "std", location)
        require(0.0 <= mean <= 1.0 and spread >= 0.0,
                f"{location}: mean must be a rate and std must be nonnegative")
        results = setting.get("results")
        require(isinstance(results, list) and results,
                f"{location}.results must be a nonempty list")

        seed_results = []
        seen_seeds = set()
        computed_rates = []
        for result_index, result in enumerate(results):
            result_location = f"{location}.results[{result_index}]"
            require(isinstance(result, dict), f"{result_location} must be an object")
            seed = result.get("seed")
            require(isinstance(seed, int) and not isinstance(seed, bool),
                    f"{result_location}.seed must be an integer")
            pair = (name, seed)
            require(pair not in seen_pairs, f"duplicate setting/seed pair: {name}/{seed}")
            seen_pairs.add(pair)
            require(seed not in seen_seeds, f"duplicate seed {seed} in setting {name}")
            seen_seeds.add(seed)

            episodes = result.get("episodes")
            successes = result.get("successes")
            require(isinstance(episodes, int) and not isinstance(episodes, bool) and episodes == 500,
                    f"{result_location}.episodes must be 500")
            require(isinstance(successes, int) and not isinstance(successes, bool)
                    and 0 <= successes <= episodes,
                    f"{result_location}.successes must be an integer in [0, 500]")
            rate = finite_number(result.get("success_rate"), "success_rate", result_location)
            peak_force = finite_number(result.get("peak_force_mean"),
                                       "peak_force_mean", result_location)
            require(0.0 <= rate <= 1.0, f"{result_location}.success_rate must be from 0 through 1")
            result_path = result.get("result_path")
            require(isinstance(result_path, str) and result_path,
                    f"{result_location}.result_path must be nonempty")
            row = {
                "name": name,
                "label": display_label(setting),
                "round": setting["round"],
                "base": setting["base"],
                "category": category_for(setting),
                "changes_from_main": setting["changes_from_main"],
                "clear_gain_over_base": setting["clear_gain_over_base"],
                "difference_status": setting.get('difference_status', 'See mean and spread'),
                "setting_mean": mean,
                "setting_std": spread,
                "seed": seed,
                "successes": successes,
                "episodes": episodes,
                "success_rate": rate,
                "peak_force_mean": peak_force,
                "result_path": result_path,
            }
            seed_results.append(row)
            all_rows.append(row)
            computed_rates.append(rate)

        # The setting-level values are retained as provided. The result-level rows
        # remain the plotted dots, so no run is replaced by a summary statistic.
        validated.append({
            "name": name,
            "label": display_label(setting),
            "round": setting["round"],
            "base": setting["base"],
            "category": category_for(setting),
            "changes_from_main": setting["changes_from_main"],
            "clear_gain_over_base": setting["clear_gain_over_base"],
            "mean": mean,
            "std": spread,
            "results": seed_results,
        })

    require(len(all_rows) == sum(len(s["results"]) for s in validated),
            "internal result-row count mismatch")
    return {
        "settings": validated,
        "rows": all_rows,
        "floor": floor,
        "ceiling": ceiling,
        "total_settings_tried": total_settings,
    }


def write_csvs(data, output_dir):
    exploration_path = output_dir / "exploration.csv"
    force_path = output_dir / "force.csv"
    fields = ["name", "label", "category", "round", "base", "changes_from_main",
              "clear_gain_over_base", "difference_status", "setting_mean", "setting_std", "seed", "successes",
              "episodes", "success_rate", "peak_force_mean", "result_path"]
    with exploration_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in data["rows"]:
            writer.writerow({key: serialize_cell(row[key]) for key in fields})

    force_fields = ["name", "label", "category", "round", "base", "seed",
                    "peak_force_mean", "successes", "episodes", "success_rate",
                    "result_path"]
    with force_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=force_fields, lineterminator="\n")
        writer.writeheader()
        for row in data["rows"]:
            writer.writerow({key: serialize_cell(row[key]) for key in force_fields})


def write_plot(data, input_path, output_path):
    mpl_config = RESULTS_ROOT / "mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = data["settings"]
    all_seeds = sorted({row["seed"] for row in data["rows"]})
    palette = plt.get_cmap("tab20")
    seed_colors = {seed: palette(i % 20) for i, seed in enumerate(all_seeds)}
    setting_count = len(settings)
    result_count = len(data["rows"])
    total_test_episodes = sum(row["episodes"] for row in data["rows"])
    height = max(6.0, 0.32 * setting_count + 2.5)
    fig, ax = plt.subplots(figsize=(12, height))

    # Keep the source setting order; no ranking or selection is performed.
    for y, setting in enumerate(settings):
        count = len(setting["results"])
        offsets = [0.0] if count == 1 else [(-0.22 + 0.44 * i / (count - 1)) for i in range(count)]
        for row, offset in zip(setting["results"], offsets):
            ax.scatter(row["success_rate"] * 100.0, y + offset,
                       color=seed_colors[row["seed"]], s=34, alpha=0.88,
                       edgecolors="black", linewidths=0.25, zorder=3)
        ax.errorbar(setting["mean"] * 100.0, y,
                    xerr=setting["std"] * 100.0,
                    fmt="D", color="black", capsize=3, markersize=5.5,
                    linewidth=1.4, zorder=4)

    ax.axvline(data["floor"] * 100.0, color="tab:red", linestyle="--",
               linewidth=1.5, label=f"Original-policy floor ({data['floor']:.1%})")
    ax.axvline(data["ceiling"] * 100.0, color="tab:green", linestyle=":",
               linewidth=1.7, label=f"No-bias ceiling ({data['ceiling']:.1%})")
    ax.set_yticks(range(setting_count))
    ax.set_yticklabels([setting["label"] for setting in settings], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-2, 102)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Development-test success rate (%)")
    ax.set_title("Goal 7 exploration results", pad=28)
    ax.text(0.5, 1.015,
            f"Settings: {setting_count}; result rows: {result_count}; test episodes: {total_test_episodes}; "
            "development test 123; final point uses 200 practice episodes",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
    ax.text(0.5, -0.065,
            "Colored dots: individual practice seeds. Black diamonds and bars: supplied setting mean and spread.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8)
    ax.grid(axis="x", alpha=0.23)

    seed_handles = [plt.Line2D([], [], marker="o", linestyle="None", markersize=5,
                               color=seed_colors[seed], label=f"Seed {seed}")
                    for seed in all_seeds]
    mean_handle = plt.Line2D([], [], marker="D", linestyle="None", markersize=5,
                             color="black", label="Setting mean / spread")
    baseline_handles, baseline_labels = ax.get_legend_handles_labels()
    handles = seed_handles + [mean_handle] + baseline_handles
    labels = [h.get_label() for h in seed_handles] + [mean_handle.get_label()] + baseline_labels
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.13),
              ncol=min(6, max(2, len(handles))), fontsize=8, frameon=True)
    fig.subplots_adjust(left=0.34, right=0.98, top=0.94, bottom=0.17)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="JSON summary with all Goal 7 settings and results")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="directory for exploration.csv, exploration.png and force.csv")
    args = parser.parse_args(argv)
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output

    try:
        try:
            output_dir.resolve().relative_to(RESULTS_ROOT.resolve())
        except ValueError as exc:
            raise InputError("Output directory must be inside results/takehome/") from exc
        data = load_summary(input_path)
        if data is None:
            return 0
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csvs(data, output_dir)
        write_plot(data, input_path, output_dir / "exploration.png")
    except (InputError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Goal 7 plotting failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
