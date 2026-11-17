#!/usr/bin/env python3
"""Build paired per-seed recovery tables and a plot for mixed camera error.

This reads registry summaries and test-only reference JSON. It does not load
experiment modules, inspect traces, run simulations, or write shared reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLLOWUP_ROOT = PROJECT_ROOT / "results" / "takehome" / "followup"
SPLITS = {
    (0.00065, 0.000375): "mostly_fixed",
    (0.00053, 0.00053): "even",
    (0.000375, 0.00065): "mostly_episode",
}
METHODS = ("aim", "heat", "movement", "combined")
METHOD_LABELS = {
    "aim": "Plain aim",
    "heat": "Across-episode heat map",
    "movement": "Plain movement",
    "combined": "Aim + movement",
}
OUTPUT_FILES = ("recovery.json", "recovery.csv", "recovery.png")


class InputError(ValueError):
    """Input data is missing or violates the predeclared reporting contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InputError(message)


def number(value: Any, where: str, *, low: float | None = None,
           high: float | None = None) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{where} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{where} must be finite")
    if low is not None:
        require(result >= low, f"{where} must be at least {low}")
    if high is not None:
        require(result <= high, f"{where} must be at most {high}")
    return result


def integer(value: Any, where: str, *, low: int | None = None) -> int:
    require(isinstance(value, int) and not isinstance(value, bool),
            f"{where} must be an integer")
    if low is not None:
        require(value >= low, f"{where} must be at least {low}")
    return value


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def sample_sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def method_for(entry: dict[str, Any]) -> str:
    name = entry.get("name")
    require(isinstance(name, str) and name, "mixed setting needs a name")
    method = name.rsplit("_", 1)[-1]
    require(method in METHODS, f"{name}: cannot identify one of {METHODS}")
    return method


def split_for(entry: dict[str, Any]) -> tuple[str, float, float, float]:
    overrides = entry.get("overrides")
    require(isinstance(overrides, dict), f"{entry.get('name')}: overrides must be an object")
    fixed = number(overrides.get("mixed_fixed_std"),
                   f"{entry['name']}.mixed_fixed_std", low=0.0)
    per_attempt = number(overrides.get("mixed_episode_std"),
                          f"{entry['name']}.mixed_episode_std", low=0.0)
    key = next((candidate for candidate in SPLITS
                if math.isclose(candidate[0], fixed, rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(candidate[1], per_attempt, rel_tol=0.0, abs_tol=1e-12)), None)
    require(key is not None,
            f"{entry['name']}: unapproved mixed split ({fixed}, {per_attempt})")
    split = SPLITS[key]
    fixed_variance_fraction = fixed ** 2 / (fixed ** 2 + per_attempt ** 2)
    require(entry.get("condition") == "mixed", f"{entry['name']}: condition must be mixed")
    return split, fixed, per_attempt, fixed_variance_fraction


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"invalid JSON in {label} {path}: {exc}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object: {path}")
    return value


def _reference_condition(reference: dict[str, Any], key: str,
                         path: Path) -> dict[str, Any]:
    conditions = reference.get("conditions")
    require(isinstance(conditions, dict), f"{path}: conditions must be an object")
    value = conditions.get(key)
    require(isinstance(value, dict), f"{path}: missing conditions.{key}")
    episodes = integer(value.get("episodes"), f"{path}: conditions.{key}.episodes", low=1)
    successes = integer(value.get("successes"), f"{path}: conditions.{key}.successes", low=0)
    require(episodes == 500, f"{path}: conditions.{key} must have exactly 500 episodes")
    require(successes <= episodes, f"{path}: conditions.{key}.successes exceeds episodes")
    rate = number(value.get("success_rate"), f"{path}: conditions.{key}.success_rate",
                  low=0.0, high=1.0)
    require(math.isclose(rate, successes / episodes, rel_tol=0.0, abs_tol=1e-12),
            f"{path}: conditions.{key} rate disagrees with successes/episodes")
    require(value.get("test_only") is True,
            f"{path}: conditions.{key}.test_only must be true")
    require(value.get("learned_from") is False,
            f"{path}: conditions.{key}.learned_from must be false")
    if "test_seed" in value:
        require(value["test_seed"] == 123, f"{path}: conditions.{key}.test_seed must be 123")
    return {"successes": successes, "episodes": episodes, "success_rate": rate}


def load_reference(path: Path, *, seed: int, fixed: float,
                   per_attempt: float) -> dict[str, Any]:
    reference = _load_json(path, "mixed reference")
    require(reference.get("practice_episodes") == 0,
            f"{path}: practice_episodes must be zero")
    require(reference.get("test_seed") == 123, f"{path}: test_seed must be 123")
    require(reference.get("test_only") is True, f"{path}: test_only must be true")
    require(reference.get("learned_from") is False, f"{path}: learned_from must be false")
    require(reference.get("episodes_per_condition") == 500,
            f"{path}: episodes_per_condition must be 500")
    require(reference.get("camera_condition") == "mixed",
            f"{path}: camera_condition must be mixed")
    require(reference.get("fixed_bias_seed") == seed,
            f"{path}: fixed_bias_seed must match seed {seed}")
    reference_fixed = number(reference.get("mixed_fixed_std"),
                             f"{path}: mixed_fixed_std", low=0.0)
    reference_per_attempt = number(reference.get("mixed_episode_std"),
                                   f"{path}: mixed_episode_std", low=0.0)
    require(math.isclose(reference_fixed, fixed, rel_tol=0.0, abs_tol=1e-12),
            f"{path}: mixed_fixed_std does not match setting")
    require(math.isclose(reference_per_attempt, per_attempt, rel_tol=0.0, abs_tol=1e-12),
            f"{path}: mixed_episode_std does not match setting")
    return {
        "path": path.relative_to(DEFAULT_FOLLOWUP_ROOT).as_posix()
        if path.is_relative_to(DEFAULT_FOLLOWUP_ROOT) else path.as_posix(),
        "floor": _reference_condition(reference, "normal", path),
        "ceiling": _reference_condition(reference, "no_camera_bias", path),
    }


def load_setting_summary(root: Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    summary_path = root / entry["phase"] / entry["name"] / "summary.json"
    summary = _load_json(summary_path, f"summary for {entry['name']}")
    rows = summary.get("per_seed")
    require(isinstance(rows, list), f"{summary_path}: per_seed must be a list")
    registry_seeds = entry.get("seeds")
    require(isinstance(registry_seeds, list), f"{entry['name']}: seeds must be a list")
    registry_seed_set = {integer(seed, f"{entry['name']}.seeds[]", low=0)
                         for seed in registry_seeds}
    require(len(registry_seed_set) == len(registry_seeds),
            f"{entry['name']}: duplicate registry seed")
    require(registry_seed_set in (set(range(3)), set(range(5))),
            f"{entry['name']}: final seeds must be exactly 0-2 or 0-4")
    require(len(rows) == len(registry_seed_set),
            f"{summary_path}: expected {len(registry_seed_set)} completed final seeds, found {len(rows)}")

    by_seed: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        where = f"{summary_path}: per_seed[{index}]"
        require(isinstance(row, dict), f"{where} must be an object")
        seed = integer(row.get("seed"), f"{where}.seed", low=0)
        require(seed not in by_seed, f"{summary_path}: duplicate result seed {seed}")
        require(row.get("test_episodes") == 500, f"{where}: test_episodes must be 500")
        require(row.get("test_seed") == 123, f"{where}: test_seed must be 123")
        require(row.get("test_only") is True, f"{where}: test_only must be true")
        require(row.get("learned_from") is False, f"{where}: learned_from must be false")
        successes = integer(row.get("successes"), f"{where}.successes", low=0)
        require(successes <= 500, f"{where}: successes exceeds 500")
        rate = number(row.get("success_percent"), f"{where}.success_percent", low=0.0, high=100.0) / 100.0
        require(math.isclose(rate, successes / 500, rel_tol=0.0, abs_tol=1e-12),
                f"{where}: success_percent disagrees with successes/500")
        by_seed[seed] = {"seed": seed, "successes": successes, "episodes": 500,
                         "success_rate": rate,
                         "result_path": row.get("result_path")}
    require(set(by_seed) == registry_seed_set,
            f"{summary_path}: result seeds do not match registry seeds")
    return [by_seed[seed] for seed in sorted(by_seed)]


def build_report(root: Path = DEFAULT_FOLLOWUP_ROOT) -> dict[str, Any]:
    """Read completed mixed settings and build serializable report data."""
    root = Path(root).resolve()
    registry_path = root / "registry.json"
    registry = _load_json(registry_path, "follow-up registry")
    entries = [entry for entry in registry.get("settings", [])
               if isinstance(entry, dict) and entry.get("phase") == "B_mixed"]
    require(len(entries) == 12, f"expected 12 B_mixed settings, found {len(entries)}")

    seen_names: set[str] = set()
    per_setting: dict[tuple[str, str], dict[str, Any]] = {}
    reference_cache: dict[tuple[str, float, float, int], dict[str, Any]] = {}
    reference_key_fn = _entry_condition_key()
    for entry in entries:
        name = entry.get("name")
        require(isinstance(name, str) and name not in seen_names,
                f"missing or duplicate B_mixed setting name: {name!r}")
        seen_names.add(name)
        require(entry.get("status") == "finished",
                f"{name}: setting status must be finished before reporting")
        split, fixed, per_attempt, fixed_variance_fraction = split_for(entry)
        method = method_for(entry)
        key = (split, method)
        require(key not in per_setting, f"duplicate setting for split/method {key}")
        seed_rows = load_setting_summary(root, entry)
        method_rows = []
        for seed_row in seed_rows:
            seed = seed_row["seed"]
            reference_identity = (split, fixed, per_attempt, seed)
            if reference_identity not in reference_cache:
                # The entry-key helper is the source of the canonical split name.
                condition_key = reference_key_fn(entry)
                path = root / "references" / f"{condition_key}_seed_{seed}_test_123.json"
                reference_cache[reference_identity] = load_reference(
                    path, seed=seed, fixed=fixed, per_attempt=per_attempt)
            reference = reference_cache[reference_identity]
            floor = reference["floor"]
            ceiling = reference["ceiling"]
            method_rate = seed_row["success_rate"]
            gap = ceiling["success_rate"] - floor["success_rate"]
            if gap <= 0.0:
                recovery_fraction = None
                recovery_reason = "ceiling_not_above_floor"
            else:
                recovery_fraction = (method_rate - floor["success_rate"]) / gap
                recovery_reason = None
            method_rows.append({
                "method": method,
                "method_label": METHODS_LABEL(method),
                "split": split,
                "fixed_std_m": fixed,
                "per_attempt_std_m": per_attempt,
                "fixed_variance_fraction": fixed_variance_fraction,
                "seed": seed,
                "successes": seed_row["successes"],
                "episodes": 500,
                "success_rate": method_rate,
                "floor_successes": floor["successes"],
                "floor_episodes": floor["episodes"],
                "floor_rate": floor["success_rate"],
                "ceiling_successes": ceiling["successes"],
                "ceiling_episodes": ceiling["episodes"],
                "ceiling_rate": ceiling["success_rate"],
                "gain_pp": 100.0 * (method_rate - floor["success_rate"]),
                "remaining_gap_pp": 100.0 * (ceiling["success_rate"] - method_rate),
                "recovery_fraction": recovery_fraction,
                "recovery_reason": recovery_reason,
                "result_path": seed_row.get("result_path"),
                "reference_path": reference["path"],
            })
        per_setting[key] = {
            "name": name,
            "method": method,
            "method_label": METHODS_LABEL(method),
            "split": split,
            "fixed_std_m": fixed,
            "per_attempt_std_m": per_attempt,
            "fixed_variance_fraction": fixed_variance_fraction,
            "seeds": method_rows,
        }

    ordered_splits = sorted(
        ((split_name, fixed, per_attempt,
          fixed ** 2 / (fixed ** 2 + per_attempt ** 2))
         for (fixed, per_attempt), split_name in SPLITS.items()),
        key=lambda item: item[3])
    expected_keys = {(split_name, method) for split_name, _, _, _ in ordered_splits
                     for method in METHODS}
    require(set(per_setting) == expected_keys,
            "B_mixed registry must contain each of four methods exactly once for each split")
    aggregates = []
    flattened = []
    for split, fixed, per_attempt, fixed_fraction in ordered_splits:
        for method in METHODS:
            setting = per_setting[(split, method)]
            rows = setting["seeds"]
            rates = [row["success_rate"] for row in rows]
            floors = [row["floor_rate"] for row in rows]
            ceilings = [row["ceiling_rate"] for row in rows]
            individual_ratios = [row["recovery_fraction"] for row in rows
                                 if row["recovery_fraction"] is not None]
            floor_mean = statistics.mean(floors)
            ceiling_mean = statistics.mean(ceilings)
            method_mean = statistics.mean(rates)
            aggregate_gap = ceiling_mean - floor_mean
            if aggregate_gap <= 0.0:
                primary_recovery = None
                primary_reason = "mean_ceiling_not_above_mean_floor"
            else:
                primary_recovery = (method_mean - floor_mean) / aggregate_gap
                primary_reason = None
            aggregates.append({
                "method": method,
                "method_label": METHODS_LABEL(method),
                "split": split,
                "fixed_std_m": fixed,
                "per_attempt_std_m": per_attempt,
                "fixed_variance_fraction": fixed_fraction,
                "seed_count": len(rows),
                "success_mean": method_mean,
                "success_sample_sd": sample_sd(rates),
                "floor_mean": floor_mean,
                "ceiling_mean": ceiling_mean,
                "primary_recovery_fraction": primary_recovery,
                "primary_recovery_reason": primary_reason,
                "individual_recovery_fraction_mean": mean(individual_ratios),
                "individual_recovery_fraction_sample_sd": sample_sd(individual_ratios),
                "individual_recovery_fraction_count": len(individual_ratios),
                "reference_path_by_seed": {str(row["seed"]): row["reference_path"] for row in rows},
            })
            flattened.extend(rows)

    return {
        "schema_version": 1,
        "source_registry": "registry.json",
        "source_directory": "B_mixed",
        "test_seed": 123,
        "test_episodes_per_seed": 500,
        "seed_sets": sorted({tuple(sorted(row["seed"] for row in setting["seeds"]))
                              for setting in per_setting.values()}),
        "fraction_definition": "(method mean rate - floor mean rate) / (zero-error ceiling mean rate - floor mean rate)",
        "individual_fraction_definition": "(method seed rate - floor seed rate) / (zero-error ceiling seed rate - floor seed rate); undefined when denominator <= 0",
        "chart_axis_labels": {
            "bottom": "Calibration component (fixed-across-seed variance fraction)",
            "top": "Per-attempt component (variance fraction)",
        },
        "settings": [per_setting[(split, method)]
                     for split, _, _, _ in ordered_splits
                     for method in METHODS],
        "aggregates": aggregates,
        "seed_rows": flattened,
    }


def _entry_condition_key():
    # Import only the stable naming helper. followup_runtime sets an isolated
    # cache path and does not import JAX, models, or simulation code.
    sys.path.insert(0, str(PROJECT_ROOT))
    from takehome.followup_paths import entry_condition_key
    return entry_condition_key


def METHODS_LABEL(method: str) -> str:
    return METHOD_LABELS[method]


CSV_FIELDS = (
    "row_type", "method", "method_label", "split", "fixed_std_m", "per_attempt_std_m",
    "fixed_variance_fraction", "seed", "successes", "episodes", "success_rate",
    "floor_successes", "floor_episodes", "floor_rate", "ceiling_successes",
    "ceiling_episodes", "ceiling_rate", "gain_pp", "remaining_gap_pp",
    "recovery_fraction", "recovery_reason", "success_mean", "success_sample_sd",
    "floor_mean", "ceiling_mean", "primary_recovery_fraction", "primary_recovery_reason",
    "individual_recovery_fraction_mean", "individual_recovery_fraction_sample_sd",
    "individual_recovery_fraction_count",
)


def render_csv(report: dict[str, Any]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for row in report["seed_rows"]:
        writer.writerow({"row_type": "seed", **row})
    for row in report["aggregates"]:
        writer.writerow({"row_type": "aggregate", **row})
    return stream.getvalue()


def plot_report(report: dict[str, Any], output_path: Path) -> None:
    followup_root = output_path.parent.parent
    os.environ.setdefault("MPLCONFIGDIR", str(followup_root / "cache" / "mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate_index = {(row["split"], row["method"]): row for row in report["aggregates"]}
    setting_index = {(row["split"], row["method"]): row for row in report["settings"]}
    split_rows = sorted(
        ((split, fixed, per_attempt,
          fixed ** 2 / (fixed ** 2 + per_attempt ** 2))
         for (fixed, per_attempt), split in SPLITS.items()),
        key=lambda row: row[3])
    x_values = [row[3] for row in split_rows]
    split_names = [row[0] for row in split_rows]
    method_offsets = dict(zip(METHODS, (-0.045, -0.015, 0.015, 0.045)))
    colors = dict(zip(METHODS, ("tab:blue", "tab:orange", "tab:green", "tab:red")))

    fig, ax = plt.subplots(figsize=(11, 6.4))
    for method in METHODS:
        xs, ys, errors = [], [], []
        for x, split in zip(x_values, split_names):
            row = aggregate_index[(split, method)]
            # The chart compares methods on the common screening seeds only;
            # five-seed extensions remain fully represented in tables/JSON.
            seed_rows = [item for item in setting_index[(split, method)]["seeds"]
                         if item["seed"] in (0, 1, 2)]
            require({item["seed"] for item in seed_rows} == {0, 1, 2},
                    f"{split}/{method}: chart requires shared screening seeds 0-2")
            screening_rates = [item["success_rate"] for item in seed_rows]
            x_method = x + method_offsets[method]
            xs.append(x_method)
            ys.append(100.0 * statistics.mean(screening_rates))
            errors.append(100.0 * sample_sd(screening_rates))
            count = len(seed_rows)
            jitter = [0.0] if count == 1 else [
                -0.009 + 0.018 * i / (count - 1) for i in range(count)
            ]
            ax.scatter([x_method + delta for delta in jitter],
                       [100.0 * item["success_rate"] for item in seed_rows],
                       color=colors[method], alpha=0.62, s=28, zorder=3)
        ax.errorbar(xs, ys, yerr=errors, color=colors[method], marker="o",
                    linewidth=1.8, capsize=4, label=f"{METHODS_LABEL(method)} mean ± sample SD",
                    zorder=4)

    floor_means = []
    ceiling_means = []
    for split in split_names:
        seed_rows = [item for item in setting_index[(split, "aim")]["seeds"]
                     if item["seed"] in (0, 1, 2)]
        require({item["seed"] for item in seed_rows} == {0, 1, 2},
                f"{split}: reference chart requires shared screening seeds 0-2")
        floor_means.append(100.0 * statistics.mean(row["floor_rate"] for row in seed_rows))
        ceiling_means.append(100.0 * statistics.mean(row["ceiling_rate"] for row in seed_rows))
    ax.plot(x_values, floor_means, color="black", linestyle="--", marker="s",
            label="Mixed-condition floor mean", zorder=2)
    ax.plot(x_values, ceiling_means, color="dimgray", linestyle=":", marker="^",
            label="Zero-error ceiling mean", zorder=2)
    for x, split in zip(x_values, split_names):
        seed_rows = [item for item in setting_index[(split, "aim")]["seeds"]
                     if item["seed"] in (0, 1, 2)]
        ax.scatter([x] * len(seed_rows), [100.0 * row["floor_rate"] for row in seed_rows],
                   marker="s", facecolors="none", edgecolors="black", s=28, zorder=2)
        ax.scatter([x] * len(seed_rows), [100.0 * row["ceiling_rate"] for row in seed_rows],
                   marker="^", facecolors="none", edgecolors="dimgray", s=30, zorder=2)

    ax.set_xticks(x_values, [f"{name.replace('_', ' ')}\n{100*x:.0f}% fixed variance"
                              for name, x in zip(split_names, x_values)])
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Calibration component (fixed-across-seed variance fraction)", labelpad=12)
    ax.set_title("Mixed-condition results by fixed/per-attempt variance split\n"
                 "Shared screening seeds 0–2; test seed 123, 500 episodes per seed")
    ax.grid(axis="y", alpha=0.25)
    top = ax.secondary_xaxis("top", functions=(lambda x: 1.0 - x,
                                                lambda x: 1.0 - x))
    top.set_xlabel("Per-attempt component (variance fraction)", labelpad=9)
    top.set_xticks(x_values, [f"{100*(1-x):.0f}%" for x in x_values])
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.11, right=0.73, bottom=0.19, top=0.82)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def write_outputs(report: dict[str, Any], root: Path) -> tuple[Path, Path, Path]:
    output_dir = root / "B_mixed"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "recovery.json"
    csv_path = output_dir / "recovery.csv"
    png_path = output_dir / "recovery.png"
    _atomic_text(json_path, json.dumps(report, indent=2, allow_nan=False) + "\n")
    _atomic_text(csv_path, render_csv(report))
    plot_report(report, png_path)
    return json_path, csv_path, png_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_FOLLOWUP_ROOT,
                        help="follow-up results root containing registry.json")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.root)
        for path in write_outputs(report, Path(args.root)):
            print(path)
    except (OSError, InputError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
