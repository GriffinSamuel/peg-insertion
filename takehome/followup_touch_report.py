#!/usr/bin/env python3
"""Render the predeclared no-learning Phase F hand-search comparisons.

This script reads only the Phase F method manifest, method result JSON, and the
preserved original-policy baseline. It does not read traces or run simulations.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Bootstrap the follow-up cache locations before importing plotting/numeric libs.
from takehome import followup_runtime  # noqa: E402,F401

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import textwrap  # noqa: E402
from typing import Any  # noqa: E402


DEFAULT_ROOT = followup_runtime.OUT
METHODS = ("blind_spiral", "twist_direction", "four_sided_probe")
METHOD_LABELS = {
    "blind_spiral": "Blind spiral",
    "twist_direction": "Twist direction",
    "four_sided_probe": "Four-sided probe",
}
CSV_FIELDS = (
    "row_type", "method", "label", "status", "reason", "successes", "episodes",
    "success_percent", "aim_error_start_mm", "aim_error_at_success_mm",
    "aim_error_final_mm", "successful_insertion_steps_mean", "peak_force_mean_n",
    "timeout_episodes", "timeout_fraction", "practice_episodes", "hand_search",
    "hidden_information", "no_learning", "test_only", "learned_from",
    "frozen_policy_unchanged", "test_data_learned_from", "test_seed", "result_path",
)


class InputError(ValueError):
    """The manifest or saved result does not match the reporting contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InputError(message)


def finite_number(value: Any, where: str, *, low: float | None = None,
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


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"invalid {label} JSON at {path}: {exc}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object: {path}")
    return value


def _reference_row(name: str, condition: dict[str, Any]) -> dict[str, Any]:
    episodes = integer(condition.get("episodes"), f"baseline.{name}.episodes", low=1)
    successes = integer(condition.get("successes"), f"baseline.{name}.successes", low=0)
    require(episodes == 500, f"baseline.{name} must have 500 test episodes")
    require(successes <= episodes, f"baseline.{name}.successes exceeds episodes")
    rate = finite_number(condition.get("success_rate"),
                         f"baseline.{name}.success_rate", low=0.0, high=1.0)
    require(math.isclose(rate, successes / episodes, rel_tol=0.0, abs_tol=1e-12),
            f"baseline.{name} rate disagrees with successes/episodes")
    require(condition.get("test_only") is True,
            f"baseline.{name}.test_only must be true")
    require(condition.get("learned_from") is False,
            f"baseline.{name}.learned_from must be false")
    require(condition.get("test_seed") == 123,
            f"baseline.{name}.test_seed must be 123")
    return {"name": name, "successes": successes, "episodes": episodes,
            "success_rate": rate, "test_seed": 123,
            "test_only": True, "learned_from": False}


def load_references(root: Path) -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "results" / "takehome" / "baseline.json"
    baseline = read_json(path, "preserved baseline")
    require(baseline.get("test_seed") == 123, "baseline test_seed must be 123")
    require(baseline.get("episodes_per_condition") == 500,
            "baseline episodes_per_condition must be 500")
    require(baseline.get("practice_episodes") == 0,
            "baseline practice_episodes must be zero")
    conditions = baseline.get("conditions")
    require(isinstance(conditions, dict), "baseline.conditions must be an object")
    result = []
    for name in ("normal", "no_camera_bias"):
        require(isinstance(conditions.get(name), dict),
                f"baseline.conditions.{name} is missing")
        result.append(_reference_row(name, conditions[name]))
    normal_bias = finite_number(conditions["normal"].get("obs_hole_bias_std"),
                                "baseline.normal.obs_hole_bias_std", low=0.0)
    zero_bias = finite_number(conditions["no_camera_bias"].get("obs_hole_bias_std"),
                              "baseline.no_camera_bias.obs_hole_bias_std", low=0.0)
    require(normal_bias > 0,
            "baseline.normal must retain camera bias")
    require(zero_bias == 0,
            "baseline.no_camera_bias must be the zero-camera-error reference")
    return result


def _finished_method(root: Path, method: str, path: Path) -> dict[str, Any]:
    result = read_json(path, f"{method} result")
    require(result.get("method") == method,
            f"{path}: result method does not match manifest method {method}")
    final = result.get("final")
    metrics = result.get("metrics")
    require(isinstance(final, dict), f"{path}: final must be an object")
    require(isinstance(metrics, dict), f"{path}: metrics must be an object")

    episodes = integer(final.get("episodes"), f"{path}: final.episodes", low=1)
    successes = integer(final.get("successes"), f"{path}: final.successes", low=0)
    require(episodes == 500, f"{path}: final must have 500 test episodes")
    require(successes <= episodes, f"{path}: final.successes exceeds episodes")
    success_rate = finite_number(final.get("success_rate"),
                                 f"{path}: final.success_rate", low=0.0, high=1.0)
    require(math.isclose(success_rate, successes / episodes,
                         rel_tol=0.0, abs_tol=1e-12),
            f"{path}: final success rate disagrees with successes/episodes")
    require(final.get("test_seed") == 123, f"{path}: final.test_seed must be 123")
    require(final.get("test_only") is True, f"{path}: final.test_only must be true")
    require(final.get("learned_from") is False,
            f"{path}: final.learned_from must be false")

    practice = integer(result.get("practice_episodes"),
                       f"{path}: practice_episodes", low=0)
    require(practice == 0, f"{path}: no-learning comparison must use zero practice episodes")
    require(result.get("no_learning") is True, f"{path}: no_learning must be true")
    require(result.get("test_data_learned_from") is False,
            f"{path}: test_data_learned_from must be false")
    require(result.get("frozen_policy_unchanged") is True,
            f"{path}: frozen_policy_unchanged must be true")
    require(result.get("hand_search") is True, f"{path}: hand_search must be true")
    require(result.get("hidden_information") is False,
            f"{path}: hidden_information must be false")

    require(metrics.get("test_episodes") == 500,
            f"{path}: metrics.test_episodes must be 500")
    require(metrics.get("test_seed") == 123, f"{path}: metrics.test_seed must be 123")
    require(metrics.get("test_only") is True, f"{path}: metrics.test_only must be true")
    require(metrics.get("learned_from") is False,
            f"{path}: metrics.learned_from must be false")
    require(metrics.get("successes") == successes,
            f"{path}: metric successes disagree with final")

    timeout_count = integer(result.get("timeout_episodes"),
                            f"{path}: timeout_episodes", low=0)
    require(timeout_count <= 500, f"{path}: timeout_episodes exceeds 500")
    rate = finite_number(metrics.get("success_percent"),
                         f"{path}: metrics.success_percent", low=0.0, high=100.0)
    require(math.isclose(rate, 100.0 * successes / 500,
                         rel_tol=0.0, abs_tol=1e-10),
            f"{path}: metrics success percent disagrees with final")
    start_error = metrics.get("initial_guess_error_mean_mm")
    success_error = metrics.get("guess_error_at_first_valid_success_mean_mm")
    steps = metrics.get("steps_mean")
    if successes:
        start_error = finite_number(start_error, f"{path}: start aim error", low=0.0)
        success_error = finite_number(success_error, f"{path}: success aim error", low=0.0)
    else:
        require(start_error is not None,
                f"{path}: start aim error must be present even with zero successes")
        start_error = finite_number(start_error, f"{path}: start aim error", low=0.0)
        require(success_error is None,
                f"{path}: success aim error must be null with zero successes")
    if steps is not None:
        steps = finite_number(steps, f"{path}: successful insertion steps", low=0.0)
    else:
        require(successes == 0,
                f"{path}: insertion steps may be null only with zero successes")
    peak_force = finite_number(metrics.get("peak_force_mean"),
                               f"{path}: peak_force_mean", low=0.0)
    final_error = finite_number(result.get("aim_final_error_mm"),
                                f"{path}: aim_final_error_mm", low=0.0)

    return {
        "method": method,
        "label": METHOD_LABELS[method],
        "status": "finished",
        "reason": None,
        "successes": successes,
        "episodes": 500,
        "success_percent": 100.0 * successes / 500,
        "aim_error_start_mm": start_error,
        "aim_error_at_success_mm": success_error,
        "aim_error_final_mm": final_error,
        "successful_insertion_steps_mean": steps,
        "peak_force_mean_n": peak_force,
        "timeout_episodes": timeout_count,
        "timeout_fraction": timeout_count / 500,
        "practice_episodes": 0,
        "hand_search": True,
        "hidden_information": False,
        "no_learning": True,
        "test_only": True,
        "learned_from": False,
        "frozen_policy_unchanged": True,
        "test_data_learned_from": False,
        "test_seed": 123,
        "result_path": path.relative_to(root).as_posix(),
    }


def build_report(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = root / "F" / "hand_manifest.json"
    manifest = read_json(manifest_path, "hand-search manifest")
    methods = manifest.get("methods")
    require(isinstance(methods, list), "manifest.methods must be a list")
    require(len(methods) == len(METHODS),
            f"manifest must contain exactly {len(METHODS)} requested methods")
    names = [row.get("name") for row in methods if isinstance(row, dict)]
    require(len(names) == len(methods), "each manifest method must be an object")
    require(len(set(names)) == len(names), "manifest contains duplicate method names")
    require(set(names) == set(METHODS),
            f"manifest methods must be exactly {list(METHODS)}")

    rows = []
    for entry in methods:
        name = entry["name"]
        status = entry.get("status")
        require(status in ("finished", "skipped", "failed"),
                f"{name}: status must be finished, skipped or failed")
        result_path = root / "F" / f"hand_{name}" / "result.json"
        if status in ("skipped", "failed"):
            reason = entry.get("reason")
            require(isinstance(reason, str) and reason.strip(),
                    f"{name}: skipped method needs a reason")
            rows.append({
                "method": name,
                "label": METHOD_LABELS[name],
                "status": status,
                "reason": reason,
                "successes": None,
                "episodes": None,
                "success_percent": None,
                "aim_error_start_mm": None,
                "aim_error_at_success_mm": None,
                "aim_error_final_mm": None,
                "successful_insertion_steps_mean": None,
                "peak_force_mean_n": None,
                "timeout_episodes": None,
                "timeout_fraction": None,
                "practice_episodes": 0,
                "hand_search": True,
                "hidden_information": False,
                "no_learning": None,
                "test_only": None,
                "learned_from": None,
                "frozen_policy_unchanged": None,
                "test_data_learned_from": None,
                "test_seed": None,
                "result_path": None,
            })
            continue
        require(not entry.get("reason"),
                f"{name}: finished method should not carry a skip reason")
        rows.append(_finished_method(root, name, result_path))

    references = load_references(root)
    return {
        "schema_version": 1,
        "source_manifest": "F/hand_manifest.json",
        "source_baseline": "../baseline.json",
        "test_seed": 123,
        "test_episodes": 500,
        "comparison_label": "Hand-written search; no learning; test-only comparison",
        "references": references,
        "methods": rows,
    }


def render_csv(report: dict[str, Any]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for row in report["methods"]:
        writer.writerow({"row_type": "method", **row})
    for reference in report["references"]:
        label = ("Original policy floor" if reference["name"] == "normal"
                 else "Zero-camera-error reference")
        writer.writerow({
            "row_type": "reference",
            "method": reference["name"],
            "label": label,
            "status": "reference",
            "successes": reference["successes"],
            "episodes": reference["episodes"],
            "success_percent": 100.0 * reference["success_rate"],
            "practice_episodes": 0,
            "hand_search": False,
            "hidden_information": False,
            "test_only": True,
            "learned_from": False,
            "test_seed": reference["test_seed"],
        })
    return stream.getvalue()


def plot_report(report: dict[str, Any], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["methods"]
    references = {item["name"]: item for item in report["references"]}
    y_positions = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    colors = {"finished": "tab:blue", "skipped": "0.55"}
    for y, row in zip(y_positions, rows):
        if row["status"] == "finished":
            ax.scatter(row["success_percent"], y, color=colors["finished"],
                       s=70, zorder=3)
        else:
            ax.text(0.02, y, textwrap.fill(f"{row['status'].capitalize()}: {row['reason']}", 75),
                    transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    floor = references["normal"]["success_rate"] * 100.0
    ceiling = references["no_camera_bias"]["success_rate"] * 100.0
    ax.axvline(floor, color="black", linestyle="--", linewidth=1.4,
               label=f"Original policy floor ({floor:.1f}%)")
    ax.axvline(ceiling, color="dimgray", linestyle=":", linewidth=1.6,
               label=f"Zero-camera-error reference ({ceiling:.1f}%)")
    ax.set_yticks(y_positions, [row["label"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Success rate (%)")
    ax.set_title("Hand-written search methods\nNo learning; 500 test episodes at seed 123")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.27),
              ncol=1, frameon=False)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.83, bottom=0.27)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=175)
    plt.close(fig)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    temporary.replace(path)


def write_outputs(report: dict[str, Any], root: Path) -> tuple[Path, Path, Path]:
    root = Path(root).resolve()
    require(root.is_relative_to(DEFAULT_ROOT.resolve()), "Outputs must stay inside the follow-up directory")
    output_dir = root / "F"
    json_path = output_dir / "hand_comparison.json"
    csv_path = output_dir / "hand_comparison.csv"
    png_path = output_dir / "hand_comparison.png"
    _atomic_write(json_path, json.dumps(report, indent=2, allow_nan=False) + "\n")
    _atomic_write(csv_path, render_csv(report))
    plot_report(report, png_path)
    return json_path, csv_path, png_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="follow-up output root containing F/hand_manifest.json")
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
