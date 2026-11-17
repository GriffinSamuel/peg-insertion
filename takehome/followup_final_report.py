"""Aggregate already completed Phase G confirmation evidence; never simulates."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set all follow-up caches before importing NumPy, Matplotlib, or JAX modules.
from takehome import followup_runtime  # noqa: E402
from takehome.followup_final_worker import configurations  # noqa: E402
from takehome.followup_metrics import summarize_seed  # noqa: E402
from takehome.followup_reference import reference_path  # noqa: E402
from takehome.followup_collect import identical_check  # noqa: E402

OUT = followup_runtime.OUT
FRESH_SEEDS = tuple(range(200, 205))
TEST_SEED = 1001
TEST_EPISODES = 500


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _choice_and_configs():
    choice_path = OUT / "G" / "choice.json"
    if not choice_path.is_file():
        raise FileNotFoundError(
            f"Phase G choice is required before aggregating confirmation: {choice_path}"
        )
    choice = json.loads(choice_path.read_text(encoding="utf-8"))
    if not isinstance(choice, dict):
        raise ValueError(f"{choice_path}: expected a JSON object")
    configs = configurations(choice)
    if len(configs) not in (1, 2):
        raise ValueError("final worker must define one reused method or chosen plus plain movement")
    if [name for name, _ in configs] == ["plain_movement"]:
        same_config_reused = True
    elif [name for name, _ in configs] == ["chosen", "plain_movement"]:
        same_config_reused = False
    else:
        raise ValueError("unexpected final worker method names/order")
    return choice_path, choice, configs, same_config_reused


def _load_method_seed(method: str, config, seed: int) -> dict[str, Any]:
    run_dir = OUT / "G" / method / f"seed_{seed}"
    summary = summarize_seed(run_dir)
    result_path = run_dir / "result.json"
    result_doc = json.loads(result_path.read_text(encoding="utf-8"))
    ref_path = reference_path(config)
    expected_cfg = config.to_dict()
    expected_cfg["baseline_path"] = str(ref_path.relative_to(ROOT))
    if result_doc.get("config") != expected_cfg:
        raise ValueError(f"{result_path}: saved configuration differs from the final worker config")
    if result_doc.get("seed") != seed or summary.get("seed") != seed:
        raise ValueError(f"{result_path}: wrong practice seed")
    if result_doc.get("practice_episodes") != 200 or summary.get("practice_episodes") != 200:
        raise ValueError(f"{result_path}: expected exactly 200 practice episodes")
    if result_doc.get("test_data_learned_from") is not False:
        raise ValueError(f"{result_path}: test data must not be learned from")
    if result_doc.get("frozen_policy_unchanged") is not True:
        raise ValueError(f"{result_path}: frozen policy must be recorded unchanged")
    final = result_doc.get("final", {})
    if final.get("episodes") != TEST_EPISODES or final.get("test_seed") != TEST_SEED:
        raise ValueError(f"{result_path}: expected 500 final tests at seed 1001")
    if final.get("test_only") is not True or final.get("learned_from") is not False:
        raise ValueError(f"{result_path}: invalid final test provenance flags")
    if summary.get("test_episodes") != TEST_EPISODES or summary.get("test_seed") != TEST_SEED:
        raise ValueError(f"{result_path}: summarized test evidence has wrong count or seed")
    if summary.get("test_only") is not True or summary.get("learned_from") is not False:
        raise ValueError(f"{result_path}: summarized test provenance is invalid")
    if summary.get("episode_success_vector_matches_final") is not True:
        raise ValueError(f"{result_path}: trace success counts disagree with the final summary")
    if result_doc.get("code_test") is True or expected_cfg.get("code_test") is True:
        raise ValueError(f"{result_path}: code-test runs cannot be confirmation evidence")
    return summary


def _aggregate_methods(configs):
    method_rows = {}
    method_summaries = {}
    for method, config in configs:
        rows = [_load_method_seed(method, config, seed) for seed in FRESH_SEEDS]
        identical_check({'name': 'G_' + method}, rows)
        rates = [float(row["success_percent"]) for row in rows]
        if len(rows) != 5:
            raise ValueError(f"{method}: expected exactly five fresh confirmation seeds")
        method_rows[method] = rows
        method_summaries[method] = {
            "method": method,
            "config": config.to_dict(),
            "seeds": rows,
            "successes_total": sum(int(row["successes"]) for row in rows),
            "episodes_total": sum(int(row["test_episodes"]) for row in rows),
            "mean_success_rate": statistics.mean(rates) / 100.0,
            "sample_sd_success_rate": statistics.stdev(rates) / 100.0,
            "mean_success_percent": statistics.mean(rates),
            "sample_sd_percentage_points": statistics.stdev(rates),
        }
    return method_rows, method_summaries


def _paired_comparison(method_rows, same_config_reused: bool):
    if same_config_reused:
        chosen = plain = method_rows["plain_movement"]
    else:
        chosen = method_rows["chosen"]
        plain = method_rows["plain_movement"]
    differences = []
    for chosen_row, plain_row in zip(chosen, plain):
        if chosen_row["seed"] != plain_row["seed"]:
            raise ValueError("paired confirmation rows must use matching practice seeds")
        count_difference = int(chosen_row["successes"]) - int(plain_row["successes"])
        percent_difference = float(chosen_row["success_percent"]) - float(plain_row["success_percent"])
        differences.append({
            "seed": int(chosen_row["seed"]),
            "chosen_successes": int(chosen_row["successes"]),
            "plain_movement_successes": int(plain_row["successes"]),
            "chosen_minus_plain_success_count": count_difference,
            "chosen_minus_plain_success_rate": percent_difference / 100.0,
            "chosen_minus_plain_percentage_points": percent_difference,
            "same_seed_data_reused": same_config_reused,
        })
    rates = [row["chosen_minus_plain_percentage_points"] for row in differences]
    return {
        "chosen_method": "plain_movement" if same_config_reused else "chosen",
        "plain_movement_method": "plain_movement",
        "same_config_reused": same_config_reused,
        "independent_method_comparison": not same_config_reused,
        "note": ("The chosen configuration equals plain movement; these are the same five runs, "
                 "not independent replicates." if same_config_reused else
                 "Differences are paired by fresh practice seed."),
        "per_seed_differences": differences,
        "mean_chosen_minus_plain_rate": statistics.mean(rates) / 100.0,
        "sample_sd_chosen_minus_plain_rate": statistics.stdev(rates) / 100.0,
        "mean_chosen_minus_plain_percentage_points": statistics.mean(rates),
        "sample_sd_chosen_minus_plain_percentage_points": statistics.stdev(rates),
    }


def _load_references(config):
    path = reference_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"fresh normal/zero-error references are required: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (doc.get("camera_condition") != "normal" or doc.get("test_seed") != TEST_SEED
            or doc.get("episodes_per_condition") != TEST_EPISODES
            or doc.get("practice_episodes") != 0 or doc.get("test_only") is not True
            or doc.get("learned_from") is not False):
        raise ValueError(f"{path}: invalid fresh reference provenance")
    conditions = doc.get("conditions")
    if not isinstance(conditions, dict) or set(("normal", "no_camera_bias")) - conditions.keys():
        raise ValueError(f"{path}: expected normal floor and no-camera-bias references")
    refs = {}
    for key, label, oracle in (
        ("normal", "Floor: original policy, normal camera error", False),
        ("no_camera_bias", "Zero-error reference", True),
    ):
        row = conditions[key]
        if (row.get("episodes") != TEST_EPISODES or row.get("test_seed") != TEST_SEED
                or row.get("test_only") is not True or row.get("learned_from") is not False
                or row.get("reference_only") is not True
                or row.get("oracle_correction") is not oracle):
            raise ValueError(f"{path}: invalid {key} reference record")
        successes = row.get("successes")
        if isinstance(successes, bool) or not isinstance(successes, int) or not 0 <= successes <= TEST_EPISODES:
            raise ValueError(f"{path}: invalid {key} success count")
        rate = _finite(row.get("success_rate"), f"{path}: {key} success_rate")
        if not math.isclose(rate, successes / TEST_EPISODES, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"{path}: {key} success rate does not match successes/episodes")
        refs[key] = {
            "label": label,
            "successes": successes,
            "episodes": TEST_EPISODES,
            "success_rate": rate,
            "success_percent": rate * 100.0,
            "test_seed": TEST_SEED,
            "test_only": True,
            "learned_from": False,
            "reference_only": True,
            "oracle_correction": oracle,
        }
    return {"path": path.relative_to(ROOT).as_posix(), "references": refs}


def _write_csv(path: Path, methods, paired, references, same_config_reused):
    fields = [
        "record_type", "method", "seed", "successes", "episodes", "success_rate",
        "success_percent", "mean_success_rate", "sample_sd_success_rate",
        "mean_success_percent", "sample_sd_percentage_points",
        "chosen_minus_plain_success_count", "chosen_minus_plain_success_rate",
        "chosen_minus_plain_percentage_points", "same_config_reused",
        "reference_key", "reference_label", "test_seed", "test_only", "learned_from",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for method, summary in methods.items():
            for row in summary["seeds"]:
                writer.writerow({
                    "record_type": "seed", "method": method, "seed": row["seed"],
                    "successes": row["successes"], "episodes": row["test_episodes"],
                    "success_rate": row["success_percent"] / 100.0,
                    "success_percent": row["success_percent"], "test_seed": row["test_seed"],
                    "test_only": row["test_only"], "learned_from": row["learned_from"],
                    "same_config_reused": same_config_reused,
                })
            writer.writerow({
                "record_type": "method_summary", "method": method,
                "mean_success_rate": summary["mean_success_rate"],
                "sample_sd_success_rate": summary["sample_sd_success_rate"],
                "mean_success_percent": summary["mean_success_percent"],
                "sample_sd_percentage_points": summary["sample_sd_percentage_points"],
                "same_config_reused": same_config_reused,
            })
        for row in paired["per_seed_differences"]:
            writer.writerow({
                "record_type": "paired_difference", "method": paired["chosen_method"],
                "seed": row["seed"], "chosen_minus_plain_success_count": row["chosen_minus_plain_success_count"],
                "chosen_minus_plain_success_rate": row["chosen_minus_plain_success_rate"],
                "chosen_minus_plain_percentage_points": row["chosen_minus_plain_percentage_points"],
                "same_config_reused": same_config_reused,
            })
        for key, ref in references["references"].items():
            writer.writerow({
                "record_type": "reference", "reference_key": key,
                "reference_label": ref["label"], "successes": ref["successes"],
                "episodes": ref["episodes"], "success_rate": ref["success_rate"],
                "success_percent": ref["success_percent"], "test_seed": ref["test_seed"],
                "test_only": ref["test_only"], "learned_from": ref["learned_from"],
                "same_config_reused": same_config_reused,
            })


def _write_chart(path: Path, method_summaries, references, same_config_reused):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_names = list(method_summaries)
    labels = []
    method_positions = {}
    y = 0.0
    for name in method_names:
        method_positions[name] = y
        labels.append(("Plain movement" if name == "plain_movement" else "Chosen learning method")
                      + (" (also the chosen method)" if same_config_reused else ""))
        y += 1.0
    reference_positions = {}
    for key in ("normal", "no_camera_bias"):
        reference_positions[key] = y
        labels.append(references["references"][key]["label"])
        y += 1.0

    fig_height = max(4.0, 1.0 + 0.8 * len(labels))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    seed_offsets = [-0.22, -0.11, 0.0, 0.11, 0.22]
    for name, summary in method_summaries.items():
        center = method_positions[name]
        ax.errorbar(summary["mean_success_percent"], center,
                    xerr=summary["sample_sd_percentage_points"], fmt="s",
                    color="black", capsize=4, markersize=7, zorder=3)
        for offset, row in zip(seed_offsets, summary["seeds"]):
            ax.plot(row["success_percent"], center + offset, "o", color="tab:blue",
                    markersize=5, zorder=4)
    for key, color in (("normal", "tab:orange"), ("no_camera_bias", "tab:green")):
        ref = references["references"][key]
        ax.plot(ref["success_percent"], reference_positions[key], marker="D",
                linestyle="none", color=color, markersize=7)
    ax.set_yticks([method_positions[name] for name in method_names] +
                  [reference_positions["normal"], reference_positions["no_camera_bias"]], labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Success rate (%)")
    ax.set_title("Fresh Phase G confirmation: method seeds and references")
    ax.grid(axis="x", alpha=0.25)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="s", linestyle="none", color="black", label="Mean ± sample SD"),
        Line2D([], [], marker="o", linestyle="none", color="tab:blue", label="Individual practice seed"),
        Line2D([], [], marker="D", linestyle="none", color="tab:orange", label="Floor reference"),
        Line2D([], [], marker="D", linestyle="none", color="tab:green", label="Zero-error reference"),
    ]
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate() -> tuple[Path, Path, Path]:
    choice_path, choice, configs, same_config_reused = _choice_and_configs()
    methods, method_summaries = _aggregate_methods(configs)
    paired = _paired_comparison(methods, same_config_reused)
    # Both configurations use the same normal-camera, fresh-test reference.
    references = _load_references(configs[0][1])
    summary = {
        "choice_path": choice_path.relative_to(ROOT).as_posix(),
        "choice": choice,
        "method_count": len(method_summaries),
        "same_config_reused": same_config_reused,
        "fresh_practice_seeds": list(FRESH_SEEDS),
        "test_seed": TEST_SEED,
        "test_episodes_per_seed": TEST_EPISODES,
        "methods": method_summaries,
        "paired_chosen_minus_plain_movement": paired,
        "references": references,
        "validation": {
            "all_method_runs_200_practice_episodes": True,
            "all_method_runs_500_final_test_episodes_seed_1001": True,
            "all_method_test_data_not_learned_from": True,
            "all_method_frozen_policies_unchanged": True,
            "both_references_500_test_episodes_seed_1001_test_only": True,
            "no_experiments_run_by_report_generator": True,
        },
    }
    folder = OUT / "G"
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / "confirmation.json"
    csv_path = folder / "confirmation.csv"
    png_path = folder / "confirmation.png"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    _write_csv(csv_path, method_summaries, paired, references, same_config_reused)
    _write_chart(png_path, method_summaries, references, same_config_reused)
    return json_path, csv_path, png_path


def main() -> int:
    for path in generate():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
