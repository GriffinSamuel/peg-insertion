"""Validate Goal 9 failure counts and create a descriptive CSV and chart."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from takehome.failure_rules import ORDER

CATEGORIES = (*ORDER, "success")
EXPECTED_AFTER_SEEDS = (100, 101, 102, 103, 104)


def resolve_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def ensure_output_path(path):
    if not path.is_relative_to(ROOT / "results" / "takehome"):
        raise ValueError(f"output must be under results/takehome: {path}")


def require_count(value, description):
    if type(value) is not int or value < 0:
        raise ValueError(f"{description} must be a non-negative integer")
    return value


def validated_group(summary, description, expected_episodes=None):
    episodes = require_count(summary.get("episodes"), f"{description}.episodes")
    successes = require_count(summary.get("successes"), f"{description}.successes")
    failures = require_count(summary.get("failures"), f"{description}.failures")
    counts = summary.get("counts")
    if not isinstance(counts, dict) or set(counts) != set(ORDER):
        raise ValueError(f"{description}.counts must contain exactly {list(ORDER)}")
    counts = {name: require_count(counts[name], f"{description}.counts.{name}")
              for name in ORDER}
    if expected_episodes is not None and episodes != expected_episodes:
        raise ValueError(f"{description} has {episodes} episodes; expected {expected_episodes}")
    if successes + failures != episodes:
        raise ValueError(f"{description}: successes + failures != episodes")
    if sum(counts.values()) != failures:
        raise ValueError(f"{description}: category counts do not sum to failures")
    if summary.get("order") is not None and tuple(summary["order"]) != ORDER:
        raise ValueError(f"{description}: category order differs from failure_rules.ORDER")
    return {**counts, "success": successes}, episodes, successes, failures


def prepare_rows(data):
    if tuple(data.get("order", ())) != ORDER:
        raise ValueError(f"input order must equal {list(ORDER)}")
    before_counts, before_n, _, _ = validated_group(data.get("before", {}), "before", 500)
    per_seed = data.get("after_per_seed")
    if not isinstance(per_seed, list) or len(per_seed) != len(EXPECTED_AFTER_SEEDS):
        raise ValueError("after_per_seed must contain exactly five records")
    seed_records = {}
    for item in per_seed:
        if not isinstance(item, dict):
            raise ValueError("each after_per_seed record must be an object")
        seed = item.get("practice_seed")
        if type(seed) is not int or seed in seed_records:
            raise ValueError(f"invalid or duplicate after practice seed: {seed!r}")
        counts, episodes, successes, failures = validated_group(
            item, f"after seed {seed}", 500)
        seed_records[seed] = (counts, episodes, successes, failures)
    if tuple(sorted(seed_records)) != EXPECTED_AFTER_SEEDS:
        raise ValueError(f"after seeds must be exactly {list(EXPECTED_AFTER_SEEDS)}")

    pooled = data.get("after_pooled", {})
    pooled_counts, pooled_n, pooled_success, pooled_failures = validated_group(
        pooled, "after_pooled", 2500)
    summed = {name: sum(seed_records[seed][0][name] for seed in EXPECTED_AFTER_SEEDS)
              for name in CATEGORIES}
    if pooled_counts != summed:
        raise ValueError("after_pooled counts do not match the sum of the five seeds")
    if pooled_n != sum(seed_records[s][1] for s in EXPECTED_AFTER_SEEDS):
        raise ValueError("after_pooled episode count does not match the five seeds")
    if pooled_success != sum(seed_records[s][2] for s in EXPECTED_AFTER_SEEDS):
        raise ValueError("after_pooled successes do not match the five seeds")
    if pooled_failures != sum(seed_records[s][3] for s in EXPECTED_AFTER_SEEDS):
        raise ValueError("after_pooled failures do not match the five seeds")

    groups = [("before", "unlearned_repeat", before_counts, before_n)]
    groups.extend(("after_seed", str(seed), seed_records[seed][0], seed_records[seed][1])
                  for seed in EXPECTED_AFTER_SEEDS)
    groups.append(("after_pooled", "pooled_2500", pooled_counts, pooled_n))
    rows = []
    for group, seed, counts, denominator in groups:
        for category in CATEGORIES:
            count = counts[category]
            rows.append({"group": group, "seed": seed, "category": category,
                         "count": count, "episodes_denominator": denominator,
                         "percent_of_all_episodes": 100.0 * count / denominator})
    chart_values = {
        "before": (before_counts, before_n),
        "after": (pooled_counts, pooled_n),
        "after_seeds": {seed: (seed_records[seed][0], seed_records[seed][1])
                        for seed in EXPECTED_AFTER_SEEDS},
    }
    return rows, chart_values


def draw_chart(path, groups):
    os.environ['MPLCONFIGDIR'] = str(ROOT / 'results/takehome/mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    y = np.arange(len(CATEGORIES), dtype=float)
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, offset, color, label in (
            ('before', -.18, 'C0', 'Before, 500 tests'),
            ('after', .18, 'C1', 'After pooled, 2500 tests')):
        counts, n = groups[name]
        rates = [100. * counts[k] / n for k in CATEGORIES]
        ax.barh(y + offset, rates, height=.30, color=color, alpha=.55, label=label)
    for i, seed in enumerate(EXPECTED_AFTER_SEEDS):
        counts, n = groups['after_seeds'][seed]
        rates = [100. * counts[k] / n for k in CATEGORIES]
        ax.scatter(rates, y + .18 + (i-2)*.045, s=30, marker=('o','s','^','v','D')[i],
                   color=f'C{i+2}', zorder=4, label=f'After seed {seed}, {n} tests')
    ax.set_yticks(y, [k.replace('_',' ') for k in CATEGORIES])
    ax.invert_yaxis()
    ax.set_xlim(-1, 101)
    ax.set_xlabel('Percent of all test episodes (including successes)')
    ax.set_title('Operational failure categories, not inferred causality')
    ax.grid(axis='x', alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5,.01))
    fig.subplots_adjust(left=.29, bottom=.24, top=.9, right=.98)
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("results/takehome/goal9/failures.json"))
    parser.add_argument("--csv", type=Path,
                        default=Path("results/takehome/goal9/failure_counts.csv"))
    parser.add_argument("--chart", type=Path,
                        default=Path("results/takehome/goal9/failure_counts.png"))
    args = parser.parse_args()
    source = resolve_path(args.input)
    csv_path, chart_path = resolve_path(args.csv), resolve_path(args.chart)
    ensure_output_path(csv_path)
    ensure_output_path(chart_path)
    if csv_path == chart_path:
        parser.error("--csv and --chart must name different files")
    if source in (csv_path, chart_path):
        parser.error("derived outputs must not overwrite the source JSON")
    data = json.loads(source.read_text(encoding="utf-8"))
    rows, groups = prepare_rows(data)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "group", "seed", "category", "count", "episodes_denominator",
            "percent_of_all_episodes"))
        writer.writeheader()
        writer.writerows(rows)
    draw_chart(chart_path, groups)
    print(json.dumps({"csv": str(csv_path.relative_to(ROOT)),
                      "chart": str(chart_path.relative_to(ROOT)),
                      "category_count": len(CATEGORIES),
                      "csv_rows": len(rows), "before_episodes": groups["before"][1],
                      "after_pooled_episodes": groups["after"][1]}, indent=2))


if __name__ == "__main__":
    main()
