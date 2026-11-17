"""Render the supplied follow-up scoreboard records without judging them."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


VERDICTS = {
    "HELPED", "PROMISING", "NO REAL DIFFERENCE", "HURT", "UNSTABLE",
    "DID NOT RUN",
}
FIELDS = (
    "phase", "name", "description", "base", "base_result", "seeds",
    "mean_percent", "spread_pp", "delta_pp", "verdict", "aim_start_mm",
    "aim_success_mm", "steps_to_insert", "peak_force_n", "practice_episodes",
    "hand_search", "hidden_information", "camera_condition", "reason",
)
CSV_FIELDS = FIELDS + ("base_mean_percent", "base_spread_pp")
NUMERIC_FIELDS = {
    "mean_percent", "spread_pp", "delta_pp", "aim_start_mm", "aim_success_mm",
    "steps_to_insert", "peak_force_n", "base_mean_percent", "base_spread_pp",
}


def _finite_number(value: Any, field: str, name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value)):
        raise ValueError(f"{name}: {field} must be a finite number or null")


def validate_records(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(data, dict) or not isinstance(data.get("summary"), str):
        raise ValueError("input must contain a string summary")
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError("input must contain a records list")
    names: set[str] = set()
    checked: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"records[{index}] must be an object")
        missing = [field for field in FIELDS if field not in record]
        if missing:
            raise ValueError(f"records[{index}] missing fields: {', '.join(missing)}")
        name = record["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"records[{index}].name must be a non-empty string")
        if name in names:
            raise ValueError(f"duplicate record name: {name}")
        names.add(name)
        if record["verdict"] not in VERDICTS:
            raise ValueError(f"{name}: invalid verdict {record['verdict']!r}")
        for field in ("phase", "description", "base", "camera_condition"):
            if not isinstance(record[field], str):
                raise ValueError(f"{name}: {field} must be a string")
        for field in ("base_result", "reason"):
            if record[field] is not None and not isinstance(record[field], str):
                raise ValueError(f"{name}: {field} must be a string or null")
        for field in ("hand_search", "hidden_information"):
            if not isinstance(record[field], bool):
                raise ValueError(f"{name}: {field} must be boolean")
        practice = record["practice_episodes"]
        if practice is not None and (isinstance(practice, bool) or not isinstance(practice, int)
                                     or practice < 0):
            raise ValueError(f"{name}: practice_episodes must be a non-negative integer or null")
        for field in NUMERIC_FIELDS:
            _finite_number(record.get(field), field, name)
        seeds = record["seeds"]
        if not isinstance(seeds, list):
            raise ValueError(f"{name}: seeds must be a list")
        seen_seeds: set[int] = set()
        for item in seeds:
            if not isinstance(item, dict) or set(("seed", "successes", "episodes")) - item.keys():
                raise ValueError(f"{name}: each seed needs seed, successes, and episodes")
            seed, successes, episodes = item["seed"], item["successes"], item["episodes"]
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (seed, successes, episodes)):
                raise ValueError(f"{name}: seed counts must be integers")
            if seed in seen_seeds:
                raise ValueError(f"{name}: duplicate seed {seed}")
            if episodes != 500 or not 0 <= successes <= episodes:
                raise ValueError(f"{name}: seed {seed} must have 0..500 successes out of exactly 500")
            seen_seeds.add(seed)
        checked.append(dict(record))
    return data["summary"], checked


def _display(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _seed_display(seeds: list[dict[str, int]]) -> str:
    return "; ".join(
        f"{item['seed']}: {item['successes']}/{item['episodes']} "
        f"({100 * item['successes'] / item['episodes']:.1f}%)"
        for item in seeds
    ) or "n/a"


def _csv_value(field: str, record: dict[str, Any]) -> str:
    value = record.get(field)
    if field == "seeds":
        return _seed_display(value)
    if field in ("hand_search", "hidden_information"):
        return "yes" if value else "no"
    return "" if value is None else str(value)


def _atomic_write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "w"
    kwargs = {} if mode == "wb" else {"encoding": "utf-8", "newline": ""}
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, mode, **kwargs) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def render_markdown(summary: str, records: list[dict[str, Any]]) -> str:
    headers = [
        "Phase", "Name", "Description", "Base", "Base result", "Seed results",
        "Mean (%)", "Spread (pp)", "Delta (pp)", "Verdict", "Aim start (mm)",
        "Aim at success (mm)", "Steps to insert", "Peak force (N)",
        "Practice episodes", "Hand search", "Hidden information", "Camera condition", "Reason",
        "Base mean (%)", "Base spread (pp)",
    ]
    keys = list(FIELDS) + ["base_mean_percent", "base_spread_pp"]
    decimal_places = {
        "mean_percent": 2, "spread_pp": 2, "delta_pp": 2,
        "base_mean_percent": 2, "base_spread_pp": 2,
        "aim_start_mm": 3, "aim_success_mm": 3,
        "steps_to_insert": 1, "peak_force_n": 1,
    }
    rows = ["# Follow-up scoreboard", "", summary, "", "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |"]
    for record in records:
        values = []
        for key in keys:
            if key == "seeds":
                value = _seed_display(record[key])
            elif key in decimal_places and record[key] is not None:
                value = f"{record[key]:.{decimal_places[key]}f}"
            else:
                value = _display(record[key])
            values.append(value.replace("|", "\\|").replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows) + "\n"


def render_csv(records: list[dict[str, Any]]) -> str:
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow({field: _csv_value(field, record) for field in CSV_FIELDS})
    return buffer.getvalue()


def _phase_slug(phase: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in phase).strip("_")
    if not slug:
        raise ValueError("phase must contain at least one letter or digit")
    return slug


def plot_phase(phase: str, records: list[dict[str, Any]], output_root: Path) -> Path | None:
    """Plot one phase; matplotlib is loaded only after cache bootstrap."""
    from takehome import followup_runtime  # noqa: F401
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import textwrap

    selected = [record for record in records if record["phase"] == phase]
    if not selected:
        return None
    fig_height = max(3.5, 0.55 * len(selected) + 1.8)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    positions = list(range(len(selected)))
    ax.set_yticks(positions, [record["name"] + (" (partial; stopped)"
        if record["verdict"]=="DID NOT RUN" and record["mean_percent"] is not None else "") for record in selected])
    ax.invert_yaxis()
    for y, record in zip(positions, selected):
        mean = record.get("mean_percent")
        spread = record.get("spread_pp")
        if mean is None:
            reason = record.get("reason") or "no run data"
            wrapped = textwrap.fill(f"{record['verdict']}: {reason}", width=60)
            ax.text(0.01, y, wrapped, transform=ax.get_yaxis_transform(),
                    va="center", fontsize=8)
        else:
            ax.errorbar(mean, y - 0.10, xerr=spread if spread is not None else 0,
                        fmt="s", color="black", capsize=3)
        seeds = record["seeds"]
        count = len(seeds)
        offsets = ([0.0] if count == 1 else
                   [(-0.22 + 0.44 * index / (count - 1)) for index in range(count)])
        for item, offset in zip(seeds, offsets):
            pct = 100 * item["successes"] / item["episodes"]
            ax.plot(pct, y + offset, "o", color="tab:blue", markersize=4)
        base_mean = record.get("base_mean_percent")
        base_spread = record.get("base_spread_pp")
        if base_mean is not None:
            ax.errorbar(base_mean, y + 0.10,
                        xerr=base_spread if base_spread is not None else 0,
                        fmt="|", color="tab:orange", capsize=3)
    ax.set_xlabel("Success rate (%)")
    ax.set_title(f"Phase {phase}: final results from completed seeds")
    bounds=[(record.get(mean_key),record.get(spread_key) or 0)
            for record in selected
            for mean_key,spread_key in (('mean_percent','spread_pp'),
                                       ('base_mean_percent','base_spread_pp'))
            if record.get(mean_key) is not None]
    ax.set_xlim(min([0]+[m-s for m,s in bounds])-1,
                max([100]+[m+s for m,s in bounds])+1)
    ax.margins(y=0.08)
    ax.grid(axis="x", alpha=0.25)
    handles = [
        Line2D([], [], marker="s", color="black", linestyle="none",
               label="Learner mean ± spread"),
        Line2D([], [], marker="o", color="tab:blue", linestyle="none",
               label="Individual seed result"),
        Line2D([], [], marker="|", color="tab:orange", linestyle="none",
               label="Base mean ± spread"),
    ]
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    folder = output_root / f"phase_{_phase_slug(phase)}"
    destination = folder / "scoreboard.png"
    folder.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".scoreboard.", suffix=".png", dir=folder)
    os.close(fd)
    try:
        fig.savefig(temp_name, dpi=160)
        os.replace(temp_name, destination)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    finally:
        plt.close(fig)
    return destination


def render(input_path: Path, output_root: Path, plot_phase_name: str | None = None) -> list[Path]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    summary, records = validate_records(data)
    outputs = [output_root / "scoreboard.md", output_root / "scoreboard.csv"]
    _atomic_write(outputs[0], render_markdown(summary, records))
    _atomic_write(outputs[1], render_csv(records))
    if plot_phase_name is not None:
        plot_path = plot_phase(plot_phase_name, records, output_root)
        if plot_path is not None:
            outputs.append(plot_path)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("results/takehome/followup/scoreboard_records.json"))
    parser.add_argument("--plot-phase", metavar="X")
    args = parser.parse_args(argv)
    try:
        from takehome import followup_runtime  # noqa: F401
        output_root = Path("results/takehome/followup")
        for path in render(args.input, output_root, args.plot_phase):
            print(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
