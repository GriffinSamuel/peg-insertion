"""Validate and plot the test-only 7 by 7 fixed-offset sweep."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Cache isolation must happen before importing NumPy or Matplotlib.
from takehome import followup_runtime  # noqa: E402,F401


def _finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def load_cells(folder: Path):
    import numpy as np

    axis = np.linspace(-0.9, 0.9, 7)
    cells = []
    seen = set()
    for index in range(49):
        path = folder / f"cell_{index:02d}" / "test_0000.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing offset sweep cell {index}: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a JSON object")
        if data.get("cell") != index:
            raise ValueError(f"{path}: cell index does not match its directory")
        grid_xy = data.get("grid_xy")
        if grid_xy != [index // 7, index % 7]:
            raise ValueError(f"{path}: grid_xy does not match its cell index")
        expected_guess = [float(axis[index // 7]), float(axis[index % 7])]
        guess = data.get("constant_guess_mm")
        if not isinstance(guess, list) or len(guess) != 2:
            raise ValueError(f"{path}: constant_guess_mm must contain X and Y")
        if any(not math.isclose(_finite_number(guess[i], f"{path}: guess[{i}]"),
                                expected_guess[i], rel_tol=0.0, abs_tol=1e-7)
               for i in range(2)):
            raise ValueError(f"{path}: constant guess is not the prescribed 7 by 7 grid value")
        if data.get("episodes") != 256:
            raise ValueError(f"{path}: expected exactly 256 test episodes")
        if data.get("test_seed") != 123:
            raise ValueError(f"{path}: expected test seed 123")
        if data.get("test_only") is not True or data.get("learned_from") is not False:
            raise ValueError(f"{path}: expected test_only=true and learned_from=false")
        if data.get("practice_episodes") != 0:
            raise ValueError(f"{path}: offset sweep must use zero practice episodes")
        successes = data.get("successes")
        if isinstance(successes, bool) or not isinstance(successes, int) or not 0 <= successes <= 256:
            raise ValueError(f"{path}: successes must be an integer from 0 through 256")
        rate = _finite_number(data.get("success_rate"), f"{path}: success_rate")
        if not math.isclose(rate, successes / 256, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"{path}: success_rate does not match successes/256")
        if index in seen:
            raise ValueError(f"duplicate cell {index}")
        seen.add(index)
        cells.append({
            "cell": index,
            "grid_x_index": index // 7,
            "grid_y_index": index % 7,
            "constant_guess_x_mm": guess[0],
            "constant_guess_y_mm": guess[1],
            "successes": successes,
            "episodes": 256,
            "success_rate": rate,
            "success_percent": rate * 100.0,
            "test_seed": 123,
            "test_only": True,
            "learned_from": False,
        })
    if len(cells) != 49 or len(seen) != 49:
        raise ValueError("offset sweep must contain exactly 49 unique cells")
    return axis, cells


def render(folder: Path) -> tuple[Path, Path, Path]:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis, cells = load_cells(folder)
    rates = np.empty((7, 7), dtype=float)
    for cell in cells:
        rates[cell["grid_x_index"], cell["grid_y_index"]] = cell["success_rate"]
    max_rate = float(max(cell["success_rate"] for cell in cells))
    best_cells = [
        {"cell": cell["cell"], "grid_xy": [cell["grid_x_index"], cell["grid_y_index"]],
         "constant_guess_mm": [cell["constant_guess_x_mm"], cell["constant_guess_y_mm"]],
         "successes": cell["successes"], "episodes": cell["episodes"],
         "success_rate": cell["success_rate"]}
        for cell in cells if cell["success_rate"] == max_rate
    ]
    # Grid ordering is X-major, then Y-major, so the centre has index 3*7+3.
    center = cells[3 * 7 + 3]
    summary = {
        "grid_shape": [7, 7],
        "cell_count": len(cells),
        "test_episodes_per_cell": 256,
        "test_seed": 123,
        "test_only": True,
        "learned_from": False,
        "practice_episodes": 0,
        "guess_axis_mm": [float(value) for value in axis],
        "success_rate_definition": "successes / 256",
        "best_success_rate": max_rate,
        "best_success_percent": max_rate * 100.0,
        "best_tied_cells": best_cells,
        "centre_cell": {
            "cell": center["cell"],
            "grid_xy": [center["grid_x_index"], center["grid_y_index"]],
            "constant_guess_mm": [center["constant_guess_x_mm"], center["constant_guess_y_mm"]],
            "successes": center["successes"], "episodes": center["episodes"],
            "success_rate": center["success_rate"],
        },
        "best_minus_centre_success_rate": max_rate - center["success_rate"],
        "best_minus_centre_percentage_points": (max_rate - center["success_rate"]) * 100.0,
        "cells": cells,
    }

    csv_path = folder / "cells.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(cells)
    summary_path = folder / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(rates.T * 100.0, origin="lower", interpolation="nearest",
                      extent=(-1.05, 1.05, -1.05, 1.05), vmin=0, vmax=100,
                      cmap="viridis", aspect="equal")
    ax.set_xticks(axis)
    ax.set_yticks(axis)
    tick_labels = ["0" if abs(float(value)) < 1e-12 else f"{value:g}" for value in axis]
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Constant aim guess X (mm); guess is subtracted from camera reading")
    ax.set_ylabel("Constant aim guess Y (mm); guess is subtracted from camera reading")
    ax.set_title("Test-only constant aim offset sweep (7 × 7)")
    for cell in cells:
        x = cell["constant_guess_x_mm"]
        y = cell["constant_guess_y_mm"]
        ax.text(x, y, f"{cell['success_percent']:.1f}%", ha="center", va="center",
                color="white" if cell["success_rate"] < 0.55 else "black", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Success rate (%)")
    fig.tight_layout()
    png_path = folder / "heatmap.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return png_path, csv_path, summary_path


def main() -> int:
    folder = followup_runtime.OUT / "B" / "offset_sweep"
    for path in render(folder):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
