#!/usr/bin/env bash
# After the sweep: re-evaluate every run with the final evaluate.py (500 episodes),
# add generalisation + diagnostic rows, then make the GIF, plots and tables.
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_ROOT"
for d in logs/pos_* logs/force_*; do
  run=$(basename "$d")
  [[ -f "$d/final_metrics.json" ]] || continue
  python scripts/evaluate.py --run "$run" > "$d/eval.log" 2>&1 && echo "eval $run ok" || echo "eval $run FAILED"
done
# Generalisation: policies trained at 2 mm, tested on tighter holes without retraining.
for s in 0 1 2; do
  for c in 1.0 0.5; do
    python scripts/evaluate.py --run pos_2mm_s$s --clearance_mm $c --tag gen${c}mm > /dev/null 2>&1 \
      && echo "gen pos_2mm_s$s @ $c ok" || echo "gen pos_2mm_s$s @ $c FAILED"
  done
done
# Diagnostic: 0.5 mm policies tested with a perfect hole estimate (no observation bias).
for r in logs/pos_0.5mm_s* logs/force_0.5mm_s*; do
  run=$(basename "$r")
  python scripts/evaluate.py --run "$run" --obs_hole_bias_std 0 --tag nobias > /dev/null 2>&1 \
    && echo "nobias $run ok" || echo "nobias $run FAILED"
done
python scripts/plot.py
python scripts/summarize.py
echo "POSTPROCESS DONE"
