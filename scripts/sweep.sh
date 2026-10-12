#!/usr/bin/env bash
# Phase 6 experiment: trains every condition sequentially (one GPU, 8 GB),
# then evaluates (500 episodes), renders a video and refreshes the plots after
# each run, so partial results are always balanced and usable.
#
# Resumable: a run whose logs/<run>/final_metrics.json exists is not retrained.
# Launch detached:  setsid nohup bash scripts/sweep.sh > logs/sweep.log 2>&1 < /dev/null &
# Extra train.py flags (shared by ALL runs) can be passed as arguments.
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_ROOT"
EXTRA_FLAGS=("$@")

# run_name clearance_mm use_force_obs seed   (priority order, see PLAN.md Phase 6)
RUNS=(
  "pos_2mm_s0 2.0 false 0"     "pos_1mm_s0 1.0 false 0"
  "pos_0.5mm_s0 0.5 false 0"   "force_0.5mm_s0 0.5 true 0"
  "pos_2mm_s1 2.0 false 1"     "pos_1mm_s1 1.0 false 1"
  "pos_0.5mm_s1 0.5 false 1"   "force_0.5mm_s1 0.5 true 1"
  "force_2mm_s0 2.0 true 0"    "force_1mm_s0 1.0 true 0"
  "pos_2mm_s2 2.0 false 2"     "pos_1mm_s2 1.0 false 2"
  "pos_0.5mm_s2 0.5 false 2"   "force_0.5mm_s2 0.5 true 2"
  "force_2mm_s1 2.0 true 1"    "force_1mm_s1 1.0 true 1"
)
if [[ -n "${SWEEP_ONLY:-}" ]]; then  # e.g. SWEEP_ONLY="pos_2mm_s0 pos_1mm_s0"
  FILTERED=()
  for r in "${RUNS[@]}"; do
    for want in $SWEEP_ONLY; do [[ "${r%% *}" == "$want" ]] && FILTERED+=("$r"); done
  done
  RUNS=("${FILTERED[@]}")
fi

note() { echo "- $1 $(date '+%F %H:%M')" | tee -a PROGRESS.md; }

for spec in "${RUNS[@]}"; do
  read -r RUN C FORCE SEED <<< "$spec"
  if [[ -f "logs/$RUN/final_metrics.json" ]]; then
    echo "skip $RUN (already finished)"
  else
    mkdir -p "logs/$RUN"
    note "STARTED $RUN (sweep)"
    echo "--clearance_mm $C --use_force_obs=$FORCE --seed $SEED ${EXTRA_FLAGS[*]}" > "logs/$RUN/args"
    python scripts/train.py --run_name "$RUN" --clearance_mm "$C" \
      --use_force_obs="$FORCE" --seed "$SEED" --norender "${EXTRA_FLAGS[@]}" \
      > "logs/$RUN/train.log" 2>&1 < /dev/null
    if [[ ! -f "logs/$RUN/final_metrics.json" ]]; then
      note "FAILED $RUN (no final_metrics.json; see logs/$RUN/train.log)"
      continue
    fi
    note "FINISHED $RUN $(python -c "import json;d=json.load(open('logs/$RUN/final_metrics.json'));print(f\"success={d['success_rate']:.3f} wall={d['wall_time_s']/60:.1f}min sps={d['sps']}\")")"
  fi
  if [[ ! -f "results/eval_$RUN.json" ]]; then
    python scripts/evaluate.py --run "$RUN" > "logs/$RUN/eval.log" 2>&1 \
      || echo "eval failed for $RUN"
  fi
  if [[ ! -f "results/videos/$RUN.mp4" ]]; then
    python scripts/render.py --run "$RUN" --episodes 3 > "logs/$RUN/render.log" 2>&1 \
      || echo "render failed for $RUN"
  fi
  python scripts/plot.py > logs/plot.log 2>&1 || echo "plot failed"
done
echo "SWEEP DONE $(date '+%F %H:%M')"
