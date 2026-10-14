#!/usr/bin/env bash
# Re-render every rollout video (26 trained runs + 2 scripted), then the GIF
# and the side-by-side comparison. Rendering only: policies and results are
# read, never changed. Usage: scripts/render_all.sh [parallel_jobs]
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_ROOT"
JOBS="${1:-3}"
mkdir -p logs/render
render_one() {
  local job="$1"
  if [[ "$job" == scripted_* ]]; then
    local c="${job#scripted_}"
    python scripts/render.py --scripted --clearance_mm "$c" --episodes 1 > "logs/render/$job.log" 2>&1
  else
    python scripts/render.py --run "$job" --episodes 3 > "logs/render/$job.log" 2>&1
  fi && echo "rendered $job" || echo "render FAILED for $job"
}
export -f render_one
{ for d in results/policies/*/; do basename "$d"; done; echo scripted_2; echo scripted_0.5; } \
  | xargs -P "$JOBS" -I{} bash -c 'render_one {}'
python scripts/make_media.py
echo "RENDER ALL DONE"
