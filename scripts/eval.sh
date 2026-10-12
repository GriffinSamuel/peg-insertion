#!/usr/bin/env bash
# Evaluate a trained run (500 episodes) and append/refresh its row in results/summary.csv.
# Usage: bash scripts/eval.sh <run_name> [evaluate.py flags...]
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_ROOT"
RUN="$1"; shift
python scripts/evaluate.py --run "$RUN" "$@"
