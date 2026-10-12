#!/usr/bin/env bash
# Launch a training run fully detached; logs go to logs/<run>/train.log.
# Usage: bash scripts/train.sh <run_name> [train.py flags...]
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_ROOT"
RUN="$1"; shift
mkdir -p logs/"$RUN"
setsid nohup python scripts/train.py --run_name "$RUN" "$@" > logs/"$RUN"/train.log 2>&1 < /dev/null &
echo $! > logs/"$RUN"/pid
echo "$*" > logs/"$RUN"/args
echo "started $RUN pid $(cat logs/"$RUN"/pid)"
