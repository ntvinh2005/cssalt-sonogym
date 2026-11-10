#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit helper}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

RUN_DIR="$SONOGYM_OUTPUT_DIR/smoke/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR/metrics"
capture_run_context "$RUN_DIR"

nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-before.csv"

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/time.txt" \
"$PYTHON" scripts/smoke_test.py \
    --task Isaac-robot-US-guidance-v0 \
    --num-envs 1 \
    --seed 20261106 \
    --steps 32 \
    --output-dir "$RUN_DIR" \
    --headless \
    --device cuda:0
SMOKE_RC=$?
set -e

nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-after.csv"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
capture_run_end "$RUN_DIR"

if [[ "$SMOKE_RC" -ne 0 ]]; then
    echo "Smoke command failed with exit code $SMOKE_RC" >&2
    exit "$SMOKE_RC"
fi
if [[ ! -s "$RUN_DIR/summary.json" ]]; then
    echo "Smoke command returned zero without writing summary.json" >&2
    exit 3
fi
"$PYTHON" -c \
    'import json, sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["status"] == "PASSED" else 3)' \
    "$RUN_DIR/summary.json"
