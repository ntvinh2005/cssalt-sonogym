#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit helper}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"
: "${SONOGYM_EVAL_CHECKPOINT:?Set SONOGYM_EVAL_CHECKPOINT in the machine-local config}"
[[ -s "$SONOGYM_EVAL_CHECKPOINT" ]] || { echo "Demo checkpoint is missing" >&2; exit 2; }

RUN_DIR="$SONOGYM_OUTPUT_DIR/demo/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR/metrics" "$RUN_DIR/video"
capture_run_context "$RUN_DIR"

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-before.csv"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/time.txt" \
"$PYTHON" scripts/record_demo.py \
    --task Isaac-robot-US-guidance-v0 \
    --checkpoint "$SONOGYM_EVAL_CHECKPOINT" \
    --seed 20261110 \
    --frames 180 \
    --output-dir "$RUN_DIR/video" \
    --headless \
    --device cuda:0
DEMO_RC=$?
set -e
kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-after.csv"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
capture_run_end "$RUN_DIR"
if [[ "$DEMO_RC" -ne 0 ]]; then
    echo "Demo recording failed with exit code $DEMO_RC" >&2
    exit "$DEMO_RC"
fi
[[ -s "$RUN_DIR/video/demo-manifest.json" ]] || { echo "Demo manifest is missing" >&2; exit 3; }
find "$RUN_DIR/video" -maxdepth 1 -type f -name '*.mp4' -size +0c -print -quit | grep -q . || {
    echo "Demo MP4 is missing" >&2
    exit 3
}
