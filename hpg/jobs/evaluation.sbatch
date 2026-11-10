#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit helper}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"
: "${SONOGYM_EVAL_CHECKPOINT:?Set SONOGYM_EVAL_CHECKPOINT in the machine-local config}"
[[ -s "$SONOGYM_EVAL_CHECKPOINT" ]] || { echo "Evaluation checkpoint is missing" >&2; exit 2; }

RUN_DIR="$SONOGYM_OUTPUT_DIR/evaluation/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR/metrics"
capture_run_context "$RUN_DIR"
cp config/evaluation_pilot.json "$RUN_DIR/context/evaluation-protocol.json"

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-before.csv"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/checkpoint-time.txt" \
"$PYTHON" scripts/evaluate_checkpoint.py \
    --task Isaac-robot-US-guidance-v0 \
    --mode checkpoint \
    --checkpoint "$SONOGYM_EVAL_CHECKPOINT" \
    --num-envs 4 \
    --episodes-per-env 5 \
    --seed 20261110 \
    --output "$RUN_DIR/checkpoint.json" \
    --headless \
    --device cuda:0
CHECKPOINT_RC=$?
ZERO_RC=99
if [[ "$CHECKPOINT_RC" -eq 0 ]]; then
    /usr/bin/time -v -o "$RUN_DIR/metrics/zero-action-time.txt" \
    "$PYTHON" scripts/evaluate_checkpoint.py \
        --task Isaac-robot-US-guidance-v0 \
        --mode zero_action \
        --num-envs 4 \
        --episodes-per-env 5 \
        --seed 20261110 \
        --output "$RUN_DIR/zero-action.json" \
        --headless \
        --device cuda:0
    ZERO_RC=$?
fi
set -e
kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-after.csv"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

if [[ "$CHECKPOINT_RC" -ne 0 ]]; then
    echo "Checkpoint evaluation failed with exit code $CHECKPOINT_RC" >&2
    exit "$CHECKPOINT_RC"
fi
if [[ "$ZERO_RC" -ne 0 ]]; then
    echo "Zero-action evaluation failed with exit code $ZERO_RC" >&2
    exit "$ZERO_RC"
fi
"$PYTHON" scripts/summarize_evaluation.py \
    --checkpoint-report "$RUN_DIR/checkpoint.json" \
    --zero-report "$RUN_DIR/zero-action.json" \
    --output "$RUN_DIR/summary.json"
capture_run_end "$RUN_DIR"
