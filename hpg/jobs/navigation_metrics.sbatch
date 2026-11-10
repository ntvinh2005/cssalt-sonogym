#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit helper}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"
: "${SONOGYM_EVAL_CHECKPOINT:?Set SONOGYM_EVAL_CHECKPOINT in the machine-local config}"
[[ -s "$SONOGYM_EVAL_CHECKPOINT" ]] || { echo "Evaluation checkpoint is missing" >&2; exit 2; }

PROTOCOL="experiments/navigation_metric_validation/protocol.yaml"
MANIFEST="experiments/navigation_metric_validation/initial_conditions.json"
RUN_DIR="$SONOGYM_OUTPUT_DIR/navigation_metric_validation/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR/metrics" "$RUN_DIR/results"
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$RUN_DIR/context/protocol.yaml"
cp "$MANIFEST" "$RUN_DIR/context/initial_conditions.json"
cp "$SONOGYM_RUNTIME_DIR/records/phase5a-preflight.json" \
    "$RUN_DIR/context/phase5a-preflight.json"
sha256sum "$MANIFEST" > "$RUN_DIR/context/initial_conditions.sha256"
sha256sum "$SONOGYM_EVAL_CHECKPOINT" > "$RUN_DIR/context/checkpoint.sha256"

nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-before.csv"
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/evaluation-time.txt" \
"$PYTHON" scripts/evaluate_navigation_metrics.py \
    --protocol "$PROTOCOL" \
    --manifest "$MANIFEST" \
    --checkpoint "$SONOGYM_EVAL_CHECKPOINT" \
    --output-dir "$RUN_DIR/results" \
    --headless \
    --device cuda:0
EVALUATION_RC=$?
SUMMARY_RC=99
if [[ "$EVALUATION_RC" -eq 0 ]]; then
    /usr/bin/time -v -o "$RUN_DIR/metrics/summary-time.txt" \
    "$PYTHON" scripts/summarize_navigation_validation.py \
        --evaluation "$RUN_DIR/results/evaluation.json" \
        --output-dir "$RUN_DIR/results"
    SUMMARY_RC=$?
fi
set -e

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-after.csv"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
{
    echo "evaluation_exit_code=$EVALUATION_RC"
    echo "summary_exit_code=$SUMMARY_RC"
} > "$RUN_DIR/context/application-exit-codes.env"
capture_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

if [[ "$EVALUATION_RC" -ne 0 ]]; then
    echo "Phase 5A evaluation failed with exit code $EVALUATION_RC" >&2
    exit "$EVALUATION_RC"
fi
if [[ "$SUMMARY_RC" -ne 0 ]]; then
    echo "Phase 5A summarization failed with exit code $SUMMARY_RC" >&2
    exit "$SUMMARY_RC"
fi
