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
EXECUTION_PROTOCOL="experiments/navigation_metric_validation/protocol_v2.yaml"
MANIFEST="experiments/navigation_metric_validation/initial_conditions.json"
RUN_DIR="$SONOGYM_OUTPUT_DIR/navigation_metric_validation_v2/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR/metrics" "$RUN_DIR/results/batches" "$RUN_DIR/results/final"
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$RUN_DIR/context/protocol.yaml"
cp "$EXECUTION_PROTOCOL" "$RUN_DIR/context/protocol_v2.yaml"
cp "$MANIFEST" "$RUN_DIR/context/initial_conditions.json"
cp "$SONOGYM_RUNTIME_DIR/records/phase5a-preflight-v2.json" \
    "$RUN_DIR/context/phase5a-preflight-v2.json"
sha256sum "$PROTOCOL" "$EXECUTION_PROTOCOL" "$MANIFEST" \
    > "$RUN_DIR/context/frozen-inputs.sha256"
sha256sum "$SONOGYM_EVAL_CHECKPOINT" > "$RUN_DIR/context/checkpoint.sha256"

nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-before.csv"
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!

EVALUATION_RC=0
AGGREGATE_RC=99
SUMMARY_RC=99
CONTROLLERS=(oracle zero_action bounded_random ppo_pilot)
set +e
for controller in "${CONTROLLERS[@]}"; do
    for batch_index in {0..7}; do
        batch_name=$(printf "batch_%02d" "$batch_index")
        batch_dir="$RUN_DIR/results/batches/$controller/$batch_name"
        mkdir -p "$batch_dir"
        echo "PHASE5A_V2_PROCESS_START controller=$controller batch=$batch_index"
        /usr/bin/time -v -o "$RUN_DIR/metrics/time-${controller}-${batch_name}.txt" \
        "$PYTHON" scripts/evaluate_navigation_metrics.py \
            --protocol "$PROTOCOL" \
            --manifest "$MANIFEST" \
            --checkpoint "$SONOGYM_EVAL_CHECKPOINT" \
            --output-dir "$batch_dir" \
            --controller "$controller" \
            --batch-index "$batch_index" \
            --headless \
            --device cuda:0
        batch_rc=$?
        echo "$batch_rc" > "$batch_dir/application-exit-code.txt"
        if [[ "$batch_rc" -ne 0 ]]; then
            EVALUATION_RC=$batch_rc
            break 2
        fi
        echo "PHASE5A_V2_PROCESS_DONE controller=$controller batch=$batch_index"
    done
done

if [[ "$EVALUATION_RC" -eq 0 ]]; then
    "$PYTHON" scripts/aggregate_navigation_batches.py \
        --protocol "$PROTOCOL" \
        --execution-protocol "$EXECUTION_PROTOCOL" \
        --batches-dir "$RUN_DIR/results/batches" \
        --output-dir "$RUN_DIR/results/final"
    AGGREGATE_RC=$?
fi
if [[ "$EVALUATION_RC" -eq 0 && "$AGGREGATE_RC" -eq 0 ]]; then
    "$PYTHON" scripts/summarize_navigation_validation.py \
        --evaluation "$RUN_DIR/results/final/evaluation.json" \
        --output-dir "$RUN_DIR/results/final"
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
    echo "aggregate_exit_code=$AGGREGATE_RC"
    echo "summary_exit_code=$SUMMARY_RC"
} > "$RUN_DIR/context/application-exit-codes.env"
capture_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

if [[ "$EVALUATION_RC" -ne 0 ]]; then
    echo "Phase 5A protocol-v2 batch evaluation failed with exit code $EVALUATION_RC" >&2
    exit "$EVALUATION_RC"
fi
if [[ "$AGGREGATE_RC" -ne 0 ]]; then
    echo "Phase 5A protocol-v2 aggregation failed with exit code $AGGREGATE_RC" >&2
    exit "$AGGREGATE_RC"
fi
if [[ "$SUMMARY_RC" -ne 0 ]]; then
    echo "Phase 5A protocol-v2 summarization failed with exit code $SUMMARY_RC" >&2
    exit "$SUMMARY_RC"
fi
