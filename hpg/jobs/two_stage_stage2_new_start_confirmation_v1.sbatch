#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
source "$SONOGYM_CONFIG_FILE"
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation"
[[ -d "$SONOGYM_RUNTIME_DIR/envs/stage1-python/sklearn" ]] || exit 2

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage2_new_start_confirmation_protocol_v1.yaml"
PROTOCOL_SEAL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage2_new_start_confirmation_protocol_v1.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage2_new_start_confirmation_v1/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/metrics" "$RUN_DIR/context"
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$PROTOCOL_SEAL" "$RUN_DIR/context/"
(cd "$RUN_DIR/context" && sha256sum -c stage2_new_start_confirmation_protocol_v1.sha256)
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-before.csv"
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!

set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_stage2_active_pilot.py" > "$RUN_DIR/tests-active-pilot.log" 2>&1
TEST_RC=$?
ACTOR_FAILURES=0
EVALUATOR_FAILURES=0
if [[ "$TEST_RC" -eq 0 ]]; then
    for START_ID in confirm_new_00 confirm_new_01 confirm_new_02 confirm_new_03; do
        START_DIR="$RUN_DIR/starts/$START_ID"
        mkdir -p "$START_DIR"
        EVALUATOR_RC=99
        /usr/bin/time -v -o "$START_DIR/time-actor.txt" \
        "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_stage2_active_pilot.py" \
            --protocol "$PROTOCOL" --start-id "$START_ID" --output-dir "$START_DIR" \
            --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" --headless --enable_cameras --device cuda:0 \
            > "$START_DIR/actor.log" 2>&1
        ACTOR_RC=$?
        if [[ "$ACTOR_RC" -eq 0 ]]; then
            "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_stage2_active_pilot.py" \
                --protocol "$PROTOCOL" --run-dir "$START_DIR" \
                --output "$START_DIR/results/evaluation_report.json" > "$START_DIR/evaluator.log" 2>&1
            EVALUATOR_RC=$?
            [[ "$EVALUATOR_RC" -eq 0 ]] || EVALUATOR_FAILURES=$((EVALUATOR_FAILURES + 1))
        else
            ACTOR_FAILURES=$((ACTOR_FAILURES + 1))
        fi
        {
            echo "actor_exit_code=$ACTOR_RC"
            echo "evaluator_exit_code=$EVALUATOR_RC"
        } > "$START_DIR/application-exit-codes.env"
    done
else
    ACTOR_FAILURES=4
fi

if [[ "$TEST_RC" -eq 0 && "$ACTOR_FAILURES" -eq 0 && "$EVALUATOR_FAILURES" -eq 0 ]]; then
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_stage2_new_starts.py" \
        --protocol "$PROTOCOL" --run-dir "$RUN_DIR" \
        --output "$RUN_DIR/results/confirmation_report.json" > "$RUN_DIR/aggregate.log" 2>&1
    AGGREGATE_RC=$?
else
    AGGREGATE_RC=99
fi
set -e

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-after.csv"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
{
    echo "active_pilot_test_exit_code=$TEST_RC"
    echo "actor_failure_count=$ACTOR_FAILURES"
    echo "evaluator_failure_count=$EVALUATOR_FAILURES"
    echo "aggregate_exit_code=$AGGREGATE_RC"
} > "$RUN_DIR/context/application-exit-codes.env"
capture_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
if [[ "$TEST_RC" -ne 0 || "$ACTOR_FAILURES" -ne 0 || "$EVALUATOR_FAILURES" -ne 0 ]]; then exit 2; fi
exit "$AGGREGATE_RC"
