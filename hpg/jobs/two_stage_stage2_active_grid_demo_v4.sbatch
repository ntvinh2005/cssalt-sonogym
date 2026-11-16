#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation"
[[ -d "$SONOGYM_RUNTIME_DIR/envs/stage1-python/sklearn" ]] || {
    echo "Pinned supervised-learning packages are missing" >&2
    exit 2
}

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage2_active_acquisition_grid_demo_protocol_v4.yaml"
PROTOCOL_SEAL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage2_active_acquisition_grid_demo_protocol_v4.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage2_active_acquisition_grid_demo_v4/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/metrics" "$RUN_DIR/video"
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$PROTOCOL_SEAL" "$RUN_DIR/context/"
(
    cd "$RUN_DIR/context"
    sha256sum -c stage2_active_acquisition_grid_demo_protocol_v4.sha256
)

nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-before.csv"
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!

set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_stage2_active_pilot.py" \
    > "$RUN_DIR/tests-active-pilot.log" 2>&1
PILOT_TEST_RC=$?
if [[ "$PILOT_TEST_RC" -eq 0 ]]; then
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_stage2_temporal_decision.py" \
        > "$RUN_DIR/tests-temporal.log" 2>&1
    TEMPORAL_TEST_RC=$?
else
    TEMPORAL_TEST_RC=99
fi
if [[ "$PILOT_TEST_RC" -eq 0 && "$TEMPORAL_TEST_RC" -eq 0 ]]; then
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_two_stage_stage2.py" \
        > "$RUN_DIR/tests-stage2-regression.log" 2>&1
    REGRESSION_RC=$?
else
    REGRESSION_RC=99
fi
if [[ "$PILOT_TEST_RC" -eq 0 && "$TEMPORAL_TEST_RC" -eq 0 && "$REGRESSION_RC" -eq 0 ]]; then
    /usr/bin/time -v -o "$RUN_DIR/metrics/time-actor.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_stage2_active_pilot.py" \
        --protocol "$PROTOCOL" --output-dir "$RUN_DIR" \
        --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" --headless --enable_cameras \
        --device cuda:0 > "$RUN_DIR/actor.log" 2>&1
    ACTOR_RC=$?
else
    ACTOR_RC=99
fi
if [[ "$ACTOR_RC" -eq 0 ]]; then
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_stage2_active_pilot.py" \
        --protocol "$PROTOCOL" --run-dir "$RUN_DIR" \
        --output "$RUN_DIR/results/evaluation_report.json" \
        > "$RUN_DIR/evaluator.log" 2>&1
    EVALUATOR_RC=$?
else
    EVALUATOR_RC=99
fi
set -e

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-after.csv"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
{
    echo "active_pilot_test_exit_code=$PILOT_TEST_RC"
    echo "temporal_test_exit_code=$TEMPORAL_TEST_RC"
    echo "stage2_regression_exit_code=$REGRESSION_RC"
    echo "actor_exit_code=$ACTOR_RC"
    echo "evaluator_exit_code=$EVALUATOR_RC"
} > "$RUN_DIR/context/application-exit-codes.env"
capture_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

if [[ "$PILOT_TEST_RC" -ne 0 || "$TEMPORAL_TEST_RC" -ne 0 || "$REGRESSION_RC" -ne 0 ]]; then
    exit 2
fi
if [[ "$ACTOR_RC" -ne 0 ]]; then exit "$ACTOR_RC"; fi
exit "$EVALUATOR_RC"
