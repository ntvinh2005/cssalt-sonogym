#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage2_coupled_smoke_protocol_v4.yaml"
PROTOCOL_SEAL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage2_coupled_smoke_protocol_v4.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage2_coupled_smoke_v4/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/metrics"
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$PROTOCOL_SEAL" "$RUN_DIR/context/"
(
    cd "$RUN_DIR/context"
    sha256sum -c stage2_coupled_smoke_protocol_v4.sha256
)

nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-before.csv"
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/time.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_stage2_orientation_smoke.py" \
        --protocol "$PROTOCOL" --output-dir "$RUN_DIR/results" \
        --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" \
        --headless --device cuda:0 > "$RUN_DIR/launcher.log" 2>&1
APPLICATION_RC=$?
set -e

RESULT_DECISION="MISSING"
if [[ -f "$RUN_DIR/results/result.json" ]]; then
    RESULT_DECISION=$("$PYTHON" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' \
        "$RUN_DIR/results/result.json")
fi

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-after.csv"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
{
    echo "application_exit_code=$APPLICATION_RC"
    echo "result_decision=$RESULT_DECISION"
} > "$RUN_DIR/context/application-exit.env"
capture_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

if [[ "$RESULT_DECISION" == "PASS" && "$APPLICATION_RC" -eq 0 ]]; then
    echo "STAGE2_COUPLED_SMOKE_PASS"
    exit 0
fi
if [[ "$RESULT_DECISION" == "FAIL" ]]; then
    echo "STAGE2_COUPLED_SMOKE_FAIL application_rc=$APPLICATION_RC"
    exit 3
fi
echo "STAGE2_COUPLED_SMOKE_ERROR decision=$RESULT_DECISION application_rc=$APPLICATION_RC"
exit 2
