#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit helper}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage1_supervised_pilot_protocol.yaml"
PROTOCOL_SEAL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage1_supervised_pilot_protocol.sha256"
CAPTURE_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/g0_g1_v2/50908807"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage1_supervised_pilot/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/metrics"
capture_cpu_run_context "$RUN_DIR"
cp "$PROTOCOL" "$RUN_DIR/context/stage1_supervised_pilot_protocol.yaml"
cp "$PROTOCOL_SEAL" "$RUN_DIR/context/stage1_supervised_pilot_protocol.sha256"
(
    cd "$RUN_DIR/context"
    sha256sum -c stage1_supervised_pilot_protocol.sha256
)

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/time-stage1-supervised-pilot.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_stage1_supervised_pilot.py" \
        --protocol "$PROTOCOL" \
        --capture-root "$CAPTURE_ROOT" \
        --output-dir "$RUN_DIR/results" \
        > "$RUN_DIR/stage1-supervised-pilot.log" 2>&1
APPLICATION_RC=$?
set -e
echo "$APPLICATION_RC" > "$RUN_DIR/context/application-exit-code.txt"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
exit "$APPLICATION_RC"
