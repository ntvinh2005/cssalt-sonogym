#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage2_threshold_calibration_protocol_v18.yaml"
PROTOCOL_SEAL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage2_threshold_calibration_protocol_v18.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage2_threshold_calibration_v18/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/metrics"
capture_cpu_run_context "$RUN_DIR"
cp "$PROTOCOL" "$PROTOCOL_SEAL" "$RUN_DIR/context/"
(
    cd "$RUN_DIR/context"
    sha256sum -c stage2_threshold_calibration_protocol_v18.sha256
)

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/time-stage2-calibration.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/calibrate_stage2_threshold.py" \
        --protocol "$PROTOCOL" --output-dir "$RUN_DIR/results" \
        > "$RUN_DIR/stage2-calibration.log" 2>&1
APPLICATION_RC=$?
set -e
echo "$APPLICATION_RC" > "$RUN_DIR/context/application-exit-code.txt"
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
exit "$APPLICATION_RC"
