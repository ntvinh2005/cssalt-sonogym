#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage2_new_start_confirmation_protocol_v1.yaml"
RUN_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage2_new_start_confirmation_v1/51668777"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage2_angular_selection_audit/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/metrics"
capture_cpu_run_context "$RUN_DIR"
cp "$PROTOCOL" "$RUN_DIR/context/"
sha256sum "$PROTOCOL" > "$RUN_DIR/context/stage2_new_start_confirmation_protocol_v1.sha256"

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/time-focused-tests.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_stage2_angular_selection_audit.py" \
    > "$RUN_DIR/tests-focused.log" 2>&1
FOCUSED_RC=$?
if [[ "$FOCUSED_RC" -eq 0 ]]; then
    /usr/bin/time -v -o "$RUN_DIR/metrics/time-stage2-regression.txt" \
        "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_two_stage_stage2.py" \
        > "$RUN_DIR/tests-stage2-regression.log" 2>&1
    REGRESSION_RC=$?
else
    REGRESSION_RC=99
fi
if [[ "$FOCUSED_RC" -eq 0 && "$REGRESSION_RC" -eq 0 ]]; then
    /usr/bin/time -v -o "$RUN_DIR/metrics/time-audit.txt" \
        "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/audit_stage2_angular_selection.py" \
            --run-root "$RUN_ROOT" \
            --protocol "$PROTOCOL" \
            --output-dir "$RUN_DIR/results" \
            > "$RUN_DIR/stage2-angular-selection-audit.log" 2>&1
    AUDIT_RC=$?
else
    AUDIT_RC=99
fi
set -e

{
    echo "focused_test_exit_code=$FOCUSED_RC"
    echo "stage2_regression_exit_code=$REGRESSION_RC"
    echo "audit_exit_code=$AUDIT_RC"
} > "$RUN_DIR/context/application-exit-codes.env"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

if [[ "$FOCUSED_RC" -ne 0 || "$REGRESSION_RC" -ne 0 ]]; then
    exit 2
fi
exit "$AUDIT_RC"
