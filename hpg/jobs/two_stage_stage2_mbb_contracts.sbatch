#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage2_mbb_contracts/$SLURM_JOB_ID"
SOURCE="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/"\
"s0010_anatomical_annotations_v2.json"
COMMITTED="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/"\
"stage2_mbb_landmark_readiness_v1.json"
GENERATED="$RUN_DIR/results/stage2_mbb_landmark_readiness_v1.json"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/metrics"
capture_cpu_run_context "$RUN_DIR"
sha256sum "$SOURCE" > "$RUN_DIR/context/s0010_anatomical_annotations_v2.sha256"

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/time-focused-tests.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_stage2_mbb_landmarks.py" \
    > "$RUN_DIR/tests-focused.log" 2>&1
FOCUSED_RC=$?
if [[ "$FOCUSED_RC" -eq 0 ]]; then
    /usr/bin/time -v -o "$RUN_DIR/metrics/time-regression-tests.txt" \
        /usr/bin/bash -c \
        '"$0" "$1/tests/test_anatomical_ground_truth.py" && \
         "$0" "$1/tests/test_two_stage_stage1.py" && \
         "$0" "$1/tests/test_two_stage_stage2.py"' \
        "$PYTHON" "$SONOGYM_PROJECT_DIR" \
        > "$RUN_DIR/tests-regression.log" 2>&1
    REGRESSION_RC=$?
else
    REGRESSION_RC=99
fi
if [[ "$FOCUSED_RC" -eq 0 && "$REGRESSION_RC" -eq 0 ]]; then
    /usr/bin/time -v -o "$RUN_DIR/metrics/time-readiness-audit.txt" \
        "$PYTHON" \
        "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/audit_stage2_mbb_readiness.py" \
        --repository-root "$SONOGYM_PROJECT_DIR" \
        --source "$SOURCE" \
        --output "$GENERATED" \
        > "$RUN_DIR/readiness-audit.log" 2>&1
    AUDIT_RC=$?
else
    AUDIT_RC=99
fi
if [[ "$AUDIT_RC" -eq 0 ]]; then
    cmp "$COMMITTED" "$GENERATED"
    REPRODUCIBILITY_RC=$?
else
    REPRODUCIBILITY_RC=99
fi
set -e

{
    echo "focused_test_exit_code=$FOCUSED_RC"
    echo "regression_test_exit_code=$REGRESSION_RC"
    echo "audit_exit_code=$AUDIT_RC"
    echo "manifest_reproducibility_exit_code=$REPRODUCIBILITY_RC"
} > "$RUN_DIR/context/application-exit-codes.env"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

if [[ "$FOCUSED_RC" -ne 0 || "$REGRESSION_RC" -ne 0 || "$AUDIT_RC" -ne 0 || \
      "$REPRODUCIBILITY_RC" -ne 0 ]]; then
    exit 2
fi
