#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/"\
"s0010_right_sap_tp_review_v2/$SLURM_JOB_ID"
PACKAGE_DIR="$RUN_DIR/reviewer_package_v2"
PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/"\
"s0010_right_sap_tp_review_protocol_v2.yaml"
ANNOTATION="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/"\
"s0010_anatomical_annotations_v2.json"
ASSET_ROOT="$SONOGYM_RUNTIME_DIR/assets/data/HumanModels"
ZIP_PATH="$SONOGYM_OUTPUT_DIR/reviewer/"\
"s0010_right_sap_tp_review_v2_${SLURM_JOB_ID}.zip"

mkdir -p "$PACKAGE_DIR" "$RUN_DIR/metrics" "$SONOGYM_OUTPUT_DIR/reviewer"
capture_cpu_run_context "$RUN_DIR"
cp "$PROTOCOL" "$PACKAGE_DIR/"
cp "$SONOGYM_PROJECT_DIR/docs/research/s0010_right_sap_tp_review_v2_instructions.md" \
    "$PACKAGE_DIR/REVIEW_INSTRUCTIONS.md"
sha256sum "$PROTOCOL" "$ANNOTATION" > "$RUN_DIR/context/input-sha256.txt"

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/time-focused-tests.txt" \
    /usr/bin/bash -c \
    '"$0" "$1/tests/test_s0010_right_landmark_review.py" && \
     "$0" "$1/tests/test_s0010_right_landmark_review_v2.py"' \
    "$PYTHON" "$SONOGYM_PROJECT_DIR" \
    > "$RUN_DIR/tests-focused.log" 2>&1
TEST_RC=$?
if [[ "$TEST_RC" -eq 0 ]]; then
    /usr/bin/time -v -o "$RUN_DIR/metrics/time-export.txt" \
        "$PYTHON" \
        "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/"\
"expand_s0010_right_landmark_review.py" \
        --side RIGHT \
        --asset-root "$ASSET_ROOT" \
        --annotation-source "$ANNOTATION" \
        --protocol "$PROTOCOL" \
        --package-root "$PACKAGE_DIR" \
        > "$RUN_DIR/export.log" 2>&1
    EXPORT_RC=$?
else
    EXPORT_RC=99
fi
if [[ "$EXPORT_RC" -eq 0 ]]; then
    (
        cd "$PACKAGE_DIR"
        find . -type f ! -name checksums.sha256 -print0 \
            | sort -z \
            | xargs -0 sha256sum > checksums.sha256
    )
    (
        cd "$RUN_DIR"
        "$PYTHON" -m zipfile -c "$ZIP_PATH" reviewer_package_v2
    )
    "$PYTHON" -m zipfile -t "$ZIP_PATH" > "$RUN_DIR/zip-test.log" 2>&1
    ZIP_RC=$?
    sha256sum "$ZIP_PATH" > "$RUN_DIR/reviewer-zip.sha256"
else
    ZIP_RC=99
fi
set -e

{
    echo "focused_test_exit_code=$TEST_RC"
    echo "export_exit_code=$EXPORT_RC"
    echo "zip_test_exit_code=$ZIP_RC"
    echo "zip_path=$ZIP_PATH"
} > "$RUN_DIR/context/application-exit-codes.env"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

if [[ "$TEST_RC" -ne 0 || "$EXPORT_RC" -ne 0 || "$ZIP_RC" -ne 0 ]]; then
    exit 2
fi
