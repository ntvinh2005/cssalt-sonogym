#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_right_sap_tp_review_v2_ultrasound/$SLURM_JOB_ID"
PACKAGE_DIR="$RUN_DIR/s0010_right_sap_tp_review_v2"
PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_right_sap_tp_review_v2_ultrasound_protocol.yaml"
EXPANDED_PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_right_sap_tp_review_protocol_v2.yaml"
ANNOTATION="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_anatomical_annotations_v2.json"
ASSET_ROOT="$SONOGYM_RUNTIME_DIR/assets/data/HumanModels"
ZIP_PATH="$SONOGYM_OUTPUT_DIR/reviewer/s0010_right_sap_tp_review_v2_${SLURM_JOB_ID}.zip"

mkdir -p "$PACKAGE_DIR" "$RUN_DIR/metrics" "$SONOGYM_OUTPUT_DIR/reviewer"
capture_cpu_run_context "$RUN_DIR"
SOURCE_COMMIT="$(git -C "$SONOGYM_PROJECT_DIR" rev-parse HEAD)"
sha256sum "$PROTOCOL" "$EXPANDED_PROTOCOL" "$ANNOTATION" > "$RUN_DIR/context/input-sha256.txt"

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/time-tests.txt" \
  "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_s0010_right_landmark_review.py" \
  > "$RUN_DIR/tests-base.log" 2>&1
BASE_RC=$?
if [[ "$BASE_RC" -eq 0 ]]; then
  /usr/bin/time -v -o "$RUN_DIR/metrics/time-tests-v2.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_s0010_right_landmark_review_v2.py" \
    > "$RUN_DIR/tests-v2.log" 2>&1
  V2_RC=$?
else
  V2_RC=99
fi
if [[ "$BASE_RC" -eq 0 && "$V2_RC" -eq 0 ]]; then
  /usr/bin/time -v -o "$RUN_DIR/metrics/time-tests-ultrasound.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_s0010_right_landmark_review_v2_ultrasound.py" \
    > "$RUN_DIR/tests-ultrasound.log" 2>&1
  US_TEST_RC=$?
else
  US_TEST_RC=99
fi
if [[ "$US_TEST_RC" -eq 0 ]]; then
  /usr/bin/time -v -o "$RUN_DIR/metrics/time-export.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_s0010_right_landmark_review_v2.py" \
    --side RIGHT \
    --asset-root "$ASSET_ROOT" \
    --annotation-source "$ANNOTATION" \
    --expanded-protocol "$EXPANDED_PROTOCOL" \
    --protocol "$PROTOCOL" \
    --package-root "$PACKAGE_DIR" \
    --source-commit "$SOURCE_COMMIT" \
    > "$RUN_DIR/export.log" 2>&1
  EXPORT_RC=$?
else
  EXPORT_RC=99
fi
if [[ "$EXPORT_RC" -eq 0 ]]; then
  (
    cd "$PACKAGE_DIR"
    find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > checksums.sha256
  )
  (
    cd "$RUN_DIR"
    "$PYTHON" -m zipfile -c "$ZIP_PATH" s0010_right_sap_tp_review_v2
  )
  "$PYTHON" -m zipfile -t "$ZIP_PATH" > "$RUN_DIR/zip-test.log" 2>&1
  ZIP_RC=$?
  sha256sum "$ZIP_PATH" > "$RUN_DIR/reviewer-zip.sha256"
else
  ZIP_RC=99
fi
set -e

{
  echo "base_test_exit_code=$BASE_RC"
  echo "v2_test_exit_code=$V2_RC"
  echo "ultrasound_test_exit_code=$US_TEST_RC"
  echo "export_exit_code=$EXPORT_RC"
  echo "zip_test_exit_code=$ZIP_RC"
  echo "zip_path=$ZIP_PATH"
} > "$RUN_DIR/context/application-exit-codes.env"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P > "$RUN_DIR/metrics/slurm.csv" || true
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

if [[ "$BASE_RC" -ne 0 || "$V2_RC" -ne 0 || "$US_TEST_RC" -ne 0 || "$EXPORT_RC" -ne 0 || "$ZIP_RC" -ne 0 ]]; then
  exit 2
fi
