#!/usr/bin/env bash
#SBATCH --job-name=s0010-sap-tp-correction
#SBATCH --account=anest-dept
#SBATCH --partition=hpg-default
#SBATCH --qos=sol
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x-%j.out
#SBATCH --error=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x-%j.err

set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

# The workload is single-process; keep linked libraries from creating independent pools.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_right_sap_tp_correction/$SLURM_JOB_ID"
PACKAGE_DIR="$RUN_DIR/s0010_right_sap_tp_correction_v1"
PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_right_sap_tp_correction_protocol_v1.yaml"
ANNOTATION="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_anatomical_annotations_v2.json"
ASSET_ROOT="$SONOGYM_RUNTIME_DIR/assets/data/HumanModels"
HISTORICAL_ZIP="$SONOGYM_OUTPUT_DIR/reviewer/s0010_right_sap_tp_review_v2_51764274.zip"
ZIP_PATH="$SONOGYM_OUTPUT_DIR/reviewer/s0010_right_sap_tp_correction_v1_${SLURM_JOB_ID}.zip"

mkdir -p "$PACKAGE_DIR" "$RUN_DIR/metrics" "$SONOGYM_OUTPUT_DIR/reviewer"
capture_cpu_run_context "$RUN_DIR"
SOURCE_COMMIT="$(git -C "$SONOGYM_PROJECT_DIR" rev-parse HEAD)"
sha256sum "$PROTOCOL" "$ANNOTATION" "$HISTORICAL_ZIP" > "$RUN_DIR/context/input-sha256.txt"

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/time-tests.txt" \
  /usr/bin/bash -c 'set -e; for test in "$@"; do "$0" "$test"; done' \
  "$PYTHON" \
  "$SONOGYM_PROJECT_DIR/tests/test_ultrasound_plane_geometry.py" \
  "$SONOGYM_PROJECT_DIR/tests/test_s0010_right_landmark_correction.py" \
  "$SONOGYM_PROJECT_DIR/tests/test_s0010_right_landmark_review.py" \
  "$SONOGYM_PROJECT_DIR/tests/test_s0010_right_landmark_review_v2.py" \
  "$SONOGYM_PROJECT_DIR/tests/test_s0010_right_landmark_review_v2_ultrasound.py" \
  > "$RUN_DIR/tests.log" 2>&1
TEST_RC=$?
if [[ "$TEST_RC" -eq 0 ]]; then
  /usr/bin/time -v -o "$RUN_DIR/metrics/time-export.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_s0010_right_landmark_correction.py" \
    --asset-root "$ASSET_ROOT" \
    --annotation-source "$ANNOTATION" \
    --protocol "$PROTOCOL" \
    --historical-package "$HISTORICAL_ZIP" \
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
    "$PYTHON" -m zipfile -c "$ZIP_PATH" "$(basename "$PACKAGE_DIR")"
  )
  "$PYTHON" -m zipfile -t "$ZIP_PATH" > "$RUN_DIR/zip-test.log" 2>&1
  ZIP_RC=$?
  sha256sum "$ZIP_PATH" > "$RUN_DIR/reviewer-zip.sha256"
else
  ZIP_RC=99
fi
set -e

{
  echo "test_exit_code=$TEST_RC"
  echo "export_exit_code=$EXPORT_RC"
  echo "zip_exit_code=$ZIP_RC"
  echo "zip_path=$ZIP_PATH"
} > "$RUN_DIR/context/application-exit-codes.env"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P > "$RUN_DIR/metrics/slurm.csv" || true
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

if [[ "$TEST_RC" -ne 0 || "$EXPORT_RC" -ne 0 || "$ZIP_RC" -ne 0 ]]; then
  exit 2
fi
