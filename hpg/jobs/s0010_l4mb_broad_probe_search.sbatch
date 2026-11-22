#!/usr/bin/env bash
#SBATCH --job-name=sg_l4mb_probe_search
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail
: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_l4mb_broad_probe_search/$SLURM_JOB_ID"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABEL_CONVERSION="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
PATIENT_SURFACE="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010"
CALIBRATION_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_l4mb_target_observability_v1_resume/52283924"
: "${SEARCH_LEVEL:=L5}"

export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$RUN_DIR"
capture_cpu_run_context "$RUN_DIR"

"$PYTHON" -m unittest -q \
  tests.test_s0010_l4mb_broad_probe_search \
  tests.test_ultrasound_plane_geometry \
  tests.test_l4_medial_branch_target \
  tests.test_s0010_l4mb_target_observability_evaluation \
  > "$RUN_DIR/tests.log" 2>&1

set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/search_s0010_l4mb_target_viewpoints.py" \
  --prediction "$PREDICTION" \
  --segmentation "$SEGMENTATION" \
  --label-conversion "$LABEL_CONVERSION" \
  --patient-surface-dir "$PATIENT_SURFACE" \
  --calibration-capture-root "$CALIBRATION_ROOT" \
  --level "$SEARCH_LEVEL" \
  --output-dir "$RUN_DIR/search" \
  > "$RUN_DIR/stdout.json" 2> "$RUN_DIR/stderr.log"
SEARCH_STATUS=$?
set -e
if [[ "$SEARCH_STATUS" -ne 0 && "$SEARCH_STATUS" -ne 2 ]]; then
  exit "$SEARCH_STATUS"
fi

capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
# Exit code 2 is a valid scientific no-candidate result from the search tool.
exit 0
