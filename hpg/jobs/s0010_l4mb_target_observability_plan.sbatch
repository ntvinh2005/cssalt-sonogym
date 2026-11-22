#!/usr/bin/env bash
#SBATCH --job-name=sg_l4mb_obs_plan
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

RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_l4mb_target_observability_plan/$SLURM_JOB_ID"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
TP_REVISION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_right_sap_tp_correction/51779234/s0010_right_sap_tp_correction_v1/boundary_revisions/editable_geometry/L5_RIGHT_transverse_process_revision.npz"
LABEL_CONVERSION="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$RUN_DIR"
capture_cpu_run_context "$RUN_DIR"

"$PYTHON" -m unittest -q \
  tests.test_s0010_l4mb_target_observability_plan \
  tests.test_l4_medial_branch_target \
  tests.test_acquisition_integrity \
  > "$RUN_DIR/tests.log" 2>&1

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/plan_s0010_l4mb_target_observability.py" \
  --reference-root "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v3/52039283/stage2" \
  --reference-root "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_development_v1/52129000/sessions" \
  --reference-root "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_followup_v1/52131349/sessions" \
  --reference-root "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_resume_v1/52182100/sessions" \
  --reference-root "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_resume_v2/52199322/sessions" \
  --prediction "$PREDICTION" \
  --segmentation "$SEGMENTATION" \
  --tp-revision "$TP_REVISION" \
  --label-conversion "$LABEL_CONVERSION" \
  --output-dir "$RUN_DIR/pose_atlas" \
  > "$RUN_DIR/stdout.json" 2> "$RUN_DIR/stderr.log"

capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
