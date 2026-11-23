#!/usr/bin/env bash
#SBATCH --job-name=sg_clin_package
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail
: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
: "${CLINICIAN_SCAN_CAPTURE_JOB_ID:?CLINICIAN_SCAN_CAPTURE_JOB_ID was not exported}"
: "${CLINICIAN_SCAN_EVALUATION_JOB_ID:?CLINICIAN_SCAN_EVALUATION_JOB_ID was not exported}"
: "${SONOANATOMY_DATASET_JOB_ID:?SONOANATOMY_DATASET_JOB_ID was not exported}"
: "${SONOANATOMY_SEGMENTATION_JOB_ID:?SONOANATOMY_SEGMENTATION_JOB_ID was not exported}"
: "${CLINICIAN_SCAN_PERCEPTION_JOB_ID:?CLINICIAN_SCAN_PERCEPTION_JOB_ID was not exported}"
: "${CLINICIAN_SCAN_REVIEW_JOB_ID:?CLINICIAN_SCAN_REVIEW_JOB_ID was not exported}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

CAPTURE_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_clinician_scan_feasibility_v2/$CLINICIAN_SCAN_CAPTURE_JOB_ID"
MOTION_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_clinician_scan_evaluation_v2/$CLINICIAN_SCAN_EVALUATION_JOB_ID"
DATASET_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sonoanatomy_dataset_v2/$SONOANATOMY_DATASET_JOB_ID/dataset"
TRAINING_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sonoanatomy_segmentation_v2r1/$SONOANATOMY_SEGMENTATION_JOB_ID/training"
PERCEPTION_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_clinician_scan_perception_v2/$CLINICIAN_SCAN_PERCEPTION_JOB_ID"
REVIEW_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_clinician_scan_review_v2/$CLINICIAN_SCAN_REVIEW_JOB_ID"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_clinician_scan_package_v2/$SLURM_JOB_ID"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"

mkdir -p "$RUN_DIR"
capture_cpu_run_context "$RUN_DIR"
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/package_s0010_clinician_scan_review.py" \
  --repo-root "$SONOGYM_PROJECT_DIR" \
  --capture-root "$CAPTURE_ROOT" --motion-root "$MOTION_ROOT" \
  --dataset-root "$DATASET_ROOT" --training-root "$TRAINING_ROOT" \
  --perception-root "$PERCEPTION_ROOT" --review-root "$REVIEW_ROOT" \
  --output-dir "$RUN_DIR/package" \
  > "$RUN_DIR/stdout.json" 2> "$RUN_DIR/stderr.log"
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
