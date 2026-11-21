#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_review
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
: "${SAP_RESULT_JOB:?SAP_RESULT_JOB must identify the completed CPU result}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

RESULT_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_negative_image_gate_development_v1/$SAP_RESULT_JOB"
REVIEW="$RESULT_ROOT/review_v2"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABELS="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"
mkdir -p "$REVIEW" "$MPLCONFIGDIR"

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/build_s0010_sap_validation_training_review.py" \
    --first-dataset "$RESULT_ROOT/dataset_initial_v2" \
    --followup-dataset "$RESULT_ROOT/dataset_followup_v2" \
    --archive-dataset "$RESULT_ROOT/dataset_archive_v3" \
    --merged-dataset "$RESULT_ROOT/dataset_merged_v2" \
    --evaluation-dataset "$RESULT_ROOT/evaluation_unacquired" \
    --training-dir "$RESULT_ROOT/training_v2" \
    --correspondence-validation "$RESULT_ROOT/correspondence_validation/correspondence_validation.json" \
    --evaluation-report "$RESULT_ROOT/heldout_evaluation_unacquired.json" \
    --prediction "$PREDICTION" --segmentation "$SEGMENTATION" \
    --label-conversion "$LABELS" \
    --execution-job "$SAP_RESULT_JOB" \
    --execution-commit "$(cat "$RESULT_ROOT/context/git-commit.txt")" \
    --source-commit "$(git -C "$SONOGYM_PROJECT_DIR" rev-parse HEAD)" \
    --output-dir "$REVIEW" > "$REVIEW/builder.log" 2>&1

ZIP="$REVIEW/s0010_right_l4l5_sap_validation_training_review.zip"
unzip -t "$ZIP" > "$REVIEW/zip-test.log"
(
    cd "$REVIEW/s0010_right_l4l5_sap_validation_training_review"
    sha256sum -c checksums.sha256
) > "$REVIEW/checksum-test.log"
sha256sum "$ZIP" > "$REVIEW/zip.sha256"
