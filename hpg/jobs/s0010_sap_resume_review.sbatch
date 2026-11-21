#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_review_final
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
: "${SAP_RESUME_JOB:?SAP_RESUME_JOB must identify the completed frozen capture}"
: "${SAP_VALIDATION_PHASE:=resume_v1}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

case "$SAP_VALIDATION_PHASE" in
    resume_v1)
        CAPTURE="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_resume_v1/$SAP_RESUME_JOB"
        ANALYSIS="$CAPTURE/analysis_v2"
        HELDOUT="$ANALYSIS/heldout_evaluation_final.json"
        ;;
    resume_v2)
        CAPTURE="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_resume_v2/$SAP_RESUME_JOB"
        ANALYSIS="$CAPTURE/analysis_v1"
        HELDOUT="$ANALYSIS/heldout_evaluation.json"
        ;;
    *)
        echo "unsupported SAP_VALIDATION_PHASE=$SAP_VALIDATION_PHASE" >&2
        exit 2
        ;;
esac
BASELINE="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_negative_image_gate_development_v1/52138127"
REVIEW="$CAPTURE/final_review"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABELS="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"
mkdir -p "$REVIEW" "$MPLCONFIGDIR"

if [[ ! -e "$HELDOUT" ]]; then
    set +e
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_s0010_sap_classifier.py" \
        --dataset-dir "$ANALYSIS/dataset_evaluation" \
        --training-dir "$ANALYSIS/frozen_checkpoints" \
        --correspondence-validation "$ANALYSIS/evaluation_correspondence/correspondence_validation.json" \
        --output "$HELDOUT" > "$ANALYSIS/heldout-final-stdout.json" \
        2> "$ANALYSIS/heldout-final-stderr.log"
    EVALUATION_RC=$?
    set -e
    echo "$EVALUATION_RC" > "$ANALYSIS/heldout-final-exit-code.txt"
fi

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/build_s0010_sap_validation_training_review.py" \
    --first-dataset "$BASELINE/dataset_initial_v2" \
    --followup-dataset "$BASELINE/dataset_followup_v2" \
    --archive-dataset "$BASELINE/dataset_archive_v3" \
    --merged-dataset "$ANALYSIS/dataset_development_merged" \
    --evaluation-dataset "$ANALYSIS/dataset_evaluation" \
    --training-dir "$ANALYSIS/frozen_checkpoints" \
    --training-result "$BASELINE/training_v2/training_result.json" \
    --correspondence-validation "$ANALYSIS/calibration_correspondence/correspondence_validation.json" \
    --evaluation-correspondence-validation "$ANALYSIS/evaluation_correspondence/correspondence_validation.json" \
    --evaluation-report "$HELDOUT" --capture-root "$CAPTURE" \
    --prediction "$PREDICTION" --segmentation "$SEGMENTATION" \
    --label-conversion "$LABELS" \
    --execution-job "$SAP_RESUME_JOB" \
    --execution-commit "$(cat "$CAPTURE/context/git-commit.txt")" \
    --source-commit "$(git -C "$SONOGYM_PROJECT_DIR" rev-parse HEAD)" \
    --output-dir "$REVIEW" > "$REVIEW/builder.log" 2>&1

ZIP="$REVIEW/s0010_right_l4l5_sap_validation_training_review.zip"
unzip -t "$ZIP" > "$REVIEW/zip-test.log"
(
    cd "$REVIEW/s0010_right_l4l5_sap_validation_training_review"
    sha256sum -c checksums.sha256
) > "$REVIEW/checksum-test.log"
sha256sum "$ZIP" > "$REVIEW/zip.sha256"
