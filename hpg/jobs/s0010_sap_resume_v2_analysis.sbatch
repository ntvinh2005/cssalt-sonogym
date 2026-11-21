#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_v2_cpu
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
: "${SAP_RESUME_JOB:?SAP_RESUME_JOB must identify the completed frozen capture}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

CAPTURE="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_resume_v2/$SAP_RESUME_JOB"
BASELINE="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_negative_image_gate_development_v1/52138127"
RUN_DIR="$CAPTURE/analysis_v1"
DEVELOPMENT="$RUN_DIR/dataset_development"
CALIBRATION_CORRESPONDENCE="$RUN_DIR/calibration_correspondence"
MERGED="$RUN_DIR/dataset_development_merged"
L4_TRAINING="$RUN_DIR/l4_training"
EVALUATION="$RUN_DIR/dataset_evaluation"
EVALUATION_CORRESPONDENCE="$RUN_DIR/evaluation_correspondence"
FROZEN_CHECKPOINTS="$RUN_DIR/frozen_checkpoints"
HELDOUT="$RUN_DIR/heldout_evaluation.json"
PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_sap_validation_resume_protocol_v2.yaml"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABELS="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
L5_CHECKPOINT="$BASELINE/training_v2/s0010_l5_sap_classifier_v1.joblib"
L5_SHA256="c173aa9812a2df25114f99604b7330ed06b435c3a77edc7e2901f2b5dd3896e5"
CORRESPONDENCE_CHECKPOINTS="$BASELINE/correspondence_validation"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"
mkdir -p "$DEVELOPMENT" "$CALIBRATION_CORRESPONDENCE" "$MERGED" "$L4_TRAINING" \
    "$EVALUATION" "$EVALUATION_CORRESPONDENCE" "$FROZEN_CHECKPOINTS" "$MPLCONFIGDIR"
capture_cpu_run_context "$RUN_DIR"
cp "$PROTOCOL" "${PROTOCOL%.yaml}.sha256" "$RUN_DIR/context/"
printf '%s  %s\n' "$L5_SHA256" "$L5_CHECKPOINT" | sha256sum -c -
cp "$L5_CHECKPOINT" "$FROZEN_CHECKPOINTS/"

run_allow_blocked() {
    local exit_path="$1"
    shift
    set +e
    "$@"
    local rc=$?
    set -e
    echo "$rc" > "$exit_path"
}

# Development construction opens only train/calibration sessions. Evaluation
# arrays remain unopened until all eligible checkpoints and thresholds freeze.
run_allow_blocked "$DEVELOPMENT/builder-exit-code.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_s0010_sap_dataset.py" \
    --protocol "$PROTOCOL" --run-root "$CAPTURE" --prediction "$PREDICTION" \
    --segmentation "$SEGMENTATION" --label-conversion "$LABELS" \
    --phase development --require-all-proposals-clear-for-negative-image \
    --output-dir "$DEVELOPMENT"

run_allow_blocked "$CALIBRATION_CORRESPONDENCE/exit-code.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_s0010_sap_correspondence.py" \
    --dataset-dir "$DEVELOPMENT" --protocol "$PROTOCOL" \
    --checkpoint-dir "$CORRESPONDENCE_CHECKPOINTS" --split calibration \
    --output-dir "$CALIBRATION_CORRESPONDENCE"

run_allow_blocked "$MERGED/merge-exit-code.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/merge_s0010_sap_datasets.py" \
    --input "$BASELINE/dataset_merged_v2" --input "$DEVELOPMENT" \
    --protocol "$PROTOCOL" \
    --correspondence-validation "$CALIBRATION_CORRESPONDENCE/correspondence_validation.json" \
    --output-dir "$MERGED"

L4_READY=$(
    "$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(str(r["readiness"]["L4"]["dataset_status"] == "READY").lower())' \
        "$MERGED/readiness_report.json"
)
echo "$L4_READY" > "$MERGED/l4-ready.txt"
if [[ "$L4_READY" == "true" ]]; then
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_s0010_sap_classifier.py" \
        --dataset-dir "$MERGED" --output-dir "$L4_TRAINING" --level L4 \
        > "$L4_TRAINING/preflight-stdout.json" 2> "$L4_TRAINING/preflight-stderr.log"
    run_allow_blocked "$L4_TRAINING/training-exit-code.txt" \
        "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_s0010_sap_classifier.py" \
        --dataset-dir "$MERGED" --output-dir "$L4_TRAINING" --fit --level L4
    L4_STATUS=$(
        "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["levels"]["L4"]["status"])' \
            "$L4_TRAINING/training_result.json"
    )
    if [[ "$L4_STATUS" == "TRAINED_CALIBRATED_HOLDOUT_UNOPENED" ]]; then
        cp "$L4_TRAINING/s0010_l4_sap_classifier_v1.joblib" "$FROZEN_CHECKPOINTS/"
    fi
fi

# The frozen L5 checkpoint is never refit or recalibrated here. Evaluation
# arrays are opened once, after the possible L4 checkpoint has frozen.
run_allow_blocked "$EVALUATION/builder-exit-code.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_s0010_sap_dataset.py" \
    --protocol "$PROTOCOL" --run-root "$CAPTURE" --prediction "$PREDICTION" \
    --segmentation "$SEGMENTATION" --label-conversion "$LABELS" \
    --phase evaluation --require-all-proposals-clear-for-negative-image \
    --output-dir "$EVALUATION"

run_allow_blocked "$EVALUATION_CORRESPONDENCE/exit-code.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_s0010_sap_correspondence.py" \
    --dataset-dir "$EVALUATION" --protocol "$PROTOCOL" \
    --checkpoint-dir "$CORRESPONDENCE_CHECKPOINTS" --split evaluation \
    --output-dir "$EVALUATION_CORRESPONDENCE"

run_allow_blocked "$RUN_DIR/heldout-exit-code.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_s0010_sap_classifier.py" \
    --dataset-dir "$EVALUATION" --training-dir "$FROZEN_CHECKPOINTS" \
    --correspondence-validation "$EVALUATION_CORRESPONDENCE/correspondence_validation.json" \
    --output "$HELDOUT"

capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
