#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_neg_gate
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
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_negative_image_gate_development_v1/$SLURM_JOB_ID"
INITIAL="$RUN_DIR/dataset_initial_v2"
FOLLOWUP="$RUN_DIR/dataset_followup_v2"
ARCHIVE="$RUN_DIR/dataset_archive_v3"
VALIDATION="$RUN_DIR/correspondence_validation"
MERGED="$RUN_DIR/dataset_merged_v2"
TRAINING="$RUN_DIR/training_v2"
INITIAL_CAPTURE="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_development_v1/52129000"
FOLLOWUP_CAPTURE="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_followup_v1/52131349"
INITIAL_PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_sap_validation_development_protocol_v2.yaml"
FOLLOWUP_PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_sap_validation_followup_protocol_v2.yaml"
ARCHIVE_PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_sap_existing_capture_development_protocol_v3.yaml"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABELS="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"
mkdir -p "$INITIAL" "$FOLLOWUP" "$ARCHIVE" "$VALIDATION" "$MERGED" "$TRAINING" "$MPLCONFIGDIR"
capture_cpu_run_context "$RUN_DIR"
cp "$INITIAL_PROTOCOL" "$FOLLOWUP_PROTOCOL" "$ARCHIVE_PROTOCOL" "$RUN_DIR/context/"
cp "${INITIAL_PROTOCOL%.yaml}.sha256" "${FOLLOWUP_PROTOCOL%.yaml}.sha256" "${ARCHIVE_PROTOCOL%.yaml}.sha256" "$RUN_DIR/context/"

build_dataset() {
    local protocol="$1"
    local run_root="$2"
    local output="$3"
    set +e
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_s0010_sap_dataset.py" \
        --protocol "$protocol" --run-root "$run_root" \
        --prediction "$PREDICTION" --segmentation "$SEGMENTATION" \
        --label-conversion "$LABELS" --phase development --output-dir "$output" \
        > "$output/builder-stdout.json" 2> "$output/builder-stderr.log"
    local rc=$?
    set -e
    echo "$rc" > "$output/builder-exit-code.txt"
}

build_dataset "$INITIAL_PROTOCOL" "$INITIAL_CAPTURE" "$INITIAL"
build_dataset "$FOLLOWUP_PROTOCOL" "$FOLLOWUP_CAPTURE" "$FOLLOWUP"
build_dataset "$ARCHIVE_PROTOCOL" "$RUN_DIR" "$ARCHIVE"

set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/validate_s0010_sap_correspondence.py" \
    --dataset-dir "$ARCHIVE" --protocol "$ARCHIVE_PROTOCOL" --output-dir "$VALIDATION" \
    > "$VALIDATION/stdout.json" 2> "$VALIDATION/stderr.log"
VALIDATION_RC=$?
set -e
echo "$VALIDATION_RC" > "$VALIDATION/exit-code.txt"

set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/merge_s0010_sap_datasets.py" \
    --input "$INITIAL" --input "$FOLLOWUP" --input "$ARCHIVE" \
    --protocol "$ARCHIVE_PROTOCOL" \
    --correspondence-validation "$VALIDATION/correspondence_validation.json" \
    --output-dir "$MERGED" \
    > "$MERGED/stdout.json" 2> "$MERGED/stderr.log"
MERGE_RC=$?
set -e
echo "$MERGE_RC" > "$MERGED/exit-code.txt"

READY_LEVELS=$(
    "$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(",".join(level for level,value in r["readiness"].items() if value["dataset_status"] == "READY"))' \
        "$MERGED/readiness_report.json"
)
echo "$READY_LEVELS" > "$MERGED/ready-levels.txt"
if [[ -z "$READY_LEVELS" ]]; then
    capture_cpu_run_end "$RUN_DIR"
    exit "$MERGE_RC"
fi

LEVEL_ARGS=()
IFS=',' read -ra LEVEL_ARRAY <<< "$READY_LEVELS"
for level in "${LEVEL_ARRAY[@]}"; do
    LEVEL_ARGS+=(--level "$level")
done
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_s0010_sap_classifier.py" \
    --dataset-dir "$MERGED" --output-dir "$TRAINING" "${LEVEL_ARGS[@]}" \
    > "$TRAINING/preflight-stdout.json" 2> "$TRAINING/preflight-stderr.log"
set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_s0010_sap_classifier.py" \
    --dataset-dir "$MERGED" --output-dir "$TRAINING" --fit "${LEVEL_ARGS[@]}" \
    > "$TRAINING/training-stdout.json" 2> "$TRAINING/training-stderr.log"
TRAIN_RC=$?
set -e
echo "$TRAIN_RC" > "$TRAINING/training-exit-code.txt"
capture_cpu_run_end "$RUN_DIR"
exit "$TRAIN_RC"
