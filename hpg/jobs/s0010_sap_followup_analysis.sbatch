#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_finish
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
: "${SAP_FOLLOWUP_JOB:?SAP_FOLLOWUP_JOB must identify the followup capture}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

FIRST_CAPTURE="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_development_v1/52129000"
FOLLOWUP_CAPTURE="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_followup_v1/$SAP_FOLLOWUP_JOB"
FIRST_DATASET="$FIRST_CAPTURE/dataset_v2"
FOLLOWUP_DEVELOPMENT="$FOLLOWUP_CAPTURE/dataset_development_v1"
MERGED="$FOLLOWUP_CAPTURE/dataset_merged_v1"
TRAINING="$FOLLOWUP_CAPTURE/training_v1"
EVALUATION="$FOLLOWUP_CAPTURE/dataset_evaluation_v1"
PROTOCOL_FIRST="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_sap_validation_development_protocol_v1.yaml"
PROTOCOL_FOLLOWUP="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_sap_validation_followup_protocol_v1.yaml"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABELS="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"
mkdir -p "$FIRST_DATASET" "$FOLLOWUP_DEVELOPMENT" "$MERGED" "$TRAINING" "$EVALUATION" "$MPLCONFIGDIR"

build_dataset() {
    local protocol="$1"
    local run_root="$2"
    local output="$3"
    local phase="$4"
    set +e
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_s0010_sap_dataset.py" \
        --protocol "$protocol" --run-root "$run_root" --prediction "$PREDICTION" \
        --segmentation "$SEGMENTATION" --label-conversion "$LABELS" \
        --phase "$phase" --output-dir "$output" \
        > "$output/builder-stdout.json" 2> "$output/builder-stderr.log"
    local rc=$?
    set -e
    echo "$rc" > "$output/builder-exit-code.txt"
}

build_dataset "$PROTOCOL_FIRST" "$FIRST_CAPTURE" "$FIRST_DATASET" development
build_dataset "$PROTOCOL_FOLLOWUP" "$FOLLOWUP_CAPTURE" "$FOLLOWUP_DEVELOPMENT" development

set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/merge_s0010_sap_datasets.py" \
    --input "$FIRST_DATASET" --input "$FOLLOWUP_DEVELOPMENT" \
    --protocol "$PROTOCOL_FOLLOWUP" --output-dir "$MERGED" \
    > "$MERGED/merge-stdout.json" 2> "$MERGED/merge-stderr.log"
MERGE_RC=$?
set -e
echo "$MERGE_RC" > "$MERGED/merge-exit-code.txt"

READY_LEVELS=$(
    "$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(",".join(level for level,value in r["readiness"].items() if value["dataset_status"] == "READY"))' \
        "$MERGED/readiness_report.json"
)
echo "$READY_LEVELS" > "$MERGED/ready-levels.txt"
if [[ -z "$READY_LEVELS" ]]; then
    exit 3
fi

LEVEL_ARGS=()
IFS=',' read -ra LEVEL_ARRAY <<< "$READY_LEVELS"
for level in "${LEVEL_ARRAY[@]}"; do
    LEVEL_ARGS+=(--level "$level")
done
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_s0010_sap_classifier.py" \
    --dataset-dir "$MERGED" --output-dir "$TRAINING" "${LEVEL_ARGS[@]}" \
    > "$TRAINING/preflight-stdout.json" 2> "$TRAINING/preflight-stderr.log"
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_s0010_sap_classifier.py" \
    --dataset-dir "$MERGED" --output-dir "$TRAINING" --fit "${LEVEL_ARGS[@]}" \
    > "$TRAINING/training-stdout.json" 2> "$TRAINING/training-stderr.log"

build_dataset "$PROTOCOL_FOLLOWUP" "$FOLLOWUP_CAPTURE" "$EVALUATION" evaluation
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_s0010_sap_classifier.py" \
    --dataset-dir "$EVALUATION" --training-dir "$TRAINING" \
    --output "$FOLLOWUP_CAPTURE/heldout_evaluation.json" \
    > "$FOLLOWUP_CAPTURE/heldout-evaluation-stdout.json" \
    2> "$FOLLOWUP_CAPTURE/heldout-evaluation-stderr.log"
