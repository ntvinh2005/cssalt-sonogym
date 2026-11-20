#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_l5_fit
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
: "${SAP_ARCHIVE_JOB:?SAP_ARCHIVE_JOB must identify the v2 archive analysis}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_correspondence_development_v1/$SLURM_JOB_ID"
VALIDATION="$RUN_DIR/correspondence_validation"
MERGED="$RUN_DIR/dataset_merged_v1"
TRAINING="$RUN_DIR/training_v1"
PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_sap_existing_capture_development_protocol_v2.yaml"
ARCHIVE_DATASET="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_existing_capture_development_v2/$SAP_ARCHIVE_JOB/dataset_v1"
INITIAL_DATASET="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_development_v1/52129000/dataset_v2"
FOLLOWUP_DATASET="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_followup_v1/52131349/dataset_development_v1"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$VALIDATION" "$MERGED" "$TRAINING"
capture_cpu_run_context "$RUN_DIR"

set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/validate_s0010_sap_correspondence.py" \
    --dataset-dir "$ARCHIVE_DATASET" --protocol "$PROTOCOL" --output-dir "$VALIDATION" \
    > "$VALIDATION/stdout.json" 2> "$VALIDATION/stderr.log"
VALIDATION_RC=$?
set -e
echo "$VALIDATION_RC" > "$VALIDATION/exit-code.txt"

set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/merge_s0010_sap_datasets.py" \
    --input "$INITIAL_DATASET" --input "$FOLLOWUP_DATASET" --input "$ARCHIVE_DATASET" \
    --protocol "$PROTOCOL" \
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
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_s0010_sap_classifier.py" \
    --dataset-dir "$MERGED" --output-dir "$TRAINING" --fit "${LEVEL_ARGS[@]}" \
    > "$TRAINING/training-stdout.json" 2> "$TRAINING/training-stderr.log"
capture_cpu_run_end "$RUN_DIR"
