#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_archive
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

RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_existing_capture_development_v1/$SLURM_JOB_ID"
DATASET="$RUN_DIR/dataset_v1"
TRAINING="$RUN_DIR/training_v1"
PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_sap_existing_capture_development_protocol_v1.yaml"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABELS="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"
mkdir -p "$DATASET" "$TRAINING" "$MPLCONFIGDIR"
capture_cpu_run_context "$RUN_DIR"
cp "$PROTOCOL" "$RUN_DIR/context/"
cp "${PROTOCOL%.yaml}.sha256" "$RUN_DIR/context/"

set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_s0010_sap_dataset.py" \
    --protocol "$PROTOCOL" --run-root "$RUN_DIR" \
    --prediction "$PREDICTION" --segmentation "$SEGMENTATION" \
    --label-conversion "$LABELS" --phase development --output-dir "$DATASET" \
    > "$DATASET/builder-stdout.json" 2> "$DATASET/builder-stderr.log"
BUILD_RC=$?
set -e
echo "$BUILD_RC" > "$DATASET/builder-exit-code.txt"

READY_LEVELS=$(
    "$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(",".join(level for level,value in r["readiness"].items() if value["dataset_status"] == "READY"))' \
        "$DATASET/readiness_report.json"
)
echo "$READY_LEVELS" > "$DATASET/ready-levels.txt"
if [[ -z "$READY_LEVELS" ]]; then
    capture_cpu_run_end "$RUN_DIR"
    exit "$BUILD_RC"
fi

LEVEL_ARGS=()
IFS=',' read -ra LEVEL_ARRAY <<< "$READY_LEVELS"
for level in "${LEVEL_ARRAY[@]}"; do
    LEVEL_ARGS+=(--level "$level")
done
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_s0010_sap_classifier.py" \
    --dataset-dir "$DATASET" --output-dir "$TRAINING" "${LEVEL_ARGS[@]}" \
    > "$TRAINING/preflight-stdout.json" 2> "$TRAINING/preflight-stderr.log"
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_s0010_sap_classifier.py" \
    --dataset-dir "$DATASET" --output-dir "$TRAINING" --fit "${LEVEL_ARGS[@]}" \
    > "$TRAINING/training-stdout.json" 2> "$TRAINING/training-stderr.log"
capture_cpu_run_end "$RUN_DIR"
