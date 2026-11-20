#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_train
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
: "${SAP_CAPTURE_JOB:?SAP_CAPTURE_JOB must identify the development capture}"
: "${SAP_LEVELS:?SAP_LEVELS must be L4, L5, or L4,L5}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

CAPTURE_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_development_v1/$SAP_CAPTURE_JOB"
DATASET="$CAPTURE_ROOT/dataset_v1"
OUTPUT="$CAPTURE_ROOT/training_v1"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$OUTPUT"

LEVEL_ARGS=()
IFS=',' read -ra LEVEL_ARRAY <<< "$SAP_LEVELS"
for level in "${LEVEL_ARRAY[@]}"; do
    LEVEL_ARGS+=(--level "$level")
done

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_s0010_sap_classifier.py" \
    --dataset-dir "$DATASET" --output-dir "$OUTPUT" "${LEVEL_ARGS[@]}" \
    > "$OUTPUT/preflight-stdout.json" 2> "$OUTPUT/preflight-stderr.log"
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_s0010_sap_classifier.py" \
    --dataset-dir "$DATASET" --output-dir "$OUTPUT" --fit "${LEVEL_ARGS[@]}" \
    > "$OUTPUT/training-stdout.json" 2> "$OUTPUT/training-stderr.log"
