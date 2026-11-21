#!/usr/bin/env bash
#SBATCH --job-name=sg_us_repr_review
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
: "${SAP_REPRESENTATION_JOB:?SAP_REPRESENTATION_JOB must identify the frozen comparison}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_ultrasound_representation/$SAP_REPRESENTATION_JOB"
OUTPUT_DIR="$RUN_DIR/review"
EVALUATION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_resume_v2/52199322/analysis_v1/dataset_evaluation"
CAPTURE_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_resume_v2/52199322"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABEL_CONVERSION="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_ultrasound_representation_protocol_v1.yaml"
REPORT="$SONOGYM_PROJECT_DIR/docs/research/sonogym_s0010_ultrasound_representation.md"
export PYTHONPATH="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/build_s0010_ultrasound_representation_review.py" \
    --experiment-dir "$RUN_DIR" \
    --evaluation-dataset "$EVALUATION" \
    --capture-root "$CAPTURE_ROOT" \
    --prediction "$PREDICTION" \
    --segmentation "$SEGMENTATION" \
    --label-conversion "$LABEL_CONVERSION" \
    --protocol "$PROTOCOL" \
    --report "$REPORT" \
    --output-dir "$OUTPUT_DIR" \
    > "$RUN_DIR/review-stdout.json" 2> "$RUN_DIR/review-stderr.log"

"$PYTHON" -m zipfile -t "$OUTPUT_DIR/s0010_ultrasound_representation_review.zip"
sha256sum "$OUTPUT_DIR/s0010_ultrasound_representation_review.zip"
