#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_corr_cal
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
: "${SAP_CORRESPONDENCE_SOURCE_JOB:?SAP_CORRESPONDENCE_SOURCE_JOB is required}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

SOURCE="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_correspondence_cpu/$SAP_CORRESPONDENCE_SOURCE_JOB"
RUN="$SOURCE/opened_calibration_shape"
DATASET="$RUN/dataset"
EVALUATION="$RUN/evaluation"
CAPTURE="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_resume_v1/52182100"
PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_sap_validation_resume_protocol_v1.yaml"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABELS="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"
mkdir -p "$DATASET" "$EVALUATION" "$MPLCONFIGDIR"

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_s0010_sap_dataset.py" \
    --protocol "$PROTOCOL" --run-root "$CAPTURE" --prediction "$PREDICTION" \
    --segmentation "$SEGMENTATION" --label-conversion "$LABELS" \
    --phase development --require-all-proposals-clear-for-negative-image \
    --output-dir "$DATASET" > "$DATASET/stdout.json" 2> "$DATASET/stderr.log" || \
    test "$?" -eq 3

set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_s0010_sap_correspondence.py" \
    --dataset-dir "$DATASET" --protocol "$PROTOCOL" \
    --checkpoint-dir "$SOURCE/selected" --split calibration \
    --output-dir "$EVALUATION" > "$EVALUATION/stdout.json" 2> "$EVALUATION/stderr.log"
EVALUATION_RC=$?
set -e
echo "$EVALUATION_RC" > "$EVALUATION/exit-code.txt"
find "$RUN" -type f -print | sort > "$RUN/files.txt"
