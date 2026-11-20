#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_data
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
: "${SAP_CAPTURE_JOB:?SAP_CAPTURE_JOB must identify the development capture}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

CAPTURE_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_development_v1/$SAP_CAPTURE_JOB"
OUTPUT="$CAPTURE_ROOT/dataset_v1"
PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_sap_validation_development_protocol_v1.yaml"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABELS="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
export PYTHONPATH="$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"
mkdir -p "$OUTPUT" "$MPLCONFIGDIR"

set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_s0010_sap_dataset.py" \
    --protocol "$PROTOCOL" --run-root "$CAPTURE_ROOT" \
    --prediction "$PREDICTION" --segmentation "$SEGMENTATION" \
    --label-conversion "$LABELS" --output-dir "$OUTPUT" \
    > "$OUTPUT/builder-stdout.json" 2> "$OUTPUT/builder-stderr.log"
BUILD_RC=$?
set -e
echo "$BUILD_RC" > "$OUTPUT/builder-exit-code.txt"
exit "$BUILD_RC"
