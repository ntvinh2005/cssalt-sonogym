#!/usr/bin/env bash
#SBATCH --job-name=sg_clin_scan_eval
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail
: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

CAPTURE_JOB_ID="${CLINICIAN_SCAN_CAPTURE_JOB_ID:-52392844}"
SCAN_DESIGN_VERSION="${CLINICIAN_SCAN_DESIGN_VERSION:-v1}"
[[ "$SCAN_DESIGN_VERSION" == "v1" || "$SCAN_DESIGN_VERSION" == "v2" ]] || exit 2
CAPTURE_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_clinician_scan_feasibility_${SCAN_DESIGN_VERSION}/$CAPTURE_JOB_ID"
PROTOCOL="$CAPTURE_ROOT/context/s0010_clinician_scan_feasibility_${SCAN_DESIGN_VERSION}.yaml"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABEL_CONVERSION="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_clinician_scan_evaluation_${SCAN_DESIGN_VERSION}/$SLURM_JOB_ID"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"

[[ -f "$CAPTURE_ROOT/context/end-time.txt" ]] || {
    echo "Capture is incomplete: $CAPTURE_ROOT" >&2
    exit 2
}
mkdir -p "$RUN_DIR" "$MPLCONFIGDIR"
capture_cpu_run_context "$RUN_DIR"
"$PYTHON" -m unittest -q tests.test_s0010_clinician_scan_evaluation \
  tests.test_sonoanatomy_labels tests.test_acquisition_integrity \
  > "$RUN_DIR/tests.log" 2>&1
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_s0010_clinician_scan.py" \
  --protocol "$PROTOCOL" --run-root "$CAPTURE_ROOT" \
  --prediction "$PREDICTION" --segmentation "$SEGMENTATION" \
  --label-conversion "$LABEL_CONVERSION" --output-dir "$RUN_DIR/evaluation" \
  > "$RUN_DIR/stdout.json" 2> "$RUN_DIR/stderr.log"
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
