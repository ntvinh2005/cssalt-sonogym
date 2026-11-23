#!/usr/bin/env bash
#SBATCH --job-name=sg_clin_perception
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
: "${CLINICIAN_SCAN_EVALUATION_JOB_ID:?CLINICIAN_SCAN_EVALUATION_JOB_ID was not exported}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

CAPTURE_JOB_ID="${CLINICIAN_SCAN_CAPTURE_JOB_ID:-52392844}"
SCAN_DESIGN_VERSION="${CLINICIAN_SCAN_DESIGN_VERSION:-v1}"
[[ "$SCAN_DESIGN_VERSION" == "v1" || "$SCAN_DESIGN_VERSION" == "v2" ]] || exit 2
CAPTURE_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_clinician_scan_feasibility_${SCAN_DESIGN_VERSION}/$CAPTURE_JOB_ID"
PROTOCOL="$CAPTURE_ROOT/context/s0010_clinician_scan_feasibility_${SCAN_DESIGN_VERSION}.yaml"
EVALUATION_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_clinician_scan_evaluation_${SCAN_DESIGN_VERSION}/$CLINICIAN_SCAN_EVALUATION_JOB_ID"
FRAME_RESULTS="$EVALUATION_ROOT/evaluation/frame_results.csv"
SEGMENTATION_JOB_ID="${SONOANATOMY_SEGMENTATION_JOB_ID:-52387251}"
SEGMENTATION_NAMESPACE="${SONOANATOMY_SEGMENTATION_NAMESPACE:-s0010_sonoanatomy_segmentation_v1}"
TRAINING_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/$SEGMENTATION_NAMESPACE/$SEGMENTATION_JOB_ID/training"
L4_CHECKPOINT="$TRAINING_ROOT/s0010_l4_sonoanatomy_unet_v1.pt"
L5_CHECKPOINT="$TRAINING_ROOT/s0010_l5_sonoanatomy_unet_v1.pt"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_clinician_scan_perception_${SCAN_DESIGN_VERSION}/$SLURM_JOB_ID"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

[[ -f "$EVALUATION_ROOT/context/end-time.txt" ]] || {
    echo "Motion evaluation is incomplete: $EVALUATION_ROOT" >&2
    exit 2
}
[[ -f "$FRAME_RESULTS" ]] || {
    echo "Missing frame evaluation: $FRAME_RESULTS" >&2
    exit 2
}
mkdir -p "$RUN_DIR"
capture_cpu_run_context "$RUN_DIR"
"$PYTHON" -m unittest -q tests.test_sonoanatomy_multiview \
  tests.test_sonoanatomy_segmentation tests.test_acquisition_integrity \
  tests.test_s0010_clinician_scan_perception \
  > "$RUN_DIR/tests.log" 2>&1
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_s0010_clinician_scan_perception.py" \
  --protocol "$PROTOCOL" --run-root "$CAPTURE_ROOT" \
  --frame-results "$FRAME_RESULTS" \
  --l4-checkpoint "$L4_CHECKPOINT" --l5-checkpoint "$L5_CHECKPOINT" \
  --output-dir "$RUN_DIR/evaluation" \
  > "$RUN_DIR/stdout.json" 2> "$RUN_DIR/stderr.log"
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
