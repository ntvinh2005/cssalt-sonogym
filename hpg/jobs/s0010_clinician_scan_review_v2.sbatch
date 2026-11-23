#!/usr/bin/env bash
#SBATCH --job-name=sg_clin_review
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
: "${CLINICIAN_SCAN_CAPTURE_JOB_ID:?CLINICIAN_SCAN_CAPTURE_JOB_ID was not exported}"
: "${CLINICIAN_SCAN_EVALUATION_JOB_ID:?CLINICIAN_SCAN_EVALUATION_JOB_ID was not exported}"
: "${CLINICIAN_SCAN_PERCEPTION_JOB_ID:?CLINICIAN_SCAN_PERCEPTION_JOB_ID was not exported}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

CAPTURE_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_clinician_scan_feasibility_v2/$CLINICIAN_SCAN_CAPTURE_JOB_ID"
MOTION_EVALUATION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_clinician_scan_evaluation_v2/$CLINICIAN_SCAN_EVALUATION_JOB_ID/evaluation"
PERCEPTION_EVALUATION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_clinician_scan_perception_v2/$CLINICIAN_SCAN_PERCEPTION_JOB_ID/evaluation"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_clinician_scan_review_v2/$SLURM_JOB_ID"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "$RUN_DIR"
capture_cpu_run_context "$RUN_DIR"
"$PYTHON" -m unittest -q tests.test_s0010_clinician_scan_review \
  tests.test_s0010_clinician_scan_perception \
  > "$RUN_DIR/tests.log" 2>&1
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/build_s0010_clinician_scan_review.py" \
  --capture-root "$CAPTURE_ROOT" \
  --motion-evaluation "$MOTION_EVALUATION" \
  --perception-evaluation "$PERCEPTION_EVALUATION" \
  --output-dir "$RUN_DIR/review" \
  --fps 8 --repeats-per-observation 2 \
  > "$RUN_DIR/stdout.json" 2> "$RUN_DIR/stderr.log"
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
