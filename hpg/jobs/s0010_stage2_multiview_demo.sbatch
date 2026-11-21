#!/usr/bin/env bash
#SBATCH --job-name=sg_mv_demo
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
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

CHECKPOINT_JOB_ID="${MULTIVIEW_RANKER_JOB_ID:?MULTIVIEW_RANKER_JOB_ID was not exported}"
DEVELOPMENT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_resume_v2/52199322/analysis_v1/dataset_development_merged"
CHECKPOINT_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_stage2_multiview_ranker/$CHECKPOINT_JOB_ID"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_stage2_multiview_demo/$SLURM_JOB_ID"
export PYTHONPATH="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"
mkdir -p "$RUN_DIR" "$MPLCONFIGDIR"
capture_cpu_run_context "$RUN_DIR"

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/render_s0010_multiview_ranker_demo.py" \
    --development-dataset "$DEVELOPMENT" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$RUN_DIR" \
    --level L5 \
    --fps 3 \
    > "$RUN_DIR/stdout.json" 2> "$RUN_DIR/stderr.log"

capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
