#!/usr/bin/env bash
#SBATCH --job-name=sg_l5_sp_review
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail
: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
: "${SPATIAL_LOCALIZER_JOB_ID:?SPATIAL_LOCALIZER_JOB_ID was not exported}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_l5_spatial_localizer_protocol_v1.yaml"
SOURCE_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v3/52039283"
ANALYSIS_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_l5_spatial_localizer/$SPATIAL_LOCALIZER_JOB_ID"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_l5_spatial_localizer_review/$SLURM_JOB_ID"
export PYTHONPATH="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$RUN_DIR"
capture_cpu_run_context "$RUN_DIR"

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/build_s0010_l5_spatial_localizer_demo.py" \
    --analysis "$ANALYSIS_RUN/analysis" \
    --run-root "$SOURCE_RUN" \
    --protocol "$PROTOCOL" \
    --output-dir "$RUN_DIR/demo" \
    > "$RUN_DIR/demo_stdout.json" 2> "$RUN_DIR/demo_stderr.log"
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/package_s0010_l5_spatial_localizer.py" \
    --analysis-run "$ANALYSIS_RUN" \
    --demo-dir "$RUN_DIR/demo" \
    --project-dir "$SONOGYM_PROJECT_DIR" \
    --output-dir "$RUN_DIR" \
    > "$RUN_DIR/package_stdout.json" 2> "$RUN_DIR/package_stderr.log"

capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
