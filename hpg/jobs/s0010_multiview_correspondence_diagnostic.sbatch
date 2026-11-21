#!/usr/bin/env bash
#SBATCH --job-name=sg_mv_diag
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
: "${MULTIVIEW_JOB:?MULTIVIEW_JOB must identify the failed frozen experiment}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_stage2_multiview_evidence_protocol_v1r1.yaml"
DEVELOPMENT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_resume_v2/52199322/analysis_v1/dataset_development_merged"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_stage2_multiview_evidence/$MULTIVIEW_JOB"
export PYTHONPATH="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/diagnose_s0010_multiview_correspondence.py" \
    --protocol "$PROTOCOL" \
    --development-dataset "$DEVELOPMENT" \
    --output "$RUN_DIR/calibration_correspondence_diagnostic.json" \
    > "$RUN_DIR/diagnostic-stdout.json" 2> "$RUN_DIR/diagnostic-stderr.log"
