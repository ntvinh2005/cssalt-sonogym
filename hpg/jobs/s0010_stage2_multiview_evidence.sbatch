#!/usr/bin/env bash
#SBATCH --job-name=sg_multiview
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_stage2_multiview_evidence_protocol_v2.yaml"
DEVELOPMENT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_resume_v2/52199322/analysis_v1/dataset_development_merged"
EVALUATION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_resume_v2/52199322/analysis_v1/dataset_evaluation"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_stage2_multiview_evidence/$SLURM_JOB_ID"
export PYTHONPATH="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"
mkdir -p "$RUN_DIR" "$MPLCONFIGDIR"
capture_cpu_run_context "$RUN_DIR"
cp "$PROTOCOL" "${PROTOCOL%.yaml}.sha256" "$RUN_DIR/context/"
(cd "$RUN_DIR/context" && sha256sum -c "$(basename "${PROTOCOL%.yaml}.sha256")")

"$PYTHON" -m unittest -q \
    tests.test_ultrasound_multiview \
    tests.test_s0010_stage2_multiview_evidence \
    tests.test_ultrasound_surface_features \
    > "$RUN_DIR/tests.log" 2>&1
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_s0010_stage2_multiview_evidence.py" \
    --protocol "$PROTOCOL" \
    --development-dataset "$DEVELOPMENT" \
    --evaluation-dataset "$EVALUATION" \
    --output-dir "$RUN_DIR" \
    --source-commit "$(git -C "$SONOGYM_PROJECT_DIR" rev-parse HEAD)" \
    > "$RUN_DIR/stdout.json" 2> "$RUN_DIR/stderr.log"

capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
