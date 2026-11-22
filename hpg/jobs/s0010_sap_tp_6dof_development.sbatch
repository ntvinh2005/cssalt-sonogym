#!/usr/bin/env bash
#SBATCH --job-name=sg_6dof_dev
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
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_sap_tp_pair_confirmation_v1r3.yaml"
SOURCE_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_tp_pair_confirmation_v1r3/52308098"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_tp_6dof_development/$SLURM_JOB_ID"
export PYTHONPATH="$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"
mkdir -p "$RUN_DIR" "$MPLCONFIGDIR"
capture_cpu_run_context "$RUN_DIR"
cp "$PROTOCOL" "$RUN_DIR/context/"
sha256sum "$PROTOCOL" > "$RUN_DIR/context/protocol.sha256"

"$PYTHON" -m unittest -q \
    tests.test_acquisition_integrity \
    tests.test_ultrasound_spatial_map \
    > "$RUN_DIR/tests.log" 2>&1

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_s0010_sap_tp_6dof_development.py" \
    --protocol "$PROTOCOL" \
    --run-root "$SOURCE_RUN" \
    --output-dir "$RUN_DIR/analysis" \
    > "$RUN_DIR/stdout.json" 2> "$RUN_DIR/stderr.log"

capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
