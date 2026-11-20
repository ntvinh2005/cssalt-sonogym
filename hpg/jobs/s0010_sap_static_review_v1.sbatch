#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_review
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_static_review_v1/$SLURM_JOB_ID"
OUTPUT="$RUN_DIR/review_packages"

mkdir -p "$RUN_DIR"/{context,metrics} "$OUTPUT"
capture_cpu_run_context "$RUN_DIR"

/usr/bin/time -v -o "$RUN_DIR/metrics/time-tests.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_s0010_sap_static_review.py" \
    > "$RUN_DIR/tests.log" 2>&1
/usr/bin/time -v -o "$RUN_DIR/metrics/time-build.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/build_s0010_sap_static_review.py" \
        --asset-root "$SONOGYM_ASSETS_DATA_DIR/HumanModels" \
        --geometry-root "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_right_sap_tp_correction/51779234" \
        --v3-root "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v3/52039283" \
        --audit-root "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_geometry_audit_v1/52053307/audit" \
        --output-root "$OUTPUT" \
        --surface-slab-tolerance-mm 1.0 \
        > "$RUN_DIR/build.log" 2>&1

capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
