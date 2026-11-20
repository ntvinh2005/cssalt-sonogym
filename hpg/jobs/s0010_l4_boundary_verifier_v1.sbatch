#!/usr/bin/env bash
#SBATCH --job-name=sg_l4_verifier
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

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_l4_boundary_verifier_v1.yaml"
PROTOCOL_SHA="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_l4_boundary_verifier_v1.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_l4_boundary_verifier_v1/$SLURM_JOB_ID"
OUTPUT="$RUN_DIR/verifier"

mkdir -p "$RUN_DIR"/{context,metrics} "$OUTPUT"
capture_cpu_run_context "$RUN_DIR"
cp "$PROTOCOL" "$PROTOCOL_SHA" "$RUN_DIR/context/"
(cd "$RUN_DIR/context" && sha256sum -c "$(basename "$PROTOCOL_SHA")")
sha256sum "$PROTOCOL" > "$RUN_DIR/context/frozen-inputs.sha256"

/usr/bin/time -v -o "$RUN_DIR/metrics/time-tests.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_s0010_l4_boundary_verifier.py" \
    > "$RUN_DIR/tests.log" 2>&1
/usr/bin/time -v -o "$RUN_DIR/metrics/time-verifier.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_s0010_l4_boundary_verifier.py" \
        --protocol "$PROTOCOL" \
        --output-dir "$OUTPUT" \
        > "$RUN_DIR/verifier.log" 2>&1

capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
