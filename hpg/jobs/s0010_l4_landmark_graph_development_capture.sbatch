#!/usr/bin/env bash
#SBATCH --job-name=sg_l4graph_cap
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail
: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
: "${L4_GRAPH_PROTOCOL:?L4_GRAPH_PROTOCOL was not exported}"
: "${EXPECTED_PROTOCOL_SHA:?EXPECTED_PROTOCOL_SHA was not exported}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

PROTOCOL="$L4_GRAPH_PROTOCOL"
PROTOCOL_SEAL="${PROTOCOL%.yaml}.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_l4_landmark_graph_development/$SLURM_JOB_ID"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

[[ "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" == "$EXPECTED_PROTOCOL_SHA" ]] || {
  echo "protocol SHA-256 changed" >&2; exit 2;
}
mkdir -p "$RUN_DIR"/{sessions,results,metrics,context}
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$PROTOCOL_SEAL" "$RUN_DIR/context/"
(cd "$RUN_DIR/context" && sha256sum -c "$(basename "$PROTOCOL_SEAL")")
"$PYTHON" -m unittest -q tests.test_two_stage_contracts \
  tests.test_acquisition_integrity tests.test_ultrasound_spatial_map \
  > "$RUN_DIR/results/preflight-tests.log" 2>&1

for session_id in \
  s0010l4graphdev_00 s0010l4graphdev_01 \
  s0010l4graphdev_02 s0010l4graphdev_03; do
  SESSION_DIR="$RUN_DIR/sessions/$session_id"
  mkdir -p "$SESSION_DIR"
  "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/collect_stage2_dataset.py" \
    --protocol "$PROTOCOL" --session-id "$session_id" --output-dir "$SESSION_DIR" \
    --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" --headless --device cuda:0 \
    > "$SESSION_DIR/launcher.log" 2>&1
  "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/verify_exact_pose_capture.py" \
    --protocol "$PROTOCOL" --session-dir "$SESSION_DIR" \
    --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" \
    --output "$SESSION_DIR/verification_report.json" \
    > "$SESSION_DIR/verifier.log" 2>&1
  "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/validate_actor_6dof_capture.py" \
    --protocol "$PROTOCOL" --run-root "$RUN_DIR" --session-id "$session_id" \
    --output "$SESSION_DIR/actor_6dof_validation.json" \
    > "$SESSION_DIR/actor_6dof_validator.log" 2>&1
done
capture_run_end "$RUN_DIR"
find -L "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
