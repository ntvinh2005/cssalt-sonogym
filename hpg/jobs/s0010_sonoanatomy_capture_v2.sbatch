#!/usr/bin/env bash
#SBATCH --job-name=sg_sono_cap2
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail
: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
: "${SONOANATOMY_CAPTURE_PROTOCOL:?SONOANATOMY_CAPTURE_PROTOCOL was not exported}"
: "${EXPECTED_PROTOCOL_SHA:?EXPECTED_PROTOCOL_SHA was not exported}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

PROTOCOL="$SONOANATOMY_CAPTURE_PROTOCOL"
PROTOCOL_SEAL="${PROTOCOL%.yaml}.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sonoanatomy_capture_v2/$SLURM_JOB_ID"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"

[[ "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" == "$EXPECTED_PROTOCOL_SHA" ]] || exit 2
mkdir -p "$RUN_DIR"/{sessions,results,metrics,context} "$MPLCONFIGDIR"
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$PROTOCOL_SEAL" "$RUN_DIR/context/"
(cd "$RUN_DIR/context" && sha256sum -c "$(basename "$PROTOCOL_SEAL")")
"$PYTHON" -m unittest -q tests.test_acquisition_integrity \
  tests.test_two_stage_contracts tests.test_sonoanatomy_labels \
  > "$RUN_DIR/results/preflight-tests.log" 2>&1

nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
  --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT

for level in l4 l5; do
  for session_index in 00 01 02 03; do
    session_id="s0010sonodevv2${level}_${session_index}"
    session_dir="$RUN_DIR/sessions/$session_id"
    mkdir -p "$session_dir"
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/collect_stage2_dataset.py" \
      --protocol "$PROTOCOL" --session-id "$session_id" --output-dir "$session_dir" \
      --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" --headless --device cuda:0 \
      > "$session_dir/launcher.log" 2>&1
    "$PYTHON" -c 'import json,sys; summary=json.load(open(sys.argv[1])); raise SystemExit(0 if summary.get("status") == "PASSED" else 2)' \
      "$session_dir/summary.json"
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/verify_exact_pose_capture.py" \
      --protocol "$PROTOCOL" --session-dir "$session_dir" \
      --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" \
      --output "$session_dir/verification_report.json" \
      > "$session_dir/verifier.log" 2>&1
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/validate_actor_6dof_capture.py" \
      --protocol "$PROTOCOL" --run-root "$RUN_DIR" --session-id "$session_id" \
      --output "$session_dir/actor_6dof_validation.json" \
      > "$session_dir/actor_6dof_validator.log" 2>&1
  done
done

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
trap - EXIT
capture_run_end "$RUN_DIR"
du -sbL "$RUN_DIR" > "$RUN_DIR/metrics/storage-bytes.txt"
find -L "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
