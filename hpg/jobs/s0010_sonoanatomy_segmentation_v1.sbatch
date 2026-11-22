#!/usr/bin/env bash
#SBATCH --job-name=sg_sono_seg
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
: "${SONOANATOMY_DATASET_JOB_ID:?SONOANATOMY_DATASET_JOB_ID was not exported}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

DATASET_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sonoanatomy_dataset_v1/$SONOANATOMY_DATASET_JOB_ID/dataset"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sonoanatomy_segmentation_v1/$SLURM_JOB_ID"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "$RUN_DIR"/{context,metrics}
capture_run_context "$RUN_DIR"
"$PYTHON" -m unittest -q tests.test_sonoanatomy_labels \
  tests.test_sonoanatomy_segmentation tests.test_acquisition_integrity \
  > "$RUN_DIR/tests.log" 2>&1
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
  --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_s0010_sonoanatomy_segmentation.py" \
  --dataset-root "$DATASET_ROOT" --output-dir "$RUN_DIR/training" \
  --device cuda --epochs 60 --batch-size 8 \
  > "$RUN_DIR/training.log" 2> "$RUN_DIR/training.stderr.log"
kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
trap - EXIT
capture_run_end "$RUN_DIR"
du -sbL "$RUN_DIR" > "$RUN_DIR/metrics/storage-bytes.txt"
find -L "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
