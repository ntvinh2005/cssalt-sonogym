#!/usr/bin/env bash
#SBATCH --job-name=sg_sono_hardprep
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
: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
: "${SONOANATOMY_CAPTURE_JOB_ID:?SONOANATOMY_CAPTURE_JOB_ID was not exported}"
: "${SONOANATOMY_DATASET_JOB_ID:?SONOANATOMY_DATASET_JOB_ID was not exported}"
: "${SONOANATOMY_PERCEPTION_JOB_ID:?SONOANATOMY_PERCEPTION_JOB_ID was not exported}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

CAPTURE_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sonoanatomy_capture_v2/$SONOANATOMY_CAPTURE_JOB_ID"
DATASET_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sonoanatomy_dataset_v3/$SONOANATOMY_DATASET_JOB_ID/dataset"
PERCEPTION_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sonoanatomy_capture_perception_v2/$SONOANATOMY_PERCEPTION_JOB_ID/evaluation"
BASE_PROTOCOL="$CAPTURE_ROOT/context/s0010_sonoanatomy_development_v2.yaml"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sonoanatomy_hard_negative_prepare_v1/$SLURM_JOB_ID"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "$RUN_DIR"
capture_cpu_run_context "$RUN_DIR"
"$PYTHON" -m unittest -q tests.test_acquisition_integrity \
  tests.test_sonoanatomy_capture_v2 > "$RUN_DIR/tests.log" 2>&1
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_s0010_sonoanatomy_hard_negative_capture.py" \
  --base-protocol "$BASE_PROTOCOL" --dataset-manifest "$DATASET_ROOT/manifest.csv" \
  --actor-frames "$PERCEPTION_ROOT/actor/frames.jsonl" \
  --source-capture-root "$CAPTURE_ROOT" \
  --implementation-commit "$(git -C "$SONOGYM_PROJECT_DIR" rev-parse HEAD)" \
  --output-dir "$RUN_DIR/prepared" > "$RUN_DIR/prepare.log" 2> "$RUN_DIR/prepare.stderr.log"
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
