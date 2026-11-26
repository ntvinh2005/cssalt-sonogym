#!/usr/bin/env bash
#SBATCH --job-name=sg_mbb_local
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept
#SBATCH --partition=hpg-turin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=30G
#SBATCH --time=00:10:00

set -euo pipefail
checkout="$PWD"
source "$checkout/hpg/jobs/common.sh"
SONOGYM_PROJECT_DIR="$checkout"
export SONOGYM_GUIDANCE_CONFIG="$checkout/config/guidance_baseline.yaml"
export PYTHONPATH="$checkout/source/spinal_surgery:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$checkout/source/spinal_surgery/spinal_surgery/evaluation:$checkout"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sonogym-mbb-local-$SLURM_JOB_ID"

study="${1:?study directory}"
reference="${2:?previous 126-frame capture context}"
stage1="${3:?Stage 1 sealed actor}"
prior="${4:?previous direct sonographic replay}"
destination="$study/capture-$SLURM_JOB_ID"
capture_run_context "$destination"
(cd "$reference" && sha256sum -c protocol.sha256 && sha256sum -c revision.sha256)
segmentation="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
"$PYTHON" workflows/two_stage_navigation/capture_simple_landmark_sweep.py \
  --protocol "$reference/protocol.yaml" --revision "$reference/revision.yaml" \
  --stage1-actor "$stage1" --stage1-session cf7_00 --seed 2026121700 \
  --robustness-arm E1-M --robustness-profile "$reference/profile.yaml" \
  --local-verification experiments/unity_two_stage_transfer/mbb_local_active_verification_v1.json \
  --verification-inputs "$prior" --headless --device cuda:0 --enable_cameras --record-robot \
  --evaluator-segmentation "$segmentation" --output "$destination/session"
capture_run_end "$destination"
