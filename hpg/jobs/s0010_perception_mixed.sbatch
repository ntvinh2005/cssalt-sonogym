#!/usr/bin/env bash
#SBATCH --job-name=sg_mix2000
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept
#SBATCH --partition=hpg-turin
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=40G
#SBATCH --time=00:25:00

set -euo pipefail
checkout="$PWD"
[[ "$(squeue -A anest-dept --states=RUNNING,COMPLETING,CONFIGURING -h -o '%C' | awk '{s+=$1} END {print s+0}')" -le 4 ]]
source hpg/jobs/common.sh
SONOGYM_PROJECT_DIR="$checkout"
export SONOGYM_GUIDANCE_CONFIG="$checkout/config/guidance_baseline.yaml"
export PYTHONPATH="$checkout/source/spinal_surgery:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$checkout/source/spinal_surgery/spinal_surgery/evaluation:$checkout/workflows/two_stage_navigation:$checkout/workflows/perception_s0010:$checkout/workflows/perception_mixed_s0010:$checkout"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sonogym-mixed-$SLURM_JOB_ID"
[[ -z "$(git status --short)" && "$(git rev-parse HEAD)" == "$MIXED_COMMIT" ]]
out="$MIXED_OUTPUT/training-$SLURM_JOB_ID"
capture_run_context "$out"
cp "$MIXED_PROFILE" "$out/context/campaign.yaml"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv --loop=10 >"$out/gpu.csv" &
telemetry=$!
trap 'kill "$telemetry" 2>/dev/null || true' EXIT
"$PYTHON" workflows/perception_mixed_s0010/train.py --protocol "$MIXED_PROTOCOL" --profile "$MIXED_PROFILE" \
    --previous "$MIXED_PREVIOUS" --data "$MIXED_DATA" --legacy-dataset "$MIXED_LEGACY_DATASET" \
    --legacy-training "$MIXED_LEGACY_TRAINING" --pilot-manifest "$MIXED_PILOT_MANIFEST" \
    --diagnosis "$MIXED_DIAGNOSIS/diagnosis.json" --source-hashes "$MIXED_DIAGNOSIS/evaluation_image_hashes.json" \
    --output "$out/fit"
capture_run_end "$out"
