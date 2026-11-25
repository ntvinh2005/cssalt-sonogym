#!/usr/bin/env bash
#SBATCH --job-name=sg_perception
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept
#SBATCH --partition=hpg-turin
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=40G
#SBATCH --time=00:20:00

set -euo pipefail
checkout="$PWD"
[[ "$(squeue -A anest-dept --states=RUNNING,COMPLETING,CONFIGURING -h -o '%C' | awk '{s+=$1} END {print s+0}')" -le 4 ]]
source "$checkout/hpg/jobs/common.sh"
SONOGYM_PROJECT_DIR="$checkout"
export SONOGYM_GUIDANCE_CONFIG="$checkout/config/guidance_baseline.yaml"
export PYTHONPATH="$checkout/source/spinal_surgery:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$checkout/source/spinal_surgery/spinal_surgery/evaluation:$checkout/workflows/two_stage_navigation:$checkout/workflows/perception_s0010:$checkout"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sonogym-perception-$SLURM_JOB_ID"
[[ -z "$(git status --short)" && "$(git rev-parse HEAD)" == "$CAMPAIGN_COMMIT" ]]
export PYTHON SONOGYM_PROJECT_DIR SONOGYM_ASSETS_DATA_DIR
if [[ "$1" == development || "$1" == confirmation ]]; then
    source workflows/perception_s0010/acquire.sh "$1"
else
    source workflows/perception_s0010/"$1".sh
fi
