#!/usr/bin/env bash
#SBATCH --job-name=sg_robust_diag
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail
[[ "$(squeue -A anest-dept --states=RUNNING,COMPLETING,CONFIGURING -h -o '%C' | awk '{s+=$1} END {print s+0}')" -le 4 ]]
checkout="$PWD"
: "${SONOGYM_CONFIG_FILE:?}"
source "$SONOGYM_CONFIG_FILE"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$checkout/source/spinal_surgery/spinal_surgery/evaluation:$checkout/workflows/two_stage_navigation:$checkout"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sonogym-robust-diagnostics-$SLURM_JOB_ID"
python="$SONOGYM_ENV_PREFIX/bin/python"
destination="$ROBUST_OUTPUT/diagnostics-$SLURM_JOB_ID"
mkdir "$destination"
git rev-parse HEAD > "$destination/execution-commit.txt"
original="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sonoanatomy_segmentation_v9/52498821/training/l5_split_manifest.csv"
"$python" workflows/two_stage_navigation/render_robust_review.py \
    --acquisition "$ROBUST_ACQUISITION" --review "$ROBUST_REVIEW" --output "$destination"
"$python" workflows/two_stage_navigation/summarize_robust_research.py \
    --review "$ROBUST_REVIEW" --original-manifest "$original" --output "$destination/summary.json"
if [[ -s "$ROBUST_REVIEW/E3-training/training_result.json" ]]; then
    "$python" workflows/two_stage_navigation/validate_robust_l5_mapping.py \
        --dataset "$ROBUST_REVIEW/E3-dataset" --training "$ROBUST_REVIEW/E3-training" \
        --protocol "$ROBUST_PROTOCOL" --output "$destination/E3_heldout_L5_mapping.json"
fi
if [[ -s "$ROBUST_REVIEW/demo/revised_systematic_sweep.mp4"
      && -s "$ROBUST_REVIEW/E3-dataset/dataset.npz" ]]; then
    "$python" workflows/two_stage_navigation/audit_robust_acquisition.py \
        --acquisition "$ROBUST_ACQUISITION" --review "$ROBUST_REVIEW" --output "$destination/acquisition_audit.json"
fi
