#!/usr/bin/env bash
#SBATCH --job-name=sg_final_mbb_registration
#SBATCH --partition=hpg-default
#SBATCH --qos=anest-dept-b
#SBATCH --account=anest-dept
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:18:00

set -euo pipefail
runtime="${1:?Specify SonoGym runtime}"
study="${2:?Specify campaign output}"
phase="${3:?Specify development or final}"
previous="${4:?Specify fixed-shape source}"
shape_previous="${5:?Specify reference source}"
[[ "$SLURM_JOB_ACCOUNT" = anest-dept ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -le 4 ]]
[[ "${SLURM_GPUS_ON_NODE:-0}" = 0 ]]
export PYTHONPATH="$PWD:$PWD/source/spinal_surgery/spinal_surgery/evaluation:$PWD/workflows/two_stage_navigation:$PWD/workflows/perception_s0010:$runtime/envs/stage1-python"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" VTK_SMP_MAX_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mbb-final-registration-$SLURM_JOB_ID"
python_exe="$runtime/envs/sonogym/bin/python"
sha256sum -c experiments/unity_two_stage_transfer/mbb_final_registration_v1.sha256
git rev-parse HEAD
git status --short
if [[ "$phase" == development ]]; then
    "$python_exe" -m workflows.mbb_final_registration.predict --stage development --previous "$previous" \
        --inputs "$study/development/inputs" --output "$study/development/predictions"
    "$python_exe" -m workflows.mbb_final_registration.evaluate development --previous "$previous" --study "$study"
elif [[ "$phase" == final ]]; then
    "$python_exe" -m workflows.mbb_final_registration.predict --stage external --previous "$previous" \
        --inputs "$study/external-confirmation/inputs" --output "$study/external-confirmation/predictions" \
        --freeze "$study/freeze.json"
    "$python_exe" -m workflows.mbb_final_registration.predict --stage transfer --previous "$previous" \
        --inputs "$previous/transfer-inputs" --output "$study/transfer-predictions" --freeze "$study/freeze.json"
    "$python_exe" -m workflows.mbb_final_registration.evaluate external --previous "$previous" --study "$study"
    "$python_exe" -m workflows.mbb_final_registration.evaluate transfer --previous "$previous" --study "$study" \
        --shape-previous "$shape_previous"
else
    printf 'Unknown phase %s\n' "$phase" >&2
    exit 2
fi
