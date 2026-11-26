#!/usr/bin/env bash
#SBATCH --job-name=sg_mbb_ssm
#SBATCH --partition=hpg-default
#SBATCH --qos=anest-dept-b
#SBATCH --account=anest-dept
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:35:00

set -euo pipefail
runtime="${1:?Specify pinned SonoGym runtime}"
study="${2:?Specify the campaign output directory}"
phase="${3:?Specify development, heldout, transfer or review}"
previous="${4:?Specify the preceding shape-completion artifact root}"
[[ "$SLURM_JOB_ACCOUNT" = anest-dept ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -le 4 ]]
[[ "${SLURM_GPUS_ON_NODE:-0}" = 0 ]]
export PYTHONPATH="$PWD:$PWD/source/spinal_surgery/spinal_surgery/evaluation:$PWD/workflows/two_stage_navigation:$PWD/workflows/perception_s0010:$runtime/envs/stage1-python"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" VTK_SMP_MAX_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mbb-ssm-$SLURM_JOB_ID"
python_exe="$runtime/envs/sonogym/bin/python"
protocol=experiments/unity_two_stage_transfer/mbb_ssm_v1.json
baseline=experiments/unity_two_stage_transfer/s0010_mbb_shape_completion_v1.json
cohort="$study/cohort-expanded"
prior="$study/prior-expanded"
model="$prior/model.npz"
freeze="$study/development/freeze.json"
sha256sum -c "${protocol%.json}.sha256"
git rev-parse HEAD
git status --short

if [[ "$phase" == development ]]; then
    "$python_exe" -m workflows.mbb_ssm.benchmark validate --cohort "$cohort" --model "$model" \
        --protocol "$protocol" --output "$study/development"
elif [[ "$phase" == heldout ]]; then
    "$python_exe" -m workflows.mbb_ssm.benchmark prepare-test --cohort "$cohort" \
        --protocol "$protocol" --output "$study/heldout"
    "$python_exe" -m workflows.mbb_ssm.fit --model "$model" --freeze "$freeze" --protocol "$protocol" \
        --inputs "$study/heldout/inputs" --output "$study/heldout/predictions" --stress
    "$python_exe" -m workflows.mbb_ssm.benchmark score-test --cohort "$cohort" \
        --protocol "$protocol" --output "$study/heldout" --baseline-protocol "$baseline" \
        --predictions "$study/heldout/predictions"
elif [[ "$phase" == transfer ]]; then
    "$python_exe" -m workflows.mbb_ssm.fit --model "$model" --freeze "$freeze" --protocol "$protocol" \
        --inputs "$study/transfer-inputs" --output "$study/transfer-predictions" --stress
    "$python_exe" -m workflows.mbb_ssm.transfer score --previous "$previous" \
        --protocol "$protocol" --output "$study/transfer" --inputs "$study/transfer-inputs" \
        --predictions "$study/transfer-predictions" --cohort "$cohort" --prior "$prior"
elif [[ "$phase" == review ]]; then
    "$python_exe" -m workflows.mbb_ssm.review --study "$study" --output "$study/figures"
else
    printf 'Unknown phase %s\n' "$phase" >&2
    exit 2
fi
