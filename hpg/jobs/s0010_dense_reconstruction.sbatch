#!/usr/bin/env bash
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --job-name=sg_stage2_dense
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00

set -euo pipefail
manifest="$1"
seal="$2"
dest="$3"
phase="$4"
runtime="$5"
checkout="$PWD"
[[ "$SLURM_JOB_ACCOUNT" = anest-dept ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -le 4 ]]
[[ "${SLURM_GPUS_ON_NODE:-0}" -eq 0 ]]
[[ -z "$(git status --short)" ]]
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export PYTHONPATH="$checkout/workflows/reconstruction_s0010:$checkout/workflows/perception_s0010:$checkout/workflows/two_stage_navigation:$checkout/source/spinal_surgery/spinal_surgery/evaluation:$runtime/envs/stage1-python"
mkdir -p "$dest"
py="$runtime/envs/sonogym/bin/python"
printf 'job_id=%s\nphase=%s\n' "$SLURM_JOB_ID" "$phase" > "$dest/$phase-run.txt"
git rev-parse HEAD >> "$dest/$phase-run.txt"
"$py" --version >> "$dest/$phase-run.txt"
date --iso-8601=seconds >> "$dest/$phase-run.txt"
if [[ "$phase" == both ]]; then
    run_phases=(oracle learned)
else
    run_phases=("$phase")
fi
for reconstruction_phase in "${run_phases[@]}"; do
    "$py" -m dense --manifest "$manifest" --manifest-sha256 "$seal" --output "$dest" --phase "$reconstruction_phase"
done
date --iso-8601=seconds >> "$dest/$phase-run.txt"
