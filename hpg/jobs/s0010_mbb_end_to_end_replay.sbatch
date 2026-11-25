#!/usr/bin/env bash
#SBATCH --job-name=sg_mbb_e2e_replay
#SBATCH --partition=hpg-default
#SBATCH --qos=anest-dept-b
#SBATCH --account=anest-dept
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail
[[ "$SLURM_JOB_ACCOUNT" == anest-dept ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -le 4 ]]
[[ "${SLURM_GPUS_ON_NODE:-0}" == 0 ]]
runtime="${1:?SonoGym runtime}"
study="${2:?Study directory}"
reconstruction="${3:?Reconstruction directory}"
reference="${4:?Frozen reference directory}"
profile=experiments/unity_two_stage_transfer/s0010_mbb_end_to_end_localization_v1.json
export PYTHONPATH="$PWD:$PWD/source/spinal_surgery/spinal_surgery/evaluation:$PWD/workflows/two_stage_navigation:$PWD/workflows/perception_s0010:$runtime/envs/stage1-python"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mbb-e2e-$SLURM_JOB_ID"
py="$runtime/envs/sonogym/bin/python"
git rev-parse HEAD
git status --short
if [[ -z "${5:-}" ]]; then
    "$py" -m workflows.mbb_end_to_end.evaluate --protocol "$profile" \
      --reconstruction "$reconstruction" --reference "$reference" --output "$study/replay"
    "$py" -m workflows.mbb_end_to_end.review recorded --protocol "$profile" --replay "$study/replay" \
      --reconstruction "$reconstruction" --output "$study/figures"
else
    session="$5"
    manifest="${6:?Source manifest}"
    cache_manifest="${7:?Historical cache manifest}"
    valid=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["valid_frame_count"])' "$session/actor_result.json")
    "$py" -m workflows.mbb_end_to_end.reconstruct --manifest "$manifest" --cache-manifest "$cache_manifest" \
      --reference "$reference" --output "$reconstruction" --session "$session" --seed 2026121700 \
      --expected-frames "$valid"
    "$py" -m workflows.mbb_end_to_end.evaluate_fresh --protocol "$profile" --reference "$reference" \
      --session "$session" --reconstruction "$reconstruction" --output "$study/fresh-evaluation"
    "$py" -m workflows.mbb_end_to_end.review fresh-video --session "$session" \
      --cloud "$reconstruction/2026121700/learned.npz" --prediction "$study/fresh-evaluation/E2_full.json" \
      --output "$study/video"
fi
