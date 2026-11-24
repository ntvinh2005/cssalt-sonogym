#!/usr/bin/env bash
#SBATCH --job-name=sg_simple_render
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:10:00

set -euo pipefail
: "${SONOGYM_CONFIG_FILE:?}"
: "${SIMPLE_MAPPING_OFFLINE_DIR:?}"
source "$SONOGYM_CONFIG_FILE"
[[ "$SLURM_JOB_ACCOUNT" == anest-dept && "${SLURM_GPUS_ON_NODE:-0}" -eq 0 ]]
running_cpus="$(squeue -A anest-dept --states=RUNNING,COMPLETING,CONFIGURING -h -o '%C' | awk '{sum+=$1} END{print sum+0}')"
[[ "$running_cpus" -le 4 ]]

export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/sonogym-simple-render-$SLURM_JOB_ID"
offline="$SIMPLE_MAPPING_OFFLINE_DIR"
python="$SONOGYM_ENV_PREFIX/bin/python"
"$python" workflows/two_stage_navigation/render_simple_landmark_mapping.py \
  --results "$offline/results" --output "$offline/visualizations"
git rev-parse HEAD > "$offline/visualizations/git-commit.txt"
(
    cd "$offline"
    tar -czf simple-landmark-mapping-review.tar.gz \
      results visualizations slurm-job.txt tests.log git-commit.txt pip-freeze.txt \
      start-time.txt end-time.txt
    sha256sum simple-landmark-mapping-review.tar.gz > simple-landmark-mapping-review.tar.gz.sha256
    tar -tzf simple-landmark-mapping-review.tar.gz > review-files.txt
)
