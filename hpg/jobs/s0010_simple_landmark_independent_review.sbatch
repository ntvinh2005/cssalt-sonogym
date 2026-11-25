#!/usr/bin/env bash
#SBATCH --job-name=sg_ind_review
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:10:00
set -euo pipefail
source "$SONOGYM_CONFIG_FILE"
[[ "$SLURM_JOB_ACCOUNT" == anest-dept && "${SLURM_GPUS_ON_NODE:-0}" -eq 0 ]]
[[ "$(squeue -A anest-dept --states=RUNNING,COMPLETING,CONFIGURING -h -o '%C' | awk '{s+=$1} END{print s+0}')" -le 4 ]]
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$PWD/source/spinal_surgery/spinal_surgery/evaluation:$PWD/workflows/two_stage_navigation:$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/sonogym-independent-review-$SLURM_JOB_ID"
campaign="$SIMPLE_MAPPING_OUTPUT_ROOT/$SIMPLE_MAPPING_CONFIRM_JOB"
protocol=experiments/unity_two_stage_transfer/s0010_simple_landmark_independent_v1.yaml
review="$SIMPLE_MAPPING_OUTPUT_ROOT/review"
renderer="${SIMPLE_MAPPING_REVIEW_CODE:-workflows/two_stage_navigation/review_independent_landmark_mapping.py}"
"$SONOGYM_ENV_PREFIX/bin/python" "$renderer" \
  --campaign "$campaign" --protocol "$protocol" --source-protocol "$SIMPLE_MAPPING_PROTOCOL" --output "$review"
cp "$protocol" "${protocol%.yaml}.sha256" "$review/"
replay="$SIMPLE_MAPPING_OUTPUT_ROOT/replay-$SIMPLE_MAPPING_REPLAY_JOB"
if [[ -f "$replay/revised_sap.json" ]]; then cp "$replay/revised_sap.json" "$review/replay_revised_sap.json"; fi
if [[ -f "$replay/results/comparison.json" ]]; then cp "$replay/results/comparison.json" "$review/replay_baseline.json"; fi
sha256sum "$renderer" > "$review/review-code.sha256"
if [[ -f "$campaign/attempts.tsv" ]]; then cp "$campaign/attempts.tsv" "$review/"; fi
if [[ -d "$campaign/context" ]]; then
    mkdir "$review/campaign-context"
    cp -a "$campaign/context/." "$review/campaign-context/"
fi
git rev-parse HEAD > "$review/review-commit.txt"
(
    cd "$SIMPLE_MAPPING_OUTPUT_ROOT"
    tar -czf simple-landmark-independent-review.tar.gz review
    tar -tzf simple-landmark-independent-review.tar.gz > review-files.txt
    sha256sum simple-landmark-independent-review.tar.gz > simple-landmark-independent-review.tar.gz.sha256
)
