#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit helper}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

RUN_DIR="$SONOGYM_OUTPUT_DIR/ppo/$SLURM_JOB_ID"
LOG_ROOT="$RUN_DIR/logs"
mkdir -p "$LOG_ROOT" "$RUN_DIR/metrics"
capture_run_context "$RUN_DIR"

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-before.csv"

/usr/bin/time -v -o "$RUN_DIR/metrics/time.txt" \
python workflows/skrl/train.py \
    --task Isaac-robot-US-guidance-v0 \
    --algorithm PPO \
    --num_envs 4 \
    --seed 20261106 \
    --max_iterations 10 \
    --log_root "$LOG_ROOT" \
    --wandb_mode disabled \
    --headless \
    --device cuda:0 \
    agent.mini_batches=4 \
    agent.experiment.checkpoint_interval=160

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-after.csv"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true

find "$LOG_ROOT" -type f -print | sort > "$RUN_DIR/files.txt"
if ! find "$LOG_ROOT" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) -print -quit | grep -q .; then
    echo "PPO pilot finished without a checkpoint" >&2
    exit 3
fi

printf '%s\n' \
    'PIPELINE_PILOT_ONLY=true' \
    'ppo_updates=10' \
    'rollout_vector_steps_per_update=32' \
    'parallel_environments=4' \
    'total_vector_steps=320' \
    'total_collected_transitions=1280' \
    > "$RUN_DIR/pilot-summary.txt"
capture_run_end "$RUN_DIR"
