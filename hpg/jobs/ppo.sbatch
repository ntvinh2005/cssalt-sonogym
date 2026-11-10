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
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/time.txt" \
"$PYTHON" workflows/skrl/train.py \
    --task Isaac-robot-US-guidance-v0 \
    --algorithm PPO \
    --num_envs 4 \
    --seed 20261106 \
    --max_iterations 10 \
    --log_root "$LOG_ROOT" \
    --wandb_mode disabled \
    --headless \
    --device cuda:0 \
    agent.agent.mini_batches=4 \
    agent.agent.experiment.checkpoint_interval=160
TRAIN_RC=$?
set -e
kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-after.csv"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true

find "$LOG_ROOT" -type f -print | sort > "$RUN_DIR/files.txt"
if [[ "$TRAIN_RC" -ne 0 ]]; then
    echo "PPO command failed with exit code $TRAIN_RC" >&2
    exit "$TRAIN_RC"
fi
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
"$PYTHON" scripts/summarize_ppo.py \
    --run-dir "$RUN_DIR" \
    --updates 10 \
    --vector-steps 320 \
    --transitions 1280
capture_run_end "$RUN_DIR"
