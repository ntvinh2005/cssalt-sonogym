#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit helper}"
# shellcheck disable=SC1090
set -a
source "$SONOGYM_CONFIG_FILE"
set +a

[[ "$SLURM_JOB_ACCOUNT" == "anest-dept" ]] || { echo "Refusing unexpected account: $SLURM_JOB_ACCOUNT" >&2; exit 2; }
[[ "${SLURM_CPUS_PER_TASK:-0}" -le 4 ]] || { echo "CPU budget exceeded" >&2; exit 2; }
[[ "${SLURM_GPUS_ON_NODE:-0}" -le 1 ]] || { echo "GPU budget exceeded" >&2; exit 2; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "Slurm did not assign a visible GPU" >&2; exit 2; }
[[ "$PWD" == /blue/* ]] || cd "$SONOGYM_PROJECT_DIR"
[[ "$PWD" == /blue/* ]] || { echo "Workloads must execute from Blue storage; got $PWD" >&2; exit 2; }

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export WANDB_MODE="${SONOGYM_WANDB_MODE:-disabled}"
export WANDB_SILENT=true
export MPLBACKEND=Agg
export PYVISTA_OFF_SCREEN=true
export QT_QPA_PLATFORM=offscreen
export XDG_CACHE_HOME="$SONOGYM_CACHE_DIR/xdg"
export PIP_CACHE_DIR="$SONOGYM_CACHE_DIR/pip"
export OMNI_USER_DIR="$SONOGYM_CACHE_DIR/omniverse"

mkdir -p "$SONOGYM_OUTPUT_DIR/slurm" "$SONOGYM_OUTPUT_DIR/smoke" \
    "$SONOGYM_OUTPUT_DIR/ppo" "$SONOGYM_CACHE_DIR/xdg" \
    "$SONOGYM_CACHE_DIR/pip" "$SONOGYM_CACHE_DIR/omniverse"

CONDA_EXE="${SONOGYM_CONDA_EXE:-$(command -v conda || true)}"
[[ -n "$CONDA_EXE" ]] || { echo "conda is not available" >&2; exit 2; }
eval "$("$CONDA_EXE" shell.bash hook)"
conda activate "$SONOGYM_ENV_PREFIX"

echo "host=$(hostname) job_id=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT pwd=$PWD"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python --version
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 1 && "${GPU_NAMES[0]}" == *L4* ]] || {
    echo "The initial pinned workflow requires exactly one L4; saw: ${GPU_NAMES[*]}" >&2
    exit 2
}
