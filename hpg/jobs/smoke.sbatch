#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

RUN_DIR="$SONOGYM_OUTPUT_DIR/smoke/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

python scripts/smoke_test.py \
    --task Isaac-robot-US-guidance-v0 \
    --num-envs 1 \
    --seed 20261106 \
    --steps 12 \
    --output-dir "$RUN_DIR" \
    --headless \
    --device cuda:0
