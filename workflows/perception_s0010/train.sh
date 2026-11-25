#!/usr/bin/env bash
set -euo pipefail
out="$CAMPAIGN_OUTPUT/training-$SLURM_JOB_ID"
capture_run_context "$out"
cp "$CAMPAIGN_PROFILE" "$out/context/campaign.yaml"
trap 'kill "$telemetry" 2>/dev/null || true' EXIT
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv --loop=10 >"$out/gpu.csv" &
telemetry=$!
"$PYTHON" workflows/perception_s0010/fit.py --protocol "$CAMPAIGN_SOURCE_PROTOCOL" --profile "$CAMPAIGN_PROFILE" \
    --data "$CAMPAIGN_DATA/render" --legacy-dataset "$CAMPAIGN_LEGACY_DATASET" --legacy-training "$CAMPAIGN_LEGACY_TRAINING" \
    --diagnosis "$CAMPAIGN_BENCHMARK/diagnosis/mapping.json" --output "$out/fit" --device cuda:0 >"$out/fit.log" 2>&1
capture_run_end "$out"
