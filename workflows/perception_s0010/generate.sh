#!/usr/bin/env bash
set -euo pipefail
out="$CAMPAIGN_OUTPUT/dataset-$SLURM_JOB_ID"
capture_run_context "$out"
cp "$CAMPAIGN_PROFILE" "$out/context/campaign.yaml"
trap 'kill "$telemetry" 2>/dev/null || true' EXIT
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv --loop=5 >"$out/gpu.csv" &
telemetry=$!
"$PYTHON" workflows/perception_s0010/render_dataset.py --protocol "$CAMPAIGN_SOURCE_PROTOCOL" --profile "$CAMPAIGN_PROFILE" \
    --previous "$CAMPAIGN_PREVIOUS" --historical "$CAMPAIGN_HISTORICAL" --legacy-training "$CAMPAIGN_LEGACY_TRAINING" \
    --output "$out/render" --benchmark-receipt "$CAMPAIGN_BENCHMARK/render/receipt.json" \
    --headless --device cuda:0 >"$out/render.log" 2>&1
capture_run_end "$out"
