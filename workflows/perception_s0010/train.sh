#!/usr/bin/env bash
set -euo pipefail
mode=${CAMPAIGN_FIT_MODE:-training}
out="$CAMPAIGN_OUTPUT/$mode-$SLURM_JOB_ID"
extra=()
if [[ "$mode" == mining ]]; then
    extra+=(--mine-only)
elif [[ -n "${CAMPAIGN_MINING:-}" ]]; then
    extra+=(--mining "$CAMPAIGN_MINING/fit")
fi
if [[ -n "${CAMPAIGN_PILOT:-}" ]]; then
    extra+=(--pilot "$CAMPAIGN_PILOT")
fi
capture_run_context "$out"
cp "$CAMPAIGN_PROFILE" "$out/context/campaign.yaml"
trap 'kill "$telemetry" 2>/dev/null || true' EXIT
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv --loop=10 >"$out/gpu.csv" &
telemetry=$!
"$PYTHON" workflows/perception_s0010/fit.py --protocol "$CAMPAIGN_SOURCE_PROTOCOL" --profile "$CAMPAIGN_PROFILE" \
    --data "$CAMPAIGN_DATA/render" --legacy-dataset "$CAMPAIGN_LEGACY_DATASET" --legacy-training "$CAMPAIGN_LEGACY_TRAINING" \
    --diagnosis "$CAMPAIGN_BENCHMARK/diagnosis/mapping.json" --output "$out/fit" --device cuda:0 "${extra[@]}" >"$out/fit.log" 2>&1
capture_run_end "$out"
