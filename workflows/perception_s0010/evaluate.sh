#!/usr/bin/env bash
set -euo pipefail
out="$CAMPAIGN_OUTPUT/evaluation-$SLURM_JOB_ID"
capture_run_context "$out"
cp "$CAMPAIGN_PROFILE" "$out/context/campaign.yaml"
extra=()
if [[ -n "${CAMPAIGN_PILOT:-}" ]]; then
    extra+=(--pilot "$CAMPAIGN_PILOT")
fi
sha256sum "$CAMPAIGN_SOURCE_PROTOCOL" "$CAMPAIGN_PROFILE" "$CAMPAIGN_LEGACY_DATASET" \
    "$CAMPAIGN_LEGACY_TRAINING/l4_split_manifest.csv" "$CAMPAIGN_LEGACY_TRAINING/l5_split_manifest.csv" \
    "$CAMPAIGN_PREVIOUS/review-52711211/E3-dataset/dataset.npz" >"$out/context/input_sha256.txt"
if [[ -s "$CAMPAIGN_PREVIOUS/E3-dataset.sha256" ]]; then
    sha256sum -c "$CAMPAIGN_PREVIOUS/E3-dataset.sha256"
fi
"$PYTHON" -m unittest -q tests.test_simple_landmark_mapping tests.test_sonoanatomy_segmentation \
    tests.test_sonoanatomy_labels tests.test_acquisition_integrity >"$out/tests.log" 2>&1
"$PYTHON" workflows/perception_s0010/selection.py --protocol "$CAMPAIGN_SOURCE_PROTOCOL" --profile "$CAMPAIGN_PROFILE" \
    --previous "$CAMPAIGN_PREVIOUS" --historical "$CAMPAIGN_HISTORICAL" --development "$CAMPAIGN_DEVELOPMENT" \
    --data "$CAMPAIGN_DATA/render" --training "$CAMPAIGN_TRAINING/fit" --output "$out/metrics" \
    --frozen "$CAMPAIGN_FROZEN_CONFIG" --device cuda:0 "${extra[@]}" >"$out/selection.log" 2>&1
"$PYTHON" workflows/perception_s0010/review.py --protocol "$CAMPAIGN_SOURCE_PROTOCOL" --profile "$CAMPAIGN_PROFILE" \
    --previous "$CAMPAIGN_PREVIOUS" --historical "$CAMPAIGN_HISTORICAL" --development "$CAMPAIGN_DEVELOPMENT" \
    --data "$CAMPAIGN_DATA/render" --evaluation "$out/metrics" --output "$out/review" \
    --device cuda:0 >"$out/review.log" 2>&1
capture_run_end "$out"
