#!/usr/bin/env bash
set -euo pipefail
: "${CAMPAIGN_PROFILE:?}" "${CAMPAIGN_OUTPUT:?}" "${CAMPAIGN_SOURCE_PROTOCOL:?}"
: "${CAMPAIGN_STAGE1:?}" "${CAMPAIGN_FROZEN_CONFIG:?}"
phase="$1"
destination="$CAMPAIGN_OUTPUT/$phase-$SLURM_JOB_ID"
mkdir -p "$destination/context"
profile="$CAMPAIGN_PROFILE"
protocol="$CAMPAIGN_SOURCE_PROTOCOL"
if [[ "$phase" == confirmation ]]; then
    [[ -s "$CAMPAIGN_FROZEN_CONFIG/ready.sha256" ]] || exit 2
    (cd "$CAMPAIGN_FROZEN_CONFIG" && sha256sum -c ready.sha256)
    protocol="$CAMPAIGN_FROZEN_CONFIG/protocol.yaml"
fi
"$PYTHON" - "$profile" "$protocol" "$phase" "$destination/context" <<'PY'
import hashlib
from pathlib import Path
import sys
import yaml

profile, protocol, phase, context = sys.argv[1:]
campaign = yaml.safe_load(Path(profile).read_text())
density = yaml.safe_load(Path("experiments/unity_two_stage_transfer/s0010_simple_landmark_robustness_v1.yaml").read_text())
seeds = campaign["development_seeds"][2:] if phase == "development" else campaign["reserved_confirmation_seeds"]
if phase == "confirmation":
    from pathlib import Path
    gate = yaml.safe_load((Path(protocol).parent / "ready.yaml").read_text())
    assert gate["readiness"] == "PASS" and gate["independent_seeds"] == seeds
density["capture_seeds"] = seeds
context = Path(context)
(context/"capture_profile.yaml").write_text(yaml.safe_dump(density))
(context/"seeds.txt").write_text("".join(f"{s}\n" for s in seeds))
expected = campaign["medium_waypoints_sha256"]
(context/"waypoints.sha256").write_text(expected + "  waypoints.json\n")
for seed in seeds:
    revision = {
        "scope": campaign["anatomy"], "source_protocol_sha256": hashlib.sha256(Path(protocol).read_bytes()).hexdigest(),
        "stage1_session": campaign["stage1"], "seeds": [seed], "demo_seed": seed,
        "acquisition": "outer_transverse_midpoints_near_l4",
        "sap_level_assignment": "projected_position_with_stage1_reference_bands",
        "tp_evidence_on_added_sap_waypoints": True,
    }
    p = context / f"{seed}.yaml"
    p.write_text(yaml.safe_dump(revision))
    p.with_suffix(".sha256").write_text(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
PY
capture_run_context "$destination"
cp "$profile" "$destination/context/campaign.yaml"
conversion=source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml
segmentation="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
while read -r seed; do
    run="$destination/$seed"
    printf '%s\tSTARTED\t%s\n' "$seed" "$(date --iso-8601=seconds)" >> "$destination/attempts.tsv"
    outcome=CAPTURE_FAILED
    if "$PYTHON" workflows/two_stage_navigation/capture_simple_landmark_sweep.py \
          --protocol "$protocol" --revision "$destination/context/$seed.yaml" \
          --robustness-arm E1-M --robustness-profile "$destination/context/capture_profile.yaml" \
          --stage1-session cf7_00 --stage1-actor "$CAMPAIGN_STAGE1" --seed "$seed" --output "$run" \
          --headless --device cuda:0 --record-robot --enable_cameras --evaluator-segmentation "$segmentation" \
          >"$destination/capture-$seed.log" 2>&1; then
        outcome=EVALUATION_FAILED
        if (cd "$run" && sha256sum -c "$destination/context/waypoints.sha256" && sha256sum actor_result.json > actor_result.json.sha256) \
           && "$PYTHON" workflows/two_stage_navigation/evaluate_simple_landmark_sweep.py \
             --session "$run" --protocol "$protocol" --label-conversion "$conversion" \
             --oracle-diagnostic "$run/oracle_diagnostic.json" >"$destination/evaluation-$seed.log" 2>&1; then
            outcome=EVALUATED
        fi
    fi
    printf '%s\t%s\t%s\n' "$seed" "$outcome" "$(date --iso-8601=seconds)" >> "$destination/attempts.tsv"
done <"$destination/context/seeds.txt"
capture_run_end "$destination"
