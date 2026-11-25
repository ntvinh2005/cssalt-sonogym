"""Measure the incremental effect of one matched fresh oblique survey."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from workflows.mbb_end_to_end.evaluate import (
    ARMS, array_file, checked, cortex_diagnostic, digest, invoke_actor, load_protocol,
    ras_to_base, read, reconstructed_file, reference_data, score, visible_coverage, write,
)


def read_frames(session):
    return [json.loads(line) for line in (session / "public/frames.jsonl").read_text().splitlines()]


def valid(frame):
    return all(frame[k] is True for k in ("image_valid", "contact_ok", "producer_advanced", "transition_valid"))


def compare_pose(fresh, archived):
    old = {f["waypoint_index"]: f for f in archived if valid(f) and f["waypoint_index"] < 102}
    pairs = [(f, old[f["waypoint_index"]]) for f in fresh if valid(f) and f["waypoint_index"] in old]
    if not pairs:
        return {"matched_waypoint_count": 0}
    position, yaw, matching_images = [], [], 0
    for new, previous in pairs:
        position.append(float(1000 * np.linalg.norm(np.asarray(new["actor_base_to_ee_position_m"]) -
                                                    previous["actor_base_to_ee_position_m"])))
        a = np.asarray(new["actor_base_to_ee_rotation_columns"])
        b = np.asarray(previous["actor_base_to_ee_rotation_columns"])
        yaw.append(float(np.rad2deg(np.arccos(np.clip((np.sum(a * b) - 1.) / 2., -1., 1.)))))
        matching_images += new["frame_sha256"] == previous["frame_sha256"]
    return {"matched_waypoint_count": len(pairs), "matching_image_content": matching_images,
            "median_pose_offset_mm": float(np.median(position)), "p95_pose_offset_mm": float(np.quantile(position, .95)),
            "median_angle_deg": float(np.median(yaw)), "p95_angle_deg": float(np.quantile(yaw, .95))}


def evaluate(opts):
    protocol, seal = load_protocol(opts.protocol)
    source = read(checked(protocol["source_manifest"]))
    receipt = checked({"path": str(opts.session / "actor_result.json"),
                       "sha256": (opts.session / "actor_result.json.sha256").read_text().split()[0]})
    capture = read(receipt)
    seed = capture["seed"]
    if seed != protocol["fresh_development_seed_if_justified"] or seed in protocol["reserved_independent_seeds"]:
        raise ValueError("fresh replay is outside the declared development seed")
    if capture["checkpoint_sha256"] != {k: v["sha256"] for k, v in source["models"].items()}:
        raise ValueError("fresh capture checkpoints do not match the development checkpoint set")
    if not capture["additional_sweep_sha256"]:
        raise ValueError("missing sealed supplemental survey")
    frames = read_frames(opts.session)
    prefix = [f["frame_sequence"] for f in frames if valid(f) and f["waypoint_index"] < 102]
    suffix = [f["frame_sequence"] for f in frames if valid(f) and f["waypoint_index"] >= 102]
    if len(prefix) + len(suffix) != capture["valid_frame_count"]:
        raise ValueError("native capture and valid frame set differ")
    reference_seal = digest(opts.reference / "checksums.sha256")
    opts.output.mkdir(parents=True, exist_ok=False)
    write(opts.output / "input_freeze.json", {"reference_receipt_sha256": reference_seal,
                                             "actor_capture_sha256": digest(receipt), "protocol_sha256": seal})
    seals = {}
    frames_digest = digest(opts.session / "public/frames.jsonl")
    for arm, label in ARMS.items():
        full = reconstructed_file(opts.reconstruction / str(seed), label, frames_digest)
        arrays = array_file(full)
        keep = np.isin(arrays["frame_sequence"], prefix)
        first = opts.output / f"{arm}_prefix_cloud.npz"
        np.savez_compressed(first, **{k: arr[keep] if arr.shape[:1] == keep.shape else arr for k, arr in arrays.items()})
        for name, path in ((f"{arm}_prefix", first), (f"{arm}_full", full)):
            prediction = opts.output / f"{name}.json"
            invoke_actor(path, opts.session / "public/frames.jsonl", prediction)
            seals[name] = {"prediction": {"path": str(prediction), "sha256": digest(prediction)},
                           "cloud": {"path": str(path), "sha256": digest(path)}}
    write(opts.output / "prediction_seals.json", seals)
    if digest(opts.reference / "checksums.sha256") != reference_seal:
        raise ValueError("reference changed during fresh localization")
    ref_info, hypotheses, members = reference_data(opts.reference)
    diagnostic = array_file(checked(members["anatomy_hypotheses.npz"]))
    native_name = "privileged/full_pose_replay.jsonl"
    public_name = "public/frames.jsonl"
    fresh_session = {"path": str(opts.session), "files": {name: digest(opts.session / name)
                                                          for name in (native_name, public_name)}}
    affine = nib.load(checked(source["target"]["segmentation"])).affine
    historical = read(checked(protocol["historical_target_for_diagnostics"]))["point sets"]["Historical midpoint"]
    legacy = ras_to_base(historical["coordinates_ras_mm"], affine, fresh_session)
    trace = ras_to_base(diagnostic["diagnostic_cortex_connection_ras_mm"], affine, fresh_session)
    refs = {k: ras_to_base(points, affine, fresh_session) for k, points in hypotheses.items()}
    gate = protocol["reference_technical_gate"]
    oracle = array_file(checked(seals["E1_full"]["cloud"]))
    old = next(s for s in source["sessions"] if s["seed"] == seed)
    outcomes = {}
    for name, entry in seals.items():
        pred = read(checked(entry["prediction"]))
        cloud = array_file(checked(entry["cloud"]))
        outcomes[name] = {
            "prediction": pred, "evaluation": score(pred.get("candidate"), refs, legacy, gate),
            "coverage_against_same_fresh_oracle_pool": visible_coverage(cloud, oracle, legacy, refs, gate),
            "cortical_trace_diagnostic": cortex_diagnostic(
                cloud, trace, diagnostic["diagnostic_cortex_connection_owner_labels"],
                gate["observed_coverage_distance_mm"],
            ),
        }
    write(opts.output / "results.json", {
        "seed": seed, "stage1": {"reused": True, "id": "cf7_00"},
        "baseline_waypoints": 102, "supplemental_waypoints": len(read(opts.session/"waypoints.json")) - 102,
        "baseline_valid_frames": len(prefix), "supplemental_valid_frames": len(suffix),
        "failed_moves": capture["failed_moves"], "acquisition_and_initialization_seconds": capture["elapsed_seconds"],
        "action_steps": capture["action_steps"], "matched_historical_pose_comparison": compare_pose(frames, read_frames(Path(old["path"]))),
        "reference_status": ref_info["status"], "arms": outcomes,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "reference", "session", "reconstruction", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
