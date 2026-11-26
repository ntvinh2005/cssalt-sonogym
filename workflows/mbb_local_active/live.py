"""Run frozen A1 on synchronized public images and request bounded views."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import numpy as np

from workflows.mbb_adaptive_cortical.tracker import adaptive_frame, anchor_brackets, candidates_for, echo_fields, track_path
from workflows.mbb_direct_sonographic.actor_pixels import visible_pixels
from workflows.mbb_direct_sonographic.localize import acquisition_views, audit_frame, distinct_poses, observed_profiles, select_views, verify_session
from workflows.mbb_direct_sonographic.prepare import POSE_FIELDS, digest, read_json, verify, write_json
from workflows.mbb_direct_sonographic.public_geometry import sweep_axes
from workflows.mbb_direct_sonographic.replay import NAMES
from two_stage_navigation.stage1_interfaces import track_bright_interfaces

BASE = Path("experiments/unity_two_stage_transfer")


def decisions(image, named, family, sign, rules, config):
    profiles = observed_profiles(image, rules)
    base = audit_frame(profiles, named, family, sign, rules)
    brackets = list(anchor_brackets(profiles, named, sign, rules)) if family == "transverse" else []
    paths = []
    if brackets:
        fields = echo_fields(image, rules)
        seeds = track_bright_interfaces(image)
    for bracket in brackets:
        if bracket["competing_on_original"]:
            paths.append({"status": "COMPETING", "rows": [None] * len(bracket["pixels"])})
            continue
        _, scores = candidates_for(bracket, seeds[bracket["profile"]], fields, named, rules, config)
        pinned = [i <= bracket["start"] or i >= bracket["stop"] for i in range(len(scores))]
        paths.append(track_path(scores, 0, pinned, rules["max_rise_per_column_px"], config, rules["pixel_mm"]))
    result = dict(adaptive_frame(base, brackets, paths, rules))
    result["joined_profiles"] = sum(path["status"] == "JOINED" for path in paths)
    result["semantic_pixels"] = {name: x[0].tolist() for name, x in named.items()}
    result["echo_paths"] = [
        {"columns": bracket["pixels"][:, 1].tolist(), **path} for bracket, path in zip(brackets, paths)
    ]
    return base, result


def for_frame(pixels, i):
    return {
        name: (pixels["pixel_row_col"][mask], pixels["confidence"][mask])
        for name in NAMES for mask in [
            (pixels["frame_sequence"] == i) & (pixels["identity"] == name)]
    }


def local_targets(public_command, limits, policy):
    items = []
    for dx, dz, yaw in policy["offsets_index_index_deg"][:policy["max_additional_frames"]]:
        command = np.asarray(public_command) + (dx, dz, np.deg2rad(yaw))
        if np.all(command >= limits[0]) and np.all(command <= limits[1]):
            items.append(("oblique", tuple(map(float, command))))
    return items


class LiveVerification:
    def __init__(self, path, archived, stage1, seed, stage1_id):
        self.policy = read_json(path)
        if (seed != self.policy["development_seed"] or stage1_id != self.policy["stage1"]
                or not 0 < self.policy["max_additional_frames"] <= 24):
            raise ValueError("wrong development context")
        self.stage1, self.out = stage1, None
        direct = BASE / "mbb_direct_sonographic_v2.json"
        adaptive = BASE / "mbb_adaptive_cortical_tracking_v1.json"
        verify(direct, self.policy["direct_config_sha256"])
        verify(adaptive, self.policy["adaptive_config_sha256"])
        self.rules, self.config = read_json(direct), read_json(adaptive)
        self.policy_sha256 = digest(path)
        self.archived = read_json(archived / "replay/frame_audits.json")
        self.thresholds = read_json(archived / "inputs/manifest.json")["recordings"][str(seed)]["source_thresholds"]
        self.historical_hashes = {p["frame_sha256"] for d in self.archived.values() for p in d["public_poses"]}
        self.frames, self.rows, self.eligible = [], [], []
        self.source_id, self.projection, self.limit = None, [], None
        self.before, self.after = None, None
        self.begin, self.started = monotonic(), None
        self.planned, self.local_evidence = [], []

    def replay_frame(self, frame, image, named, view):
        family, sign = view
        if not frame["image_valid"]:
            invalid = {"status": "INVALID_ACQUISITION", "direct_candidate": False, "joined_profiles": 0}
            return invalid, invalid
        return decisions(image, named, family, sign, self.rules, self.config)

    def start_local(self, frames, images, probabilities, limits, planned_initial):
        if planned_initial != self.policy["initial_waypoints"]:
            raise ValueError("matched initial sweep differs from frozen protocol")
        self.limit = len(frames)
        self.frames = [{key: frame[key] for key in POSE_FIELDS} for frame in frames]
        self.started = monotonic()
        views, _ = acquisition_views(self.frames, self.rules)
        self.right, self.cranial = sweep_axes(self.frames, self.rules)
        named = visible_pixels(self.frames, images, probabilities, self.stage1, self.thresholds, self.rules,
                               cranial=self.cranial)
        baseline_rows = []
        for i, frame in enumerate(self.frames):
            base, row = self.replay_frame(frame, images[i], for_frame(named, i), views[i])
            baseline_rows.append(base)
            self.rows.append(row)
        # Invalid rows do not enter ranking.
        if any(not frame["image_valid"] for frame in self.frames):
            # The historical shortlist scorer requires the full original fields.
            self.eligible = [i for i, frame in enumerate(self.frames) if frame["image_valid"]]
        else:
            self.eligible, _ = select_views(self.frames, baseline_rows, self.rules)
        self.before = verify_session(self.frames, self.rows, self.eligible, self.rules)
        self.after = self.before
        source = next((i for i in self.eligible if self.rows[i]["direct_candidate"]), None)
        if source is None or self.before["accepted"]:
            return []
        self.source_id = source
        self.source_image = images[source].copy()
        self.projection = self.before["regions"][source]["candidate_xyz_mm"]
        self.planned = local_targets(self.frames[source]["requested_relative_pose_index"], limits, self.policy)
        return self.planned

    def add_view(self, record, image, probabilities):
        frame = {key: record[key] for key in POSE_FIELDS}
        self.frames.append(frame)
        i = len(self.frames) - 1
        direction = np.asarray(frame["actor_base_to_ee_rotation_columns"])[:, 0]
        lateral = float(direction @ self.right)
        family = "transverse" if abs(lateral) >= np.cos(np.deg2rad(self.rules["max_transverse_angle_deg"])) else "incompatible"
        named = visible_pixels([frame], [image], probabilities, self.stage1, self.thresholds, self.rules,
                               cranial=self.cranial)
        _, decision = self.replay_frame(frame, image, for_frame(named, 0), (family, 1 if lateral >= 0 else -1))
        self.rows.append(decision)
        novel = frame["frame_sha256"] not in self.historical_hashes and all(
            p["frame_sha256"] != frame["frame_sha256"] for p in self.frames[:-1])
        distinct = bool(distinct_poses(self.frames, self.rules)[self.source_id, i])
        if frame["image_valid"] and novel and distinct:
            self.eligible.append(i)
        result = verify_session(self.frames, self.rows, self.eligible, self.rules)
        region = result["regions"].get(i)
        stop = bool(result["accepted"] and any(j >= self.limit for j in result["support_frames"]))
        self.after = result
        self.local_evidence.append({
            "sequence": i, "fresh_producer": record["producer_advanced"], "contact_ok": record["contact_ok"],
            "new_hash_vs_archive_and_session": novel, "pose_distinct_from_source": distinct,
            "raw_mae_vs_fresh_source": float(np.mean(np.abs(image - self.source_image))),
            "requested": record["requested_relative_pose_index"], "executed_fk": record["actor_base_to_ee_position_m"],
            "rotation_columns": record["actor_base_to_ee_rotation_columns"], "detection": decision["status"],
            "contributed_candidate": region is not None,
            "verification": result["status"], "region": region,
        })
        return stop

    def seal(self, out, failed):
        file = out / "local_actor.json"
        write_json(file, {
            "label": "Fresh matched development sweep and candidate-driven local verification",
            "initial_frame_count": self.limit, "source_sequence": self.source_id, "source_projected_mm": self.projection,
            "before": self.before, "after": self.after, "eligible_sequences": self.eligible,
            "frames": self.rows, "planned_local": self.planned, "additional_frames": self.local_evidence,
            "failed_moves": failed, "policy_sha256": self.policy_sha256,
            "initial_elapsed_seconds": None if self.started is None else self.started - self.begin,
            "local_elapsed_seconds": None if self.started is None else monotonic() - self.started,
            "independent_target_status": "TARGET_LOCALIZATION_NOT_EVALUABLE",
        })
        seal = out / "local_actor.sha256"
        entries = [file, out / "public/frames.jsonl", out / "public/ultrasound.npz"]
        seal.write_text("".join(f"{digest(p)}  {p.relative_to(out)}\n" for p in entries))
