"""Detect and verify a directly supported TP–SAP cortical region in B-mode."""

from __future__ import annotations

from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial.distance import cdist, pdist

from two_stage_navigation.simple_landmark_mapping import project_image
from two_stage_navigation.stage1_interfaces import track_bright_interfaces
from workflows.mbb_observed_surface.localize import distinct_poses

from .public_geometry import sweep_axes

TP, SAP = "RIGHT_L5_TP", "RIGHT_L5_SAP"
COMPETING = ("RIGHT_L4_TP", "RIGHT_L4_SAP")


def acquisition_views(frames, rules):
    right, cranial = sweep_axes(frames, rules)
    superior = np.asarray(cranial.direction)
    result = []
    for frame in frames:
        column_direction = np.asarray(frame["actor_base_to_ee_rotation_columns"])[:, 0]
        lateral = float(np.dot(column_direction, right))
        longitudinal = float(np.dot(column_direction, superior))
        phase = frame["phase"]
        if (phase == "longitudinal"
                and abs(longitudinal) >= np.cos(np.deg2rad(rules["max_longitudinal_angle_deg"]))):
            family = "longitudinal"
        elif (phase in ("transverse", "oblique")
              and abs(lateral) >= np.cos(np.deg2rad(rules["max_transverse_angle_deg"]))):
            family = "transverse"
        else:
            family = "incompatible"
        result.append((family, 1 if lateral >= 0 else -1))
    return result, cranial.maximum_fit_residual_index


def observed_profiles(image, rules):
    """Re-sample existing smooth tracks onto measured echo peaks column by column."""
    low, high = np.quantile(image, (.05, .95))
    normalized = (image - low) / max(float(high - low), 1e-6)
    smooth = gaussian_filter1d(normalized, .7, axis=0)
    peak = smooth == maximum_filter1d(smooth, 3, axis=0)
    height, _ = image.shape
    profiles = []
    for track in track_bright_interfaces(image):
        if not track.valid:
            continue
        columns = np.asarray(track.columns, dtype=int)
        rough_rows = np.rint(track.rows).astype(int)
        radius = rules["track_row_radius_px"]
        choices = np.clip(rough_rows[:, None] + np.arange(-radius, radius + 1), 5, height - 10)
        echoes = smooth[choices, columns[:, None]]
        pre = np.mean(smooth[choices[..., None] - np.arange(2, 6), columns[:, None, None]], axis=2)
        post = np.mean(smooth[choices[..., None] + np.arange(3, 10), columns[:, None, None]], axis=2)
        contrast, shadow = echoes - pre, echoes - post
        real = (peak[choices, columns[:, None]]
                & (echoes >= rules["minimum_relative_echo"])
                & (contrast >= rules["minimum_proximal_contrast"])
                & (shadow >= rules["minimum_distal_contrast"]))
        choice = np.argmax(np.where(real, np.minimum(contrast, shadow), -np.inf), axis=1)
        rows = choices[np.arange(len(columns)), choice]
        supported = np.any(real, axis=1)
        profiles.append((
            np.column_stack((rows, columns)), supported,
            shadow[np.arange(len(columns)), choice],
        ))
    return profiles


def profile_evidence(pixels, supported, annotations, rules):
    confidence, labeled = {}, {}
    for name, samples in annotations.items():
        xy, probability = samples
        if not len(xy):
            matched = np.zeros(len(pixels), dtype=bool)
            q = np.zeros(len(pixels), dtype=float)
        else:
            within = cdist(pixels, xy) <= rules["track_label_radius_px"]
            matched = supported & within.any(axis=1)
            q = np.max(np.where(within, probability, 0.0), axis=1)
        labeled[name] = matched
        confidence[name] = q[matched]
    return labeled, confidence


def contiguous_runs(mask, minimum):
    indices = np.flatnonzero(mask)
    return [run for run in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
            if len(run) >= minimum]


def pair_regions(pixels, supported, labeled, lateral_sign, rules):
    """Require a continuous measured interface and a shallowing turn toward medial SAP."""
    if lateral_sign == 1:
        # Scan in anatomical order from lateral TP to medial SAP.
        pixels, supported = pixels[::-1], supported[::-1]
        labeled = {name: matches[::-1] for name, matches in labeled.items()}

    tp = contiguous_runs(labeled[TP], rules["anchor_span_px"])
    sap = contiguous_runs(labeled[SAP], rules["anchor_span_px"])
    competing = np.logical_or.reduce([labeled[name] for name in COMPETING])
    rows = pixels[:, 0].astype(float)
    candidates = []
    counters = dict(same_profile_pairs=len(tp) * len(sap), ordered_pairs=0,
                    continuous_pairs=0, transition_pairs=0)
    nearest_gap = np.inf
    for a in tp:
        for b in sap:
            start, stop = int(a[-1]), int(b[0])
            if stop <= start:
                continue
            nearest_gap = min(nearest_gap, np.linalg.norm(pixels[stop] - pixels[start]) * rules["pixel_mm"])
            flank = rules["turn_window_px"]
            lo, hi = start - flank + 1, stop + flank
            bracket = slice(lo, hi)
            if (lo < 0 or hi > len(pixels)
                    or (stop - start) * rules["pixel_mm"] > rules["max_anchor_span_mm"]
                    or (np.median(rows[a[-flank:]]) - np.median(rows[b[:flank]])) * rules["pixel_mm"]
                    < rules["min_anchor_depth_rise_mm"]):
                continue
            counters["ordered_pairs"] += 1
            if (not supported[bracket].all() or competing[bracket].any()
                    or np.max(np.abs(np.diff(rows[bracket]))) > rules["max_rise_per_column_px"]):
                continue
            counters["continuous_pairs"] += 1
            # A shallowing transition at the medial end of the TP can occur only
            # where a local change of slope is actually sampled on the B-mode ridge.
            bends = []
            for i in range(start, stop + 1):
                previous = (rows[i] - rows[i - flank + 1]) / (flank - 1)
                following = (rows[i + flank - 1] - rows[i]) / (flank - 1)
                bend = previous - following
                if following < 0 and bend >= rules["min_turn_slope_change"]:
                    bends.append((i, bend))
            if not bends:
                continue
            counters["transition_pairs"] += 1
            strongest = max(strength for _, strength in bends)
            for i, strength in bends:
                if strength >= strongest - rules["min_turn_slope_change"]:
                    candidates.append({
                        "candidate": pixels[i - 1:i + 2],
                        TP: pixels[a[-flank:]],
                        SAP: pixels[b[:flank]],
                        "observed_bridge": pixels[start:stop + 1],
                    })
    return candidates, counters, nearest_gap


def region_distance(a, b):
    distances = cdist(a, b)
    return max(float(np.percentile(distances.min(axis=0), 95)),
               float(np.percentile(distances.min(axis=1), 95)))


def audit_frame(profiles, annotations, family, sign, rules):
    started = perf_counter()
    row = {
        "view": family, "n_tp_pixels": len(annotations[TP][0]),
        "n_sap_pixels": len(annotations[SAP][0]), "n_l4_tp_pixels": len(annotations[COMPETING[0]][0]),
        "n_l4_sap_pixels": len(annotations[COMPETING[1]][0]), "track_count": len(profiles),
        "tp_columns": 0, "sap_columns": 0, "l4_tp_columns": 0, "tp_confidence": 0.,
        "sap_confidence": 0., "maximum_supported_run": 0, "shadow_contrast": None,
        "same_profile_pairs": 0, "ordered_pairs": 0, "continuous_pairs": 0, "transition_pairs": 0,
        "nearest_ordered_anchor_gap_mm": None, "direct_candidate": False,
        "candidate_pixels": [], "profile_pixels": [], "support_pixels": {},
        "candidate_image_diameter_mm": None,
    }
    candidate_parts = []
    anchor_span = rules["anchor_span_px"]
    best_profile = (-1, -1)
    for pixels, supported, shadow in profiles:
        labeled, confidence = profile_evidence(pixels, supported, annotations, rules)
        runs = [max((len(chunk) for chunk in contiguous_runs(labeled[key], 1)), default=0)
                for key in (TP, SAP, COMPETING[0])]
        for key, run in zip(("tp_columns", "sap_columns", "l4_tp_columns"), runs):
            row[key] = max(row[key], run)
        for name, key in ((TP, "tp_confidence"), (SAP, "sap_confidence")):
            if len(confidence[name]):
                row[key] = max(row[key], float(np.mean(confidence[name])))
        max_run = max((len(chunk) for chunk in contiguous_runs(supported, 1)), default=0)
        row["maximum_supported_run"] = max(row["maximum_supported_run"], max_run)
        available_shadow = float(np.median(shadow[supported])) if supported.any() else None
        rank = (min(runs[:2]) if family == "transverse" else runs[0], max_run)
        if rank > best_profile:
            best_profile = rank
            row["profile_pixels"] = pixels[supported].tolist()
            row["shadow_contrast"] = available_shadow
        if family == "transverse":
            candidates, counters, nearest = pair_regions(pixels, supported, labeled, sign, rules)
            candidate_parts.extend(candidates)
            for key, value in counters.items():
                row[key] += value
            current = row["nearest_ordered_anchor_gap_mm"]
            if np.isfinite(nearest) and (current is None or nearest < current):
                row["nearest_ordered_anchor_gap_mm"] = float(nearest)
    if family == "longitudinal":
        status = "LONGITUDINAL_CONTEXT" if row["tp_columns"] >= anchor_span else "L5_TP_NOT_SUPPORTED"
    elif family != "transverse":
        status = "INCOMPATIBLE_VIEW"
    elif min(row["tp_columns"], row["sap_columns"]) < anchor_span:
        if max(row["tp_columns"], row["sap_columns"]) < anchor_span:
            status = "NEITHER_L5_INTERFACE_SUPPORTED"
        else:
            status = "L5_SAP_NOT_SUPPORTED" if row["sap_columns"] < anchor_span else "L5_TP_NOT_SUPPORTED"
    elif not row["ordered_pairs"]:
        status = "NO_COMPATIBLE_ORDERED_PROFILE"
    elif not row["continuous_pairs"]:
        status = "DISCONTINUOUS_OR_CONFLICTING_PROFILE"
    elif not candidate_parts:
        status = "NO_LOCAL_CORTICAL_TURN"
    else:
        pixels = np.unique(np.concatenate([p["candidate"] for p in candidate_parts]), axis=0)
        diameter = float(pdist(pixels).max() * rules["pixel_mm"]) if len(pixels) > 1 else 0.
        row["candidate_image_diameter_mm"] = diameter
        if diameter > rules["max_region_diameter_mm"]:
            status = "AMBIGUOUS_IMAGE_REGIONS"
        else:
            status = "DIRECT_IMAGE_CANDIDATE"
            row["direct_candidate"] = True
            row["candidate_pixels"] = pixels.tolist()
            row["support_pixels"] = {
                name: np.unique(np.concatenate([p[name] for p in candidate_parts]), axis=0).tolist()
                for name in (TP, SAP, "observed_bridge")
            }
    row["status"] = status
    row["observation_ambiguity"] = (
        "multiple or overwide image regions" if status == "AMBIGUOUS_IMAGE_REGIONS"
        else "incomplete or incompatible local evidence" if status != "DIRECT_IMAGE_CANDIDATE"
        else "provisional bony pattern only"
    )
    row["runtime_seconds"] = perf_counter() - started
    return row


def search_quality(row, rules):
    """Rank only measured echo continuity and learned semantic support."""
    tp = min(row["tp_columns"] / rules["anchor_span_px"], 1.0)
    sap = min(row["sap_columns"] / rules["anchor_span_px"], 1.0)
    continuity = min(row["maximum_supported_run"] / 15, 1.0)
    if row["view"] == "longitudinal":
        return 2 * tp + row["tp_confidence"] + continuity
    if row["view"] == "transverse":
        return 3 * min(tp, sap) + tp + sap + continuity + (row["tp_confidence"] + row["sap_confidence"]) / 2
    return 0.


def select_views(frames, evidence, rules):
    """Freeze one learned shortlist for both A1 and A2."""
    positions = 1000 * np.asarray([f["actor_base_to_ee_position_m"] for f in frames])
    attitude = np.asarray([f["actor_base_to_ee_rotation_columns"] for f in frames]).reshape(-1, 9)
    angles = np.rad2deg(np.arccos(np.clip((attitude @ attitude.T - 1) / 2, -1, 1)))
    nearby = (cdist(positions, positions) <= rules["neighbor_position_mm"]) & (angles <= rules["neighbor_angle_deg"])
    families = np.asarray([row["view"] for row in evidence])
    nearby &= families[:, None] == families
    quality = np.asarray([search_quality(row, rules) for row in evidence])
    supported_quality = []
    for i in range(len(frames)):
        neighbors = np.flatnonzero(nearby[i] & (np.arange(len(frames)) != i))
        support = float(np.median(quality[neighbors])) if len(neighbors) else 0.
        supported_quality.append(float(quality[i]) + .25 * support)
    # Preserve different executed poses when several near-duplicates rank highly.
    distinct = distinct_poses(frames, rules)
    selected = []
    for family in ("longitudinal", "transverse"):
        ranked = sorted(np.flatnonzero(families == family), key=lambda k: (-supported_quality[k], k))
        keep = []
        for i in ranked:
            if all(distinct[i, j] for j in keep):
                keep.append(int(i))
            if len(keep) == rules[f"shortlist_{family}"]:
                break
        selected.extend(keep)
    return sorted(selected), supported_quality


def verify_session(frames, evidence, selected, rules):
    """Project only candidate and labeled evidence pixels, then verify distinct views."""
    projected = {}
    for i in selected:
        row = evidence[i]
        if not row["direct_candidate"]:
            continue
        coordinates = project_image(frames[i], tuple(rules["image_shape"]), rules["pixel_mm"])
        projected[i] = {
            "candidate_xyz_mm": coordinates[tuple(np.asarray(row["candidate_pixels"]).T)].tolist(),
            "support_xyz_mm": {
                name: coordinates[tuple(np.asarray(pixels, dtype=int).reshape(-1, 2).T)].tolist()
                for name, pixels in row["support_pixels"].items()
            },
        }
    candidate = {i: np.asarray(item["candidate_xyz_mm"]) for i, item in projected.items()}
    context = [i for i in selected if evidence[i]["status"] == "LONGITUDINAL_CONTEXT"]
    result = {
        "accepted": False, "longitudinal_context_frames": context, "candidate_frames": list(projected),
        "support_frames": [], "regions": projected, "pairs": [],
        "region_xyz_mm": [], "diameter_mm": None, "max_disagreement_mm": None,
        "uncertainty": "No accepted region. Potential views are retained in regions.",
    }
    if not candidate:
        result["status"] = "INSUFFICIENT_ANATOMICAL_EVIDENCE"
        return result
    distinct = distinct_poses(frames, rules)
    pairs = [(i, j, region_distance(candidate[i], candidate[j]))
             for i, j in combinations(candidate, 2) if distinct[i, j]]
    result["pairs"] = [{"a": i, "b": j, "p95_mm": d} for i, j, d in pairs]
    if not context:
        result["status"] = "LONGITUDINAL_CONTEXT_NOT_SUPPORTED"
        return result
    if any(d > rules["max_disagreement_mm"] for _, _, d in pairs):
        # A similarly plausible separated region is not silently removed.
        result["status"] = "AMBIGUOUS_MULTIVIEW_REGIONS"
        return result
    independent = []
    for i in candidate:
        if all(distinct[i, j] for j in independent):
            independent.append(i)
    if len(independent) < rules["min_views"]:
        result["status"] = "INSUFFICIENT_DISTINCT_TRANSVERSE_VIEWS"
        return result
    region = np.concatenate([candidate[i] for i in independent])
    diameter = float(pdist(region).max())
    if diameter > rules["max_region_diameter_mm"]:
        result["status"] = "MULTIVIEW_REGION_TOO_WIDE"
        return result
    result.update(
        accepted=True, status="SUPPORTED_BONY_LANDMARK_REGION", support_frames=independent,
        region_xyz_mm=region.tolist(), diameter_mm=diameter,
        max_disagreement_mm=max(d for _, _, d in pairs),
        uncertainty="Observed extent and cross-view disagreement. No inferred geometry or calibrated confidence.",
    )
    return result
