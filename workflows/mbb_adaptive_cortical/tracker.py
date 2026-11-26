"""Search bounded, image-supported paths between frozen sonographic anchors."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial.distance import cdist, pdist

from workflows.mbb_direct_sonographic.localize import COMPETING, SAP, TP, contiguous_runs, profile_evidence


def echo_fields(image, rules):
    low, high = np.quantile(image, (.05, .95))
    smooth = gaussian_filter1d((image - low) / max(float(high - low), 1e-6), .7, axis=0)
    peak = smooth == maximum_filter1d(smooth, 3, axis=0)
    depth = np.arange(5, len(image) - 9)
    intensity = smooth[depth]
    proximal = intensity - smooth[depth[:, None] - np.arange(2, 6)].mean(axis=1)
    distal = intensity - smooth[depth[:, None] + np.arange(3, 10)].mean(axis=1)
    qualified = (peak[depth] & (intensity >= rules["minimum_relative_echo"])
                 & (proximal >= rules["minimum_proximal_contrast"])
                 & (distal >= rules["minimum_distal_contrast"]))
    return depth, intensity, proximal, distal, qualified


def anchor_brackets(profiles, annotations, lateral, rules):
    """Use the original endpoint support, direction, horizontal span and depth rise."""
    flank = rules["turn_window_px"]
    for j, (pixels, support, _) in enumerate(profiles):
        labels, _ = profile_evidence(pixels, support, annotations, rules)
        if lateral == 1:
            pixels, support = pixels[::-1], support[::-1]
            labels = {name: mask[::-1] for name, mask in labels.items()}
        tp = contiguous_runs(labels[TP], rules["anchor_span_px"])
        sap = contiguous_runs(labels[SAP], rules["anchor_span_px"])
        for a in tp:
            for b in sap:
                start, stop = int(a[-1]), int(b[0])
                lo, hi = start - flank + 1, stop + flank
                rise = (np.median(pixels[a[-flank:], 0]) - np.median(pixels[b[:flank], 0]))
                if (stop <= start or lo < 0 or hi > len(pixels)
                        or (stop - start) * rules["pixel_mm"] > rules["max_anchor_span_mm"]
                        or rise * rules["pixel_mm"] < rules["min_anchor_depth_rise_mm"]):
                    continue
                yield {
                    "profile": j, "pixels": pixels[lo:hi], "baseline_support": support[lo:hi],
                    "competing_on_original": np.logical_or.reduce([labels[k] for k in COMPETING])[lo:hi].any(),
                    "start": start - lo, "stop": stop - lo,
                }


def candidates_for(bracket, track, fields, annotations, rules, config):
    depth, intensity, proximal, distal, qualified = fields
    window = round(config["corridor_half_width_mm"] / rules["pixel_mm"])
    centers = dict(zip(track.columns, np.rint(track.rows).astype(int)))
    competing = np.concatenate([annotations[name][0] for name in COMPETING])
    candidates, scores = [], []
    for i, (old_row, col) in enumerate(bracket["pixels"]):
        selected = qualified[:, col] & (np.abs(depth - centers[col]) <= window)
        if i <= bracket["start"] or i >= bracket["stop"]:
            selected &= depth == old_row
        rows = depth[selected]
        if len(rows) and len(competing):
            keep = cdist(np.column_stack((rows, np.full(len(rows), col))), competing).min(axis=1)
            selected[selected] = keep > rules["track_label_radius_px"]
        values = np.column_stack((intensity[selected, col], proximal[selected, col], distal[selected, col]))
        score = 1.0 - np.clip(values, 0, 1) @ np.array([.5, .25, .25])
        candidates.append([[int(row), *value.tolist()] for row, value in zip(depth[selected], values)])
        scores.append(dict(zip(map(int, depth[selected]), map(float, score))))
    return candidates, scores


def pass_dp(scores, max_gap, step_limit, step_cost, gap_cost, pinned):
    """Return minimum-cost prefixes indexed by column, depth and missing-column count."""
    states = [{(row, 0): (cost, None) for row, cost in scores[0].items()}]
    for c in range(1, len(scores)):
        current = {}
        for step in range(1, min(c, max_gap + 1) + 1):
            if any(pinned[c - step + 1:c]):
                continue
            for (previous_row, old_gap), (cost, _) in states[c - step].items():
                gap = old_gap + step - 1
                if gap > max_gap:
                    continue
                for row, echo_cost in scores[c].items():
                    depth_change = abs(row - previous_row)
                    if depth_change > step_limit * step:
                        continue
                    total = cost + echo_cost + step_cost * depth_change + (step - 1) * gap_cost
                    state = (row, gap)
                    if state not in current or total < current[state][0]:
                        current[state] = (total, (c - step, previous_row, old_gap))
        states.append(current)
    return states


def backtrack(states, node):
    path = [None] * len(states)
    while node is not None:
        column, row, missing = node
        path[column] = row
        node = states[column][(row, missing)][1]
    return path


def best_node(states):
    column = next((i for i in reversed(range(len(states))) if states[i]), None)
    if column is None:
        return None
    row, missing = min(states[column], key=lambda key: states[column][key][0])
    return column, row, missing


def track_path(scores, max_gap, pinned, step_limit, config, spacing):
    options = (max_gap, step_limit, config["step_cost_per_pixel"], config["gap_cost_per_column"])
    forward = pass_dp(scores, *options, pinned)
    backward = pass_dp(scores[::-1], *options, pinned[::-1])
    end = best_node(forward)
    prefix = backtrack(forward, end)
    if end is None or end[0] != len(scores) - 1:
        suffix = backtrack(backward, best_node(backward))[::-1]
        # Keep representations of separately measured end portions, not one joined path.
        return {"status": "UNSUPPORTED", "rows": prefix, "suffix_rows": suffix,
                "alternative_rows": None, "cost": None, "alternative_cost": None}
    cost = forward[-1][end[1:]][0]
    alternate = None
    for c, options_at_column in enumerate(forward):
        if prefix[c] is None:
            continue
        for (row, left_gap), (left_cost, _) in options_at_column.items():
            if abs(row - prefix[c]) * spacing < config["ambiguity_depth_mm"]:
                continue
            for (right_row, right_gap), (right_cost, _) in backward[len(scores) - 1 - c].items():
                if right_row != row or left_gap + right_gap > max_gap:
                    continue
                total = left_cost + right_cost - scores[c][row]
                if alternate is None or total < alternate[0]:
                    alternate = total, c, row, left_gap, right_gap
    alternative_rows, alternate_cost = None, None
    if alternate is not None:
        alternate_cost, c, row, left_gap, right_gap = alternate
        left = backtrack(forward, (c, row, left_gap))
        right = backtrack(backward, (len(scores) - 1 - c, row, right_gap))[::-1]
        alternative_rows = left[:c] + right[c:]
    return {
        "status": "AMBIGUOUS" if alternate_cost is not None and alternate_cost <= cost + config["ambiguity_cost_margin"]
        else "JOINED",
        "rows": prefix, "suffix_rows": None, "alternative_rows": alternative_rows,
        "cost": cost, "alternative_cost": alternate_cost,
    }


def turn_regions(path, bracket, rules):
    """Apply the historical bend rule within an entirely observed local window."""
    if path["status"] != "JOINED":
        return []
    rows = path["rows"]
    flank = rules["turn_window_px"] - 1
    bends = []
    for i in range(bracket["start"], bracket["stop"] + 1):
        if any(value is None for value in rows[i - flank:i + flank + 1]):
            continue
        previous = (rows[i] - rows[i - flank]) / flank
        following = (rows[i + flank] - rows[i]) / flank
        if following < 0 and previous - following >= rules["min_turn_slope_change"]:
            bends.append((i, previous - following))
    strongest = max((strength for _, strength in bends), default=0.)
    return [i for i, strength in bends if strength >= strongest - rules["min_turn_slope_change"]]


def adaptive_frame(original, brackets, results, rules):
    """Keep the historical status outside eligible brackets and gate the new paths."""
    if not brackets:
        return original
    output = {**original, "direct_candidate": False, "candidate_pixels": [], "support_pixels": {}}
    if any(result["status"] == "AMBIGUOUS" for result in results):
        return {**output, "status": "AMBIGUOUS"}
    support = {TP: [], SAP: [], "observed_bridge": [], "candidate": []}
    for bracket, path in zip(brackets, results):
        turns = turn_regions(path, bracket, rules)
        if not turns:
            continue
        start, stop = bracket["start"], bracket["stop"]
        cols = bracket["pixels"][:, 1]
        observed = [[row, int(cols[c])] if row is not None else None for c, row in enumerate(path["rows"])]
        for i in turns:
            support["candidate"].extend(observed[i - 1:i + 2])
        support[TP].extend(observed[:start + 1])
        support[SAP].extend(observed[stop:])
        support["observed_bridge"].extend(p for p in observed[start:stop + 1] if p is not None)
    if not support["candidate"]:
        status = "NO_LOCAL_CORTICAL_TURN" if any(p["status"] == "JOINED" for p in results) else "UNSUPPORTED"
        return {**output, "status": status}
    points = np.unique(support.pop("candidate"), axis=0)
    diameter = float(pdist(points).max() * rules["pixel_mm"]) if len(points) > 1 else 0.
    if diameter > rules["max_region_diameter_mm"]:
        return {**output, "status": "AMBIGUOUS_IMAGE_REGIONS", "candidate_image_diameter_mm": diameter}
    return {
        **output, "status": "DIRECT_IMAGE_CANDIDATE", "direct_candidate": True,
        "candidate_pixels": points.tolist(), "candidate_image_diameter_mm": diameter,
        "support_pixels": {name: np.unique(pixels, axis=0).tolist() for name, pixels in support.items()},
    }
