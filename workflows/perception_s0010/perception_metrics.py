"""Measure connected-component acceptance against the frozen 3 px visible interface."""

from collections import Counter, defaultdict

import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects, label

from replay import STRUCTURE


SEMANTIC = ((1, "TP"), (2, "SAP"))


def pick_components(probability, exclusive, threshold, minimum, distance=None):
    groups, count = label((probability >= threshold) & exclusive, STRUCTURE)
    components = []
    for index, bounds in enumerate(find_objects(groups, count), start=1):
        rows = np.column_stack(np.where(groups[bounds] == index))
        if len(rows) < minimum:
            continue
        rows += (bounds[0].start, bounds[1].start)
        weights = probability[tuple(rows.T)]
        center = np.average(rows, axis=0, weights=weights)
        r, c = rows[np.argmin(np.sum((rows - center)**2, axis=1))]
        components.append({
            "pixel": (int(r), int(c)), "confidence": float(weights.mean()),
            "distance_px": float(distance[r,c]) if distance is not None else None,
            "pixels": int(len(rows)),
        })
    return components


def readout(components):
    accepted = bool(components)
    correct = any(c["distance_px"] is not None and c["distance_px"] <= 3.0 for c in components)
    return accepted, correct


def cases_for(probs, labels, records, level, rules, thresholds=None):
    cases = {name:[] for _,name in SEMANTIC}
    for probabilities, target, record in zip(probs, labels, records):
        winner = probabilities.argmax(0)
        frame = {}
        for code, name in SEMANTIC:
            actual = target == code
            tp = int((actual & (winner == code)).sum())
            fp = int(((winner == code) & (target != 255) & ~actual).sum())
            area = int(actual.sum())
            distance = distance_transform_edt(~actual) if area else None
            rule = rules[name]
            cutoff = rule["threshold"] if thresholds is None else thresholds[name]
            components = pick_components(probabilities[code], winner == code, cutoff, rule["minimum_pixels"], distance)
            accepted, correct = readout(components)
            genuine_wrong_bone = bool(any(target[tuple(c["pixel"])] == 3 for c in components
                                          if c["distance_px"] is None or c["distance_px"] > 3.0))
            metadata = record["labels"][level]
            clear_negative = not area and not metadata[f"projected_{name.lower()}_pixels"]
            frame[name] = {
                "positive": bool(area), "accepted": accepted, "correct": correct, "truth_pixels": area,
                "hard_positive": bool(area) and not correct,
                "hard_negative": genuine_wrong_bone,
                "clear_negative": clear_negative,
                "pixel_tp": tp, "pixel_fp": fp, "pixel_fn": area-tp,
                "components": components,
            }
            cases[name].append(frame[name])
    return cases


def statistics(cases):
    tp = sum(c["correct"] for c in cases)
    any_accept = sum(c["accepted"] for c in cases)
    present = sum(c["positive"] for c in cases)
    absent = len(cases) - present
    false_on_absent = sum(c["accepted"] and not c["positive"] for c in cases)
    clear_total = sum(c["clear_negative"] for c in cases)
    clear_false = sum(c["accepted"] and c["clear_negative"] for c in cases)
    pix = [sum(c[k] for c in cases) for k in ("pixel_tp","pixel_fp","pixel_fn")]
    tp_pix, fp_pix, fn_pix = pix
    distances = [min(d["distance_px"] for d in c["components"] if d["distance_px"] is not None)
                 for c in cases if c["positive"] and c["accepted"]]
    good_components = sum(d["distance_px"] is not None and d["distance_px"] <= 3.0
                          for c in cases for d in c["components"])
    all_components = sum(len(c["components"]) for c in cases)
    hard_negative = sum(c["hard_negative"] for c in cases)
    return {
        "frames": len(cases), "positive": present, "absent": absent,
        "correct_frames": tp, "incorrect_or_misaligned_accepted_frames": any_accept-tp,
        "missed_or_misaligned_visible_frames": present-tp, "accepted_frames": any_accept,
        "precision": tp/any_accept if any_accept else None, "recall": tp/present if present else None,
        "false_positive_rate": false_on_absent/absent if absent else None,
        "false_negative_rate": (present-tp)/present if present else None,
        "false_on_absent": false_on_absent, "clear_negative_false_accepts": [clear_false,clear_total],
        "correct_interface_components": [good_components,all_components],
        "borderline_representative_3_to_4_px": sum(3 < d <= 4 for d in distances),
        "closest_representative_px": {"count":len(distances), "median":float(np.median(distances)) if distances else None,
                                      "p95":float(np.quantile(distances,.95)) if distances else None},
        "pixel_confusion": pix, "pixel_dice": 2*tp_pix/(2*tp_pix+fp_pix+fn_pix) if any(pix) else None,
        "pixel_iou": tp_pix/sum(pix) if any(pix) else None,
        "other_bone_false_component_frames": hard_negative,
    }


def eval_fixed(probs, labels, records, level, rules):
    cases = cases_for(probs, labels, records, level, rules)
    return {name:statistics(rows) for name,rows in cases.items()}, cases


def curves(probs, target, records, level, rules, grid):
    accum = {name:{t:[] for t in grid} for _,name in SEMANTIC}
    bins = {name:defaultdict(lambda:[0,0]) for _,name in SEMANTIC}
    for prob, labels, record in zip(probs,target,records):
        winner = prob.argmax(0)
        for code, name in SEMANTIC:
            oracle = labels == code
            qualifies = bool(oracle.any())
            distance = distance_transform_edt(~oracle) if qualifies else None
            clear = not qualifies and not record["labels"][level][f"projected_{name.lower()}_pixels"]
            other = not qualifies and bool((labels == 3).any())
            floor = rules[name]["minimum_pixels"]
            for cutoff in grid:
                rows = pick_components(prob[code], winner==code, cutoff, floor, distance)
                a,c = readout(rows)
                accum[name][cutoff].append((qualifies,a,c,clear,other))
            best = max(pick_components(prob[code],winner==code,grid[0],floor,distance),
                       key=lambda item:item["confidence"], default=None)
            if best:
                confidence = best["confidence"]
                cell = min(9, int(confidence*10))
                bins[name][cell][0] += 1
                bins[name][cell][1] += int(best["distance_px"] is not None and best["distance_px"] <= 3)
    result = {}
    for _,name in SEMANTIC:
        points = []
        for threshold,v in accum[name].items():
            present,accepted,correct,clear,other = np.asarray(v,dtype=bool).T
            tp,called,positive = map(int,(correct.sum(),accepted.sum(),present.sum()))
            fp_negative = int((accepted & ~present).sum())
            total_negative = int((~present).sum())
            points.append({
                "threshold":threshold,"correct":tp,"accepted":called,"positive":positive,
                "precision":tp/called if called else None,"recall":tp/positive if positive else None,
                "false_on_absent": [fp_negative,total_negative],
                "clear_negative": [int((accepted & clear).sum()),int(clear.sum())],
                "other_bone_negative": [int((accepted & other).sum()),int(other.sum())],
            })
        result[name] = {"curve":points,"reliability_correct_counts":dict(bins[name])}
    return result


def stability(cases, records):
    report = {}
    for semantic,rows in cases.items():
        repeat = defaultdict(list)
        for idx,row in enumerate(records):
            repeat[(row["source_acquisition"],row["perturbation"])].append(idx)
        groups = [group for group in repeat.values() if len(group)>1]
        visible = [g for g in groups if all(rows[i]["positive"] for i in g)]
        present_changes = sum(1 < len(set(rows[i]["positive"] for i in g)) for g in groups)
        candidates = []
        for group in visible:
            picks = []
            for i in group:
                correct = [c for c in rows[i]["components"] if c["distance_px"] is not None and c["distance_px"] <= 3]
                if correct:
                    picks.append(max(correct,key=lambda c:c["confidence"])["pixel"])
            if len(picks)>1:
                candidates.append(float(np.linalg.norm(np.array(picks[0])-picks[1])*.5))
        report[semantic] = {
            "pose_groups_repeated_speckle":len(groups),
            "oracle_presence_changes":present_changes,
            "visible_all_fields":len(visible),
            "correct_under_all_fields":sum(all(rows[i]["correct"] for i in g) for g in visible),
            "paired_correct_representative_displacement_mm":{
                "pairs":len(candidates),"median":float(np.median(candidates)) if candidates else None,
                "p95":float(np.quantile(candidates,.95)) if candidates else None,
            },
        }
        perturb = defaultdict(dict)
        for idx,item in enumerate(records):
            perturb[(item["source_acquisition"],item["field"])][item["perturbation"]] = idx
        targets = [(d[0], i) for d in perturb.values() if 0 in d for k,i in d.items() if k]
        eligible = [(i,j) for i,j in targets if rows[i]["positive"] and rows[j]["positive"]]
        report[semantic]["same_field_pose_comparison"] = {
            "pairs":len(targets),"both_visible":len(eligible),
            "both_correct":sum(rows[i]["correct"] and rows[j]["correct"] for i,j in eligible),
        }
    return report
