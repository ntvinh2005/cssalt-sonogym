"""Assemble the matched 2,000-image L5 training manifests from existing data."""

from collections import Counter, defaultdict
import csv
import hashlib
import json

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist

from perception_metrics import SEMANTIC, cases_for, pick_components
from replay import infer
from run_simple_landmark_mapping import sha256, write_json


def presence(row):
    labels = row["labels"]["L5"]
    return tuple(bool(labels[f"{name.lower()}_interface_pixels"]) for _, name in SEMANTIC)


def diverse_order(indices, records, rng):
    """Round-robin physical poses with their shuffled speckle repeats."""
    groups = defaultdict(list)
    for i in indices:
        groups[records[i]["geometry_sha256"]].append(i)
    buckets = [list(rng.permutation(group)) for group in groups.values()]
    return [int(i) for depth in range(max(map(len, buckets), default=0))
            for i in rng.permutation([group[depth] for group in buckets if depth < len(group)])]


def representative(indices, records, count, rng):
    """Sample proportional to the natural phase and interface-presence strata."""
    groups = defaultdict(list)
    for i in indices:
        groups[(records[i]["phase"], presence(records[i]))].append(i)
    quotas = {k: count * len(group) / len(indices) for k, group in groups.items()}
    ordered = sorted(groups, key=lambda key: quotas[key] % 1, reverse=True)
    integers = {k: int(n) for k, n in quotas.items()}
    for k in ordered[:count-sum(integers.values())]:
        integers[k] += 1
    return sorted(i for k, group in groups.items() for i in diverse_order(group, records, rng)[:integers[k]])


def legacy_selection(path, combined, geometry, settings, rng):
    with path.open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "train" and r["included"] == "True"
                and r["label_provenance"] == "SIMULATOR_WEAK_ANATOMICAL_LABEL"]
    positions = [json.loads(r["split_pose_position_m"]) for r in rows]
    radius = settings["group_buffer_mm"] / 1000
    safe = cdist(positions, [r["position_m"] for r in combined]).min(axis=1) >= radius
    conflicts = cdist(positions, [r["position_m"] for r in geometry]) < radius
    groups = [r["pose_group"] for r in geometry]
    blockers = [frozenset(groups[j] for j in np.flatnonzero(conflicts[i])) for i in range(len(rows))]
    weights = dict(Counter(groups))
    flags = np.asarray([presence(r) for r in geometry])
    target = settings["legacy_images"]
    excluded = set()
    while sum(bool(ok and missing <= excluded) for ok, missing in zip(safe, blockers)) < target:
        proposals = {missing-excluded for ok, missing in zip(safe, blockers) if ok and missing-excluded}
        viable = []
        for proposal in proposals:
            keep = np.asarray([g not in excluded | proposal for g in groups])
            if np.any(flags[keep].sum(axis=0) < settings["minimum_geometry_validation_positive_frames_per_class"]):
                continue
            unlocked = [i for i, (ok, missing) in enumerate(zip(safe, blockers))
                        if ok and missing <= excluded | proposal and not missing <= excluded]
            cost = sum(weights[g] for g in proposal)
            sap = sum(rows[i]["has_sap_interface"] == "True" for i in unlocked)
            viable.append(((len(unlocked) / cost, sap / cost, -cost, tuple(sorted(proposal))), proposal))
        if not viable:
            raise ValueError("400 legacy training images cannot be separated from grouped validation")
        excluded.update(max(viable)[1])
    eligible = [r for r, ok, missing in zip(rows, safe, blockers) if ok and missing <= excluded]
    jitter = rng.random(len(eligible))
    ranked = sorted(range(len(eligible)), key=lambda i: (eligible[i]["has_sap_interface"] != "True", jitter[i]))
    selected = sorted((eligible[i] for i in ranked[:target]), key=lambda r: int(r["archive_index"]))
    keep = [i for i, group in enumerate(groups) if group not in excluded]
    audit = {"original_training_rows":len(rows),"eligible_without_buffer_conflict":len(eligible),
             "removed_checkpoint_selection_groups":sorted(excluded),"geometry_validation_before":len(geometry),
             "geometry_validation_after":len(keep),"geometry_positive_after":flags[keep].sum(axis=0).tolist(),
             "legacy_positive":{name:sum(r[f"has_{name.lower()}_interface"]=="True" for r in selected)
                                for _,name in SEMANTIC}}
    return selected, keep, audit


def verified_training_candidates(models, images, targets, records, eligible, settings, device):
    evidence = {i: {"positive": [], "negative": []} for i in eligible}
    probabilities_examples = {}
    for name, (model, rules) in models.items():
        for i, ps in zip(eligible, infer(model, images[eligible], device)):
            target = targets[i]
            cases = cases_for([ps], [target], [records[i]], "L5", rules)
            winner = ps.argmax(0)
            for code, semantic in SEMANTIC:
                actual = target == code
                assessed = cases[semantic][0]
                if assessed["hard_positive"]:
                    evidence[i]["positive"].append(
                        {"model":name, "class":semantic, "visible_pixels":int(actual.sum())})
                distance = distance_transform_edt(~actual) if actual.any() else None
                candidates = pick_components(ps[code], winner == code, settings["hard_candidate_probability"],
                                             settings["hard_candidate_minimum_pixels"], distance)
                for component in candidates:
                    at = tuple(component["pixel"])
                    delta = component["distance_px"]
                    if delta is not None and delta <= 3.0:
                        continue
                    if target[at] == 3:
                        reason = "labeled_other_bone"
                    elif assessed["clear_negative"] and target[at] != 255:
                        reason = "no_visible_or_projected_target"
                    elif delta is not None and delta >= 5 and target[at] != 255:
                        reason = "mislocalized_visible_target"
                    else:
                        continue
                    evidence[i]["negative"].append(dict(component, model=name, semantic=semantic, reason=reason))
            if name == "pilot" and len(probabilities_examples) < 12 and (evidence[i]["positive"] or evidence[i]["negative"]):
                probabilities_examples[i] = ps
    return evidence, probabilities_examples


def choose_hard(indices, records, evidence, kind, target, rng):
    availability = [i for i in indices if evidence[i][kind]]
    diversity_rank = dict(zip(diverse_order(availability, records, rng), range(len(availability))))
    if kind == "negative":
        def priority(i):
            rows = evidence[i][kind]
            return (any(r["semantic"] == "TP" for r in rows),
                    max(r["confidence"] for r in rows))
    else:
        def priority(i):
            rows = evidence[i][kind]
            tp = [r for r in rows if r["class"] == "TP"]
            return (bool(tp), len(tp or rows) > 1, min(r["visible_pixels"] for r in tp or rows) <= 8)
    # Unique physical poses receive priority before additional speckle realizations.
    ranked = sorted(availability, key=lambda i: (priority(i), -diversity_rank[i]), reverse=True)
    seen = set()
    first, repeat = [], []
    for i in ranked:
        pose = records[i]["geometry_sha256"]
        (repeat if pose in seen else first).append(i)
        seen.add(pose)
    return (first + repeat)[:target]


def array_hash(image):
    return hashlib.sha256(np.ascontiguousarray(image, dtype=np.float32)).hexdigest()


def describe(items):
    generated = [r for r in items if r["category"] != "legacy"]
    return {"images":len(items),"category_counts":dict(Counter(r["category"] for r in items)),
            "combined_foreground_frames":{name:sum(
                r[f"has_{name.lower()}_interface"]=="True" if r["category"]=="legacy" else presence(r)[code-1]
                for r in items) for code,name in SEMANTIC},
            "generated_poses":len({r["geometry_sha256"] for r in generated}),
            "generated_source_groups":len({r["pose_group"] for r in generated}),
            "generated_speckle_fields":sorted({r["field"] for r in generated}),
            "generated_phase_counts":dict(Counter(r["phase"] for r in generated))}


def assemble(args, profile, grouped, arrays, specs, rules, output):
    settings = profile["inputs"]
    rng = np.random.default_rng(settings["selection_seed"])
    index_path = args.legacy_training/"l5_split_manifest.csv"
    archive_path = args.legacy_dataset/"dataset.npz"
    for path,key in ((index_path,"legacy_index_sha256"),(archive_path,"legacy_archive_sha256")):
        if sha256(path) != settings[key]:
            raise ValueError(f"archive checksum mismatch: {path}")
    write_json(output/"provenance.json",{"pilot_training_manifest_sha256":sha256(args.pilot_manifest),
        "legacy_index_sha256":sha256(index_path),"legacy_archive_sha256":sha256(archive_path),
        "robust_manifest_sha256":sha256(args.data/"manifest.jsonl"),"profile_sha256":sha256(args.profile),
        "source_protocol_sha256":sha256(args.protocol)})
    legacy, validation_keep, audit = legacy_selection(index_path, grouped["combined"], grouped["unseen_geometry"],
                                                       settings,rng)
    with np.load(archive_path) as old:
        legacy_ids = [int(r["archive_index"]) for r in legacy]
        legacy_images = old["ultrasound"][legacy_ids].astype(np.float32)
        legacy_labels = old["labels"][legacy_ids].astype(np.uint8)
    legacy = [dict(r, category="legacy", training_image_sha256=array_hash(img),
                   dataset=str(archive_path)) for r,img in zip(legacy, legacy_images)]
    if any(int((mask==code).sum()) != int(row[f"{name.lower()}_interface_pixels"])
           for mask,row in zip(legacy_labels,legacy) for code,name in SEMANTIC):
        raise ValueError("legacy label masks do not match their original oracle manifest")
    if not set(np.unique(legacy_labels)).issubset({0, 1, 2, 3, 255}):
        raise ValueError("legacy label vocabulary differs from robustness labels")
    records = grouped["train"]
    excluded = settings["diagnosed_medium_source_groups_excluded"]
    excluded_groups = {row["pose_group"] for row in records if row["source_seed"] == 2026121700
                       and row["waypoint"] in excluded}
    # A separate geometric buffer excludes analogous poses from other sources.
    bad = [row["position_m"] for row in records if row["pose_group"] in excluded_groups]
    near = cdist([r["position_m"] for r in records],bad).min(axis=1) < settings["group_buffer_mm"]/1000
    eligible = [i for i,r in enumerate(records) if not near[i] and r["pose_group"] not in excluded_groups]
    train_images, train_labels = arrays["train"]
    evidence, examples = verified_training_candidates(
        specs,train_images,train_labels,records,eligible,settings,args.device)
    np.savez_compressed(output/"mining_examples.npz",indices=list(examples),
                        ultrasound=train_images[list(examples)],labels=train_labels[list(examples)],
                        pilot_probability=list(examples.values()))
    count_b = settings["B_robustness_images"]
    batch_b = representative(eligible,records,count_b,rng)
    negative = choose_hard(eligible,records,evidence,"negative",settings["C_hard_negative_images"],rng)
    remaining = sorted(set(eligible)-set(negative))
    positive = choose_hard(remaining,records,evidence,"positive",settings["C_hard_positive_images"],rng)
    remaining = sorted(set(remaining)-set(positive))
    count_c = settings["C_representative_images"]+(settings["C_hard_positive_images"]-len(positive))+(
        settings["C_hard_negative_images"]-len(negative))
    batch_c = representative(remaining,records,count_c,rng)
    frozen_a = json.loads(args.pilot_manifest.read_text())
    base = [r["archive_index"] for r in frozen_a["records"]]
    if any(records[i]["image_sha256"] != r["image_sha256"] for i,r in zip(base,frozen_a["records"])):
        raise ValueError("Dataset A does not match archived robustness data")
    assignments = {"A_matched":[(i,"representative") for i in base],
                   "T2":[(i,"representative") for i in batch_b],
                   "T3":[(i,"representative") for i in batch_c] +
                        [(i,"hard_positive") for i in positive] + [(i,"hard_negative") for i in negative]}
    validation = {"unseen_speckle":list(range(len(grouped["unseen_speckle"]))),
                  "unseen_geometry":validation_keep}
    selection_rows = {split:[grouped[split][i] for i in ids] for split,ids in validation.items()}
    write_json(output/"validation_manifest.json",{"selection":selection_rows,"opened_comparison":grouped["combined"]})
    forbid = {array_hash(image) for split,(images, _) in arrays.items() if split!="train"
              for index,image in enumerate(images) if split=="combined" or index in validation[split]}
    # The re-rendered pool shares some poses with acquisition development. Raw source
    # images and the opened 99-image comparison remain disjoint from these manifests.
    evaluation_hashes = set()
    for filename in args.source_hashes:
        evaluation_hashes.update(json.loads(filename.read_text()))
    forbid |= evaluation_hashes
    train_sets = {}
    for arm, assigns in assignments.items():
        ids = [i for i,_ in assigns]
        old = arm != "A_matched"
        manifest = [dict(records[i],category=cat,training_image_sha256=array_hash(train_images[i]))
                    for i,cat in assigns] + (legacy if old else [])
        hashes = {r["training_image_sha256"] for r in manifest}
        if len(manifest) != profile["maximum_unique_training_images"] or len(hashes) != len(manifest) or hashes & forbid:
            raise ValueError(f"{arm} duplicate, over-budget or evaluation overlap")
        out = output/arm
        out.mkdir()
        write_json(out/"training_manifest.json",manifest)
        images = np.concatenate((train_images[ids],legacy_images)) if old else train_images[ids]
        targets = np.concatenate((train_labels[ids],legacy_labels)) if old else train_labels[ids]
        train_sets[arm] = images, targets
    write_json(output/"mining.json",{"eligible":len(eligible),"excluded_diagnosed_groups":sorted(excluded_groups),
        "excluded_count":len(records)-len(eligible),"candidate_count":{kind:sum(bool(r[kind]) for r in evidence.values())
                                                                    for kind in ("positive","negative")},
        "legacy_buffer_audit":audit,"candidates":evidence,"input_thresholds":rules,
        "selected_hard_positive":positive,"selected_hard_negative":negative})
    write_json(output/"leakage_audit.json",{"exact_saved_replay_image_hashes_checked":len(evaluation_hashes),
        "exact_raw_training_overlap":0,"legacy_source":"original L5 train included subset only",
        "legacy_unseen_geometry_excluded_whole_pose_groups":audit["removed_checkpoint_selection_groups"],
        "generated_train_to_opened_and_geometry_minimum_mm":settings["group_buffer_mm"],
        "familiar_geometry_validation":"unseen speckle fields on familiar poses by design",
        "medium_development_geometry":"Archived A and source renders can share physical neighborhoods with recorded development trajectories."})
    write_json(output/"composition.json",{
        arm:describe(json.loads((output/arm/"training_manifest.json").read_text()))
        for arm in assignments})
    return train_sets, validation
