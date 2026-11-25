"""Audit the 2000-image split and replay compatible historical ultrasound."""

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
import yaml

from replay import CLASSES, CONVERSION, STRUCTURE, context, infer, models_for, route
from run_simple_landmark_mapping import load_evaluator, load_public, read_jsonl, score_map, sha256, unet_observations, write_json
from two_stage_navigation.incremental_landmark_map import IncrementalLandmarkMapConfig
from two_stage_navigation.simple_landmark_mapping import extract_observations, simple_map, stage1_region


def audit_split(args, campaign):
    original=[]
    for level in ("l4","l5"):
        with (args.legacy_training/f"{level}_split_manifest.csv").open() as handle:
            original.extend(json.loads(r["split_pose_position_m"]) for r in csv.DictReader(handle)
                            if r["included"]=="True" and r["split"]=="train")
    earlier=json.loads((args.previous/"review-52711211/E3-training/training_manifest.json").read_text())
    original.extend(r["position_m"] for r in earlier["training_records"])
    actual=json.loads((args.training/"training_split.json").read_text())
    training=actual["records"]
    full=read_jsonl(args.data/"manifest.jsonl")
    roles={s:[r for r in full if r["split"]==s] for s in ("unseen_speckle","unseen_geometry","combined")}
    roles["train"]=training
    excluded=set(campaign["reserved_confirmation_seeds"])
    assert len({r["image_sha256"] for r in training})==len(training)==2000
    assert not actual["original_examples_allowed"]
    assert all(r["source_seed"] not in excluded for r in full)
    positions=cKDTree([r["position_m"] for r in training])
    geometry={s:{r["geometry_sha256"] for r in rows} for s,rows in roles.items()}
    fields={s:{r["field"] for r in rows} for s,rows in roles.items()}
    source_groups={s:{r["source_acquisition"] for r in rows} for s,rows in roles.items()}
    images={s:{r["image_sha256"] for r in rows} for s,rows in roles.items()}
    disjoint={}
    for name in ("unseen_geometry","combined"):
        disjoint[name]={
            "min_mm_from_training": float(1000*positions.query([r["position_m"] for r in roles[name]])[0].min()),
            "shared_geometry_hashes":len(geometry[name]&geometry["train"]),
            "shared_source_acquisitions":len(source_groups[name]&source_groups["train"]),
            "shared_image_hashes":len(images[name]&images["train"]),
        }
        assert disjoint[name]["min_mm_from_training"]>=campaign["data"]["legacy_and_split_buffer_mm"]
        assert all(disjoint[name][k]==0 for k in
                   ("shared_geometry_hashes","shared_source_acquisitions","shared_image_hashes"))
    assert not (fields["combined"] & (fields["train"]|fields["unseen_geometry"]|fields["unseen_speckle"]))
    assert not (fields["unseen_speckle"] & fields["train"])
    assert geometry["unseen_speckle"] <= geometry["train"]
    legacy_dist = 1000*cKDTree(original).query([r["position_m"] for r in roles["combined"]])[0]
    assert legacy_dist.min()>=campaign["data"]["legacy_and_split_buffer_mm"]
    counts={}
    for source,rows in roles.items():
        counts[source]={
            "images":len(rows),"poses":len(geometry[source]),"pose_groups":len(source_groups[source]),
            "field_counts":dict(Counter(r["field"] for r in rows)),
            "phase_counts":dict(Counter(r["phase"] for r in rows)),
            "class_positive_images":{
                f"{level}_{name}":sum(r["labels"][level][f"has_{name.lower()}_interface"] for r in rows)
                for level in ("L4","L5") for name in ("TP","SAP")},
        }
    record={
        "splits":counts, "separation":disjoint, "combined_min_mm_from_pretraining_and_E3":float(legacy_dist.min()),
        "known_geometry_new_speckle":len(geometry["unseen_speckle"]),
        "all_generated_images":len(full),
        "surplus_generated_training_images_not_used":sum(r["split"]=="train" for r in full)-len(training),
        "reserved_seed_intersection":[],
        "training_manifest_sha256":sha256(args.training/"training_split.json"),
    }
    mining={}
    for level in ("L4","L5"):
        path=args.training/level/"mining_used.json"
        if path.exists():
            used=json.loads(path.read_text())["candidates"]
            mining[level]={
                "unique_hard_positive_images":sum(bool(r["positive"]) for r in used),
                "unique_hard_negative_images":sum(bool(r["negative"]) for r in used),
                "positive_per_class":dict(Counter(k for r in used for k in r["positive"])),
                "negative_per_class":dict(Counter(k for r in used for k in r["negative"])),
            }
    record["mined_on_the_selected_2000"]=mining
    write_json(args.output/"split_audit.json",record)


def legacy_replay(args,protocol,specs):
    baseline=json.loads((args.legacy_replay/"comparison.json").read_text())
    stage1=json.loads(args.stage1.read_text())
    config=IncrementalLandmarkMapConfig(**protocol["landmark_map"])
    output=[]
    for src in baseline["sessions"]:
        session=Path(src["source"])
        frames,images,valid,coords,_,_=load_public(session,protocol["geometry"]["ultrasound_pixel_resolution_mm"])
        if valid!=src["valid_frames"]:
            raise ValueError("incompatible v6r19 validity masks")
        actor=json.loads((session/"results/actor_result.json").read_text())
        region=stage1_region(stage1,actor["stage1_session_id"],protocol)
        visible,ref={},{}
        with np.load(args.legacy_replay/src["session"]/"oracle/visible_interface_labels.npz") as saved:
            for level in ("L4","L5"):
                target=saved[level]
                for code,kind in ((1,"TP"),(2,"SAP")):
                    identity=f"RIGHT_{level}_{kind}"
                    points=[]
                    for i in valid:
                        pixels=target[i]==code
                        if pixels.any():
                            visible[(identity,i)]=cKDTree(np.argwhere(pixels))
                            points.append(coords[i][pixels])
                    if points:
                        ref[identity]=cKDTree(np.concatenate(points))
        entry={"name":src["session"],"frames":len(valid),"arms":{}}
        for name,spec in specs.items():
            models=models_for(spec)
            get,ambiguous,_,_=unet_observations(frames,images,valid,coords,region,models,8)
            result=simple_map(get,ambiguous,config)
            score=score_map(result,get,visible,ref,protocol["evaluator"]["maximum_semantic_interface_distance_pixels"])
            entry["arms"][name]=score
        output.append(entry)
        print("v6r19 historical replay",src["session"],flush=True)
    write_json(args.output/"v6r19_regression.json",{
        "scope":"Frozen historical frame-center level routing and point map; no 102-waypoint claim",
        "historical_baseline": baseline["comparison"]["B"],"cases":output})


def representative_case(ctx,protocol,models,peak):
    observed,ambiguous=defaultdict(list),defaultdict(list)
    config=IncrementalLandmarkMapConfig(**protocol["landmark_map"])
    for level,(model,rules) in models.items():
        for i,prob in zip(ctx["valid"],infer(model,ctx["images"][ctx["valid"]],batch_size=1)):
            f, xyz=ctx["frames"][i],ctx["coords"][i]
            rows,masks=extract_observations(prob.argmax(0),prob,rules,f,xyz)
            labelled={}
            if peak:
                for c in (1,2):
                    labelled[c]=label(masks==c,STRUCTURE)[0]
            for r in rows:
                if peak:
                    c=CLASSES.index(r.semantic_class)
                    membership=labelled[c]
                    coord=np.argwhere(membership==membership[r.row,r.column])
                    best=coord[np.argmax(prob[c][tuple(coord.T)])]
                    r=replace(r,row=int(best[0]),column=int(best[1]),coordinate_mm=tuple(xyz[tuple(best)]))
                n=route(r,f,level,ctx,True)
                if n:
                    (observed if n==1 else ambiguous)[f"RIGHT_{level}_{r.semantic_class}"].append(r)
    mapped=simple_map(observed,ambiguous,config)
    score=score_map(mapped,observed,ctx["visible"],ctx["ref"],
                    protocol["evaluator"]["maximum_semantic_interface_distance_pixels"])
    return {"score":score,"complete_with_integrity":score["complete_map"] and len(ctx["valid"])==102
            and ctx["projection"]["passes"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("protocol","profile","stage1","previous","historical","legacy-training","legacy-replay",
                 "development","data","training","evaluation","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    protocol=yaml.safe_load(args.protocol.read_text())
    campaign=yaml.safe_load(args.profile.read_text())
    audit_split(args,campaign)
    pre=(args.evaluation/"selected_before_combined.json")
    assert sha256(pre)==(args.evaluation/"selected_before_combined.sha256").read_text().split()[0]
    fixed=json.loads(pre.read_text())["models"]
    original=protocol["models"]["sonoanatomy"]
    pilot=json.loads((args.previous/"pilot-recovery-52713119/metrics/training_result.json").read_text())
    specs={"original":original,"E3":dict(original,L5={"path":pilot["checkpoint"],"sha256":pilot["checkpoint_sha256"]}),
           "selected":fixed}
    legacy_replay(args,protocol,specs)
    evaluator=load_evaluator(protocol,CONVERSION)
    models=models_for(fixed)
    groups=[]
    for seed in campaign["development_seeds"]:
        p=(args.previous/"acquisition-52710240"/f"E1_{seed}" if seed in campaign["development_seeds"][:2]
           else args.development/str(seed))
        ctx=context(p,protocol,evaluator,"medium")
        reference=representative_case(ctx,protocol,models,False)
        peak=representative_case(ctx,protocol,models,True)
        groups.append({"seed":seed,"weighted_center":reference,"maximum_probability":peak})
    write_json(args.output/"representative_sensitivity.json",{
        "scope":"Separate fixed-checkpoint replay; maximum-probability representative was not a selection candidate",
        "cases":groups})


if __name__=="__main__":
    main()
