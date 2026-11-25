"""Trace selected L5 observations back to the recorded ultrasound and masks."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
import yaml

from replay import CONVERSION, STRUCTURE, context, infer, models_for, replay
from run_simple_landmark_mapping import load_evaluator, load_public, sha256, write_json
from two_stage_navigation.incremental_landmark_map import IncrementalLandmarkMapConfig
from two_stage_navigation.simple_landmark_mapping import extract_observations, simple_map, stage1_region


def correspondences(prob, truth, context_row, coords, rules, class_code):
    rows, mask = extract_observations(prob.argmax(0), prob, rules, context_row, coords)
    actual = np.argwhere(truth == class_code)
    tree = cKDTree(actual) if len(actual) else None
    group = label(mask == class_code, STRUCTURE)[0]
    result = []
    for row in rows:
        if row.semantic_class != ("TP" if class_code == 1 else "SAP"):
            continue
        pix = row.row, row.column
        points = np.argwhere(group == group[pix])
        points_have_interface = int(np.count_nonzero(truth[tuple(points.T)] == class_code))
        min_dist = float(tree.query(points)[0].min()) if tree is not None else None
        distance = float(tree.query(pix)[0]) if tree is not None else None
        provisional = int(truth[pix])
        if tree is None:
            error = "FALSE_POSITIVE_WITH_NO_VISIBLE_TP" if class_code == 1 else "FALSE_POSITIVE_WITH_NO_VISIBLE_SAP"
        elif distance <= 3:
            error = "CORRECT"
        elif min_dist <= 3:
            error = "REPRESENTATIVE_POINT_ERROR"
        elif provisional == 3:
            error = "WRONG_ANATOMICAL_STRUCTURE"
        elif provisional == 255:
            error = "LABEL_AMBIGUITY"
        else:
            error = "SEGMENTATION_SPATIAL_SHIFT"
        result.append({
            **asdict(row), "representative_probability":float(prob[class_code][pix]),
            "interface_distance_px":distance,"component_min_distance_px":min_dist,
            "exact_oracle_pixel_overlap":points_have_interface,"representative_proxy_label":provisional,
            "other_bone_overlap":int((truth[tuple(points.T)]==3).sum()),"error":error,
        })
    return result, mask


def diagnosis_case(name, ctx, support, arms, targets, model_specs, output, class_code, base_batch=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    support = sorted(support)
    models=models_for(model_specs)
    if base_batch != 1:
        predicted = {m:{i:p for i,p in zip(ctx["old_valid"],infer(model,ctx["images"][ctx["old_valid"]],
                        batch_size=base_batch)) if i in support} for m,(model,rules) in models.items()}
    else:
        predicted = {m:{i:p for i,p in zip(support,infer(model,ctx["images"][support],batch_size=1))}
                     for m,(model,rules) in models.items()}
    fig,axes=plt.subplots(len(support),3,figsize=(8,2.25*len(support)),layout="constrained",squeeze=False)
    archive={}
    seen=[]
    for axes_for_frame,i in zip(axes,support):
        image=ctx["images"][i]
        truth=targets[i]
        f=ctx["frames"][i]
        record={"sequence":i,"waypoint":f.get("waypoint_index"),"frame_sha256":f["frame_sha256"],
                "acquisition_id":f["acquisition_id"],"phase":f.get("phase",f.get("trajectory_id")),
                "oracle_visible_pixels":int((truth==class_code).sum()),"models":{}}
        archive[f"{i}_us"]=image
        archive[f"{i}_oracle"]=truth
        if "raw" in ctx:
            archive[f"{i}_native_labels"]=ctx["raw"][i]
        for col,(model_name,(_,rules)) in enumerate(models.items(),start=1):
            probability=predicted[model_name][i]
            values,mask=correspondences(probability,truth,f,ctx["coords"][i],rules,class_code)
            for v in values:
                if "raw" in ctx:
                    v["simulator_structure_id"]=int(ctx["raw"][i,v["row"],v["column"]])
                v["in_selected_hypothesis"]=any(
                    row["frame_sequence"]==i and row["row"]==v["row"] and row["column"]==v["column"]
                    for row in arms[model_name])
            record["models"][model_name]={"components":values,
                "max_probability_on_oracle":float(probability[class_code][truth==class_code].max())
                if (truth==class_code).any() else None}
            archive[f"{i}_{model_name}_probability"]=probability
            archive[f"{i}_{model_name}_mask"]=mask
            ax=axes_for_frame[col]
            ax.imshow(probability[class_code],cmap="magma",vmin=0,vmax=1)
            if (truth==class_code).any():
                ax.contour(truth==class_code,levels=[.5],colors=["cyan"],linewidths=.8)
            if (mask==class_code).any():
                ax.contour(mask==class_code,levels=[.5],colors=["orange"],linewidths=.7)
            for r in values:
                ax.plot(r["column"],r["row"],"x",c="white" if r["in_selected_hypothesis"] else "#92e2f0",ms=5)
            ax.set_title(f"{model_name} probability and mask",fontsize=8)
        left=axes_for_frame[0]
        left.imshow(image,cmap="gray",vmin=0,vmax=np.quantile(image,.99))
        if (truth==class_code).any():
            left.contour(truth==class_code,levels=[.5],colors=["cyan"],linewidths=1)
        left.set_title(f"Frame {i} | {record['oracle_visible_pixels']} visible px",fontsize=8)
        for ax in axes_for_frame:
            ax.axis("off")
        seen.append(record)
    fig.suptitle(f"{name} | cyan oracle · orange actor mask · white selected representative")
    fig.savefig(output/f"{name}.png",dpi=135)
    plt.close(fig)
    np.savez_compressed(output/f"{name}.npz",**archive)
    return seen


def selected(entry, identity, include_competing=False):
    mapping=entry["mapping"][identity]
    clusters=([mapping["selected"]] if mapping["selected"] else [])
    if include_competing:
        clusters += mapping["competing_clusters"][:1]
    return [r for cluster in clusters for r in cluster["supporting_observations"]]


def historical_context(source, stored, stage1, protocol, model_specs):
    frames,images,valid,coords,_,_=load_public(source,protocol["geometry"]["ultrasound_pixel_resolution_mm"])
    with np.load(stored/"oracle/visible_interface_labels.npz") as data:
        targets=data["L5"]
    actor=json.loads((source/"results/actor_result.json").read_text())
    region=stage1_region(stage1,actor["stage1_session_id"],protocol)
    indices=[i for i in valid if "L5" in region.levels_at(frames[i]["executed_relative_pose_index"][1])]
    config=IncrementalLandmarkMapConfig(**protocol["landmark_map"])
    maps={}
    for name,(model,rules) in models_for(model_specs).items():
        obs,uncertain=defaultdict(list),defaultdict(list)
        for i,prob in zip(indices,infer(model,images[indices],batch_size=8)):
            rows,_=extract_observations(prob.argmax(0),prob,rules,frames[i],coords[i])
            destination=obs if len(region.levels_at(frames[i]["executed_relative_pose_index"][1]))==1 else uncertain
            for row in rows:
                destination[f"RIGHT_L5_{row.semantic_class}"].append(row)
        maps[name]={"mapping":simple_map(obs,uncertain,config)}
    return dict(frames=frames,images=images,coords=coords,old_valid=indices),targets,maps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("protocol","profile","previous","legacy-replay","stage1","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    protocol=yaml.safe_load(args.protocol.read_text())
    profile=yaml.safe_load(args.profile.read_text())
    assert sha256(args.protocol)==profile["source_protocol_sha256"]
    prior=args.previous/"evaluation-52723344/metrics"
    candidate=json.loads((prior/"selected_before_combined.json").read_text())
    assert candidate["models"]["L5"]["sha256"]==profile["previous_pilot_l5_sha256"]
    original=protocol["models"]["sonoanatomy"]
    model_specs={"original":original["L5"],"pilot":candidate["models"]["L5"]}
    retained=json.loads((prior/"downstream.json").read_text())
    evaluator=load_evaluator(protocol,CONVERSION)
    result={}
    wanted=[c for c in retained["cases"] if (c["kind"],c["seed"]) in (("medium",2026121700),("historical",2026121703))]
    for c in wanted:
        kind,name,code=("TP","medium_l5_tp_1700",1) if c["kind"]=="medium" else ("SAP","historical_l5_sap_1703",2)
        key="RIGHT_L5_"+kind
        ctx=context(Path(c["source"]),protocol,evaluator,c["kind"])
        combos={"original":c["arms"]["original"],"pilot":c["arms"][candidate["selected"]]}
        supports={n:selected(entry,key,code==2) for n,entry in combos.items()}
        ids={r["frame_sequence"] for r in supports["pilot"]}
        if code==2:
            ids|={r["frame_sequence"] for r in supports["original"]}
        result[name]={
            "source":c["source"],"map":{n:e["mapping"][key] for n,e in combos.items()},
            "score":{n:e["score"]["landmarks"][key] for n,e in combos.items()},
            "frames":diagnosis_case(name,ctx,ids,supports,ctx["labels"]["L5"],model_specs,args.output,code),
        }
        print("Diagnosed",name,flush=True)
    old=json.loads((args.legacy_replay/"comparison.json").read_text())
    stage1=json.loads(args.stage1.read_text())
    audited=json.loads((args.previous/"supplement-52725009/v6r19_regression.json").read_text())
    for row in audited["cases"]:
        key="RIGHT_L5_SAP"
        if row["arms"]["selected"]["landmarks"][key]["outcome"]!="UNRESOLVED":
            continue
        src=next(x for x in old["sessions"] if x["session"]==row["name"])
        ctx,target,maps=historical_context(Path(src["source"]),args.legacy_replay/src["session"],stage1,protocol,model_specs)
        support={n:selected(e,key,True) for n,e in maps.items()}
        ids={v["frame_sequence"] for values in support.values() for v in values}
        name=row["name"]+"_l5_sap"
        result[name]={"source":src["source"],"map":{n:e["mapping"][key] for n,e in maps.items()},
                      "score":{n:row["arms"][k]["landmarks"][key] for n,k in (("original","original"),("pilot","selected"))},
                      "frames":diagnosis_case(name,ctx,ids,support,target,model_specs,args.output,2,8)}
        print("Diagnosed",name,flush=True)
    write_json(args.output/"diagnosis.json",result)


if __name__=="__main__":
    main()
