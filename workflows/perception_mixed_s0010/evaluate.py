"""Replay the fixed acquisition and isolate observation representatives offline."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import label
from scipy.spatial.distance import cdist
import yaml

from fit import verified_archive
from perception_metrics import SEMANTIC, curves, eval_fixed
from replay import CONVERSION, STRUCTURE, context, infer, models_for, oracle_map, replay, route
from review import draw_map
from run_simple_landmark_mapping import load_evaluator, read_jsonl, score_map, sha256, write_json
from selection import downstream_summary, rates_from_rich
from supplement import legacy_replay
from two_stage_navigation.incremental_landmark_map import IncrementalLandmarkMapConfig
from two_stage_navigation.simple_landmark_mapping import CLASSES, extract_observations, simple_map


POINT_POLICIES = ("weighted_center", "weighted_medoid", "maximum_probability")


def relocate(point, mode, pixels, probability, xyz):
    if mode == "weighted_center":
        return point
    weights = probability[tuple(pixels.T)]
    if mode == "weighted_medoid":
        idx = np.argmin(cdist(pixels,pixels) @ weights)
    else:
        idx = np.argmax(weights)
    r,c = map(int,pixels[idx])
    return replace(point,row=r,column=c,coordinate_mm=tuple(xyz[r,c]))


def point_diagnostic(ctx, protocol, models, fields):
    buckets = {name:(defaultdict(list),defaultdict(list)) for name in POINT_POLICIES}
    moves = {name:[] for name in POINT_POLICIES}
    for lvl, (_, rules) in models.items():
        for i, prob in zip(ctx["valid"],fields[lvl]):
            frame,xyz = ctx["frames"][i],ctx["coords"][i]
            rows, mask = extract_observations(prob.argmax(0),prob,rules,frame,xyz)
            membership = {code:label(mask==code,STRUCTURE)[0] for code,_ in SEMANTIC}
            for point in rows:
                code = CLASSES.index(point.semantic_class)
                components = membership[code]
                region = np.argwhere(components==components[point.row,point.column])
                identity = f"RIGHT_{lvl}_{point.semantic_class}"
                oracle = ctx["visible"].get((identity,i))
                for name in POINT_POLICIES:
                    item = point if lvl=="L4" else relocate(point,name,region,prob[code],xyz)
                    eligibility = route(item,frame,lvl,ctx,True)
                    if eligibility:
                        buckets[name][eligibility != 1][identity].append(item)
                    if lvl=="L5":
                        moves[name].append({"frame":i,"waypoint":frame["waypoint_index"],"identity":identity,
                            "pixel":[item.row,item.column],"original_pixel":[point.row,point.column],
                            "eligible":eligibility==1,"confidence":item.confidence,
                            "distance_px":float(oracle.query((item.row,item.column))[0]) if oracle is not None else None})
    config = IncrementalLandmarkMapConfig(**protocol["landmark_map"])
    gate = protocol["evaluator"]["maximum_semantic_interface_distance_pixels"]
    results = {}
    for name,(accepted,ambiguous) in buckets.items():
        current = simple_map(accepted,ambiguous,config)
        score = score_map(current,accepted,ctx["visible"],ctx["ref"],gate)
        results[name] = {"mapping":current,"score":score,"changes":moves[name]}
    return results


def point_metrics(records, gate):
    result = {}
    for name in POINT_POLICIES:
        result[name] = {}
        for arm in ("pilot","selected"):
            for _,semantic in SEMANTIC:
                points = [r for case in records for r in case[arm][name]["changes"]
                          if r["identity"]=="RIGHT_L5_"+semantic and r["eligible"]]
                distances = [r["distance_px"] for r in points if r["distance_px"] is not None]
                good = sum(d <= gate for d in distances)
                result[name].setdefault(arm,{})[semantic] = {
                    "correct_over_eligible":[good,len(points)],"visible_distance_count":len(distances),
                    "no_visible_target":len(points)-len(distances),"visible_beyond_3px":len(distances)-good,
                    "median_px":float(np.median(distances)) if distances else None,
                    "p95_px":float(np.quantile(distances,.95)) if distances else None}
    return result


def historical_summary(results, arm):
    cases = [r["arms"][arm] for r in results]
    landmarks = [x["landmarks"] for x in cases]
    count = Counter(row["outcome"] for group in landmarks for row in group.values())
    return {"cases":len(cases),"correct":count["CORRECT"],"wrong":count["WRONG"],
            "unresolved":count["UNRESOLVED"],"complete":sum(x["complete_map"] for x in cases),
            "correct_by_landmark":{k:sum(group[k]["outcome"]=="CORRECT" for group in landmarks)
                                   for k in landmarks[0]}}


def plots(fit, image_metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for filename in sorted(fit.glob("*/history.json")):
        history = json.loads(filename.read_text())
        x = [r["epoch"] for r in history]
        axes[0].plot(x,[r["train_loss"] for r in history],label=filename.parent.name)
        axes[1].plot(x,[r["validation_score"] for r in history],label=filename.parent.name)
    for ax, title in zip(axes,("Training objective","Grouped validation interface F1")):
        ax.set(xlabel="Epoch",ylabel=title)
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(output/"learning_curves.png",dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,(_,semantic) in zip(axes,SEMANTIC):
        for arm,data in image_metrics.items():
            curve = [p for p in data["curves"][semantic]["curve"] if p["precision"] is not None]
            ax.plot([p["recall"] for p in curve],[p["precision"] for p in curve],"o-",ms=3,label=arm)
        ax.set(xlim=(0,1.05),ylim=(0,1.05),xlabel="Correct-interface recall",
               ylabel="Correct-interface precision",title="L5 "+semantic)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(output/"precision_recall_99.png",dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","profile","stage1","previous","historical","legacy-replay","data","fit","output"):
        parser.add_argument("--"+key,required=True,type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg = yaml.safe_load(args.profile.read_text())
    if sha256(args.protocol) != cfg["source_protocol_sha256"]:
        raise ValueError("source protocol does not match the frozen comparison")
    protocol = yaml.safe_load(args.protocol.read_text())
    original = protocol["models"]["sonoanatomy"]
    prior = args.previous/"evaluation-52723344/metrics"
    local = json.loads((prior/"selected_before_combined.json").read_text())
    if local["models"]["L5"]["sha256"] != cfg["previous_pilot_l5_sha256"]:
        raise ValueError("previous pilot hash mismatch")
    specs = {"original":original,"pilot":local["models"]}
    for path in sorted(args.fit.glob("*/result.json")):
        result = json.loads(path.read_text())
        if result["status"]=="completed" and result["provenance"]["num_training"]==2000:
            specs[path.parent.name] = dict(original,L5={"path":result["checkpoint"], "sha256":result["checkpoint_sha256"]})
    if not {"T2","T3"} <= set(specs):
        raise ValueError("T2 and T3 must complete before comparative evaluation")
    l4 = models_for({"L4":original["L4"]})["L4"]
    l5 = {name:models_for({"L5":spec["L5"]})["L5"] for name,spec in specs.items()}
    if any(rules != l5["original"][1] for _,rules in l5.values()):
        raise ValueError("all checkpoints must preserve original semantic rules")
    receipt = json.loads((args.data/"receipt.json").read_text())
    records = [r for r in read_jsonl(args.data/"manifest.jsonl") if r["split"]=="combined"]
    if len(records)!=99:
        raise ValueError("incorrect 99-image opened evaluation set")
    with verified_archive(args.data,"combined",receipt) as archive:
        images,targets = archive["images"],archive["L5"]
    image_results = {}
    for arm,(model,rules) in l5.items():
        probabilities = list(infer(model,images,"cpu"))
        fixed,cases = eval_fixed(probabilities,targets,records,"L5",rules)
        if arm=="original":
            reference_hard = cases
        image_results[arm] = {"fixed":fixed,"hard_rate":rates_from_rich(cases,reference_hard),
                "curves":curves(probabilities,targets,records,"L5",rules,[.50,.55,.60,.65,.70,.75,.80,.85,.90,.95])}
    write_json(args.output/"opened_99_images.json",image_results)
    plots(args.fit,image_results,args.output)

    evaluator = load_evaluator(protocol,CONVERSION)
    archived = json.loads((prior/"downstream.json").read_text())
    cases = []
    for item in archived["cases"]:
        path,kind = Path(item["source"]),item["kind"]
        if item["seed"] in cfg["reserved_seeds"]:
            raise ValueError("reserved seed present in replay plan")
        ctx = context(path,protocol,evaluator,kind)
        saved_l4 = list(infer(l4[0],ctx["images"][ctx["valid"]],"cpu",batch_size=1))
        entry = {"source":str(path),"seed":item["seed"],"kind":kind,"valid":len(ctx["valid"]),
                 "projection":ctx["projection"],"arms":{}}
        for name,(model,rules) in l5.items():
            models = {"L4":l4,"L5":(model,rules)}
            fields = {"L4":saved_l4,"L5":list(infer(model,ctx["images"][ctx["valid"]],"cpu",batch_size=1))}
            entry["arms"][name],_ = replay(ctx,protocol,models,tp_all=kind=="medium",probabilities=fields,diagnose=True)
        entry["arms"]["oracle"] = oracle_map(ctx,protocol,tp_all=kind=="medium")
        if kind=="medium":
            entry["arms"]["oracle_routed"] = oracle_map(ctx,protocol,tp_all=True,strict_routing=True)
        cases.append(entry)
        write_json(args.output/"downstream.json",cases)
        print("Completed frozen replay",kind,item["seed"],flush=True)
    subsets = {name:{key:downstream_summary(cases,name,key) for key in ("medium","historical")} for name in specs}
    subsets["oracle"] = {"medium":downstream_summary(cases,"oracle","medium")}
    subsets["oracle_routed"] = {"medium":downstream_summary(cases,"oracle_routed","medium")}
    write_json(args.output/"replay_summary.json",subsets)
    legacy_replay(SimpleNamespace(legacy_replay=args.legacy_replay,stage1=args.stage1,output=args.output),protocol,specs)
    old = json.loads((args.output/"v6r19_regression.json").read_text())["cases"]
    for name in specs:
        subsets[name]["historical_816"] = historical_summary(old,name)
    checks = {}
    gate = cfg["readiness"]
    old_tp = image_results["pilot"]["fixed"]["TP"]
    for name in specs:
        tp = image_results[name]["fixed"]["TP"]
        profile = subsets[name]
        medium = profile["medium"]
        historical = [profile[t] for t in ("historical","historical_816")]
        checks[name] = {
            "correct_landmarks":medium["correct"]>=gate["correct_landmarks"],
            "complete_maps":medium["complete"]>=gate["complete_maps"],
            "zero_wrong_development":medium["wrong"]<=gate["wrong_landmarks"],
            "no_material_historical_sap_regression":all(a["correct_by_landmark"].get("RIGHT_L5_SAP",0)>=
                                                        gate["historical_sap_correct_each"] for a in historical),
            "comparable_tp_precision":bool(tp["precision"] and
                tp["precision"]>=old_tp["precision"]-gate["evaluation_99_l5_tp_precision_loss_absolute"]),
            "comparable_tp_recall":tp["correct_frames"]>=old_tp["correct_frames"]-
                                                         gate["evaluation_99_l5_tp_recall_loss_frames"],
        }
    def ranking(name):
        sets = subsets[name]
        dev = sets["medium"]
        return (all(checks[name].values()),dev["wrong"]==0,dev["complete"],dev["correct"],
                sum(sets[k]["correct_by_landmark"].get("RIGHT_L5_SAP",0) for k in ("historical","historical_816")),
                image_results[name]["fixed"]["TP"]["recall"] or 0,-dev["wrong"])
    new_arms = [name for name in specs if name not in ("original","pilot")]
    selected = max(new_arms,key=ranking)
    best_any = max(specs,key=ranking)
    write_json(args.output/"replay_summary.json",subsets)
    write_json(args.output/"selection.json",{"selection":selected,"best_including_baselines":best_any,
        "ready":all(checks[selected].values()),"checks":checks,"specs":specs,"profile_sha256":sha256(args.profile),
        "source_baselines_from_archive":{"original":archived["summary"].get("original") if "summary" in archived else None,
                                         "pilot":local["selected"]},
        "note":"The 99-image comparison is opened development evidence; no reserved confirmation was run."})
    points = []
    for item in cases:
        if item["kind"]!="medium":
            continue
        ctx = context(Path(item["source"]),protocol,evaluator,"medium")
        field_l4 = list(infer(l4[0],ctx["images"][ctx["valid"]],"cpu",batch_size=1))
        diagnostics = {"seed":item["seed"]}
        for arm,name in (("pilot","pilot"),("selected",selected)):
            models = {"L4":l4,"L5":l5[name]}
            fields = {"L4":field_l4,"L5":list(infer(l5[name][0],ctx["images"][ctx["valid"]],"cpu",batch_size=1))}
            diagnostics[arm] = point_diagnostic(ctx,protocol,models,fields)
            if diagnostics[arm]["weighted_center"]["score"] != item["arms"][name]["score"]:
                raise ValueError("fixed primary representative did not reproduce downstream mapping")
        points.append(diagnostics)
    write_json(args.output/"representatives.json",{"cases":points,
               "observation_metrics":point_metrics(points,protocol["evaluator"]["maximum_semantic_interface_distance_pixels"])})
    # This recording's on-screen flags already match the all-102-waypoint replay.
    session = Path(next(r["source"] for r in cases if r["kind"]=="medium" and r["seed"]==2026121900))
    draw_map(session,protocol,evaluator,specs[selected],args.output,"cpu")


if __name__=="__main__":
    main()
