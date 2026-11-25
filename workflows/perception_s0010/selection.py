"""Select on grouped development evidence, then read the combined holdout once."""

import argparse
from collections import Counter
from copy import deepcopy
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from fit import verified_archive
from perception_metrics import curves, eval_fixed, stability, statistics
from replay import CONVERSION, context, infer, models_for, oracle_map, replay
from run_simple_landmark_mapping import load_evaluator, read_jsonl, sha256, write_json
from two_stage_navigation.sonoanatomy_segmentation import load_segmentation_checkpoint


DEV_SPLITS = ("unseen_speckle","unseen_geometry")


def rates_from_rich(cases, hard):
    output = {}
    for semantic, values in cases.items():
        mask = hard[semantic]
        hard_positive=[v for v,m in zip(values,mask) if m["hard_positive"]]
        hard_negative=[v for v,m in zip(values,mask) if m["hard_negative"]]
        output[semantic] = {
            "correct_on_original_hard_positives": [sum(z["correct"] for z in hard_positive),len(hard_positive)],
            "accepted_on_original_hard_negatives": [sum(z["accepted"] for z in hard_negative),len(hard_negative)],
        }
    return output


def threshold_selection(candidate, old, fallback, cfg):
    need = max(cfg["precision_floor"], (old["precision"] or 0)-cfg["precision_drop_allowed"])
    count, total = old["clear_negative_false_accepts"]
    far = count/total if total else 0
    admissible = []
    for row in candidate:
        sn,sd = row["clear_negative"]
        if (row["accepted"] >= cfg["min_positive_calls"] and row["recall"] is not None
                and row["precision"] is not None and row["precision"] >= need
                and (not sd or sn/sd <= far+cfg["extra_false_accept_rate_allowed"])):
            admissible.append(row)
    if not admissible:
        return {"threshold":fallback,"status":"retained_original_no_admissible_cutoff","precision_floor":need}
    winner = max(admissible, key=lambda x:(x["recall"],x["precision"],x["threshold"]))
    return {"threshold":winner["threshold"],"status":"calibrated_on_grouped_development",
            "precision_floor":need,"development_point":winner}


def downstream_summary(cases, arm, key):
    rows = [r for r in cases if r["kind"] == key]
    counts=Counter()
    good=Counter()
    wrong=Counter()
    complete = 0
    for row in rows:
        m=row.get("arms",{}).get(arm)
        if m is None:
            counts["UNRESOLVED"] += 4
            continue
        complete+=m["complete"]
        for identity, value in m["score"]["landmarks"].items():
            outcome=value["outcome"]
            counts[outcome]+=1
            good[identity]+=int(outcome=="CORRECT")
            wrong[identity]+=int(outcome=="WRONG")
    return {"cases":len(rows),"complete":complete,"correct":counts["CORRECT"],"wrong":counts["WRONG"],
            "unresolved":counts["UNRESOLVED"],"correct_by_landmark":dict(good),"wrong_by_landmark":dict(wrong)}


def safe_reliable(summary, baseline, historical_minimum):
    keys=("RIGHT_L4_TP","RIGHT_L5_SAP")
    return all(summary["medium"]["correct_by_landmark"].get(k,0)
               >= baseline["medium"]["correct_by_landmark"].get(k,0)
               and summary["historical"]["correct_by_landmark"].get(k,0) >= historical_minimum[k]
               and not summary["historical"]["wrong_by_landmark"].get(k,0) for k in keys)


def plots(output, training, data, chosen, combined, baseline_combined):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for path in sorted(training.glob("L*/*/history.json")):
        rows=json.loads(path.read_text())
        epochs=[r["epoch"] for r in rows]
        name=path.parent.parent.name+" "+path.parent.name
        axs[0].plot(epochs,[r["train_loss"] for r in rows],label=name)
        axs[1].plot(epochs,[r["validation_score"] for r in rows],label=name)
    for ax,name in zip(axs,("Training objective","Grouped development interface F1")):
        ax.set(xlabel="Epoch",ylabel=name)
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.savefig(output/"learning_curves.png",dpi=155)
    plt.close(fig)
    rows=((lvl,semantic) for lvl in ("L4","L5") for semantic in ("TP","SAP"))
    fig,axs=plt.subplots(2,2,figsize=(9,7),layout="constrained")
    for ax,(lvl,semantic) in zip(axs.flat,rows):
        for data,arm in ((baseline_combined,"Original"),(combined,"Selected")):
            branch=data[lvl]["curve"][semantic]["curve"]
            evaluated=[r for r in branch if r["precision"] is not None and r["recall"] is not None]
            ax.plot([r["recall"] for r in evaluated],[r["precision"] for r in evaluated],"o-",ms=3,label=arm)
        ax.set(xlabel="Correct-interface recall",ylabel="Correct-interface precision",title=lvl+" "+semantic,
               xlim=(-.02,1.02),ylim=(-.02,1.02))
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(output/"combined_precision_recall.png",dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","profile","previous","historical","development","data","training","output","frozen"):
        parser.add_argument("--"+key,required=True,type=Path)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    campaign=yaml.safe_load(args.profile.read_text())
    protocol=yaml.safe_load(args.protocol.read_text())
    gate=campaign["readiness"]
    profile_hash=sha256(args.profile)
    original=protocol["models"]["sonoanatomy"]
    pilot=json.loads((args.previous/"pilot-recovery-52713119/metrics/training_result.json").read_text())
    specs={"L4":{"original":original["L4"]},
           "L5":{"original":original["L5"],"E3":{"path":pilot["checkpoint"],"sha256":pilot["checkpoint_sha256"]}}}
    for path in sorted(args.training.glob("L*/*/result.json")):
        receipt=json.loads(path.read_text())
        if receipt["status"] == "completed":
            specs[path.parent.parent.name][path.parent.name] = {
                "path":receipt["checkpoint"],"sha256":receipt["checkpoint_sha256"]}
    if len(specs["L5"])<=2:
        raise ValueError("no completed L5 campaign arm")
    models={level:{name:models_for({level:spec})[level] for name,spec in candidates.items()}
            for level,candidates in specs.items()}
    data_receipt=json.loads((args.data/"receipt.json").read_text())
    assert data_receipt["profile_sha256"]==profile_hash
    records=read_jsonl(args.data/"manifest.jsonl")
    assert all(r["source_seed"] not in campaign["reserved_confirmation_seeds"] for r in records)
    by_split={key:[r for r in records if r["split"]==key] for key in data_receipt["splits"]}
    images=[]; targets={"L4":[],"L5":[]}
    development_records=[]
    for split in DEV_SPLITS:
        with verified_archive(args.data,split,data_receipt) as data:
            images.append(data["images"])
            for level in targets:
                targets[level].append(data[level])
        development_records.extend(by_split[split])
    images=np.concatenate(images)
    targets={level:np.concatenate(t) for level,t in targets.items()}
    dev_measurements={}
    thresholds={}
    grid=campaign["calibration"]["threshold_candidates"]
    for level,candidates in models.items():
        dev_measurements[level]={}
        thresholds[level]={}
        for name,(model,rules) in candidates.items():
            probs=list(infer(model,images,args.device))
            fixed, cases=eval_fixed(probs,targets[level],development_records,level,rules)
            if name == "original":
                baseline=fixed
                hard=cases
            curve=curves(probs,targets[level],development_records,level,rules,grid)
            chosen_development={semantic:threshold_selection(curve[semantic]["curve"],baseline[semantic],
                                rules[semantic]["threshold"],campaign["calibration"])
                                for semantic in ("TP","SAP")}
            calibrated=deepcopy(rules)
            for semantic,row in chosen_development.items():
                calibrated[semantic]["threshold"]=row["threshold"]
            thresholds[level][name]=calibrated
            after, after_cases=eval_fixed(probs,targets[level],development_records,level,calibrated)
            details={}
            for split in DEV_SPLITS:
                kept=[i for i,row in enumerate(development_records) if row["split"]==split]
                subrecords=[development_records[i] for i in kept]
                before={k:[r[i] for i in kept] for k,r in cases.items()}
                following={k:[r[i] for i in kept] for k,r in after_cases.items()}
                details[split]={
                    "fixed":{k:statistics(r) for k,r in before.items()},"fixed_stability":stability(before,subrecords),
                    "calibrated":{k:statistics(r) for k,r in following.items()},
                    "calibrated_stability":stability(following,subrecords),
                }
            dev_measurements[level][name]={"fixed":fixed,"curve":curve,"threshold_selection":chosen_development,
                                            "calibrated":after,"by_split":details,
                                            "hard_example_comparison":rates_from_rich(cases,hard)}
            print("Calibrated",level,name,flush=True)
    write_json(args.output/"calibration.json",dev_measurements)
    del images,targets

    design={"original":("original","original","fixed"),"E3":("original","E3","fixed")}
    for a,b,threshold_mode in itertools.product(specs["L4"],specs["L5"],("fixed","calibrated")):
        if b not in ("original","E3"):
            design[f"L4_{a}_L5_{b}_{threshold_mode}"]=(a,b,threshold_mode)
    sources=[(args.previous/"acquisition-52710240"/f"E1_{seed}","medium") for seed in campaign["development_seeds"][:2]]
    sources += [(args.development/str(seed),"medium") for seed in campaign["development_seeds"][2:]]
    sources += [(path,"historical") for path in sorted(args.historical.glob("session_*"))]
    evaluator=load_evaluator(protocol,CONVERSION)
    cases=[]
    for path,kind in sources:
        if not (path/"actor_result.json").exists():
            cases.append({"source":str(path),"kind":kind,"status":"capture_failed"})
            continue
        ctx=context(path,protocol,evaluator,kind)
        fields={lvl:{name:list(infer(model,ctx["images"][ctx["valid"]],args.device,batch_size=1))
                     for name,(model,rules) in models[lvl].items()} for lvl in models}
        entry={"source":str(path),"seed":ctx["actor"]["seed"],"kind":kind,"arms":{},
               "projection":ctx["projection"],"valid":len(ctx["valid"])}
        for name,(l4,l5,mode) in design.items():
            active={}
            for lvl,variant in (("L4",l4),("L5",l5)):
                model,rules=models[lvl][variant]
                active[lvl]=model,(thresholds[lvl][variant] if mode=="calibrated" else rules)
            result,_=replay(ctx,protocol,active,tp_all=kind=="medium",
                            probabilities={"L4":fields["L4"][l4],"L5":fields["L5"][l5]})
            entry["arms"][name]=result
        if kind=="medium":
            entry["oracle"]=oracle_map(ctx,protocol,True)
            entry["oracle_routed"]=oracle_map(ctx,protocol,True,strict_routing=True)
            original_arm={k:models[k]["original"] for k in models}
            entry["original_50_tp"],_=replay(ctx,protocol,original_arm,tp_all=False,
                                    probabilities={lvl:f["original"] for lvl,f in fields.items()})
        cases.append(entry)
        write_json(args.output/"downstream.json",{"design":design,"cases":cases})
        print("Mapped",kind,entry["seed"],flush=True)
    summary={a:{k:downstream_summary(cases,a,k) for k in ("medium","historical")} for a in design}
    baseline=summary["original"]
    eligible=[x for x in design if x not in ("original","E3")
              and safe_reliable(summary[x],baseline,gate["historical_reliable_required_correct"])]
    pool=eligible or [x for x in design if x not in ("original","E3")]
    def rank(name):
        medium=summary[name]["medium"]
        l4,l5,mode=design[name]
        sag=dev_measurements["L5"][l5]["calibrated" if mode=="calibrated" else "fixed"]["TP"]["recall"] or 0
        return (-medium["wrong"],medium["complete"],-medium["unresolved"],sag,-int(mode=="calibrated"))
    selected=max(pool,key=rank)
    write_json(args.output/"summary.json",{"summary":summary,"ranked":sorted(pool,key=rank,reverse=True),
                                          "preserved_reliable_landmarks":eligible,"selected":selected})
    l4,l5,mode=design[selected]
    reference_specs={}
    for level,name in (("L4",l4),("L5",l5)):
        reference_specs[level]=dict(specs[level][name])
        if mode=="calibrated":
            reference_specs[level]["thresholds"]=thresholds[level][name]
    write_json(args.output/"selected_before_combined.json",{
        "selected":selected,"models":reference_specs,"profile_sha256":profile_hash,
        "combined_archive_sha256":data_receipt["splits"]["combined"]["archive_sha256"],
        "test_used_for_selection":False,
    })
    candidate_sha=sha256(args.output/"selected_before_combined.json")
    (args.output/"selected_before_combined.sha256").write_text(candidate_sha+"  selected_before_combined.json\n")
    combined_report={}
    with verified_archive(args.data,"combined",data_receipt) as data:
        test_images=data["images"]
        for arm,arm_specs in (("original",original),("E3",dict(original,L5=specs["L5"]["E3"])),
                              ("selected",reference_specs)):
            results={}
            for level,(model,rules) in models_for(arm_specs).items():
                targets=data[level]
                probs=list(infer(model,test_images,args.device))
                values, truth_cases=eval_fixed(probs,targets,by_split["combined"],level,rules)
                result={"fixed_evaluation":values,"curve":curves(probs,targets,by_split["combined"],level,rules,grid),
                        "stability":stability(truth_cases,by_split["combined"])}
                results[level]=result
            combined_report[arm]=results
    for level in combined_report["original"]:
        for name in ("TP","SAP"):
            original_point=combined_report["original"][level]["fixed_evaluation"][name]
            fixed_precision=original_point["precision"]
            fixed_recall=original_point["recall"]
            for arm in ("E3","selected"):
                item=combined_report[arm][level]["curve"][name]
                curve=item["curve"]
                matched=[point for point in curve if point["precision"] is not None
                         and fixed_precision is not None and point["precision"]>=fixed_precision]
                item["descriptive_best_recall_at_original_precision"]=max(
                    (point["recall"] for point in matched if point["recall"] is not None), default=None)
                matched=[point for point in curve if point["recall"] is not None
                         and fixed_recall is not None and point["recall"]>=fixed_recall]
                item["descriptive_best_precision_at_original_recall"]=max(
                    (point["precision"] for point in matched if point["precision"] is not None),default=None)
    write_json(args.output/"combined_validation.json",combined_report)
    u=combined_report["original"]
    v=combined_report["selected"]
    oldtp=u["L5"]["fixed_evaluation"]["TP"]
    newtp=v["L5"]["fixed_evaluation"]["TP"]
    comparison=summary[selected]["medium"]
    checks={
        "four_development_sessions":comparison["cases"]==gate["required_development_sessions"],
        "complete_development_maps":comparison["complete"]>=gate["minimum_development_complete_maps"],
        "wrong_development_landmarks":comparison["wrong"]<=gate["maximum_development_wrong_landmarks"],
        "reliable_landmarks":selected in eligible,
        "combined_l5_tp_positives":newtp["positive"]>=gate["combined_test_min_positives"],
        "combined_l5_tp_absolute_recall_gain":oldtp["recall"] is not None and newtp["recall"] is not None
             and newtp["recall"]-oldtp["recall"]>=gate["combined_test_l5_tp_absolute_recall_gain"],
        "combined_l5_tp_precision":newtp["precision"] is not None and oldtp["precision"] is not None
             and newtp["precision"]>=oldtp["precision"]-gate["combined_test_max_precision_loss"],
    }
    for lvl, semantic in (("L4","TP"),("L5","SAP")):
        a=u[lvl]["fixed_evaluation"][semantic]["recall"]
        b=v[lvl]["fixed_evaluation"][semantic]["recall"]
        checks[f"combined_{lvl}_{semantic}_recall_preservation"]=a is not None and b is not None and (
            b>=a-gate["max_recall_loss_l4_tp_or_l5_sap"])
    readiness="PASS" if all(checks.values()) else "FAIL"
    write_json(args.output/"readiness.json",{
        "readiness":readiness,"checks":checks,"selected":selected,
        "selected_pre_combined_sha256":candidate_sha,
        "medium":comparison,"historical":summary[selected]["historical"],
        "reserved_confirmation_seeds":campaign["reserved_confirmation_seeds"],
    })
    if readiness=="PASS":
        args.frozen.mkdir(exist_ok=False,parents=True)
        frozen_protocol=deepcopy(protocol)
        immutable={}
        for lvl,spec in reference_specs.items():
            model,bundle=load_segmentation_checkpoint(spec["path"],expected_level=lvl)
            if sha256(Path(spec["path"]))!=spec["sha256"]:
                raise ValueError("checkpoint seal mismatch at freeze")
            target=args.frozen/(lvl+".pt")
            torch.save(dict(bundle,thresholds=spec.get("thresholds",bundle["thresholds"])),target)
            immutable[lvl]={"path":str(target.resolve()),"sha256":sha256(target)}
        frozen_protocol["models"]["sonoanatomy"]=immutable
        out=args.frozen/"protocol.yaml"
        out.write_text(yaml.safe_dump(frozen_protocol,sort_keys=False))
        (args.frozen/"protocol.sha256").write_text(sha256(out)+"  protocol.yaml\n")
        ready=args.frozen/"ready.yaml"
        ready.write_text(yaml.safe_dump({"readiness":readiness,"selected":selected,"checks":checks,
            "independent_seeds":campaign["reserved_confirmation_seeds"],"profile_sha256":profile_hash,
            "checkpoint_specs":immutable,"design":campaign["medium_waypoints_sha256"],
            "selection_seal":candidate_sha}))
        (args.frozen/"ready.sha256").write_text("".join(f"{sha256(f)}  {f.name}\n"
              for f in (ready,out,args.frozen/"L4.pt",args.frozen/"L5.pt")))
    plots(args.output,args.training,args.data,selected,v,u)
    print("Readiness",readiness,"Selected",selected,flush=True)


if __name__=="__main__":
    main()
