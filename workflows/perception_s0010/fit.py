"""Train the original level-specific U-Net under controlled sampling and loss arms."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
import yaml

from perception_metrics import eval_fixed
from replay import infer, models_for
from run_simple_landmark_mapping import read_jsonl, sha256, write_json
from two_stage_navigation.sonoanatomy_segmentation import (
    interface_loss, negative_presence_loss, normalized_ultrasound,
)


def verified_archive(data, split, receipt):
    path = data / (split + ".npz")
    if sha256(path) != receipt["splits"][split]["archive_sha256"]:
        raise ValueError(f"archive checksum mismatch {split}")
    return np.load(path)


def original_examples(legacy_dataset, legacy_training, level, forbidden, radius, seed):
    with (legacy_training/f"{level.lower()}_split_manifest.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "train" and r["included"] == "True"
                and r["label_provenance"] == "SIMULATOR_WEAK_ANATOMICAL_LABEL"]
    xyz = [json.loads(r["split_pose_position_m"]) for r in rows]
    index = cKDTree(forbidden)
    distant = index.query(xyz)[0] >= radius
    kept = defaultdict(list)
    for row, keep in zip(rows,distant):
        if keep:
            kept[(row["has_tp_interface"],row["has_sap_interface"])].append(row)
    rng = np.random.default_rng(seed)
    chosen = [group[i] for group in kept.values() for i in rng.permutation(len(group))[:72]]
    ids = [int(r["archive_index"]) for r in chosen]
    with np.load(legacy_dataset) as inputs:
        return inputs["ultrasound"][ids].astype(np.float32), inputs["labels"][ids].astype(np.uint8), chosen


def tversky_interface_loss(logits, target, weights, false_positive_cost):
    valid = target != 255
    truth = torch.nn.functional.one_hot(target.masked_fill(~valid,0), num_classes=4).permute(0,3,1,2)
    truth = truth[:,1:].to(logits.dtype)
    probs = logits.softmax(1)[:,1:]
    match = (truth * probs * valid[:,None]).sum(dim=(0,2,3))
    fp = ((1-truth) * probs * valid[:,None]).sum(dim=(0,2,3))
    fn = (truth * (1-probs) * valid[:,None]).sum(dim=(0,2,3))
    overlap = (match+1.0)/(match + false_positive_cost*fp + (1-false_positive_cost)*fn + 1.0)
    return torch.nn.functional.cross_entropy(logits,target,weight=weights,ignore_index=255) + (1-overlap).mean()


def save_checkpoint(path, model, metadata, provenance):
    file = path.with_suffix(".pt.partial")
    torch.save(dict(metadata, state_dict={k:z.detach().cpu() for k,z in model.state_dict().items()},
                    campaign_training=provenance), file)
    file.replace(path)


def make_mining_examples(images, labels, cases, level, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2,4,figsize=(12,6),layout="constrained")
    for i, semantic in enumerate(("TP","SAP")):
        code = i+1
        row = cases[semantic]
        predicates = (lambda r:r["hard_positive"],lambda r:r["hard_negative"],lambda r:r["correct"],
                      lambda r:r["clear_negative"] and not r["accepted"])
        for axis, pred, title in zip(axes[i],predicates,("Hard positive","Hard negative","Correct","Clear negative")):
            axis.axis("off")
            indices = [k for k,r in enumerate(row) if pred(r)]
            if not indices:
                continue
            index = indices[len(indices)//2]
            image,mask = images[index],labels[index]
            axis.imshow(image,cmap="gray",vmin=0,vmax=np.quantile(image,.99))
            for category,color in ((code,"cyan"),(3,"#92ec77")):
                if (mask==category).any():
                    axis.contour(mask==category,levels=[.5],colors=[color],linewidths=.75)
            for component in row[index]["components"]:
                r,c = component["pixel"]
                axis.plot(c,r,"x",color="orange",ms=6)
            axis.set_title(f"{level} {semantic} | {title}\ntraining frame {index}",fontsize=9)
    figure.suptitle("Training-only examples | cyan target · green other bone · orange predicted representative")
    figure.savefig(output,dpi=145)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","profile","data","legacy-dataset","legacy-training","diagnosis","output"):
        parser.add_argument("--"+key,required=True,type=Path)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    cfg=yaml.safe_load(args.profile.read_text())
    settings=cfg["training"]
    protocol=yaml.safe_load(args.protocol.read_text())
    spec=protocol["models"]["sonoanatomy"]
    diagnosis=json.loads(args.diagnosis.read_text())
    diagnostic_rows=[r for case in diagnosis["cases"] if not case.get("failed")
                     for r in case["policies"]["all_tp"]["original"]["diagnostics"]]
    missed=Counter()
    visible=Counter()
    for row in diagnostic_rows:
        if row["truth_pixels"]:
            identity=row["identity"]
            visible[identity]+=1
            if row["reason"] not in ("correct_interface","level_routing_or_ambiguous"):
                missed[identity]+=1
    levels={"L5":["T1","T2"]}
    if visible["RIGHT_L5_TP"] >= 12 and missed["RIGHT_L5_TP"]/visible["RIGHT_L5_TP"] >= .25:
        levels["L5"].append("T3")
    if visible["RIGHT_L4_SAP"] >= 12 and missed["RIGHT_L4_SAP"]/visible["RIGHT_L4_SAP"] >= .25:
        levels["L4"]=["T1","T2"]
    write_json(args.output/"training_decision.json",{"arms":levels,"diagnostic_visible":dict(visible),
                                                   "diagnostic_perception_missed":dict(missed)})
    receipt=json.loads((args.data/"receipt.json").read_text())
    if receipt["profile_sha256"] != sha256(args.profile):
        raise ValueError("training profile does not match data")
    records=read_jsonl(args.data/"manifest.jsonl")
    grouped={name:[r for r in records if r["split"]==name] for name in receipt["splits"]}
    assert all(r["source_seed"] not in cfg["reserved_confirmation_seeds"] for r in records)
    dev_splits=("unseen_speckle","unseen_geometry")
    forbidden=[r["position_m"] for n in ("combined","unseen_geometry") for r in grouped[n]]
    train_records=grouped["train"]
    with verified_archive(args.data,"train",receipt) as source:
        train_new = source["images"]
        new_labels = {level:source[level] for level in levels}
    eval_images, eval_labels = {}, {}
    for split in dev_splits:
        with verified_archive(args.data,split,receipt) as source:
            eval_images[split]=source["images"]
            eval_labels[split]={level:source[level] for level in levels}
    for level, arm_names in levels.items():
        out=args.output/level
        out.mkdir()
        rules=models_for({level:spec[level]})[level][1]
        model,_=models_for({level:spec[level]})[level]
        probs=list(infer(model,train_new,args.device,settings["batch_size"]))
        training_metrics,cases=eval_fixed(probs,new_labels[level],train_records,level,rules)
        del probs
        mine=[]
        for i,record in enumerate(train_records):
            hard_positives=[n for n in cases if cases[n][i]["hard_positive"]]
            hard_negatives=[n for n in cases if cases[n][i]["hard_negative"]]
            mine.append({"index":i,"pose_group":record["pose_group"],"positive":hard_positives,"negative":hard_negatives})
        write_json(out/"mining.json",{"summary":training_metrics,"candidates":mine})
        make_mining_examples(train_new,new_labels[level],cases,level,out/"mining.png")
        images_old,labels_old,source_old=original_examples(
            args.legacy_dataset,args.legacy_training,level,forbidden,
            cfg["data"]["legacy_and_split_buffer_mm"]/1000,settings["seed"],
        )
        write_json(out/"legacy_selection.json",source_old)
        images=np.concatenate((train_new,images_old))
        labels=np.concatenate((new_labels[level],labels_old))
        natural = np.arange(len(labels))
        positive=[i for i,r in enumerate(mine) if r["positive"]]
        negative=[i for i,r in enumerate(mine) if r["negative"]]
        pool=[natural,np.asarray(positive),np.asarray(negative)]
        group_weights=[]
        for items in pool:
            group=[mine[i]["pose_group"] if i < len(mine) else source_old[i-len(mine)]["trajectory_id"] for i in items]
            counter=Counter(group)
            weights=np.asarray([1/counter[g] for g in group],dtype=float)
            group_weights.append(weights/weights.sum() if len(weights) else weights)
        counts=np.bincount(labels[labels!=255],minlength=4).astype(float)
        if not np.all(counts[1:]):
            raise ValueError(f"{level} lacks foreground labels")
        weights=np.sqrt(np.median(counts[1:])/counts).clip(.25,8.)
        weights[0]=.05
        weights=torch.tensor(weights,dtype=torch.float32,device=args.device)
        for arm in arm_names:
            target=out/arm
            target.mkdir()
            if arm != "T1" and not (positive and negative):
                write_json(target/"result.json",{"status":"insufficient_hard_examples","hard_positive":len(positive),
                                                "hard_negative":len(negative)})
                continue
            seed=settings["seed"]+int(level[-1])
            torch.manual_seed(seed)
            rng=np.random.default_rng(seed)
            # Preserve the original checkpoint's inference metadata.
            from two_stage_navigation.sonoanatomy_segmentation import load_segmentation_checkpoint
            model,bundle=load_segmentation_checkpoint(spec[level]["path"],expected_level=level)
            model.to(args.device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=settings["learning_rate"],weight_decay=settings["weight_decay"])
            provenance={"level":level,"arm":arm,"initial_sha256":spec[level]["sha256"],
                        "data_manifest_sha256":sha256(args.data/"manifest.jsonl"),
                        "settings":settings,"effective_seed":seed,"num_training":len(labels)}
            history=[]
            best=-np.inf
            stale=0
            for epoch in range(1,settings["epochs"]+1):
                start=time.perf_counter()
                if arm=="T1":
                    epoch_indices=rng.permutation(natural)
                else:
                    fractions=settings["sample_fraction_natural_hard_positive_hard_negative"]
                    sizes=[int(len(labels)*f) for f in fractions]
                    sizes[0]+=len(labels)-sum(sizes)
                    epoch_indices=rng.permutation(np.concatenate([
                        rng.choice(indices,size,replace=True,p=w) for indices,size,w in zip(pool,sizes,group_weights)
                    ]))
                model.train()
                losses=[]
                for group in np.array_split(epoch_indices,int(np.ceil(len(epoch_indices)/settings["batch_size"]))):
                    x=torch.from_numpy(images[group]).to(args.device)
                    y=torch.from_numpy(labels[group]).long().to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits=model(normalized_ultrasound(x)[:,None])
                    loss=(tversky_interface_loss(logits,y,weights,settings["foreground_tversky_false_positive_weight"])
                          if arm=="T3" else interface_loss(logits,y,weights))
                    loss+=settings["negative_presence_weight"]*negative_presence_loss(logits,y)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                by_split={}
                validation_losses=[]
                for split in dev_splits:
                    probs=list(infer(model,eval_images[split],args.device,settings["batch_size"]))
                    with torch.no_grad():
                        for group in np.array_split(np.arange(len(probs)),int(np.ceil(len(probs)/settings["batch_size"]))):
                            p=torch.as_tensor(np.stack([probs[k] for k in group]),device=args.device)
                            t=torch.as_tensor(eval_labels[split][level][group],dtype=torch.long,device=args.device)
                            log_probability=p.clamp(min=1e-7).log()
                            validation_losses.append(float(interface_loss(log_probability,t,weights) +
                                                           settings["negative_presence_weight"]*negative_presence_loss(log_probability,t)))
                    metrics,_=eval_fixed(probs,eval_labels[split][level],grouped[split],level,rules)
                    by_split[split]=metrics
                f_values=[]
                for split,metrics in by_split.items():
                    for measured in metrics.values():
                        if measured["positive"]:
                            f_values.append(2*measured["correct_frames"]/(measured["accepted_frames"]+measured["positive"]))
                score=float(np.mean(f_values))
                if score>best+1e-9:
                    best,stale=score,0
                    save_checkpoint(target/"best.pt",model,bundle,dict(provenance,epoch=epoch,validation_score=score))
                else:
                    stale+=1
                history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),
                                "validation_original_objective":float(np.mean(validation_losses)),
                                "validation_score":score,"splits":by_split,
                                "learning_rate":settings["learning_rate"],"seconds":time.perf_counter()-start})
                save_checkpoint(target/"last.pt",model,bundle,dict(provenance,epoch=epoch,validation_score=score))
                write_json(target/"history.json",history)
                print(level,arm,epoch,f"loss={history[-1]['train_loss']:.4f} grouped_interface_f1={score:.4f}",flush=True)
                if epoch>=settings["minimum_epochs"] and stale>=settings["patience"]:
                    break
            write_json(target/"result.json",{"status":"completed","selection":"mean grouped development interface F1",
                                            "epochs":len(history),"best_score":best,"checkpoint":str(target/"best.pt"),
                                            "checkpoint_sha256":sha256(target/"best.pt"),"legacy_examples":len(labels_old),
                                            "training_examples":len(labels),"cpu_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                            "gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated() if "cuda" in args.device else None,
                                            "provenance":provenance})


if __name__=="__main__":
    main()
