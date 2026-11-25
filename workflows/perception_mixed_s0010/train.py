"""Train three matched data-composition arms with immutable perception and mapping rules."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy.spatial.distance import cdist
import torch
import yaml

from composition import array_hash, assemble
from fit import save_checkpoint, tversky_interface_loss, verified_archive
from perception_metrics import eval_fixed
from replay import infer, models_for
from run_simple_landmark_mapping import read_jsonl, sha256, write_json
from two_stage_navigation.sonoanatomy_segmentation import (
    load_segmentation_checkpoint, negative_presence_loss, normalized_ultrasound,
)


def read_inputs(args, cfg):
    receipt = json.loads((args.data/"receipt.json").read_text())
    records = read_jsonl(args.data/"manifest.jsonl")
    if any(r["source_seed"] in cfg["reserved_seeds"] for r in records):
        raise ValueError("reserved-seed samples in input archive")
    grouped = {split:[r for r in records if r["split"]==split] for split in receipt["splits"]}
    if len(grouped["combined"]) != 99:
        raise ValueError("opened development set must remain 99 images")
    source_groups = {split:{r["pose_group"] for r in rows} for split,rows in grouped.items()}
    if any(source_groups["train"] & source_groups[s] for s in ("unseen_geometry","combined")):
        raise ValueError("training source overlap with unseen geometry")
    radius = cfg["inputs"]["group_buffer_mm"]/1000
    for split in ("unseen_geometry","combined"):
        distance = cdist([r["position_m"] for r in grouped["train"]], [r["position_m"] for r in grouped[split]])
        if distance.min() + 1e-8 < radius:
            raise ValueError(f"new data pool violates the {split} geometry buffer")
    fields = {split:{r["field"] for r in rows} for split,rows in grouped.items()}
    if fields["train"] & (fields["unseen_speckle"] | fields["combined"]):
        raise ValueError("training speckle overlap with speckle holdout")
    arrays = {}
    for split, rows in grouped.items():
        with verified_archive(args.data,split,receipt) as data:
            images, labels = data["images"], data["L5"]
        if len(images) != len(rows) or [r["archive_index"] for r in rows] != list(range(len(rows))):
            raise ValueError(f"archive index mismatch in {split}")
        if any(array_hash(i) != r["image_sha256"] for i,r in zip(images,rows)):
            raise ValueError(f"image checksum mismatch in {split}")
        if any(hashlib.sha256(np.ascontiguousarray(mask)).hexdigest() != row["labels"]["L5"]["label_sha256"]
               for mask,row in zip(labels,rows)):
            raise ValueError(f"label checksum mismatch in {split}")
        arrays[split] = (images, labels)
    return grouped, arrays


def train_arm(out, images, labels, validation, records, model_spec, rules, settings, device, weights, auxiliary=None):
    seed = settings["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, bundle = load_segmentation_checkpoint(model_spec["path"], expected_level="L5")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    counts = np.bincount(labels[labels != 255], minlength=4).astype(float)
    if not all(counts):
        raise ValueError("training manifest misses a semantic class")
    provenance = {"level":"L5","arm":out.name,"settings":settings,"initial_sha256":model_spec["sha256"],
                  "training_split_sha256":sha256(out/"training_manifest.json"),
                  "validation_split_sha256":sha256(out.parent/"validation_manifest.json"),
                  "num_training":len(labels),"class_pixel_counts":counts.astype(int).tolist(),
                  "common_class_weights":weights.tolist()}
    if auxiliary is not None:
        provenance["auxiliary_loss"] = auxiliary.settings
    history = []
    best = -np.inf
    stale = 0
    for epoch in range(1, settings["epochs"]+1):
        started = time.monotonic()
        permutation = rng.permutation(len(labels))
        groups = np.array_split(permutation, int(np.ceil(len(labels)/settings["batch_size"])))
        model.train()
        train_loss = []
        for batch in groups:
            x = torch.from_numpy(images[batch]).to(device)
            y = torch.from_numpy(labels[batch]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(normalized_ultrasound(x)[:,None])
            loss = tversky_interface_loss(pred,y,weights,settings["foreground_tversky_false_positive_weight"])
            loss += settings["negative_presence_weight"]*negative_presence_loss(pred,y)
            if auxiliary is not None:
                loss += auxiliary(pred,batch)
            loss.backward()
            optimizer.step()
            train_loss.append(float(loss.detach()))
        model.eval()
        metrics, valid_loss, terms = {}, [], []
        for split,(valid_images,valid_labels) in validation.items():
            probs = list(infer(model,valid_images,device,settings["batch_size"]))
            with torch.no_grad():
                for offset in range(0,len(probs),settings["batch_size"]):
                    end = offset+settings["batch_size"]
                    p = torch.from_numpy(np.asarray(probs[offset:end])).to(device)
                    z = p.clamp_min(1e-7).log()
                    y = torch.from_numpy(valid_labels[offset:end]).long().to(device)
                    value = tversky_interface_loss(z,y,weights,settings["foreground_tversky_false_positive_weight"])
                    value += settings["negative_presence_weight"] * negative_presence_loss(z,y)
                    valid_loss.append(float(value))
            metrics[split], _ = eval_fixed(probs,valid_labels,records[split],"L5",rules)
            terms.extend(2*v["correct_frames"]/(v["accepted_frames"]+v["positive"])
                         for v in metrics[split].values() if v["positive"])
        score = float(np.mean(terms))
        epoch_provenance = dict(provenance,epoch=epoch,validation_score=score)
        if score > best + 1e-9:
            best, stale = score, 0
            save_checkpoint(out/"best.pt",model,bundle,epoch_provenance)
        else:
            stale += 1
        save_checkpoint(out/"last.pt",model,bundle,epoch_provenance)
        history.append({"epoch":epoch,"train_loss":float(np.mean(train_loss)),
                        "validation_loss":float(np.mean(valid_loss)),"validation_score":score,
                        "learning_rate":settings["learning_rate"],"optimizer_steps":len(groups),
                        "seconds":time.monotonic()-started,"splits":metrics})
        write_json(out/"history.json",history)
        print(out.name, epoch, f"train={history[-1]['train_loss']:.4f} grouped-interface-F1={score:.4f}", flush=True)
        if epoch >= settings["minimum_epochs"] and stale >= settings["patience"]:
            break
    result = {"status":"completed","epochs":len(history),"checkpoint":str(out/"best.pt"),
              "checkpoint_sha256":sha256(out/"best.pt"),"best_score":best,
              "best_epoch":int(np.argmax([r["validation_score"] for r in history]))+1,
              "cpu_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
              "provenance":provenance}
    write_json(out/"result.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","profile","previous","data","legacy-dataset","legacy-training","pilot-manifest",
                "diagnosis","output"):
        parser.add_argument("--"+key,required=True,type=Path)
    parser.add_argument("--source-hashes",nargs="+",required=True,type=Path)
    parser.add_argument("--device",default="cuda:0")
    args = parser.parse_args()
    profile = yaml.safe_load(args.profile.read_text())
    if sha256(args.protocol) != profile["source_protocol_sha256"]:
        raise ValueError("source protocol differs from the frozen baseline")
    if not json.loads(args.diagnosis.read_text()) or not all(p.is_file() for p in args.source_hashes):
        raise ValueError("finish failure diagnosis and audit saved acquisitions before training")
    original = yaml.safe_load(args.protocol.read_text())["models"]["sonoanatomy"]["L5"]
    previous = json.loads((args.previous/"evaluation-52723344/metrics/selected_before_combined.json").read_text())
    pilot = previous["models"]["L5"]
    if pilot["sha256"] != profile["previous_pilot_l5_sha256"]:
        raise ValueError("pilot model differs from the frozen baseline")
    model_specs = {"original":original,"pilot":pilot}
    specs = {name:models_for({"L5":spec})["L5"] for name,spec in model_specs.items()}
    args.output.mkdir(parents=True,exist_ok=False)
    grouped, arrays = read_inputs(args, profile)
    datasets, selected = assemble(args,profile,grouped,arrays,specs,specs["original"][1],args.output)
    validation = {split:(arrays[split][0][ids], arrays[split][1][ids]) for split,ids in selected.items()}
    records = {split:[grouped[split][i] for i in ids] for split,ids in selected.items()}
    rules = specs["original"][1]
    reference_labels = datasets["A_matched"][1]
    counts = np.bincount(reference_labels[reference_labels != 255],minlength=4)
    if not all(counts):
        raise ValueError("Dataset A cannot define a common objective for every class")
    values = np.sqrt(np.median(counts[1:])/counts).clip(.25,8.)
    values[0] = .05
    weights = torch.from_numpy(values.astype(np.float32)).to(args.device)
    del specs, arrays
    summary = {}
    for arm in ("T2","T3","A_matched"):
        images,labels = datasets[arm]
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        summary[arm] = train_arm(args.output/arm,images,labels,validation,records,original,
                                 rules,profile["training"],args.device,weights)
        write_json(args.output/"training_summary.json",summary)


if __name__ == "__main__":
    main()
