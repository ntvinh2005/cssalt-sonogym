"""Train a separate L5 arm with a visible-TP boundary penalty."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import yaml

from composition import array_hash
from replay import models_for
from run_simple_landmark_mapping import sha256, write_json
from train import read_inputs, train_arm


class TPBoundaryLoss:
    """Concentrate high TP probability near a visible TP without changing class labels."""

    def __init__(self, labels, settings, device):
        self.settings = settings
        self.device = device
        self.present = (labels==1).any(axis=(1,2))
        self.distance = np.zeros(labels.shape,dtype=np.float32)
        limit = settings["maximum_interface_distance_px"]
        for i in np.flatnonzero(self.present):
            self.distance[i] = np.minimum(distance_transform_edt(labels[i]!=1),limit)

    def __call__(self, logits, batch):
        visible = self.present[batch]
        if not visible.any():
            return logits.sum()*0
        margin = self.settings["zero_cost_interface_distance_px"]
        limit = self.settings["maximum_interface_distance_px"]
        distance = torch.from_numpy(self.distance[batch[visible]]).to(self.device,torch.float32)
        cost = (distance-margin).clamp_min(0)/(limit-margin)
        probability = logits[torch.from_numpy(visible).to(self.device)].softmax(1)[:,1]
        mass = probability.pow(self.settings["foreground_probability_power"])
        mean_cost = ((mass*cost).sum((1,2))/mass.sum((1,2)).clamp_min(1e-6)).mean()
        return mean_cost*self.settings["auxiliary_weight"]


def restore_manifest(rows, arrays, legacy_file):
    images, labels = arrays["train"]
    with np.load(legacy_file) as archive:
        legacy_images = archive["ultrasound"]
        legacy_labels = archive["labels"]
    x, y = [], []
    for row in rows:
        index = int(row["archive_index"])
        source_images, source_labels = (legacy_images,legacy_labels) if row["category"]=="legacy" else (images,labels)
        x.append(source_images[index])
        y.append(source_labels[index])
    return np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("profile","boundary-profile","protocol","selection","fit","data","legacy-archive","output"):
        parser.add_argument("--"+option,required=True,type=Path)
    parser.add_argument("--device",default="cuda:0")
    args = parser.parse_args()
    profile = yaml.safe_load(args.profile.read_text())
    extra = yaml.safe_load(args.boundary_profile.read_text())
    if extra["parent_profile_sha256"] != sha256(args.profile) or sha256(args.protocol) != profile["source_protocol_sha256"]:
        raise ValueError("boundary experiment requires the frozen parent profile")
    selection = json.loads(args.selection.read_text())
    arm = selection["selection"]
    if arm not in ("A_matched","T2","T3") or selection["profile_sha256"] != sha256(args.profile):
        raise ValueError("boundary experiment data must match the prior completed selection")
    args.output.mkdir(exist_ok=False,parents=True)
    out = args.output/"T4"
    out.mkdir()
    parent_result = json.loads((args.fit/arm/"result.json").read_text())
    manifests = ((args.fit/arm/"training_manifest.json",out/"training_manifest.json","training_split_sha256"),
                 (args.fit/"validation_manifest.json",args.output/"validation_manifest.json","validation_split_sha256"))
    for source,target,key in manifests:
        if sha256(source) != parent_result["provenance"][key]:
            raise ValueError("boundary experiment must reuse exact control manifests")
        shutil.copyfile(source,target)
    grouped, arrays = read_inputs(args,profile)
    manifest = json.loads((out/"training_manifest.json").read_text())
    if (len(manifest) != extra["maximum_unique_training_images"]
        or sha256(args.legacy_archive) != profile["inputs"]["legacy_archive_sha256"]):
        raise ValueError("invalid training size or legacy archive")
    images,labels = restore_manifest(manifest,arrays,args.legacy_archive)
    if any(array_hash(image) != row["training_image_sha256"] for image,row in zip(images,manifest)):
        raise ValueError("restored images differ from matched arm")
    records = json.loads((args.output/"validation_manifest.json").read_text())["selection"]
    validation = {split:tuple(data[[row["archive_index"] for row in rows]] for data in arrays[split])
                  for split,rows in records.items()}
    del grouped, arrays
    original = yaml.safe_load(args.protocol.read_text())["models"]["sonoanatomy"]["L5"]
    rules = models_for({"L5":original})["L5"][1]
    weights = torch.tensor(parent_result["provenance"]["common_class_weights"],device=args.device)
    auxiliary = TPBoundaryLoss(labels,extra,args.device)
    write_json(args.output/"matched_boundary_protocol.json",{"compared_arm":arm,"control_checkpoint":parent_result,
        "selection_sha256":sha256(args.selection),"boundary_profile_sha256":sha256(args.boundary_profile),
        "visible_tp_training_images":int(auxiliary.present.sum())})
    train_arm(out,images,labels,validation,records,original,rules,profile["training"],args.device,weights,
              auxiliary=auxiliary)


if __name__ == "__main__":
    main()
