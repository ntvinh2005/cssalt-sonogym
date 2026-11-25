"""Write the Arm B report from the completed reconstruction outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

METHODS = ("B0", "B1", "B2", "B3")
IDENTITIES = ("RIGHT_L4_TP", "RIGHT_L4_SAP", "RIGHT_L5_TP", "RIGHT_L5_SAP")


def table(head, rows):
    return "\n".join(["| "+" | ".join(head)+" |","| "+" | ".join(["---"]*len(head))+" |"]+
                      ["| "+" | ".join(map(str,row))+" |" for row in rows])


def percent(stats):
    n,d=stats["covered"],stats["total"]
    return f"{n:,} / {d:,} ({100*n/d:.1f}%)" if d else "0 / 0 (N/A)"


def pooled_errors(folder, cases, radius):
    values={m:[] for m in METHODS}
    for case in cases:
        with np.load(folder/f'plot_{case["seed"]}.npz') as f:
            center=cKDTree(f["midpoint"])
            for m in METHODS:
                xyz, ident=f[m+"_xyz"],f[m+"_code"]
                local=center.query(xyz)[0]<=radius
                for i,k in enumerate(IDENTITIES):
                    out=xyz[local & (ident==i)]
                    ref=f["surface_"+k]
                    ref=ref[center.query(ref)[0]<=radius]
                    values[m].extend(cKDTree(ref).query(out)[0].tolist() if len(ref) else [float("inf")]*len(out))
    report={}
    for m,v in values.items():
        finite=np.asarray(v)[np.isfinite(v)]
        report[m]={"median_mm":float(np.median(finite)) if len(finite) else None,
                   "p95_mm":float(np.quantile(finite,.95)) if len(finite) else None,
                   "finite":len(finite),"total":len(v),"outliers_or_missing":sum(x>1.5 for x in v)}
    return report


def mm(value):
    return f"{value:.2f}" if value is not None else "N/A"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("manifest","results","slurm","commit","runtime"):
        p.add_argument("--"+k,required=True)
    p.add_argument("--presentation-commit")
    p.add_argument("--presentation-slurm")
    opt=p.parse_args()
    folder=Path(opt.results)
    x=json.loads((folder/"summary.json").read_text())
    manifest=json.loads(Path(opt.manifest).read_text())
    pool=x["pooled"]
    data=[]
    for m in METHODS:
        s=pool[m]
        data.append([m,percent(s["global"]),
                     *[percent(s["local"][r]["all"]["coverage_eligible_visible"]) for r in ("5","10","20")]])
    coverage=table(["Method","Global","5 mm","10 mm","20 mm"],data)
    data=[]
    for r in ("5","10","20"):
        s=pool["B0"]["local"][r]
        data.append([r,*[percent(s[k]["full_ct_boundary_observed_all_visible"])
                         for k in ("RIGHT_L5_TP","RIGHT_L5_SAP","all")],
                     percent(s["all"]["full_ct_boundary_observed_eligible"])])
    full=table(["Radius (mm)","L5 TP all-visible","L5 SAP all-visible","All all-visible","All eligible"],data)
    data=[]
    all_errors={}
    for r in (5,10,20):
        all_errors[str(r)]=pooled_errors(folder,x["sessions"],r)
        for m,v in all_errors[str(r)].items():
            by_seed=[s["methods"][m]["local"][str(r)]["same_frame_semantics"] for s in x["sessions"]]
            n=sum(s["comparisons"] for s in by_seed)
            matched=sum(s["correct_within_gate"] for s in by_seed)
            wrong=sum(s["wrong_nearest_identity"] for s in by_seed)
            data.append([r,m,mm(v["median_mm"])+" / "+mm(v["p95_mm"]),
                         f'{v["outliers_or_missing"]}/{v["total"]}',f"{matched}/{n}",f"{wrong}/{n}"])
    errors=table(["Radius (mm)","Method","CT error median / P95 (mm)","CT outliers or unsupported",
                  "Same-frame correct","Same-frame wrong identity"],data)
    a=x["hidden_annotation_sensitivity"]
    endpoints=table(["Seed","Frames","Unique SAP core voxels hit","Unique TP core voxels hit",
                     "SAP hit frames","TP hit frames"],
                    [[s["seed"],s["frames"],
                      *[f'{s["strict_endpoint_coverage"][k]["first_visible_voxels"]}/{s["strict_endpoint_coverage"][k]["core_ct_voxels"]}'
                        for k in ("SAP","TP")],
                      *[s["strict_endpoint_coverage"][k]["first_visible_frames"] for k in ("SAP","TP")]]
                     for s in x["sessions"]])
    data=[]
    for s in x["sessions"]:
        for m in METHODS:
            values={k:v["observed_feature"]["closest_visible_pair_mm"]
                    for k,v in s["methods"][m]["local"]["20"]["leave_out"].items()}
            one=[v for k,v in values.items() if k.startswith("leave_frame") and v is not None]
            data.append([s["seed"],m,mm(values["baseline"]),
                         f"{mm(min(one))}–{mm(max(one))}" if one else "N/A",
                         sum(k.startswith("leave_frame") and v is None for k,v in values.items()),
                         mm(values.get("leave_family_longitudinal")),mm(values.get("leave_family_transverse"))])
    views=table(["Seed","Method","All views (mm)","Leave one (mm)","Leave-one pair absent",
                 "Omit longitudinal (mm)","Omit transverse (mm)"],data)
    variants=table(["Hidden SAP annotation","Changed voxels","Minimum surface separation (mm)",
                    "Midpoint-set Hausdorff (mm)","Centroid shift (mm)"],
                   [[v["variant"].replace("_"," "),v["changed_sap_annotation_voxels"],
                     mm(v["minimum_separation_mm"]),mm(v["comparison_to_reference"]["hausdorff_mm"]),
                     mm(v["comparison_to_reference"]["centroid_shift_mm"])] for v in a["variants"]])
    compared=max(a["pairs"],key=lambda q:q["hausdorff_mm"]) if a["pairs"] else None
    extreme="No candidate passed the visibility and topology constraints."
    if compared:
        with np.load(folder/"annotation_sensitivity_geometry.npz") as f:
            separation=cKDTree(f[compared["first"]]).query(f[compared["second"]])[0].min()
        extreme=(f'Maximum pairwise midpoint-set Hausdorff distance was {compared["hausdorff_mm"]:.2f} mm '
                 f'for {compared["first"].replace("_"," ")} versus {compared["second"].replace("_"," ")}. '
                 f'The nearest two samples were {separation:.2f} mm apart. '
                 f'{compared["first_within_3_mm"]}/{compared["first_total"]} midpoint samples in the first set '
                 f'and {compared["second_within_3_mm"]}/{compared["second_total"]} in the second '
                 'were within 3 mm of a sample in the other set. This measures extent changes for one '
                 'tested hidden annotation family and fixed CT bone. The distance tolerance regions overlap.')
    report=f"""# Local reconstruction and target identifiability on s0010

Four recorded 102-waypoint scans test the local evidence for a provisional
right L4 medial branch bony surrogate at right L5. The local reference and
its variants are evaluator-only.

## Result

**TARGET_LOCALIZATION_NOT_EVALUABLE on the recorded cohort.**

| Step | Measured | Verdict |
|---|---|---|
| Local support | Global, local observed and full CT coverage on the same 408 frames | Evaluate separately |
| Historical visibility gate | Strict SAP and TP cores | No first-visible SAP core |
| Annotation sensitivity | Bounded within-bone perturbations with first-hit labels held fixed | Conditional on this tested family |

## 1. Local reconstruction — does global coverage carry over to the target region?

The reconstruction exports enter the sealed evaluator without new inference.

**Criterion.** No absolute success threshold. Compare 5, 10 and 20 mm
regions using the same 1.5 mm same-identity distance gate and the same
source views for all four configurations.

### 1.0 Quantities

| Item | Source |
|---|---|
| ROI radii and geometry metric | Frozen research manifest |
| Calibration and robot base coordinates | ultrasound_spatial_map.py |
| First-visible Oracle interface | sonoanatomy_labels.py |

### 1.1 Method

- Both query and reference points are clipped to the stated distance from
  the nearest provisional midpoint sample.
- Eligible observed coverage measures the fraction of admitted Oracle pixels
  near a same-class reconstructed point. Full CT coverage measures the
  fraction of all local surface voxel centers near a genuinely observed,
  same-class Oracle sample.
- All-visible evidence includes pixels rejected by pose routing. A voxel
  within the 1.5 mm geometric gate can still lack a direct first-bone hit.
- Fused semantic comparisons use each selected source viewpoint. Local error
  uses nearest same-class CT boundary voxel centers, with nonfinite distances
  omitted from quantiles and retained as unsupported in counts.
- L4 observations supply adjacent anatomy for the L5 bone-level proxy.

### 1.2 Setup

| Item | Value |
|---|---|
| Base | {manifest["base_commit"]} |
| Execution commit | {opt.commit} |
| Slurm | {opt.slurm}, 2 CPUs and no GPU |
| Figures and report | {opt.presentation_commit or opt.commit} · Slurm {opt.presentation_slurm or opt.slurm} |
| Analysis runtime and peak process RSS | {x["wall_seconds"]:.2f} s and {x["peak_process_rss_mib"]:.1f} MiB |
| Manifest | {opt.manifest} |
| Manifest SHA-256 | {x["manifest_sha256"]} |
| Methods | B0 Oracle raw, B1 U-Net raw, B2 Oracle fused, B3 U-Net fused |
| L4 checkpoint | {manifest["models"]["L4"]["sha256"]} |
| T4 L5 checkpoint | {manifest["models"]["L5"]["sha256"]} |
| Sample accounting | 408 / 408 across 2026121700, 2026121702, 2026121900 and 2026121901 |

Run from the Arm B worktree with the pinned compute allocation.

    sbatch hpg/jobs/s0010_mbb_identifiability.sbatch {opt.manifest} NEW_OUTPUT {opt.runtime}

### 1.3 Results

**Eligible observed-surface coverage**

{coverage}

**Provisional CT boundary near actual observed evidence**

{full}

**Local errors and semantic counts**

{errors}

**Verdict.** Eligible observed coverage measures recovery of the evidence
available to reconstruction. Full local reference coverage measures how much
of the proposed anatomical surface was observed.

### 1.4 Reading it

- 0 / 0 means the relevant observed group contains no samples.
- Opened development data, one anatomy and unvalidated CT-dependent semantic
  labels limit inference to this recorded simulator study.

### 1.5 Files

| File | Contents |
|---|---|
| summary.json | Coverage, shape error, observed views and gaps split by seed, radius, and class |
| pooled_local_errors.json | Raw count and pooled geometric quantiles |
| coverage.png | Global and local coverage and full reference support |
| dense_local_comparison.png | Oracle and U-Net local raw and fused geometry |
| reference_visibility.png | Separate observed and evaluator-only reference panels |
| sparse_dense_local.png | Same-view sparse versus dense comparison |
| local_errors.png | Distribution of 10 mm ROI error |

## 2. Identifiability — what changes when local evidence is removed or altered?

**Criterion.** Preserve the historical minimum of two distinct paired-core
image-supported frames in each of two sessions. Use the existing 3 mm
development distance only to compare the finite annotation family.

### 2.1 Method

- The repository first defined the right L4 medial branch bony apposition
  surrogate in commit d01f5c2. It takes CT model labels 6 and 5 inside right
  L5 source bone label 27. Strict endpoint cores are the boundary subsets
  within the minimum SAP–TP boundary separation plus one CT spacing. The
  provisional region consists of representative nearest-core midpoints.
- The strict paired criterion operationalizes direct evidence. It does not
  prove whether a validated partial-anatomy prior could recover the relation.
  The target proposal lacks a validated nerve reference and no dedicated
  clinician sign-off is recorded in the recovered material. The clinical
  scan inspiration and prior broad SAP/TP evidence do not validate the
  proxy. The L5 dorsal ramus is a separate L5–S1 taxonomy.
- Remove each contributing view or a whole scan family and recompute voxel
  fusion when applicable. Report nearest visible SAP–TP distance as a
  descriptive observation, with no proxy inferred from it.
- Expand or retract the hidden apposition-facing SAP annotation by one or two
  voxel steps. Source bone, acoustic inputs, CT intensities, TP, component
  count and first-visible SAP membership remain fixed. The uncertainty shell
  around the CT-model labels is unchanged at every sampled first-hit bone
  voxel. The original midpoint construction is applied unchanged to each
  allowable annotation. This is an annotation sensitivity, not a physical
  shape completion or new acoustic simulation.

### 2.2 Setup

| Item | Value |
|---|---|
| Target definition | l4_medial_branch_target.py:build_target_contract_geometry |
| Historical rule | s0010_l4mb_target_observability_v1.yaml |
| Protected distinct first-hit voxels | {a["protected_first_hit_voxels_all_frames"]:,} |
| Cohort | Same four sessions, input manifest and invocation as §1 |

### 2.3 Results

**Exact first-visible endpoint coverage**

{endpoints}

**Nearest observed local SAP–TP distance under missing views, radius 20 mm**

{views}

**Hidden annotation sensitivity**

{variants}

{extreme}

**Verdict.** The paired visibility gate fails and the existing recorded
anatomy does not establish identifiability of the provisional target.

### 2.4 Reading it

- The proximal-facing gaps can reflect acquisition evidence and weak target
  definition together, so high reconstruction coverage does not remove them.
- Separate clinical/anatomical verification of the reference and a small
  blinded test of local evidence recovery would discriminate these causes.

### 2.5 Files

| File | Contents |
|---|---|
| summary.json | Exact core counts, source removal trials and sensitivity |
| missing_view_sensitivity.png | Per-session and per-method missing-view figure |
| annotation_sensitivity.png | Evaluator-only label alternatives |
| annotation_sensitivity_geometry.npz | The tested provisional geometries |
"""
    (folder/"REPORT.md").write_text(report)
    (folder/"pooled_local_errors.json").write_text(json.dumps(all_errors,indent=2)+"\n")
    print(f"Wrote {folder/'REPORT.md'}")


if __name__=="__main__":
    main()
