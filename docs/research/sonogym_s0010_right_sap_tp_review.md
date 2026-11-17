# SonoGym s0010 RIGHT SAP/TP review package

This record verifies the bounded anatomical-review export for RIGHT L4/L5 SAP and TP candidates.
It does not promote annotations, train a model, construct an MBB target, or run Isaac Sim.

## Result

**The review package is complete, but all four candidate decisions remain `PENDING_INDEPENDENT_REVIEW`; Stage 2 remains `MBB_STAGE2_BLOCKED_GROUND_TRUTH`.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| package export | source integrity, exact candidate reconstruction, CT/mesh views, ZIP integrity | all hashes/counts match; four figures; ZIP test passes | 4 / 4 candidates; 4 / 4 figures; 5 / 5 tests; ZIP valid | PASS |
| semantic readiness | independently reviewed SAP/TP boundaries | reviewer identity, decision, geometry and uncertainty for each candidate | 0 / 4 independently reviewed | PENDING |

## 1. RIGHT landmark review export — is the evidence package complete without promoting labels?

`review_s0010_right_landmarks.py`, run by Slurm job `51706459`. It reconstructs the v2
geometry candidates exactly and renders three mesh projections plus three CT planes per landmark.

**Criterion.** The user-approved package gate requires matching source hashes, exact v2 vertex
counts, four complete review figures, fail-closed metadata, and a valid ZIP. Semantic acceptance is
not defined until an independent reviewer returns a per-landmark decision and uncertainty.

### 1.0 Quantities

| symbol | definition | source |
|---|---|---|
| MBB landmark evidence | causal actor-side landmark evidence, not the privileged review overlays | [`docs/notation.md`](../notation.md#mbb-landmark-evidence) |
| `PENDING_INDEPENDENT_REVIEW` | geometry candidate exists but no qualified review decision is recorded | review protocol v1 |

### 1.1 Method

- Freeze patient `s0010`, facet L4-L5, and laterality `RIGHT`.
- Verify SHA-256 for CT, segmentation, L4/L5 meshes, and annotation v2.
- Reconstruct the existing TP/SAP mesh-quantile candidates and require exact vertex counts.
- Render evaluator-only posterior, right-lateral, superior, axial, sagittal, and coronal views.
- Keep TP `PROVISIONAL`, SAP `UNRESOLVED`, and all candidates actor-ineligible.
- Package decisions as `ACCEPT_BOUNDARY`, `REVISE_BOUNDARY`, or `REJECT_CANDIDATE` for a reviewer;
  do not infer a decision from image appearance automatically.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| run | Slurm `51706459`; commit `72d9cfdb177f386d3f442f5c4c85ef034f460f3f` | CPU-only export |
| protocol | `s0010_right_sap_tp_review_protocol_v1.yaml` | SHA-256 `bfb6eada5683614e55765a459283393dbc59270868dea26e27f4c68f69ee1059` |
| input annotation | `s0010_anatomical_annotations_v2.json` | SHA-256 `25df75aadb83b6d1faf39cfd8b376fb8976a08dbd834e302bad8f38e9de023c3` |
| resources | `anest-dept`; 4 CPU; 16 GB requested; 0 GPU | elapsed 17 s; peak RSS 449,608 KiB |
| runtime | Python 3.10.22 | pinned SonoGym environment |

```bash
sbatch --parsable --account=anest-dept --qos=anest-dept --partition=hpg-default \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:20:00 \
  --job-name=sg-right-sap-tp-review \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_s0010_right_landmark_review.sbatch
```

Nothing was dropped.

### 1.3 Results

| candidate | v2 status | reconstructed vertices | review result |
|---|---|---:|---|
| L4 RIGHT SAP | `UNRESOLVED` | 2,198 | `PENDING_INDEPENDENT_REVIEW` |
| L4 RIGHT TP | `PROVISIONAL` | 4,833 | `PENDING_INDEPENDENT_REVIEW` |
| L5 RIGHT SAP | `UNRESOLVED` | 2,422 | `PENDING_INDEPENDENT_REVIEW` |
| L5 RIGHT TP | `PROVISIONAL` | 4,343 | `PENDING_INDEPENDENT_REVIEW` |

The TP candidates visually cover the expected lateral processes but remain broad. The SAP
candidates occupy plausible posterior-superior regions but may contain adjacent pars/lamina; the
figures do not independently establish semantic boundaries.

**Verdict: package PASS; semantic review PENDING.** All engineering checks passed, while 0 / 4
candidate boundaries were promoted and the target region remains unresolved.

### 1.4 Reading it

- This package is ready for review, not detector training.
- CT/segmentation overlays are privileged and remain forbidden at inference.
- Reviewer corrections must return explicit geometry in mesh-index coordinates.
- Missing nerve segmentation does not prevent bony-landmark review but prevents a verified nerve
  target claim.

### 1.5 Files

| file | contents |
|---|---|
| `experiments/unity_two_stage_transfer/s0010_right_sap_tp_review_v1_result.json` | small execution/result manifest |
| `docs/research/s0010_right_sap_tp_review_instructions.md` | reviewer checklist and figure legend |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/s0010_right_sap_tp_review_v1_51706459.zip` | four figures, review manifest, protocol, instructions, checksums |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_right_sap_tp_review/51706459/` | complete CPU job evidence |
