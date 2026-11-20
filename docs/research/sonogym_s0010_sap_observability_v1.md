# s0010 RIGHT L4/L5 SAP ultrasound observability pilot

This report records a bounded simulator-only observability test of the frozen
model-generated RIGHT L4/L5 SAP proposals. It does not approve SAP anatomy,
train a classifier, define an MBB target or authorize an intervention.

## Result

**Fresh, exactly mapped ultrasound was acquired successfully, but neither
RIGHT L4 nor RIGHT L5 has a clean-image cortical interface aligned with the
projected pseudo-SAP core. Both structures therefore receive the frozen
`AMBIGUOUS` verdict.**

| criterion | measured result | verdict |
|---|---:|---|
| synchronized acquisition and replay | 12/12 frames pass; 12 distinct image hashes | pass |
| target projection coverage | 7/8 target frames pass the frozen 20-pixel and 4-pixel-margin gate | partial |
| L4 aligned visible evidence | 0/4 target views; 0/2 sessions | `AMBIGUOUS` |
| L5 aligned visible evidence | 0/4 target views; 0/2 sessions | `AMBIGUOUS` |
| context controls | 4/4 show a reviewer-marked bone interface with zero pseudo-SAP projection | informative control |

The clean-image review marked one L4 target view visible and three ambiguous;
all four L5 target views were marked visible. None of those marked interfaces
was within the frozen four-pixel tolerance of the projected pseudo-SAP core.
The privileged overlays place the core substantially deeper in the image,
usually beneath or beyond the visible cortical interfaces. This demonstrates a
specific correspondence problem between the CT pseudo-SAP volume and an
actor-visible cortical echo; it does not by itself establish whether the CT
proposal, scan plane, or acoustic occlusion is the dominant cause.

## 1. Acquisition integrity

**Criterion.** Every acquired frame must be fresh, physically distinct within
its session, hash-consistent, and exactly reproducible from the synchronized
full pose and native pixel-to-volume coordinates.

### 1.1 Method

- Freeze 12 geometry-privileged waypoints before capture: two target views for
  each level in each of two reset sessions plus one context control per level
  and session.
- Use the coupled physical controller; do not teleport.
- Persist the ultrasound, raw label image, full acquisition transform, native
  pixel map, acquisition identity and hashes from the same observation.
- Reload the saved files and apply the unchanged exact-pose replay gates.

### 1.2 Reproducible setup

| item | value |
|---|---|
| anatomy | fixed simulator anatomy `s0010` |
| execution protocol | `sonogym-s0010-sap-observability-v1r1` |
| protocol SHA-256 | `5a78091b205f59579199b4a88e222ffc07cfcd5a5c331604df933c0da05cf179` |
| execution commit | `b1e7f3ba5ca216550fcb1738ef8beed301b9ee9c` |
| implementation commit | `7107f9e3df1fd2e812a0755debc6095271c68cb5` |
| Slurm job | `52092083`; one NVIDIA L4, four CPUs, 30 GB; 1 min 32 s |
| split | `opened_development_observability` |
| model proposal | frozen Spine_Subregions label 6 intersected with the matching whole vertebra and RIGHT RAS half |
| proposed core | one six-connected voxel erosion |
| uncertainty band | one-voxel dilation minus the core |

The original V1 job `52091634` failed before capture because an unused
visibility threshold was set to zero even though the existing configuration
contract requires a positive value. V1R1 changed only that threshold to the
established `0.05`; masks, waypoints, projection gates and review criteria were
unchanged. The failed job produced zero scientific observations and remains
preserved.

```bash
sbatch hpg/jobs/s0010_sap_observability_v1r1.sbatch
```

### 1.3 Sample accounting

| session | frames | distinct hashes | physical actions | exact replay |
|---|---:|---:|---:|---:|
| `s0010sapobs_00` | 6 | 6 | 361 | 6/6 |
| `s0010sapobs_01` | 6 | 6 | 387 | 6/6 |
| **total** | **12** | **12** | **748** | **12/12** |

No frame, reset, session or failed acquisition was dropped. The legacy
`valid_frame_count` field refers to the old L4 geometric target-pose label and
is not an observability or technical-image gate for this experiment.

### 1.4 Results

All five verifier gates pass in both sessions:
`NATIVE_MAP_REPLAY`, `SYNCHRONIZED_FULL_POSE_REPLAY`,
`ACQUISITION_FRESHNESS`, `VIEWPOINT_DIVERSITY` and
`ARRAY_HASH_INTEGRITY`. Raw-label agreement and bone-mask IoU are 1.0 for all
12 frames, no sample is clipped, and the maximum full-pose/native coordinate
disagreement is `3.7269968e-05` volume index, below the frozen `0.001` limit.

### 1.5 Reading it

The acquisition and mapping problem is resolved for these frames. This result
does not establish semantic SAP identity or ultrasound visibility.

## 2. Blind clean-image observability review

**Criterion.** A level can pass only when at least two projection-gated target
views across both sessions have a clean-image `VISIBLE` interface whose marked
location is within four pixels of the projected pseudo-SAP core centroid.

### 2.1 Method

- Review randomized clean images without geometry overlays.
- Record `VISIBLE`, `AMBIGUOUS` or `NOT_VISIBLE` and interface coordinates.
- Commit the decision file before opening the privileged projection manifest
  or overlays.
- Join the locked decisions with projection evidence and run the frozen
  finalizer.

This is a single-agent, phase-separated review, not independent anatomical or
clinical review. The locked decision CSV SHA-256 is
`b9c43a9d82484d563317af23183f8441b674f4d9998762f8d3b4eb31c7b7df28`.

### 2.2 Results

| level | target views | projection-gated | clean visible | clean ambiguous | aligned visible | qualifying sessions | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| RIGHT L4 | 4 | 3 | 1 | 3 | 0 | 0/2 | `AMBIGUOUS` |
| RIGHT L5 | 4 | 4 | 4 | 0 | 0 | 0/2 | `AMBIGUOUS` |

The projected L4 core centroids lie at image rows 178.5–184.1, while the
reviewed interfaces lie around rows 70–122. The projected L5 centroids lie at
rows 134.7–171.5, while the reviewed interfaces lie around rows 78–115. One L4
target frame reached the intended anatomy but failed the frozen four-pixel
image-margin gate after physical tracking placed the core at the image border.

All four context controls contained reviewer-marked cortical interfaces despite
zero pseudo-SAP pixels. This makes a global technical-image failure unlikely;
the evidence loss is specific to the proposed SAP location and scan geometry.

### 2.3 Reading it

- `AMBIGUOUS` is fail-closed: no SAP classifier training labels are released.
- L5's four `VISIBLE` clean decisions mean that bone interfaces are visible,
  not that SAP is visible. The privileged join shows that they are not aligned
  with the pseudo-SAP core.
- The result does not invalidate the CT semantic proposal. A deeper SAP surface
  may be acoustically shadowed, or the chosen volumetric core may not represent
  the actor-visible cortical surface.
- The result does not support a SAP-derived MBB target, injection point or
  Stage 3 action.

## 3. Engineering and scientific outcome

**Engineering outcome: pass.** The new capture path produced 12 fresh,
distinct, exactly replayable observations and a reproducible blind-to-privileged
review join. All 33 focused observability/acquisition tests and 92 Stage 2
contract regressions pass.

**Scientific outcome: `SAP_ULTRASOUND_OBSERVABILITY_INCONCLUSIVE`.** The
specific frozen pseudo-SAP proposal was not confirmed as an actor-visible
cortical interface for either RIGHT L4 or RIGHT L5. The next meaningful change
is not classifier training. It is a geometry-privileged redesign of the
observation target around the first visible cortical surface along each ray,
with explicit review of whether that surface is a valid SAP surrogate before
any new capture or model fitting.

## 4. Files

| file | purpose |
|---|---|
| `experiments/unity_two_stage_transfer/s0010_sap_observability_v1_result.json` | concise machine-readable result |
| runtime `sessions/*/verification_report.json` | per-frame synchronized replay evidence |
| runtime `review/.../blind_decisions.csv` | locked clean-image decisions |
| runtime `review/.../audit_result.json` | final joined result |
| review ZIP | clean images, separate overlays, manifests, decisions and checksums |

