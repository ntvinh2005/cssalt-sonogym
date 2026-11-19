# s0010 L4 TP surface-core proxy improvement cycle

## Result

**SCIENTIFIC_NO_GO_WITH_DIAGNOSIS.** Existing-data one-vs-rest learning removed
seven clean false-positive images but lost two true-positive images. A new targeted,
synchronized supplement improved calibration, but did not improve primary full-image
metrics on the sealed development session. Active-acquisition integration remains
**HOLD**.

This experiment measures simulator geometry-proxy recognition on fixed patient s0010.
It does not establish whole-TP or SAP recognition, an MBB target, patient
generalization, or Stage 3 control.

## Existing-data ablation

Job 51961297 compared four frozen ExtraTrees variants. Calibration selected
l4_ovr_hn10 without reading development.

| opened development | TP | FP | FN | precision | recall |
|---|---:|---:|---:|---:|---:|
| frozen v1 | 12 | 7 | 5 | 0.6316 | 0.7059 |
| selected OVR | 10 | 0 | 7 | 1.0000 | 0.5882 |

The candidate retained 10 TP, below the predeclared floor of 11, so the immutable
result is OPENED_DEVELOPMENT_NO_GO. L5 was not modified and remained TP=16, FP=4,
FN=3 on these opened data.

## Targeted supplement

The first GPU execution, job 51992362, captured 48 valid train frames but its verifier
crashed while writing metadata because protocol v11 omitted source.implementation_commit.
Its evidence is preserved and was not mixed into the corrected experiment.

Protocol v12 retained the scientific gates and waypoint banks, added explicit source
provenance, and used new session IDs. Its SHA-256 is
a699216f29cec1b4ef543b1dbc18c5bc056d320440add93a8fae9ab63859668e.
CPU gate job 51992778 passed 93/93 tests.

GPU job 51992814 collected 48 train, 24 calibration, and 24 sealed-development
frames. All 96 images had unique hashes. Native-map replay, synchronized full-pose
replay, freshness, viewpoint diversity, and serialized-array hash integrity passed
for every session. Dataset preparation produced 925 proposals and 132 verified
candidate intersections.

The 240-tree L4 one-vs-rest candidate fitted 490 train-only samples from 364 source
images. New data contributed five L4-positive images, five frozen-v1 false-trigger
images, and seven competing-L5 images. Threshold 0.50 and the 774-value image-only
feature contract stayed unchanged. The saved checkpoint SHA-256 is
ab0ca634796c1af9e4b2824999e54f2096b68142854a51ff29516f6b79b1afa9;
save/reload was exact and no development feature was loaded before calibration.

### Calibration

| model | TP | FP | FN | precision | recall | ambiguous activations |
|---|---:|---:|---:|---:|---:|---:|
| frozen v1 | 2 | 2 | 0 | 0.5000 | 1.0000 | 1 |
| candidate | 2 | 1 | 0 | 0.6667 | 1.0000 | 0 |

The predeclared calibration gate passed.

### Sealed development

| model | TP | FP | FN | precision | recall | ambiguous activations |
|---|---:|---:|---:|---:|---:|---:|
| frozen v1 | 1 | 1 | 2 | 0.5000 | 0.3333 | 2 |
| candidate | 1 | 1 | 2 | 0.5000 | 0.3333 | 0 |

The candidate removed both ambiguous-image activations and reduced accepted proposals
inside the clean false-positive image from two to one, but did not remove that image-level
FP. It recovered neither FN: one score rose from 0.3708 to 0.4125, while the other fell
from 0.2792 to 0.2500. The split also failed the predeclared evidence sufficiency gate:
baseline had one TP and one FP, versus required diagnostic support of at least three TP
and two FP images.

Frozen L5 was unchanged. On the new sessions it scored TP=5, FP=0, FN=0 in calibration
and TP=2, FP=1, FN=0 in sealed development.

## Diagnosis and next minimum change

Clear hard negatives suppress ambiguous and duplicate triggers, but do not yet improve
actor-visible L4 localization. The main measured limitation is positive coverage: only
five new train images, two calibration positives, and three sealed-development positives
passed the complete L4 proxy contract. Another threshold or margin search does not
address this.

The next meaningful change is a redesigned privileged waypoint planner with a CPU
preflight that predicts substantially greater exact native-plane L4-core coverage before
another capture. A future protocol must freeze that coverage requirement, retain clear
confusers, and use a new development split.

## Reproducibility

| item | value |
|---|---|
| execution commit | dc4d26d13b194ad84182ae3ccbf5a9e42bd9d250 |
| CPU ablation | 51961297; 4 CPU; 44 s |
| v12 CPU gate | 51992778; 4 CPU; 6 s; 93/93 tests |
| failed v11 GPU job | 51992362; 1 L4; 93 s |
| completed v12 GPU job | 51992814; 1 L4; 226 s |
| new artifact storage | 124,239,412 bytes |
| run root | outputs/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_targeted_supplement_v1/51992814/ |

The machine-readable result is
experiments/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_improvement_cycle_result.json.
