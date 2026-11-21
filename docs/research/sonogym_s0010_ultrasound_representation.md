# s0010 ultrasound representation investigation

This experiment asks whether complete cortical-interface arrangements and image context are more informative for simulator-only lumbar localization than the frozen local SAP patch classifier. It uses only recorded clean ultrasound and actor-available motion at inference. CT, segmentation, pseudo-SAP masks, raw labels, and pixel-to-volume coordinates remain evaluator-only.

## Result

**Promising but inconclusive.** On the previously opened L5 evaluation sessions, the selected full-image representation improved clear-label recall from `9/13` to `13/13` while preserving `0` false positives. It also activated `17/36` ambiguous images, versus `8/36` for the frozen patch baseline, and selected a positive top-ranked view in only one of two sessions. A post-selection interfaces-only diagnostic retained the baseline's `8/36` ambiguous activations and improved clear-label recall to `10/13`. This supports better representation of visible bone arrangements, but not independent confirmation or Stage 2 integration.

RIGHT L4 remains unresolved: its evaluation set contains only one clear positive image, so its apparent `1/1` recall is not meaningful readiness evidence.

The terms and metric definitions are inherited from the frozen SAP validation protocol and [project notation](../notation.md).

## Step 1 — Information-content investigation

### Criterion

Determine whether the renderer provides structure-specific intensity or whether localization must rely on morphology, image context, and changes across views. Do not infer semantic identity from geometry intersection alone.

### Method

The audit traced the ultrasound label conversion and the frozen SAP validation data. It compared clean images, first-visible-bone evaluator masks, clear proxy labels, ambiguous cases, and recorded probe motion. It also checked which inputs are available to an actor at inference.

### Reproducible setup

```bash
rg -n "25:|26:|27:|28:|29:|30:|31:" \
  source/spinal_surgery/spinal_surgery/lab/sensors/ultrasound/label_conversion.yaml
python -m unittest tests.test_ultrasound_surface_features
```

- Anatomy: `s0010` only.
- Recorded source: synchronized SAP validation captures from job `52199322`.
- Runtime inputs used by the prototype: clean ultrasound, the immediately preceding clean image in the same trajectory, and actor-visible executed motion.
- Privileged inputs excluded from features: CT, segmentation, pseudo-SAP geometry, raw labels, native coordinate maps, and target coordinates.

### Results

Whole vertebral source labels map to the same bone acoustic class. The rendered intensity therefore does not encode a unique SAP identifier. The usable signal is the arrangement of bright interfaces, their depth, shape, relative spacing, intensity contrast, and shadow-like context. Single images contain useful evidence but remain non-unique: multiple bone surfaces can produce compatible bright contours.

### Reading it

This finding rules out a structure-specific intensity shortcut. It does not prove that SAP-related views are unobservable. It predicts that an image-only method needs spatial context or multiple views, and that ambiguous images must remain explicit.

### Files

- Renderer mapping: `source/spinal_surgery/spinal_surgery/lab/sensors/ultrasound/label_conversion.yaml`.
- Feature implementation: `source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/ultrasound_surface_features.py`.
- Tests: `tests/test_ultrasound_surface_features.py`.

## Step 2 — Frozen representation comparison

### Criterion

Select one candidate using training and calibration only. The candidate must meet calibration precision and recall of at least `0.80` and false-positive rate at most `0.20`. On evaluation, a better representation must exceed the frozen L5 baseline recall and F1 without adding clear-label false positives. Previously opened evaluation sessions cannot establish independent confirmation.

### Method

Two predeclared actor-visible candidates were evaluated:

- complete tracked interfaces plus low-resolution full-image intensity and gradient context;
- the same representation plus differences from the previous observation and executed relative motion.

Each level uses a 240-tree balanced ExtraTrees classifier. Fitting uses training sessions only. Candidate and threshold selection use calibration only. The selected representation and threshold were written to `frozen_selection.json` before evaluation rows were loaded.

### Reproducible setup

```bash
source .sonogym.env
export SONOGYM_CONFIG_FILE="$PWD/.sonogym.env"
sbatch hpg/jobs/s0010_ultrasound_representation.sbatch
```

- Job: `52203168`.
- Resources: four CPUs, zero GPUs, 16 GB requested RAM, 2 minutes 24 seconds elapsed.
- Protocol: `s0010_ultrasound_representation_protocol_v1.yaml`.
- Protocol SHA-256: see the adjacent `.sha256` seal.
- Model seed: `20261121`.
- Evaluation sessions: `s0010sapval_evaluation_04` and `s0010sapval_evaluation_05`; both were already opened in earlier work.

### Results

| L5 method | Clear TP/FP/FN/TN | Precision | Recall | F1 | Ambiguous activated | Positive top-ranked sessions |
|---|---:|---:|---:|---:|---:|---:|
| Frozen patch baseline | 9/0/4/31 | 1.000 | 0.692 | 0.818 | 8/36 | not reconstructed here |
| Selected full-image context | 13/0/0/31 | 1.000 | 1.000 | 1.000 | 17/36 | 1/2 |

The selected L5 threshold was `0.40`. Top-three ranking contained a clear positive in both sessions, while top-one ranking did so in one session. L4 produced `1/0/0/30` on clear evaluation images, but only one clear positive was present and the protocol had already marked L4 exploratory.

### Reading it

The clear-label improvement is real for these recorded observations, but it is incomplete. Full-image context recovers all four baseline false negatives while responding to nine additional ambiguous images. The result supports candidate ranking or evidence accumulation, not direct promotion to a semantic SAP recognizer.

### Files

- Runner: `workflows/two_stage_navigation/run_s0010_ultrasound_representation_experiment.py`.
- Slurm entry point: `hpg/jobs/s0010_ultrasound_representation.sbatch`.
- Runtime result: `s0010_ultrasound_representation/52203168/result.json`.
- Frozen selection: `s0010_ultrasound_representation/52203168/frozen_selection.json`.
- Checkpoints: `s0010_l4_single_full_image.joblib` and `s0010_l5_single_full_image.joblib` in the same runtime directory.

## Step 3 — Post-selection feature diagnosis

### Criterion

Use the already-opened evaluation data only to explain the selected model's behavior. Do not use this analysis to revise the frozen candidate, threshold, or primary verdict. Compare interface geometry, image context, and prior-view information under the same fit/calibration procedure.

### Method

The diagnostic separately fit:

- interface arrangement only;
- low-resolution image context only;
- interface arrangement and image context with the previous view and executed motion.

The selected combined single-image result from Step 2 remained unchanged. The diagnostic retained a separate checkpoint for the interfaces-only L5 model because it is the lowest-ambiguity follow-up candidate.

### Reproducible setup

```bash
source .sonogym.env
export SONOGYM_CONFIG_FILE="$PWD/.sonogym.env"
export SAP_REPRESENTATION_JOB=52203168
sbatch hpg/jobs/s0010_ultrasound_representation_diagnostic.sbatch
```

- Initial diagnostic job: `52203502`.
- Deterministic checkpoint-preserving rerun: `52203646`.
- Resources per job: four CPUs, zero GPUs, 12 GB requested RAM; final run elapsed 3 minutes 19 seconds.
- Interfaces-only checkpoint SHA-256: `a6f11ccf487de168118685d86b5ce42f43cbdadcebfe6646d9482453c9e2b15e`.

### Results

| L5 representation | Clear TP/FP/FN/TN | Recall | F1 | Ambiguous activated | Positive top-ranked sessions | Positive in top three |
|---|---:|---:|---:|---:|---:|---:|
| Interfaces only | 10/0/3/31 | 0.769 | 0.870 | 8/36 | 1/2 | 2/2 |
| Image context only | 12/0/1/31 | 0.923 | 0.960 | 12/36 | 1/2 | 2/2 |
| Selected combined single image | 13/0/0/31 | 1.000 | 1.000 | 17/36 | 1/2 | 2/2 |
| With prior view and motion | 13/0/0/31 | 1.000 | 1.000 | 13/36 | 1/2 | 2/2 |

Interfaces only adds one clear true positive over the frozen patch baseline without increasing clear false positives or ambiguous activations. Image context supplies most of the recall gain but also most of the additional ambiguous responses. Previous-view information removes four ambiguous activations relative to the selected single image, but it does not resolve the top-ranked-view failure.

### Reading it

The evidence favors representing complete visible surface arrangements and accumulating several ranked views. It does not support replacing the baseline with an unrestricted full-image classifier. Multi-view information helps, but the current immediately preceding view is insufficient to identify the correct top-ranked surface reliably.

### Files

- Diagnostic: `workflows/two_stage_navigation/diagnose_s0010_ultrasound_representation.py`.
- Result: `s0010_ultrasound_representation/52203168/post_selection_diagnostic.json`.
- Checkpoint: `s0010_ultrasound_representation/52203168/s0010_l5_topology_post_selection.joblib`.

## Step 4 — Closed-loop suitability assessment

### Criterion

Recommend Stage 2 integration only if actor-visible evidence reliably ranks the correct view across sessions and ambiguous activations are controlled. Development evidence on previously opened sessions is insufficient for an integration claim.

### Method

The assessment compared clear-label classification, ambiguous coverage, session-level top-one and top-three ranking, L4 sample sufficiency, and the information boundary. No geometry was supplied to any runtime feature. No new capture or GPU job was run.

### Reproducible setup

The assessment uses the frozen and diagnostic outputs from Steps 2 and 3. It requires no additional execution.

### Results

**Outcome: promising but inconclusive.** L5 surface and context representations improve clear-label discrimination, and both evaluation sessions contain a clear positive within the top three ranked observations. Neither selected model nor diagnostic reliably puts that observation first. L4 lacks enough clear positive evaluation evidence. Stage 2 integration therefore remains on hold.

The smallest decisive follow-up is to freeze the interfaces-only representation together with a multi-view consistency rule that operates on the top-ranked observations, then evaluate it once on newly reset, session-separated captures. That capture is not authorized or performed by this experiment.

### Reading it

The experiment changes the preferred research direction: reason over complete surface arrangements and multiple views rather than classify isolated bright patches. It does not establish SAP semantic identity, MBB targeting, or patient-level generalization.

### Files

- Frozen protocol and results listed above.
- Visual demonstration and compact review package are generated by `build_s0010_ultrasound_representation_review.py`.

