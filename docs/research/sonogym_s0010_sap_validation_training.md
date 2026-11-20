# s0010 RIGHT L4/L5 SAP ultrasound validation and conditional training

## Goal

Determine whether clean simulated ultrasound supports level-conditioned RIGHT L4/L5 SAP-related cortical-interface labels, and fit the frozen ExtraTrees patch baseline only when the annotation and dataset gates pass.

## Reproducible setup

- Anatomy: `s0010` only.
- CT pseudo-SAP prediction: frozen job `52078067`, SHA-256 `a78daeae9159f0df28ba6fffcb8e8d7ef6c2e87f490c9598acfcb82efad1677f`.
- Development capture: job `52129000`, one NVIDIA L4, four CPUs, 30 GB requested memory, 4 minutes 11 seconds.
- Targeted follow-up capture: job `52131349`, one NVIDIA L4, four CPUs, 30 GB requested memory, 4 minutes 23 seconds.
- Final CPU analysis: job `52131929`, four CPUs, 16 GB requested memory, 1 minute 15 seconds.
- Development and follow-up protocols were checksum-sealed before capture. The two evaluation sessions in the follow-up protocol were not acquired because capture stopped fail-closed during calibration.

## Method

Image-only interface tracks are generated before privileged comparison. A positive proxy requires a smooth clean-image interface satisfying the frozen gradient, contrast, posterior-shadow, continuity and alignment criteria against the first-hit pseudo-SAP surface. Clear other-level or other-bone patches may be negative. Boundary or identity ambiguity is excluded.

Development annotation readiness requires, per level:

- matched acceptance rate at least 0.50;
- mismatched false-alignment rate at most 0.10;
- correct-projection ranking rate at least 0.70;
- at least 30 positive and 30 negative unique training images;
- at least 10 positive and 10 negative unique calibration images.

The model path is a level-conditioned 48-by-80 patch ExtraTrees classifier with 240 trees. Geometry is used only to construct and evaluate proxy labels, never as an inference feature.

## Results

The initial capture produced 160 verified images across three training sessions and one calibration session. The follow-up produced 180 verified images across three additional training sessions. The next calibration session stopped when realized motion was 1.9915 mm, below the frozen 2.0 mm distinct-viewpoint gate. No tolerance was changed and the evaluation sessions remained unacquired.

| Level | Train positive | Train negative | Calibration positive | Calibration negative | Matched acceptance | Mismatched alignment | Ranking |
|---|---:|---:|---:|---:|---:|---:|---:|
| RIGHT L4 SAP | 21 | 149 | 1 | 20 | 22/170 (12.9%) | 37/1020 (3.6%) | 39/170 (22.9%) |
| RIGHT L5 SAP | 52 | 149 | 3 | 19 | 55/170 (32.4%) | 67/1020 (6.6%) | 39/170 (22.9%) |

L5 obtained enough training positives, but not enough calibration positives and did not pass correspondence specificity/ranking. L4 missed both the training-positive quota and the calibration/correspondence gates.

## Engineering outcome

- Ambiguous clean-image decisions no longer become positive labels because of nearby privileged geometry.
- Matched, other-level and distinct-pose correspondence controls are recorded explicitly.
- Development, calibration and evaluation sessions are assigned before capture.
- The dataset builder excludes incomplete sessions and records their exact failure.
- Train-only fitting, calibration-only threshold selection, full-image proposal evaluation and checkpoint save/reload are implemented.
- Conditional execution correctly did not call `fit()` because neither level passed readiness.

## Scientific verdict

- RIGHT L4 SAP: `SIMULATOR_PROXY_SAP_LABEL_NOT_READY`; classifier `SAP_TRAINING_BLOCKED_DATA`.
- RIGHT L5 SAP: `SIMULATOR_PROXY_SAP_LABEL_NOT_READY`; classifier `SAP_TRAINING_BLOCKED_DATA`.
- Overall: `SAP_RECOGNITION_STAGE2_NOT_READY`.

The experiment establishes that synchronized captures and fail-closed proxy construction work, and that targeted L5 views increase the number of geometrically supported image candidates. It does not establish reliable image-only SAP identity or held-out performance.

## Limitations and minimum remedy

No clinician or real-patient claim is made. The model SAP masks remain simulator-only pseudo labels. The frozen two-job GPU allowance was exhausted. A future approved capture must first repair the calibration waypoint that produced only 1.9915 mm realized motion, then acquire at least seven additional clear L5 calibration positives and nine L4 calibration positives. L4 also needs at least nine additional training positives. More importantly, new views must improve matched-projection ranking, not merely increase frame count. Evaluation sessions should be acquired only after these development gates pass.

## Reproducibility

Protocols:

- `experiments/unity_two_stage_transfer/s0010_sap_validation_development_protocol_v1.yaml`
- `experiments/unity_two_stage_transfer/s0010_sap_validation_followup_protocol_v1.yaml`

Primary workflows:

- `workflows/two_stage_navigation/prepare_s0010_sap_dataset.py`
- `workflows/two_stage_navigation/merge_s0010_sap_datasets.py`
- `workflows/two_stage_navigation/train_s0010_sap_classifier.py`
- `workflows/two_stage_navigation/evaluate_s0010_sap_classifier.py`

The final review ZIP and large capture arrays remain in the runtime output tree rather than Git.
