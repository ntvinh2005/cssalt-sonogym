# Project notation

This is the single registry for project-specific methods, metrics, states, and named objects used in research documentation.

### s0010_sap_geometry_audit_v1
- **Kind:** object
- **Definition:** The privileged, CPU-only inventory of the fixed-`s0010` RIGHT L4/L5 SAP source geometry, semantic labels, exact-pose ultrasound-plane correspondence, and unresolved-boundary evidence; it creates no actor input or semantic training label.
- **Intuition:** It distinguishes available whole-vertebra geometry and working projection from a verified SAP boundary.
- **Source:** approved by user, 2026-11-19.

### SAP_CANDIDATES_NEED_REVIEW
- **Kind:** object
- **Definition:** The anatomy-audit verdict used when the required vertebral geometry and coordinate transforms exist, but the RIGHT L4/L5 SAP boundaries remain heuristic or uncertain and therefore cannot support semantic labels or MBB target-region coordinates.
- **Intuition:** The source anatomy is inspectable, but independent boundary review remains necessary.
- **Source:** approved by user, 2026-11-19.

### s0010_l4_boundary_verifier_v1
- **Kind:** object
- **Definition:** The session-separated fixed-`s0010` experiment that fits an image-only three-state L4 TP surface-core verifier on training sessions, selects its operating threshold on calibration sessions, and restricts opened development and V3 observations to retrospective diagnosis.
- **Intuition:** It tests whether ultrasound morphology can distinguish proxy-positive views from boundary-ambiguous and proxy-negative views without runtime geometry.
- **Source:** approved by user, 2026-11-19.

### VERIFIER_IMPROVED
- **Kind:** object
- **Definition:** The L4 verifier verdict requiring a frozen image-only rule to improve the declared target-view failure measure on previously unopened session-separated observations without violating its recall, false-positive, abstention, L5-control, or information-boundary gates.
- **Intuition:** Calibration or opened-data gains alone cannot establish this verdict.
- **Source:** approved by user, 2026-11-19.

### VERIFIER_INCONCLUSIVE
- **Kind:** object
- **Definition:** The L4 verifier verdict used when implementation and exploratory evidence are available but no previously unopened session-separated evaluation supports or refutes an improvement claim.
- **Intuition:** It preserves a testable candidate without promoting retrospective diagnosis to independent evidence.
- **Source:** approved by user, 2026-11-19.

### VERIFIER_NO_IMPROVEMENT
- **Kind:** object
- **Definition:** The L4 verifier verdict used when a frozen candidate is evaluated on the declared independent sessions and fails its predeclared improvement gate.
- **Intuition:** It records a negative scientific result without changing the frozen V3 baseline.
- **Source:** approved by user, 2026-11-19.

### END_TO_END_PASS
- **Kind:** object
- **Definition:** The fixed-`s0010` MBB milestone verdict requiring every declared start to complete ultrasound anatomical discovery, both RIGHT L4/L5 target localizations, valid-view return and fresh confirmation, validated MBB target-region construction, simulated intervention, and independent final evaluation.
- **Intuition:** Module-level success cannot substitute for an executed complete procedure.
- **Source:** approved by user, 2026-11-19.

### PARTIAL_PASS
- **Kind:** object
- **Definition:** The fixed-`s0010` MBB milestone verdict requiring all declared Stage 1 maps and all RIGHT L4/L5 TP surface-core target-view gates to pass while the validated target-region or simulated intervention remains unavailable and fail-closed.
- **Intuition:** It records a complete ultrasound localization subsystem without claiming MBB procedure completion.
- **Source:** approved by user, 2026-11-19.

### BLOCKED
- **Kind:** object
- **Definition:** The fixed-`s0010` MBB milestone verdict used when either the declared ultrasound localization gates fail or required anatomical/intervention contracts are unavailable, so neither END_TO_END_PASS nor PARTIAL_PASS is supported.
- **Intuition:** It preserves the maximum verified milestone without inventing missing anatomy or procedure capability.
- **Source:** approved by user, 2026-11-19.

### Unity Stage 1
- **Kind:** method
- **Definition:** Physical longitudinal ultrasound acquisition followed by supervised image-patch scoring, aggregation of image-derived evidence in a Home-relative spatial frame, ordered landmark inference, and abstention when the accumulated evidence is insufficient.
- **Intuition:** It converts ultrasound morphology observations and executed relative movement into an anatomical landmark map without evaluator coordinates at inference.
- **Source:** Thoracic-US-Coach-Standalone commit f7e182b5e2ac71cc73ac41f8b2a928ce10e220c0, approved by user, 2026-11-12.

### Unity Stage 2
- **Kind:** method
- **Definition:** Deterministic bounded spatial and angular exploration scored by a supervised validity classifier and quality regressor, followed by physical return to the selected candidate, fresh-frame confirmation, and independent evaluation.
- **Intuition:** It acquires the best acceptable view near an inferred anatomical target while keeping learned perception separate from physical search.
- **Source:** Thoracic-US-Coach-Standalone commit f7e182b5e2ac71cc73ac41f8b2a928ce10e220c0, approved by user, 2026-11-12.

### Stage 2 candidate memory
- **Kind:** object
- **Definition:** Actor-side causal records of physically observed candidate views containing only executed relative pose, frame identity, frozen image-model predictions, action feedback, and supporting-observation identities from the active trajectory.
- **Intuition:** It retains evidence needed for selection without storing evaluator labels or target coordinates.
- **Source:** approved by user, 2026-11-15.

### MBB-oriented Stage 2
- **Kind:** method
- **Definition:** Causal ultrasound acquisition and executed-motion aggregation for unilateral L4-L5 anatomical localization, requiring separate evidence for the L3 medial-branch-related L4 SAP/TP site and the L4 medial-branch-related L5 SAP/TP site, with explicit uncertainty and fail-closed abstention.
- **Intuition:** It replaces one maximum-quality probe pose with procedure-specific multi-view landmark evidence while remaining separate from instrument control.
- **Source:** approved by user, 2026-11-16.

### MBB landmark evidence
- **Kind:** object
- **Definition:** An actor-side landmark hypothesis containing image coordinates, optional validated patient-relative coordinates, uncertainty, confidence, and causal supporting frame/viewpoint identities; CT, segmentation, evaluator coordinates, and hidden target locations are excluded.
- **Intuition:** It records what ultrasound and executed motion actually support without converting a single detection into a 3-D anatomical claim.
- **Source:** approved by user, 2026-11-16.

### MBB target-region interface
- **Kind:** method
- **Definition:** A branch-specific mapping from independently supported SAP/TP landmark estimates to a bounded bony-landmark-derived surrogate region; it remains undefined and returns `BLOCKED_GROUND_TRUTH` until the anatomical mapping and uncertainty coverage are independently validated.
- **Intuition:** It prevents vertebral centroids, mesh quantiles, or classifier confidence from becoming invented nerve or injection coordinates.
- **Source:** approved by user, 2026-11-16.

### TP_SURFACE_CORE
- **Kind:** object
- **Definition:** The versioned, conservative distal surface subset of a RIGHT L4 or L5 transverse-process candidate, excluding its unresolved base and carrying separate anatomical-approval and image-semantic-review states.
- **Intuition:** It supports a narrowly named feasibility detector without claiming whole-TP segmentation, SAP localization, an MBB target, or injection readiness.
- **Source:** approved by user, 2026-11-17.

### SIMULATOR_GEOMETRY_PROXY
- **Kind:** method
- **Definition:** Offline supervision that assigns a TP_SURFACE_CORE patch class only from a verified simulator projection, the versioned surface-core geometry, zero overlap with every declared uncertainty region and competing core, and the experiment's projected-support rule; it does not encode anatomical or clinical approval.
- **Intuition:** It permits a bounded engineering test of whether image morphology predicts the simulator-defined projected regions while keeping that result separate from anatomical recognition.
- **Source:** approved by user, 2026-11-18.

### s0010_tp_surface_core_simulator_proxy_exploratory_v1
- **Kind:** object
- **Definition:** The fixed-patient three-class ExtraTrees experiment using SIMULATOR_GEOMETRY_PROXY labels for BACKGROUND_OR_OTHER_STRUCTURE, L4_RIGHT_TP_SURFACE_CORE, and L5_RIGHT_TP_SURFACE_CORE, with train-only fitting, calibration-only threshold selection, and development-only evaluation.
- **Intuition:** It is an exploratory simulator-label pilot, not the verified-label protocol or evidence of anatomical recognition.
- **Source:** approved by user, 2026-11-18.

### EXPLORATORY_TRAINING_PASS
- **Kind:** object
- **Definition:** The s0010_tp_surface_core_simulator_proxy_exploratory_v1 engineering decision indicating that the sealed proxy labels, split-isolated ExtraTrees fitting, checkpoint persistence, and reported development measurements completed as specified.
- **Intuition:** It permits failure analysis of the frozen exploratory checkpoint without promoting simulator-proxy performance to anatomical recognition.
- **Source:** approved by user, 2026-11-18.

### ACTIVE_ACQUISITION_INTEGRATION_HOLD
- **Kind:** object
- **Definition:** The decision that the frozen simulator-proxy checkpoint must not enter active acquisition while its whole-image proposal errors remain insufficiently diagnosed or controlled.
- **Intuition:** It separates a completed exploratory fit from readiness to control physical evidence acquisition.
- **Source:** approved by user, 2026-11-18.

### IMPROVEMENT_DEMONSTRATED
- **Kind:** object
- **Definition:** The autonomous L4 proxy improvement-cycle result in which a protocol-frozen candidate clears its calibration and evidence-sufficiency gates, retains at least 90% of the frozen baseline's true-positive development images, strictly reduces false-positive development images, and does not increase PROXY_AMBIGUOUS activations.
- **Intuition:** It records a reproducible improvement in the intended simulator-proxy failure mode without promoting that result to anatomical, clinical, or active-acquisition validation.
- **Source:** approved by user, 2026-11-19.

### L4 margin
- **Kind:** metric
- **Definition:** For one actor-visible proposal, the frozen L4 class probability minus the larger of the frozen background and L5 class probabilities; an added decision rule may require this value to exceed a calibration-selected bound after the unchanged 0.50 L4 threshold and L4 argmax rule pass.
- **Intuition:** It tests whether L4 evidence separates from its strongest competing class without refitting the checkpoint or modifying L5 decisions.
- **Source:** approved by user, 2026-11-18.

### s0010_tp_surface_core_proxy_l4_margin_calibration_v1
- **Kind:** object
- **Definition:** The calibration-only fixed-checkpoint experiment that applies the L4 margin on whole-image actor-visible proposals, requires at least 90% retention of baseline L4 true-positive images and a reduction in false-positive images whose accepted proposals contain no projected core or uncertainty, and leaves L5 unchanged.
- **Intuition:** It determines whether a one-parameter L4 decision rule is worth freezing before any new development acquisition.
- **Source:** approved by user, 2026-11-18.

### PROXY_POSITIVE
- **Kind:** object
- **Definition:** An L4 proposal or image state with verified projection and at least one proposal satisfying the complete frozen L4 SIMULATOR_GEOMETRY_PROXY contract.
- **Intuition:** It is eligible to act as L4-positive proxy evidence.
- **Source:** approved by user, 2026-11-19.

### PROXY_NEGATIVE
- **Kind:** object
- **Definition:** An L4 proposal or image state with verified projection, no L4 core intersection and no L4 uncertainty intersection within the evaluated proposal support.
- **Intuition:** It is eligible to act as L4-negative proxy evidence; competing L5 geometry is recorded separately.
- **Source:** approved by user, 2026-11-19.

### PROXY_AMBIGUOUS
- **Kind:** object
- **Definition:** An L4 proposal or image state that is neither PROXY_POSITIVE nor PROXY_NEGATIVE because it intersects L4 uncertainty, contains L4 core evidence that does not satisfy the complete positive contract, or lacks verified projection.
- **Intuition:** It is excluded from primary positive/negative metrics and reported separately.
- **Source:** approved by user, 2026-11-19.

### s0010_tp_surface_core_proxy_tristate_hard_negative_v1
- **Kind:** object
- **Definition:** The CPU-only preparation experiment that applies PROXY_POSITIVE, PROXY_NEGATIVE and PROXY_AMBIGUOUS to existing s0010 proposals, then selects at most one high-L4-probability PROXY_NEGATIVE actor-visible interface patch per unique training image while excluding every declared uncertainty region.
- **Intuition:** It measures whether existing training captures contain enough clear L4 confusers to justify a separately approved L4 v2 fit.
- **Source:** approved by user, 2026-11-19.

### L4 one-vs-rest proxy classifier
- **Kind:** method
- **Definition:** A binary image-patch classifier whose positive class is the complete L4 RIGHT TP surface-core proxy contract and whose negative class contains only verified L4 proxy-negative patches, including L5-positive patches when they contain no L4 core or L4 uncertainty.
- **Intuition:** It asks only whether a patch contains the L4 proxy, so L5 evidence is a valid L4-negative confuser without being relabelled as three-class background.
- **Source:** approved by user, 2026-11-19.

### s0010_tp_surface_core_proxy_l4_existing_data_ablation_v1
- **Kind:** object
- **Definition:** The bounded CPU experiment that compares the frozen proxy-v1 detector with one multiclass hard-negative arm and three L4 one-vs-rest arms, selects an arm using calibration images only, and evaluates the selected arm once on previously opened development images.
- **Intuition:** It tests whether the ten clear training hard negatives and existing L5-positive patches can improve L4 full-image proxy recognition before additional capture.
- **Source:** approved by user, 2026-11-19.

### s0010_tp_surface_core_proxy_l4_targeted_supplement_v1
- **Kind:** object
- **Definition:** The fixed-anatomy simulator-proxy experiment that captures preassigned L4-positive, posterior-bone-confuser and L5-confuser train, calibration and evaluation views, fits the selected L4 one-vs-rest method on train only, gates it on calibration and evaluates it once on the sealed evaluation session.
- **Intuition:** It tests whether targeted real simulator observations recover L4 recall while retaining the false-positive reduction seen in the existing-data ablation.
- **Source:** approved by user, 2026-11-19.

### Stage 2 temporal evidence
- **Kind:** method
- **Definition:** Support for a candidate from causally ordered, technically admissible observations in the same trajectory and local actor-relative neighborhood, after duplicate image hashes are given at most one vote.
- **Intuition:** It tests prediction stability over acquired observations but does not by itself establish different physical viewpoints.
- **Source:** approved by user, 2026-11-15.

### Stage 2 multi-view evidence
- **Kind:** method
- **Definition:** Stage 2 temporal evidence whose counted observations are pairwise separated by the protocol-defined executed relative translation or rotation.
- **Intuition:** It requires physically different observed viewpoints while making no claim that their model errors are statistically independent.
- **Source:** approved by user, 2026-11-15.

### Stage 2 fresh confirmation
- **Kind:** method
- **Definition:** Ultrasound observations acquired after verified physical return to a remembered candidate, with increasing acquisition identity, distinct frame hashes, valid motion and technical feedback, and predictions consistent with the remembered view.
- **Intuition:** It verifies reacquisition and same-pose repeatability separately from nearby multi-view corroboration.
- **Source:** approved by user, 2026-11-15.

### Stage 2 abstention waterfall
- **Kind:** metric
- **Definition:** A fixed-order count of the same trajectory population remaining after each declared observation, movement, neighborhood, quality, hash-diversity, viewpoint-diversity, and decision requirement; every stage retains the original trajectory count as its reporting denominator.
- **Intuition:** It localizes where evidence disappears without hiding losses through denominator changes.
- **Source:** approved by user, 2026-11-15.

### Clamp-consistent diagnostic
- **Kind:** method
- **Definition:** An opened-data Stage 2 replay arm that preserves the frozen observations, model predictions, threshold, neighborhood, quality, freshness, and viewpoint requirements, but admits an angle-only clamped frame only when its frozen acquisition protocol explicitly allowed ceiling-limited capture and its complete action record remains finite, mapping-valid, surface-valid, nonterminal, nontruncated, and position-unclamped.
- **Intuition:** It measures the narrow collector-versus-replay contract mismatch without creating a global rule that ignores angular clamps.
- **Source:** approved by user, 2026-11-15.

### Stage 2 active-acquisition pilot
- **Kind:** method
- **Definition:** A development-only fixed-anatomy execution that receives a sealed actor-only Phase 1 target-level estimate, performs one continuous deterministic coarse, fine, and micro ultrasound search with frozen image validity and quality models, requires Stage 2 multi-view evidence, physically returns to the selected view, captures Stage 2 fresh confirmation, and exposes evaluator geometry only after the actor result is sealed.
- **Intuition:** It tests whether continuous physical acquisition supplies the evidence absent from reset-separated replay without claiming independent confirmation or changing perception models.
- **Source:** approved by user, 2026-11-16.

### Stage 2 bounded orientation refinement
- **Kind:** method
- **Definition:** A development-only continuation of a Stage 2 active-acquisition pilot that samples the protocol-declared angular offsets around the actor-selected candidate, clips them to declared physical angle limits, scores only the resulting fresh ultrasound observations with the frozen validity and quality models, and selects before evaluator geometry is opened.
- **Intuition:** It resolves orientation at a finer angular scale without reading target orientation or changing the image models.
- **Source:** experiments/unity_two_stage_transfer/stage2_orientation_refinement_protocol_v6r1.yaml, approved by user, 2026-11-16.

### Stage 2 new-start confirmation
- **Kind:** method
- **Definition:** Four protocol-sealed executions of Stage 2 bounded orientation refinement using previously unused reset seeds and distinct realized initial reset poses on fixed anatomy `s0010`; the aggregate passes only when all four executions independently pass the unchanged per-run actor, return, geometry, image, and fresh-confirmation gates.
- **Intuition:** It tests sensitivity to new physical reset starts without claiming patient-held-out generalization.
- **Source:** approved by user, 2026-11-16.

### D0
- **Kind:** metric
- **Definition:** Passes when replay observations, frame identities, frozen predictions, trajectory boundaries, and executed relative motion can be reconstructed without joining evidence across resets.
- **Intuition:** It checks replay integrity before decision performance is interpreted.
- **Source:** approved by user, 2026-11-15.

### D1
- **Kind:** metric
- **Definition:** Passes when every replay decision uses only actor-allowed fields available by the current causal step.
- **Intuition:** It prevents evaluator leakage and future-frame access.
- **Source:** approved by user, 2026-11-15.

### D2
- **Kind:** metric
- **Definition:** Passes when tested decision logic distinguishes duplicate same-view observations, distinct hashes at one pose, invalid motion, reset boundaries, and physically different viewpoints.
- **Intuition:** It checks that repeated records cannot masquerade as independent physical evidence.
- **Source:** approved by user, 2026-11-15.

### D3
- **Kind:** metric
- **Definition:** An exploratory classification of whether Stage 2 multi-view evidence reduces false accepted candidates relative to frozen quality-ranked selection while retaining the protocol-defined minimum acceptance coverage on opened datasets.
- **Intuition:** It determines whether a new independent physical experiment is justified, not whether Stage 2 is validated.
- **Source:** approved by user, 2026-11-15.

### D4
- **Kind:** metric
- **Definition:** Passes when the isolated decision implementation is deterministic, information-boundary tested, regression tested, and specified sufficiently for later controller integration.
- **Intuition:** It establishes engineering readiness without claiming live execution.
- **Source:** approved by user, 2026-11-15.

### D5
- **Kind:** metric
- **Definition:** Passes only after a separately authorized physical return-to-best and Stage 2 fresh confirmation experiment in SonoGym.
- **Intuition:** CPU replay cannot satisfy this gate.
- **Source:** approved by user, 2026-11-15.

### D6
- **Kind:** metric
- **Definition:** Passes only after coarse, fine, and micro physical search, return-to-best, Stage 2 fresh confirmation, and independent evaluation all complete under a frozen protocol.
- **Intuition:** It is the full Unity Stage 2 equivalent and remains distinct from replay implementation readiness.
- **Source:** approved by user, 2026-11-15.

### Shared anatomical-structure detector
- **Kind:** method
- **Definition:** A binary ExtraTrees classifier fitted on normalized intensity and gradient features from tracked-interface-centred ultrasound patches, with positive offline labels requiring Group A raw-identity overlap in the corresponding patch.
- **Intuition:** It tests recurrent vertebral morphology without assuming that each vertebral level has a unique appearance.
- **Source:** approved by user, 2026-11-13.

### Position-only control
- **Kind:** method
- **Definition:** An ExtraTrees classifier whose inputs are executed Home-relative x/z displacement, relative angle, acquisition direction, and normalized frame sequence; raw anatomical identity and ultrasound intensity features are excluded, although candidate rows originate from the common tracked-interface extractor.
- **Intuition:** It measures how much fixed-anatomy location and scan history predict the offline label without patch morphology.
- **Source:** approved by user, 2026-11-13.

### Shuffled-image control
- **Kind:** method
- **Definition:** Evaluation after deterministically permuting ultrasound morphology features within each development session while preserving labels and position/history rows.
- **Intuition:** It destroys the image-label association without changing the session-level image distribution.
- **Source:** approved by user, 2026-11-13.

### Image evidence gate
- **Kind:** metric
- **Definition:** Passes when the shared anatomical-structure detector has precision and recall at least 0.90, image-only seven-level macro recall is at least 0.70, and image-only macro recall exceeds the shuffled-image control by at least 0.20 on session-separated development data.
- **Intuition:** It permits spatial reconstruction only when actor-visible pixels carry reproducible anatomical development signal.
- **Source:** experiments/unity_two_stage_transfer/stage1_supervised_pilot_protocol.yaml, approved by user, 2026-11-13.

### L4 localization error
- **Kind:** metric
- **Definition:** The absolute difference in millimetres between the actor-emitted Home-relative L4 coordinate and the evaluator-only L4 reference coordinate.
- **Intuition:** It scores localization without exposing the reference coordinate to actor inference.
- **Source:** experiments/unity_two_stage_transfer/stage1_supervised_pilot_protocol.yaml, approved by user, 2026-11-13.

### Phase 1 confirmation
- **Kind:** method
- **Definition:** Four preregistered physical forward/reverse SonoGym scans from new session starts, evaluated with the frozen Stage 1 checkpoint, spatial threshold, and actor inputs; actor output is written and hashed before evaluator-only L4 reference data are opened.
- **Intuition:** It tests the selected s0010 L4 method on new acquisition sessions without further fitting or threshold selection.
- **Source:** approved by user, 2026-11-13.

### Phase 1 demo
- **Kind:** object
- **Definition:** A short visualization generated from one preregistered Phase 1 confirmation session, showing real ultrasound observations, executed Home-relative scan progress, accumulated actor evidence, the frozen actor result, and an explicitly separated evaluator-only comparison.
- **Intuition:** It makes the completed Phase 1 scan and inference auditable without creating a separate scientific run.
- **Source:** approved by user, 2026-11-13.

### Group A annotation
- **Kind:** object
- **Definition:** A privileged, geometry-supported annotation for sacrum, S1, or L1--L5 that is required to supervise or evaluate reproduction of Unity Stage 1 and Unity Stage 2 in patient s0010.
- **Intuition:** These annotations are the minimum anatomical reference layer for the immediate two-stage SonoGym experiment.
- **Source:** approved by user, 2026-11-12.

### Group B annotation
- **Kind:** object
- **Definition:** A privileged annotation or unresolved record for fine lumbar bony geometry investigated for a later medial branch block extension.
- **Intuition:** These records preserve future procedure-relevant geometry without delaying the immediate two-stage navigation experiment.
- **Source:** approved by user, 2026-11-12.

### PROVISIONAL
- **Kind:** object
- **Definition:** Annotation status indicating that source geometry supports a candidate point, region, or surface with recorded evidence, uncertainty, and limitations, but without clinician validation.
- **Intuition:** The annotation is usable for simulator engineering while remaining explicitly non-clinical.
- **Source:** approved by user, 2026-11-12.

### UNRESOLVED
- **Kind:** object
- **Definition:** Annotation status indicating that the available source data do not support identifying the claimed structure with sufficient evidence.
- **Intuition:** The record prevents an ambiguous structure from being converted into a fabricated coordinate.
- **Source:** approved by user, 2026-11-12.

### REJECTED
- **Kind:** object
- **Definition:** Annotation status indicating that an investigated candidate contradicts geometric or anatomical checks.
- **Intuition:** The record preserves a negative finding and prevents reuse of a disproven candidate.
- **Source:** approved by user, 2026-11-12.

### A0
- **Kind:** metric
- **Definition:** Passes when raw anatomical identities, coordinate frames, image orientation, CT-to-segmentation alignment, and mesh-to-segmentation geometry are consistent for their intended simulator use.
- **Intuition:** It checks source-anatomy integrity before labels are used.
- **Source:** approved by user, 2026-11-12.

### A1
- **Kind:** metric
- **Definition:** Passes when sacrum, S1, and L1--L5 each have a source-linked Group A annotation with geometry, uncertainty, and supporting evidence.
- **Intuition:** It checks coverage of the major ordered lumbar and sacral references.
- **Source:** approved by user, 2026-11-12.

### A2
- **Kind:** metric
- **Definition:** Passes when the requested fine bony structures have been inspected, with supported candidates marked PROVISIONAL and unsupported candidates marked UNRESOLVED or REJECTED.
- **Intuition:** It checks fine-bony-geometry feasibility without requiring every structure to be resolved.
- **Source:** approved by user, 2026-11-12.

### A3
- **Kind:** metric
- **Definition:** Passes only when ultrasound observations establish discriminative, repeatable actor-visible anatomical evidence rather than merely evaluator-only raw-label presence in the image field.
- **Intuition:** It separates geometric visibility from learnable ultrasound recognition.
- **Source:** approved by user, 2026-11-12.

### A4
- **Kind:** metric
- **Definition:** A separate READY, PROVISIONAL, or BLOCKED classification for Unity Stage 1 localization, L4 best-valid-view acquisition, provisional bony medial branch block target evaluation, simulated instrument positioning, and nerve-specific validation.
- **Intuition:** It prevents readiness for the immediate navigation experiment from being conflated with later procedure simulation.
- **Source:** approved by user, 2026-11-12.

### Medial branch block
- **Kind:** method
- **Definition:** A procedure targeting the medial branches of the dorsal rami that innervate lumbar facet joints; this project currently models only provisional simulator bony relationships and does not annotate the nerve or prescribe a trajectory.
- **Intuition:** It is the longer-term research context, not the present two-stage ultrasound-navigation endpoint.
- **Source:** [Consensus lumbar facet guidelines](https://pmc.ncbi.nlm.nih.gov/articles/PMC7362874/), approved by user, 2026-11-12.

### MBB_TARGET_UNRESOLVED
- **Kind:** object
- **Definition:** A Group B annotation status/name pair indicating that no bounded medial branch block target region can be supported from the current bone geometry without nerve data or a validated anatomical construction.
- **Intuition:** It records that procedure-target inference is blocked while leaving Group A work usable.
- **Source:** approved by user, 2026-11-12.

### EXACT_POSE_REPLAY
- **Kind:** method
- **Definition:** Reconstruction of an ultrasound image plane from a recorded full end-effector transform or the exact sampled volume coordinates used by the renderer.
- **Intuition:** It can reproduce the recorded plane without estimating missing orientation components.
- **Source:** approved by user, 2026-11-17.

### REDUCED_POSE_RECONSTRUCTION
- **Kind:** method
- **Definition:** Approximate reconstruction of an ultrasound image plane from recorded surface x, surface z, in-plane angle, and patient surface geometry when the full end-effector transform is unavailable.
- **Intuition:** It is testable against stored labels but is not exact recorded-pose replay.
- **Source:** approved by user, 2026-11-17.

### PROJECTION_PARITY_PASS
- **Kind:** metric
- **Definition:** The CPU reference implements native width/elevation/depth ordering and inverse projection, and at least one existing frame passes the unchanged exact-label-agreement and bone-IoU gates.
- **Intuition:** It distinguishes mathematical parity plus recovered real data from synthetic-test success alone.
- **Source:** approved by user, 2026-11-17.

### PROJECTION_REPAIR_BLOCKED
- **Kind:** metric
- **Definition:** Native-coordinate tests pass but no existing frame satisfies the unchanged correspondence gates, or required pose metadata are missing.
- **Intuition:** It prevents a synthetic coordinate test from being reported as recovered CT-to-ultrasound correspondence.
- **Source:** approved by user, 2026-11-17.

### PROPOSED_REVISION_PENDING_ANATOMICAL_REVIEW
- **Kind:** object
- **Definition:** An agent-generated surface-region revision with editable geometry and explicit uncertainty that has not received documented qualified anatomical approval.
- **Intuition:** It is review material, not semantic detector ground truth.
- **Source:** approved by user, 2026-11-17.

### UNRESOLVED_BOUNDARY
- **Kind:** object
- **Definition:** Boundary-review outcome indicating that the available CT, segmentation, and surface mesh do not defensibly separate the requested structure from adjacent anatomy.
- **Intuition:** It leaves the full candidate as uncertainty instead of presenting a cosmetic re-cut as anatomy truth.
- **Source:** approved by user, 2026-11-17.
