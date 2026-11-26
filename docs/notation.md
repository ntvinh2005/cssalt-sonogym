# Project notation

### Incremental multiview anatomical landmark mapping
- **Kind:** method
- **Definition:** An actor-owned persistent map that associates fresh pose-synchronized ultrasound observations of RIGHT L4/L5 SAP and TP with spatial hypotheses in one common 3D frame, confirms each landmark from distinct probe poses, uses relative SAP-TP geometry to reject inconsistent associations, and never requires SAP and TP to be visible in the same image.
- **Intuition:** It reconstructs a shared anatomical relationship from purposeful probe exploration instead of treating one paired ultrasound frame as the target.
- **Source:** approved by user, 2026-11-23; `incremental_landmark_map.py:IncrementalAnatomicalLandmarkMap`.

### Clinician-inspired lumbar ultrasound scanning
- **Kind:** method
- **Definition:** A simulator-only sequence that uses longitudinal, transverse, cranial-caudal and physically different supporting acquisitions to identify lumbar bony landmarks from clean ultrasound and actor-visible robot motion; privileged anatomy may design or evaluate an opened feasibility capture but is excluded from actor decisions.
- **Intuition:** It tests whether controlled anatomical transitions provide information that isolated images do not, without implementing an invasive procedure.
- **Source:** approved by user, 2026-11-22; ASRA, *Ultrasound-Guided Lumbar Medial Branch and Intra-Articular Facet Injections* (2019).

### Longitudinal scanning
- **Kind:** method
- **Definition:** Probe acquisition with the ultrasound image plane aligned with the lumbar long axis while the probe is translated from midline toward lateral posterior elements.
- **Intuition:** The observed bony interfaces are expected to progress from spinous processes to articular pillars and then transverse processes.
- **Source:** approved by user, 2026-11-22; ASRA (2019).

### Transverse scanning
- **Kind:** method
- **Definition:** Probe acquisition with the ultrasound image plane rotated toward the lumbar cross axis at a declared vertebral level, preserving the actual robot orientation limit and its deviation from an exact orthogonal view.
- **Intuition:** It exposes depth and step-off relationships among posterior bony structures without pretending the simulator reaches an unavailable orientation.
- **Source:** approved by user, 2026-11-22; ASRA (2019).

### Cranial-caudal transition
- **Kind:** method
- **Definition:** A forward and reverse probe translation along the lumbar cranial-caudal direction that records the disappearance and reappearance of an image-supported TP interface while preserving contact and acquisition freshness.
- **Intuition:** Direction-dependent change can support a TP boundary hypothesis more strongly than one static cortical appearance.
- **Source:** approved by user, 2026-11-22; ASRA (2019).

### Reversible longitudinal medial-lateral sweep
- **Kind:** method
- **Definition:** A longitudinal-plane acquisition sequence that starts with image-supported TP-region evidence, moves toward the right-lateral boundary, reverses on a physically distinct parallel path, and continues medially toward an evaluator-labeled AP-region candidate; the candidate region is not promoted to SAP identity.
- **Intuition:** Ordered loss, return, and neighboring-region evidence can test a TP/AP boundary hypothesis more strongly than independent static frames.
- **Source:** approved by user, 2026-11-23; supplied clinician-video analyses A and B.

### Posterior acoustic-shadow fraction
- **Kind:** metric
- **Definition:** For the first actor-visible bright-interface track in a clean ultrasound frame, the fraction of normalized pixels below intensity 0.20 in the 42-pixel distal band beginning three pixels beneath the tracked interface.
- **Intuition:** It measures how much low-intensity posterior shadow follows the selected cortical interface without using simulator anatomy.
- **Source:** approved by user, 2026-11-23; implementation operationalization of the supplied clinician-video analysis B13.

### Deep-window visibility
- **Kind:** metric
- **Definition:** The mean actor-visible robust-normalized ultrasound intensity in the deepest 28% of image rows.
- **Intuition:** It provides a simple image-only measure of whether deeper tissue remains visible during a motion-conditioned cortical transition.
- **Source:** approved by user, 2026-11-23; implementation operationalization of the supplied clinician-video analysis B13.

### SPATIAL_GEOMETRY_VALIDATED
- **Kind:** object
- **Definition:** The spatial-reconstruction verdict used when an actor-accessible calibrated probe transform reproduces saved native physical coordinates within the frozen coordinate-error gate and mapped cortical candidates pass the declared bone-surface consistency controls.
- **Intuition:** It validates the spatial transform and observed-surface geometry, not anatomical target identity.
- **Source:** approved by user, 2026-11-21.

### SPATIAL_GEOMETRY_PARTIAL
- **Kind:** object
- **Definition:** The spatial-reconstruction verdict used when calibrated geometry is technically accurate or a reduced representation is physically informative, but actor-record availability, surface consistency, or target-specific coverage prevents the complete validation gate from passing.
- **Intuition:** Some spatial information is reliable, but the complete runtime reconstruction claim is not supported.
- **Source:** approved by user, 2026-11-21.

### SPATIAL_GEOMETRY_UNSUPPORTED
- **Kind:** object
- **Definition:** The spatial-reconstruction verdict used when neither calibrated nor reduced actor-side geometry clears the frozen physical-consistency controls.
- **Intuition:** Executed motion and public calibration do not support a useful spatial reconstruction under the tested formulation.
- **Source:** approved by user, 2026-11-21.

### L5_SPATIAL_LOCALIZER_READY
- **Kind:** object
- **Definition:** The RIGHT L5 TP surface-core verdict requiring the frozen map-driven localizer to satisfy target selection, wrong-acceptance, top-k, localization-error, and distinct-support gates on the declared final comparison sessions.
- **Intuition:** The localizer is ready for a separately authorized Stage 2 closed-loop validation, not already proven in a new live execution.
- **Source:** approved by user, 2026-11-21.

### L5_SPATIAL_LOCALIZER_PROMISING
- **Kind:** object
- **Definition:** The RIGHT L5 TP surface-core verdict used when map-driven target selection improves or adds reproducible spatial evidence on opened development sessions but does not satisfy every frozen readiness gate.
- **Intuition:** The method warrants a decisive follow-up without being ready for deployment.
- **Source:** approved by user, 2026-11-21.

### L5_SPATIAL_LOCALIZER_FAILED
- **Kind:** object
- **Definition:** The RIGHT L5 TP surface-core verdict used when the frozen map-driven localizer cannot control wrong selections or meet its minimum correct-selection and localization gates on the declared comparison.
- **Intuition:** Spatial support did not produce a defensible target-view decision.
- **Source:** approved by user, 2026-11-21.

### STAGE2_INTEGRATION_INTERFACE_READY
- **Kind:** object
- **Definition:** The integration verdict used when a leakage-tested CPU interface accepts clean ultrasound, actor-available executed pose feedback, a Stage 1 prior and localizer state, then returns ranked target views and an acceptance or abstention decision in recorded replay.
- **Intuition:** The software boundary is ready for a later physical-controller experiment; no new closed-loop success is implied.
- **Source:** approved by user, 2026-11-21.

### STAGE2_INTEGRATION_BLOCKED
- **Kind:** object
- **Definition:** The integration verdict used when the localizer evidence or software contract is insufficient to support a fail-closed Stage 2 replay interface.
- **Intuition:** Another scientific or engineering correction is required before physical integration.
- **Source:** approved by user, 2026-11-21.

### Motion-compensated spatial cortical mapping
- **Kind:** method
- **Definition:** An actor-side 2.5D representation that transforms clean-ultrasound cortical candidates into a Home-relative spatial frame using public image calibration and executed reduced probe-pose feedback, then counts support only across distinct acquisitions and viewpoints; privileged native coordinates are evaluator-only.
- **Intuition:** It tests whether physical consistency can reject visually similar but spatially incompatible bone interfaces without claiming a complete 3D reconstruction or SAP identity.
- **Source:** approved by user, 2026-11-21.

### INVALIDATED_AXIS_ORDERING
- **Kind:** object
- **Definition:** The correction status assigned when native ultrasound coordinates were flattened width-major but reconstructed by a direct depth-by-width reshape, invalidating every derived pixel position while preserving per-pixel membership counts and source acquisitions.
- **Intuition:** It withdraws spatial conclusions without discarding intact recorded frames.
- **Source:** approved by user, 2026-11-20.

### SAP_ULTRASOUND_OBSERVABILITY_NOT_EVALUABLE_PROJECTION_BUG
- **Kind:** object
- **Definition:** The SAP observability verdict used when acquired ultrasound and synchronized native coordinates are intact but a demonstrated width/depth rasterization defect invalidates projected SAP pixel locations, projection gates, and image-interface alignment measurements.
- **Intuition:** It preserves usable acquisitions while withdrawing every spatial observability claim derived from the invalid projection.
- **Source:** approved by user, 2026-11-20.

### SAP_SURFACE_CANDIDATE_VISIBLE
- **Kind:** object
- **Definition:** A simulator-only per-frame state in which the first source-label bone hit along at least one ultrasound image column belongs to the versioned pseudo-SAP volume and a clean-image cortical polyline lies within the frozen four-pixel image tolerance of those first-hit pixels.
- **Intuition:** It records provisional agreement between a visible interface and privileged geometry without promoting the pseudo-SAP mask to verified anatomy.
- **Source:** approved by user, 2026-11-20.

### SAP_SURFACE_VISIBLE_PROVISIONAL
- **Kind:** object
- **Definition:** A simulator-only structure verdict requiring at least two projection-gated SAP_SURFACE_CANDIDATE_VISIBLE target frames drawn from two distinct reset sessions.
- **Intuition:** It requires repeatable surface evidence while remaining weaker than semantic SAP ground truth.
- **Source:** approved by user, 2026-11-20.

### OCCLUDED_BY_OTHER_BONE
- **Kind:** object
- **Definition:** A per-frame or structure-level SAP observability state in which a non-pseudo-SAP source-label bone hit occurs earlier along every evaluated target ray than the projected pseudo-SAP surface and no first-hit pseudo-SAP surface is available.
- **Intuition:** The target geometry intersects the scan but another bone surface blocks its direct acoustic path.
- **Source:** approved by user, 2026-11-20.

### NO_VISIBLE_INTERFACE
- **Kind:** object
- **Definition:** A SAP observability state in which the clean ultrasound review identifies no cortical interface supporting the evaluated target after corrected spatial mapping.
- **Intuition:** It distinguishes absent clean-image evidence from geometric occlusion and unresolved identity.
- **Source:** approved by user, 2026-11-20.

### INCONCLUSIVE
- **Kind:** object
- **Definition:** The corrected SAP observability verdict used when the available target frames do not satisfy SAP_SURFACE_VISIBLE_PROVISIONAL and do not consistently support OCCLUDED_BY_OTHER_BONE or NO_VISIBLE_INTERFACE.
- **Intuition:** The observations contain partial or conflicting evidence and release no SAP training label.
- **Source:** approved by user, 2026-11-20.

This is the single registry for project-specific methods, metrics, states, and named objects used in research documentation.

### s0010_right_l4l5_sap_review_v2
- **Kind:** object
- **Definition:** The second static, simulator-only external-review package for anatomy `s0010`, containing a source-STL quality audit, a separately reconstructed segmentation surface, paired-facet geometric context, exact-pose ultrasound correspondence, and versioned RIGHT L4/L5 SAP region proposals without anatomical promotion.
- **Intuition:** It tests whether the original SAP candidates can be partitioned from actual source evidence instead of repeating their coordinate-selected boundaries.
- **Source:** approved by user, 2026-11-19.

### PROPOSED_SAP_SURFACE_CORE
- **Kind:** object
- **Definition:** The subset of an original RIGHT SAP candidate whose unique physical locations are supported by valid source faces on the lower vertebra and coincide with valid source-face locations on the articulating upper vertebra; it is a simulator geometric articulation surrogate pending anatomical review.
- **Intuition:** It isolates the strongest paired-facet evidence without claiming a complete SAP or accepted anatomy.
- **Source:** approved by user, 2026-11-19.

### BOUNDARY_UNCERTAIN
- **Kind:** object
- **Definition:** The subset of an original RIGHT SAP candidate not assigned to PROPOSED_SAP_SURFACE_CORE and not supported strongly enough to classify as a non-SAP neighbor from the available whole-vertebra labels and CT resolution.
- **Intuition:** It retains unresolved extent rather than forcing it into positive or negative anatomy.
- **Source:** approved by user, 2026-11-19.

### LIKELY_NON_SAP_NEIGHBOR
- **Kind:** object
- **Definition:** The subset of an original RIGHT SAP candidate supported by explicit adjacent-structure evidence as belonging to a neighboring posterior element rather than SAP; it remains empty when those semantics are unavailable.
- **Intuition:** It separates evidence-based exclusion from merely uncertain boundary extent.
- **Source:** approved by user, 2026-11-19.

### SOURCE_SEMANTICS_INSUFFICIENT
- **Kind:** object
- **Definition:** The review verdict used when whole-vertebra meshes, CT labels, adjacent-level geometry, and exact-pose projection support a reproducible geometric candidate but do not identify SAP, IAP, lamina, pars, or their boundaries well enough to establish a semantic SAP annotation.
- **Intuition:** Geometry can narrow review targets while still being insufficient to promote them to semantic ground truth.
- **Source:** approved by user, 2026-11-19.

### s0010_right_l4l5_sap_review_v1
- **Kind:** object
- **Definition:** The static, simulator-only external-review package for the versioned RIGHT L4/L5 SAP candidates on anatomy s0010, containing privileged geometry, curated exact-pose ultrasound correspondence, explicit uncertainty, and no promoted semantic label or MBB target.
- **Intuition:** It lets a reviewer judge the current candidates without running repository code or receiving an interactive tool.
- **Source:** approved by user, 2026-11-19.

### ACCEPT_PROVISIONAL
- **Kind:** object
- **Definition:** A review decision that the exact versioned candidate is sufficiently plausible for the declared simulator-only provisional use while remaining non-clinical and subject to its recorded uncertainty.
- **Intuition:** It accepts only the reviewed candidate version, not an actual nerve location or injection target.
- **Source:** approved by user, 2026-11-19.

### REVISE
- **Kind:** object
- **Definition:** A review decision that the exact versioned candidate has useful anatomical support but requires a specified boundary correction before provisional acceptance.
- **Intuition:** It preserves the evidence while preventing the uncorrected geometry from being promoted.
- **Source:** approved by user, 2026-11-19.

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

### Stage 2A
- **Kind:** method
- **Definition:** RIGHT L4/L5 TP and SAP localization in a shared actor-owned 3D map. Each successful level requires correct TP and SAP identities under the simulator evaluator, independent viewpoints, spatial consistency, fresh reacquisition of both final hypotheses, and resolution of competing identities.
- **Intuition:** It evaluates the anatomical map and the observations that actually supported it.
- **Source:** approved by user, 2026-11-24, Stage 2 decision and viewpoint-selection task.

### Stage 2B
- **Kind:** method
- **Definition:** Localization of a separately defined bony target region from anatomically validated Stage 2A landmark evidence.
- **Intuition:** It requires a target-region definition and evaluation beyond the historical TP-core view benchmark.
- **Source:** approved by user, 2026-11-24, Stage 2 decision and viewpoint-selection task.

### MBB target-region interface
- **Kind:** method
- **Definition:** A branch-specific mapping from independently supported SAP/TP landmark estimates to a bounded bony-landmark-derived surrogate region; it remains undefined and returns `BLOCKED_GROUND_TRUTH` until the anatomical mapping and uncertainty coverage are independently validated.
- **Intuition:** It prevents vertebral centroids, mesh quantiles, or classifier confidence from becoming invented nerve or injection coordinates.
- **Source:** approved by user, 2026-11-16.

### ANATOMICAL_TARGET_CONTRACT
- **Kind:** object
- **Definition:** The RIGHT L4 medial-branch target-definition verdict on fixed anatomy `s0010`: `VALIDATED_SURROGATE` requires an independently supported bony surrogate with bounded uncertainty, `PARTIAL` permits evaluator-only exploratory geometry with unresolved semantics, and `BLOCKED` means no defensible target region can be constructed.
- **Intuition:** It separates a reproducible bone relationship from an annotated nerve or clinically validated intervention endpoint.
- **Source:** approved by user, 2026-11-21.

### ULTRASOUND_TARGET_LOCALIZATION
- **Kind:** object
- **Definition:** The verdict for ultrasound-based localization of the RIGHT L4 medial-branch bony target at RIGHT L5: `PASS`, `DEVELOPMENT_ONLY`, `FAIL`, or `NOT_RUN`, with `PASS` requiring frozen evaluation evidence beyond already opened development sessions.
- **Intuition:** It prevents TP detection or a privileged geometry projection from being reported as target-region localization.
- **Source:** approved by user, 2026-11-21.

### ROBOT_NAVIGATION
- **Kind:** object
- **Definition:** The execution verdict for the first-target robot workflow: `NEW_CLOSED_LOOP_PASS`, `NEW_CLOSED_LOOP_FAIL`, `HISTORICAL_REPLAY_ONLY`, or `NOT_RUN`.
- **Intuition:** It distinguishes actual new physical simulator control from offline analysis and historical replay.
- **Source:** approved by user, 2026-11-21.

### FULL_MBB_INTERVENTION
- **Kind:** object
- **Definition:** The separate intervention verdict, which remains `BLOCKED` until a validated target, needle/injection contract, executed simulator intervention, and independent evaluation all exist.
- **Intuition:** Ultrasound navigation alone cannot establish an MBB procedure result.
- **Source:** approved by user, 2026-11-21.

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

### Multi-view cortical correspondence
- **Kind:** method
- **Definition:** Actor-side association of image-derived cortical-interface hypotheses across Stage 2 multi-view evidence using clean ultrasound, acquisition order, executed relative translation, executed relative rotation, and frame identity; evaluator geometry and semantic labels are excluded until scoring is frozen.
- **Intuition:** It tests whether repeated surface arrangements across physical viewpoints carry more localization evidence than isolated image patches.
- **Source:** approved by user, 2026-11-21.

### Top-k candidate selection
- **Kind:** metric
- **Definition:** For each session, rank actor-visible candidate views by the frozen decision score and report whether at least one clear supported positive occurs among the first k candidates; ambiguous evidence remains separate and is never counted as negative.
- **Intuition:** It measures whether a bounded return-and-confirm policy can recover a supported view even when top-1 ranking remains imperfect.
- **Source:** approved by user, 2026-11-21.

### Learned multi-view ranking
- **Kind:** method
- **Definition:** Supervised ranking of Stage 2 candidate views using only actor-visible single-frame scores, Multi-view cortical correspondence, executed relative motion, support counts, and acquisition identity; base scores used to fit the ranker are generated out-of-fold by session, and ambiguous evidence is excluded from hard labels.
- **Intuition:** It tests whether a small learned combination of repeated-view evidence can generalize better than a fixed correspondence average without introducing privileged geometry at inference.
- **Source:** approved by user, 2026-11-21.

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

### Spine_Subregions
- **Kind:** method
- **Definition:** The MICN-Lab pretrained nnU-Net model that assigns each CT voxel one of eight labels: background, vertebral body, pedicles, lamina, spinous process, transverse process, superior articular process, or inferior articular process.
- **Intuition:** It supplies semantic subregion proposals that are independent of the project's coordinate-based SAP candidates.
- **Source:** Da Mutten et al., *Journal of Imaging Informatics in Medicine* (2026), https://doi.org/10.1007/s10278-026-02169-7; approved by user, 2026-11-20.

### s0010_right_l4l5_sap_semantic_review_v3
- **Kind:** object
- **Definition:** The static review package that compares Spine_Subregions predictions for s0010 with the frozen RIGHT L4/L5 SAP V2 surface proposals in physical coordinates.
- **Intuition:** It introduces independent pretrained-model evidence without promoting either method to semantic ground truth.
- **Source:** approved by user, 2026-11-20.

### MODEL_V2_AGREEMENT
- **Kind:** object
- **Definition:** Review category for a Spine_Subregions SAP prediction that overlaps or is spatially consistent with the corresponding frozen V2 SAP core.
- **Intuition:** It marks agreement between two imperfect proposal methods, not accuracy against ground truth.
- **Source:** approved by user, 2026-11-20.

### MODEL_ONLY_SAP
- **Kind:** object
- **Definition:** Review category for a Spine_Subregions SAP prediction outside the spatial support of the corresponding frozen V2 SAP core.
- **Intuition:** It exposes semantic-model evidence that the V2 geometry did not propose.
- **Source:** approved by user, 2026-11-20.

### V2_ONLY_CORE
- **Kind:** object
- **Definition:** Review category for a frozen V2 SAP core location without spatial support from the corresponding Spine_Subregions SAP prediction.
- **Intuition:** It exposes V2 geometric evidence that the semantic model does not reproduce.
- **Source:** approved by user, 2026-11-20.

### UNRESOLVED_OR_CONFLICT
- **Kind:** object
- **Definition:** Review category for meaningful disagreement, registration uncertainty, anatomical ambiguity, or a competing Spine_Subregions class involving a proposed SAP region.
- **Intuition:** It preserves uncertainty instead of forcing disagreement into a positive or negative label.
- **Source:** approved by user, 2026-11-20.

### MODEL_SAP_PROPOSAL_READY_FOR_REVIEW
- **Kind:** object
- **Definition:** V3 verdict indicating that Spine_Subregions produced geometrically aligned and anatomically plausible SAP/IAP proposals with sufficient static evidence for independent review.
- **Intuition:** It authorizes review, not semantic-ground-truth acceptance.
- **Source:** approved by user, 2026-11-20.

### MODEL_PREDICTION_UNRELIABLE
- **Kind:** object
- **Definition:** V3 verdict indicating that inference completed but the prediction has serious anatomical, spatial, or domain-shift defects.
- **Intuition:** It preserves completed inference as negative evidence.
- **Source:** approved by user, 2026-11-20.

### BLOCKED_INPUT_OR_MODEL
- **Kind:** object
- **Definition:** V3 verdict indicating that the actual CT intensity volume, usable pretrained checkpoint, or authorized model access is unavailable.
- **Intuition:** It stops model-dependent execution without substituting fabricated data.
- **Source:** approved by user, 2026-11-20.

### ACCEPTED_SIMULATOR_PROXY_POSITIVE
- **Kind:** object
- **Definition:** A simulator-only SAP patch label requiring a technically valid acquisition, a clean-image decision of `VISIBLE`, an image-only cortical-interface candidate, privileged first-bone pseudo-SAP correspondence within the frozen spatial tolerance, no competing target identity, and no unresolved boundary condition.
- **Intuition:** It prevents ambiguous clean images or geometry-only intersections from becoming positive training labels.
- **Source:** approved by user, 2026-11-20; `workflows/two_stage_navigation/correct_s0010_sap_observability_v1.py:frame_status`.

### CLEAR_NEGATIVE_OR_OTHER_BONE
- **Kind:** object
- **Definition:** A technically valid simulator-only SAP patch label with image-visible interface or background evidence and verified absence of the requested pseudo-SAP core and its uncertainty region from the evaluated patch.
- **Intuition:** It separates supported other-bone or background negatives from unknown anatomy.
- **Source:** approved by user, 2026-11-20.

### BOUNDARY_OR_IDENTITY_AMBIGUOUS
- **Kind:** object
- **Definition:** A simulator-only SAP annotation state in which clean-image visibility, boundary support, requested-structure identity, or matched correspondence is unresolved.
- **Intuition:** These examples remain auditable but are excluded from hard binary targets.
- **Source:** approved by user, 2026-11-20.

### TECHNICALLY_INVALID
- **Kind:** object
- **Definition:** A SAP annotation state for an acquisition that fails freshness, synchronized mapping, pose, array-integrity, or image-validity requirements.
- **Intuition:** Technical failures are excluded before semantic evidence is considered.
- **Source:** approved by user, 2026-11-20.

### SIMULATOR_PROXY_SAP_LABEL_READY
- **Kind:** object
- **Definition:** A per-level simulator-only annotation verdict requiring the frozen correspondence gate, unique-image train and calibration quotas, ambiguity exclusions, and information-boundary checks to pass.
- **Intuition:** It permits bounded classifier fitting without promoting the labels to clinical or independently verified anatomy.
- **Source:** approved by user, 2026-11-20; `workflows/two_stage_navigation/merge_s0010_sap_datasets.py`.

### TRAINED_CALIBRATED_HOLDOUT_UNOPENED
- **Kind:** object
- **Definition:** A per-level classifier status indicating train-only fitting, calibration-only threshold selection, a passing frozen calibration gate, reload-verified checkpoint persistence, and no opened held-out evaluation.
- **Intuition:** It separates a calibrated development candidate from an independently evaluated detector.
- **Source:** approved by user, 2026-11-20; `workflows/two_stage_navigation/train_s0010_sap_classifier.py`.

### SAP_TRAINED_HELDOUT_PASS
- **Kind:** object
- **Definition:** A per-level classifier verdict requiring the frozen checkpoint and threshold to pass the declared full-image precision, recall and false-positive gates on the preassigned session-disjoint evaluation captures opened exactly once.
- **Intuition:** It is the minimum simulator-only evidence for later Stage 2 integration review, not clinical validation.
- **Source:** approved by user, 2026-11-20; `workflows/two_stage_navigation/evaluate_s0010_sap_classifier.py`.

### BLOCKED_ALIGNMENT
- **Kind:** object
- **Definition:** V3 verdict indicating that a model prediction cannot be mapped reliably into the existing simulator anatomy.
- **Intuition:** It separates inference availability from physical-coordinate validity.
- **Source:** approved by user, 2026-11-20.

### Simple anatomical landmark mapping

- **Kind:** method
- **Definition:** A fixed longitudinal and transverse ultrasound survey followed by complete-link clustering of projected visible-interface component representatives, separately for each RIGHT L4/L5 TP and SAP hypothesis. Distinct pose support is counted after spatial clustering. A medoid is the supporting representative with the smallest sum of distances to the other supporting representatives.
- **Intuition:** Multiple distinct views can support one surface location. A structure spanning several clusters can retain competing locations.
- **Source:** Approved experiment request, 2026-11-24. `simple_landmark_mapping.py:simple_map`.

### Anatomical mapping status

- **Kind:** object
- **Definition:** `NOT_OBSERVED` denotes no eligible observation. `CANDIDATE` denotes a selected cluster with insufficient distinct-view support. `AMBIGUOUS` denotes unresolved vertebral level or a competitor with equal distinct-view support and confidence within the declared margin. `MAPPED` denotes sufficient support for the selected cluster under the actor's rules. The experiment's evaluator scores separate anatomical correctness by strict majority of selected views within the declared image-distance gate.
- **Intuition:** Map acceptance and evaluator-derived identity are separate measurements.
- **Source:** Approved experiment request, 2026-11-24. `simple_landmark_mapping.py:simple_map` and `run_simple_landmark_mapping.py:score_map`.

### Distance to the visible interface

- **Kind:** metric
- **Definition:** Euclidean distance in millimeters from the selected map position to the nearest publicly calibrated 3D pixel labeled as a first-hit, image-supported provisional right-side cortical interface of the same landmark in the session.
- **Intuition:** It measures proximity to the observed surface sampled during the session. A distance of zero is expected for an oracle medoid drawn from those same visible pixels.
- **Source:** Approved experiment request, 2026-11-24. `run_simple_landmark_mapping.py:score_map`.

### Local shape completion

- **Kind:** method
- **Definition:** Estimate a bounded missing cortical surface from nearby ultrasound-supported anatomical observations under an explicit local surface model. Preserve measured points and inferred samples as separate layers.
- **Source:** approved ultrasound-only shape-completion task, protocol s0010_mbb_shape_completion_v1.

### Completion ensemble range

- **Kind:** metric
- **Definition:** The maximum minus the minimum supported fitted height at a query location across declared removal, neighborhood, chart, pose and smoothing variants.
- **Intuition:** A sensitivity diagnostic. Its fraction of withheld heights enclosed is measured separately.
- **Source:** s0010_mbb_shape_completion_v1 and the controlled held-out surface benchmark.

### Statistical shape model

- **Kind:** method
- **Definition:** A mean surface and principal displacement modes estimated from corresponding L5 vertices in independent external anatomies. The meshes retain physical size. Inference fits rigid pose and a bounded set of mode coefficients to observed surface points.
- **Source:** approved multi-anatomy statistical-shape campaign, protocol mbb_ssm_v1.
