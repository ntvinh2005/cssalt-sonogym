# Project notation

This is the single registry for project-specific methods, metrics, states, and named objects used in research documentation.

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
