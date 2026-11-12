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
