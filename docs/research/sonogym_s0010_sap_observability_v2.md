# s0010 RIGHT L4/L5 pseudo-SAP observability V2

This experiment asks whether fresh ultrasound observations contain a cortical interface spatially consistent with the corrected first-visible surface of the frozen RIGHT L4/L5 CT pseudo-SAP proposals. It does not validate SAP anatomy, train a detector, define an MBB target or authorize Stage 3.

## Result

**Engineering passes. Both levels meet the frozen two-session surface-evidence gate and receive `SAP_SURFACE_VISIBLE_PROVISIONAL`; the evidence is opened development, not independent semantic confirmation.**

| level | fresh target frames | projection-gated | surface candidate visible | qualifying | sessions | verdict |
|---|---:|---:|---:|---:|---:|---|
| RIGHT L4 SAP | 6 | 4 | 4 | 3 | 2 / 2 | `SAP_SURFACE_VISIBLE_PROVISIONAL` |
| RIGHT L5 SAP | 6 | 4 | 5 | 4 | 2 / 2 | `SAP_SURFACE_VISIBLE_PROVISIONAL` |

The evidence supports proceeding to a separately frozen, simulator-proxy SAP image-label review. It does not release a semantic SAP training label or an MBB target.

## 1. Archived-pose planning

`plan_s0010_sap_observability_followup.py` scanned the 336 synchronized frames from MBB PoC V3 with the corrected native rasterization. Candidate eligibility required the unchanged core projection gate, at least five first-hit pseudo-SAP pixels, and physical pose diversity. Ranking used image-normalized gradient at the first-hit pseudo-SAP surface, followed by surface support and image margin.

### Setup

| item | value |
|---|---|
| planner job | `52118753` |
| planner commit | `7173ebe19e6aa5ac43770b0451c00d455ec5a834` |
| source frames | 336 in four opened-development sessions |
| eligible candidates | L4: 12; L5: 62 |
| frozen selection | six L4 and six L5 poses, divided across two waypoint banks |

No evaluator geometry enters actor-visible inference; geometry is used only for offline pose design and evaluation.

## 2. Fresh capture and replay

Job `52120622` executed the sealed V2R1 protocol at commit `acbc28de9339f08402e5b2919599d79220eaa3d9` with one NVIDIA L4, four CPUs and 30 GB requested memory. It completed in 90 seconds.

### Method and gate

- Two separately reset sessions, each with three L4 then three L5 waypoints.
- Every observation requires a new acquisition identity, distinct image hash, native movement acceptance and exact saved-array integrity.
- Native coordinate replay must reproduce the stored raw label exactly.
- Full-pose replay must retain raw-label agreement at least 0.985, bone IoU at least 0.90 and coordinate error at most 0.001 volume index.

### Results

| measurement | result |
|---|---:|
| captured frames | 12 / 12 |
| distinct frame hashes | 12 / 12 |
| exact-pose replay | 12 / 12 |
| largest coordinate error | `3.6943e-05` volume index |
| session verifier status | 2 / 2 passed |

Job `52119650` is a preserved zero-frame technical failure: its three-frame segment declared 650 maximum actions against the 600-step episode horizon. V2R1 changed only trajectory reset boundaries; masks, pose order, seeds, gates and resource bounds remained frozen.

## 3. Clean-image surface evaluation

The evaluator reconstructs the corrected 200-row by 150-column projection using `flat.reshape(width, depth).T`. For each image column it finds the shallowest source label mapped to acoustic bone and checks whether that first hit belongs to the pseudo-SAP proposal.

The preliminary evaluation job `52121202` is superseded because hand-entered clean-image polylines were displaced 10–30 rows below the actual cortical gradient ridge. V2R1 preserves the capture and four-pixel gate, and applies one image-only rule to all frames: trace the strongest local gradient ridge by column within each clean-image interface region. That rule uses no projection or target geometry, but it was defined after opening this development set and therefore is not independent confirmation.

### Per-level result

| level | gate-passing aligned frames by session | unresolved | interpretation |
|---|---|---:|---|
| RIGHT L4 | session 00: 1; session 01: 2 | 2 / 6 | repeatable first-hit pseudo-SAP surface candidate |
| RIGHT L5 | session 00: 2; session 01: 2 | 1 / 6 | repeatable first-hit pseudo-SAP surface candidate |

Distances for qualifying frames are at most four pixels. Frames whose candidate core fails the 20-pixel/four-pixel-margin projection gate do not count toward the structure-level verdict even if a local interface is aligned.

## 4. Validation and reproducibility

Evaluation job `52121821` ran at commit `56505856aab2ff6e0f2f862d4cd621c43ebf2336` with one CPU, 8 GB requested memory, zero GPUs and 33 seconds elapsed.

```bash
sbatch --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/s0010_sap_observability_v2.sbatch

sbatch hpg/jobs/s0010_sap_observability_v2_evaluation.sbatch
```

| validation | result |
|---|---:|
| focused CPU tests | 41 / 41 passed |
| exact raw-label replay | 12 / 12 passed |
| package checksums | 59 / 59 passed |
| ZIP integrity | pass |

The package contains clean ultrasound, native coordinates, raw labels, full poses, corrected binary masks, frame comparisons, per-frame metrics and decision sheets.

## 5. Scientific boundary and next action

The new evidence overturns only the claim that the archived scan geometry cannot show a pseudo-SAP surface. It does not show that the pretrained CT region is anatomically correct, that SAP can be distinguished from adjacent bone using image-only features, or that an MBB endpoint is defined.

The next bounded action is a simulator-proxy SAP image-label review using these qualifying frames plus supported negatives, followed by a no-fit dataset preflight. Independent confirmation must use unopened sessions and a clean-image rule frozen before overlays are inspected.

No SAP classifier training, MBB target release or Stage 3 work was performed.
