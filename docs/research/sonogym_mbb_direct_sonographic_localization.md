# Direct sonographic landmark feasibility on s0010

The experiment asks whether simulated B-mode supports a directly visible bony
landmark associated with the right L4 medial branch at L5. The candidate must
come from a measured cortical interface near the cranial right L5 transverse
process (TP) root and lateral base of the superior articular process (SAP).

## 1. References — What is the sonographic evidence?

The criterion is a recognizable local cortical relationship with complementary
level context. Clinical placement accuracy and response were assessed outside
this simulation.

### 1.1 Method

- Greher's sonoanatomic study used paravertebral long-axis and cross-axis
  sonograms and evaluated the anatomical approach in cadavers and volunteers.
  It motivates the use of bone echoes as landmarks. [Greher et al., 2004, sonoanatomy](https://pubmed.ncbi.nlm.nih.gov/15114223/)
- The CT-confirmation study reports the longitudinal TP pattern for level
  context, then a transverse image of the cranial TP margin adjacent to the SAP
  base. Its L5 example is associated with the right L4 medial branch. Its
  placement accuracy used a separate CT assessment. [Greher et al., 2004, CT verification](https://pubs.asahq.org/anesthesiology/article-pdf/101/5/1195/357034/0000542-200411000-00020.pdf)
- Shim identifies a groove along the cranial TP edge beside the SAP. The
  study verified segment and placement using fluoroscopy. [Shim et al., 2006, fluoroscopy verification](https://pubmed.ncbi.nlm.nih.gov/16952818/)
- The review describes the distinct roles of sagittal or longitudinal level
  identification and transverse local landmark recognition. It also documents
  variation in reference and confirmation methods. [Ashmore et al., 2022, systematic review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9113209/)

### 1.2 Setup

The nerve of interest is the right L4 medial branch. The reference vertebra
is L5. The simulator protocol studies the bony image pattern only.

### 1.3 Results

The four references support complementary view roles and a location near the
cranial TP root and lateral SAP base. An arbitrary TP–SAP juxtaposition is
insufficient to identify that location. The study has no blinded expert
assessment of the simulated B-mode images.

### 1.4 Interpretation

The two plane families have separate functions. The longitudinal replay
tests whether image-supported L5 TP can corroborate the reused Stage 1
context. It does not independently count levels from the sacrum. The
transverse replay tests the local cortical relationship in the sonogram.

### 1.5 Files

- [Notation](../notation.md)
- [Frozen configuration](../../experiments/unity_two_stage_transfer/mbb_direct_sonographic_v2.json)

## 2. Replay — Does a visible cortical relationship support localization?

Each Oracle and learned replay uses the same input images and executed
robot-base poses. The gates require at least 4/5 accepted Oracle sessions
and 4/5 accepted learned sessions. Accepted learned regions on distinct
102-frame recordings must have median pairwise bidirectional P95 at most
2.0 mm. The independent-reference median error gate is 2.0 mm when evaluable.

### 2.1 Method

1. The existing Stage 1 bright-interface tracker proposes image profiles.
   Each sampled column must have a local echo maximum within two pixels of
   the track. A peak is compared with nearby echoes above and below it.
   Intensity is scaled using the frame's 5th and 95th percentiles. Accepted
   ridges need the frozen minimum proximal and distal contrasts. The
   distal contrast is an image feature, without asserting the physical
   cause of shadowing. The thresholds are engineering constraints.
2. TP and SAP anchors each require at least three consecutive supported
   columns within two pixels of a named interface. The local transverse
   bracket must run from lateral TP toward medial SAP, exhibit at least
   0.5 mm shallowing, and span at most 15 mm. All columns spanning the
   bracket and the local slope windows must have measured support, with no
   competing L4 annotation within the same local track. The tracker cannot
   fill a missing echo for the detector.
3. The image candidate comprises measured ridge samples around its local
   shallowing turn. The code retains comparably strong turns and abstains if
   the result exceeds the frozen 6 mm diameter. It does not synthesize
   points between disconnected interfaces. The detected turn is an
   engineering test for a relationship, without independent validation as
   the cranial groove.
4. A0 uses all eligible recordings and the Oracle-visible pixel identities.
   A1 uses Oracle-visible identities on a fixed shortlist. A2 uses learned
   identities on the same shortlist. The shortlist is computed once using
   U-Net semantic support, measured echo continuity and nearby executed
   poses. It retains up to eight longitudinal and 24 transverse-compatible
   views per recording. The Oracle provides no input to ranking. Selection
   can improve the fraction of informative views but cannot supply missing
   surface echoes.
5. A longitudinal view must be aligned within 30 degrees of the Stage 1
   sweep's fitted cranial axis. Transverse and previously recorded oblique
   views must be within 40 degrees of the lateral axis. The planned family
   is checked against the executed probe orientation.
6. The existing calibration projects only the candidate region and the
   supporting pixels into robot-base millimeters. It uses 0.5 mm native
   pixels, the width-centered lateral origin and the established 133 mm
   tool-to-image origin along the probe axis. The exact same public pose
   rotation and origin apply to both perception arms.
7. Each session requires longitudinal L5 context and at least two compatible
   transverse regions from distinct image hashes and poses separated by at
   least 2 mm translation or 1 degree of rotation. Candidate regions with
   greater than 2 mm bidirectional P95 disagreement cause abstention. The
   final diameter must remain within 6 mm. Sample overlap is reported
   directionally as the fraction of points within one calibrated pixel
   (0.5 mm) of the other region. Repeatability and region size do not
   measure anatomical correctness or calibrated confidence.

### 2.2 Setup

- Anatomy s0010. Stage 1 actor reference reused from `cf7_00`. The corpus
  consists of four 102-frame development records and a 126-frame extension
  of `2026121700`. The extended record's first 102 images and admitted
  pose fields must exactly match the original recording.
- The original L4 U-Net and T4 L5 U-Net are pinned by their complete
  checkpoint hashes and original thresholds. Raw cached probabilities are
  hash-verified for the four 102-frame recordings. Only the 24 suffix images
  missing from that cache require CPU inference.
- The source selection receipt contains a legacy relative pose derived
  from the simulator's human-frame transform. The final actor never uses
  that field or pre-routed learned geometry. It fits anatomical directions
  from Stage 1-relative **requested** scan coordinates and measured
  robot-base poses, then applies the existing level-specific TP/SAP
  routing rules in that public frame. Historical motion execution is
  unchanged. The diagnostic Oracle reads only image-supported first-visible
  pixel identities from the previously sealed cache.
- All original images are content-hash checked. In-session producer order,
  pose validity, contact and fresh-acquisition flags are checked. The
  published record of calibrated projection parity is copied into the
  manifest. No full mesh, CT volume, diagnostic path or simulator native
  transform enters the actor or reference review.
- No additional eligible untouched matched ultrasound record was verified.
  Older physician-style scans and their failure modes were already reviewed.
  The supported live capture job requests a GPU. This campaign is CPU-only
  and uses the existing 24 oblique acquisitions for its complementary-view
  evidence. No new images were requested from the simulator.
- Runtime is the existing `sonogym` environment. The Slurm wrapper requests
  two CPUs and zero GPUs on `anest-dept` for each stage.

### 2.3 Results

**FAIL.** All three frozen configurations return 0/5 accepted sessions. No
2D target region passes the image criteria. No anatomical region is
projected in 3D. This is a result on the recorded B-mode and the specified
recognition rules.

The study processed 534 recording rows with 432 distinct B-mode hashes.
The view gate accepts 70 longitudinal and 452 transverse-compatible rows.
These correspond to 56 and 364 distinct B-mode images. The other 12 rows
come from previously recorded oblique planes outside the orientation gate.

| Arm | Retained rows | L5 TP longitudinal context | Both transverse labels present | Both interfaces supported by B-mode | Ordered local profile | Continuous measured profile | 2D region | Rejected retained views | Accepted sessions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 all Oracle-visible | 534 | 28/70 | 32/452 | 9/452 | 6/452 | 0/452 | 0/452 | 506 | 0/5 |
| A1 selected Oracle-visible | 160 | 18/40 | 29/120 | 9/120 | 6/120 | 0/120 | 0/120 | 142 | 0/5 |
| A2 same selection, U-Net | 160 | 3/40 | 13/120 | 7/120 | 7/120 | 0/120 | 0/120 | 157 | 0/5 |

The rejected count includes retained views that provide neither longitudinal
context nor a localized transverse region. Rows outside the A1/A2
shortlist are reported separately. Nine of the 28 Oracle longitudinal
contexts have a supported L4 TP as well as L5 TP, compared with 0/3 learned
contexts on all eligible rows. On the matched transverse shortlist, five
frames have both B-mode-supported names with either perception source, four
with Oracle only and two with U-Net only. Of 120 matched rows, 109 support
neither pair of names. An Oracle/learned disagreement does not disambiguate
anatomical ground truth.

| Recording | Frame rows | Oracle view context A0 | Oracle view context A1 | U-Net view context A2 | A0/A1/A2 regions | Session disposition |
|---|---:|---:|---:|---:|---:|---|
| 2026121700 | 102 | 3/14 | 3/8 | 1/8 | 0/0/0 | insufficient evidence |
| 2026121702 | 102 | 10/14 | 5/8 | 0/8 | 0/0/0 | insufficient evidence |
| 2026121900 | 102 | 6/14 | 4/8 | 0/8 | 0/0/0 | insufficient evidence |
| 2026121901 | 102 | 6/14 | 3/8 | 1/8 | 0/0/0 | insufficient evidence |
| 2026121700-126 | 126 | 3/14 | 3/8 | 1/8 | 0/0/0 | insufficient evidence |

The 126-frame extension reproduces the original prefix's 102/102 frame
decisions for each perception source. In the suffix, 12/24 frames pass the
transverse-compatible orientation gate. Among those, 1/12 has both
Oracle-visible interface runs and 0/12 has both learned runs. The Oracle
case is oblique frame 107 and has no compatible ordered profile. The
excluded 12 frames contain two with both B-mode-supported names per source.
Those frames were left out of localization by the frozen view gate. They
cannot be used to conclude that all oblique views are uninformative.

Replay wall time including shared tracking and all three arms is **21.95 s**
after staging. Median shared B-mode tracking time per recording is 3.89 s.
The acquisition-independent learned preprocessing stage takes 4.51–5.49 s
per recording, plus 2.20 s for CPU inference on the 24 uncached frames.
There are zero projected candidates, so region extent, directed overlap
and inter-view and inter-session disagreement are all unavailable.

### 2.4 Interpretation

The earliest shared failure is the lack of a continuously supported
cortical relationship on an eligible transverse image. Shortlisting keeps
all nine Oracle frames with an image-supported TP and SAP, including all
six with an admissible ordered bracket. All six brackets from five distinct
images have unsupported columns. Their longest uninterrupted unsupported
spans range from **4.5 to 10.5 mm**, with a median of **5.75 mm**. Three also
exceed the maximum adjacent depth change. None is rejected by competing L4
evidence at this stage. All eight learned brackets from seven recorded rows
and six distinct images have unsupported columns. Their longest spans are
**3.0–10.5 mm**. Reviewing the raw B-mode in Figure 4 confirms separate bright
interfaces and interruptions in the measured profile. It is an engineering
image review.

The weaker learned longitudinal evidence is a separate limitation. The
shared transverse failure occurs before a turn, projection or cross-view
check can influence acceptance. The replay cannot isolate scan coverage from
acoustic signal limitations or from the conservative single-profile and
column support rules. The Oracle supplies only weak visible-interface
identities, and may miss or misidentify the local TP root or lateral SAP
base. Human readability of an interrupted profile remains untested. The
available data cannot establish that further views or a physician would fail.

### 2.5 Files

- `frozen-run/inputs/manifest.json` records input provenance and hashes.
- `frozen-run/replay/frames.csv` and `frame_audits.json` contain per-frame
  evidence, pose, view quality, annotations and abstention.
- `frozen-run/replay/sessions.csv`, `inter_session.csv` and
  `prediction_seal.json` record all dispositions and the prediction seal.
- `frozen-run/ordered_profile_failures.csv` reports the ordered brackets.
  Its counts are checked against the sealed predictions.
- `frozen-run/figures/` contains seven PNG figures. Figure 6 has no
  estimated region because no image passed.

## 3. Decision — Has localization feasibility been established?

The frozen decision includes the information-boundary, observation and
repeatability gates. Accuracy requires an independent anatomical reference.

### 3.1 Method

Verify the actor prediction hashes before reading the existing evaluator
reference audit. Do not use its endpoint hypotheses or historical
diagnostic path either to generate or score candidate locations.

### 3.2 Setup

The evaluator audit is `reference/reference.json` from the previously
sealed end-to-end package. The reference SHA-256 is
`2f24e1f01d2f0d8542cf431880921a0b66bbdc47ab6a7d94a121939c4323b36d`.
The declared anatomical contour remains unresolved. No subject geometry
is needed to confirm that the audit supplies no accepted region.

### 3.3 Results

| Gate | Status | Evidence |
|---|---|---|
| G1 Sonographic observability | Not established | Partial isolated cortical evidence; 0/452 supported complete Oracle patterns; no expert readability assessment |
| G2 Oracle feasibility | Fail | A0 and A1 each accept 0/5; threshold 4/5 |
| G3 Learned feasibility | Fail | A2 accepts 0/5; threshold 4/5 |
| G4 Spatial repeatability | Not evaluable | No accepted regions in the distinct recordings or shared-prefix pair |
| G5 Anatomical accuracy | Not evaluable | No accepted independent anatomical reference |
| G6 Information boundary | Pass for replay | B-mode, fixed U-Net, Stage 1 regional references, recorded scan commands, public robot poses and calibration; no evaluator geometry |

Independent ultrasound confirmation was unavailable in the verified pool.
No reserved seeds were consumed. The diagnostic cortical path remains a
historical geometric artifact. No new proxy was created from any actor
output. Target localization remains `TARGET_LOCALIZATION_NOT_EVALUABLE`.

### 3.4 Interpretation

Close this Stage 2 proof of concept at the demonstrated capability of
retaining ultrasound-observed cortical interfaces and identifying partial
bony context. The frozen direct estimator has no supported localization
on these recordings. The proximal cause is a gap in the measured local
cortical relationship, and its physical cause has not been isolated.
No new research direction was started.

### 3.5 Files and reproduction

Historical source jobs were `52710240`, `52719450` and `52801423`.
The final replay job was `52847252` on commit
`ff9b74cc628c6d0e40a47197fdacca3ae6cd3e33`. It ran for 81 seconds on
two allocated CPUs and zero GPUs. Slurm reports 68.35 CPU seconds.
The staged report was untracked during execution. The invoked method and
replay configuration were committed. The configuration SHA-256 is
`e5a004be64ab2bf4fe1b0a905c0eda48fa2c0ac7e4f6a9fdad4d6f69e5cecf8d`.

Initial staging job `52845317` completed, then the associated pending replay
and review jobs `52845759` and `52845921` were canceled before execution
because the legacy relative pose field was found to have simulator-frame
provenance. Public input staging job `52846324` failed at an omitted
acquisition ID required by the existing component interface. Its pending
dependencies `52846398` and `52846437` were canceled. The retry retained
that public identifier and preserved the scientific criteria and all inputs.
No earlier job generated a scientific outcome.

From the experimental worktree on HiPerGator:

```bash
repo=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym
runtime=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime
prior="$repo/artifacts/mbb-end-to-end-210237"
cache="$repo/artifacts/reconstruction-103750/shared/cache"
output=/path/to/unused/project-output
sbatch hpg/jobs/s0010_mbb_direct_sonographic.sbatch "$runtime" "$prior" "$cache" "$output"
```

Run the attribution audit against the sealed output in the same runtime
and with the same Python path as the Slurm script:

```bash
"$runtime/envs/sonogym/bin/python" -m workflows.mbb_direct_sonographic.explain \
  --inputs "$output/inputs" --replay "$output/replay" \
  --config experiments/unity_two_stage_transfer/mbb_direct_sonographic_v2.json \
  --output "$output/ordered_profile_failures.csv"
```

The 46 passed unit and regression checks are recorded in `tests-final.log`.
The code and prepared inputs permit replay without the U-Net probability
files or rerendering. The observation audit and figures retain original
B-mode and visible-only metadata.
