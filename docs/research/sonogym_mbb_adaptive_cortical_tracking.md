# Adaptive cortical-interface tracking on s0010

## 1. Frozen replay — Does adaptive tracking recover measured cortex?

### 1.1 Method

A0 uses the fixed-window tracker. A1 retains the original zero-gap rule
and searches a ±4 mm corridor by first-order dynamic programming. A2
allows at most 1 or 2 mm total unsupported horizontal distance, as two
diagnostic settings. Skips store no depth sample or anatomical evidence.
The unchanged bend rule examines only fully observed windows.

The original normalized intensity, depth-local maximum, and proximal and
distal contrast thresholds qualify peaks. The data cost is one minus the
weighted sum of peak intensity, proximal contrast and distal contrast,
respectively weighted 0.50, 0.25 and 0.25 after clipping each to [0,1].
The step penalty is 0.07 per depth pixel. Adjacent sampled columns retain
the original maximum 4-pixel depth step. For an A2 skip the endpoint depth
change is bounded by four pixels times the number of column intervals.
The gap penalty is 1.25 per skipped column. TP and SAP flank pixels remain
fixed; eligible brackets and semantic exclusions follow the historical
rules. A full alternative differing by at least 2 mm at a common observed
column within 0.30 total cost triggers ambiguity. Separate best anchored
end portions are recorded when no joined path exists. They are excluded
from localization.

### 1.2 Setup

Replay commit `2550b488a2f26c73b90816c4635f579d98fe3ae0`.
The 534 rows represent 432 unique images across five development sessions.
Sessions 2026121700 and 2026121700-126 share their first 102 records.
The same cached U-Net and visible Oracle pixels, Stage 1 `cf7_00`, probe
poses, calibration, eligibility and historical session verification
are used. Primary session comparisons use the shared frozen shortlist.
The all-frame sensitivity comparison is recorded in `sessions.csv`.

```bash
root=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym
prior="$root/artifacts/mbb-direct-sonographic-localization-023312"
runtime="$root-runtime"
out=/path/to/unused/result
sbatch hpg/jobs/s0010_mbb_adaptive_cortical.sbatch "$runtime" "$prior" "$out"
config=experiments/unity_two_stage_transfer/mbb_adaptive_cortical_tracking_v1.json
sbatch hpg/jobs/s0010_mbb_observed_surface.sbatch "$runtime" \
  -m workflows.mbb_adaptive_cortical.evaluate --result "$out" --source "$prior" --adaptive-config "$config"
sbatch hpg/jobs/s0010_mbb_observed_surface.sbatch "$runtime" \
  -m workflows.mbb_adaptive_cortical.summarize --result "$out"
sbatch hpg/jobs/s0010_mbb_observed_surface.sbatch "$runtime" \
  -m workflows.mbb_adaptive_cortical.review --result "$out" --prior "$prior" --config "$config" --output "${out}-figures"
```

Run the four commands sequentially after each prior command completes.
The actor seal is verified before the evaluator opens the synchronized
raw simulator-label rasters. The evaluator matches selected points in the
same image column within 1 mm of the first visible bone sample. A named
L5 TP or SAP match additionally requires matching a provisional visible
interface there. Other L5 bone, unknown L5 subregion, competing bone,
possible nonbone label boundaries and unmatched points remain separate.
The evaluator does not repair actor outputs.

### 1.3 Results

The baseline reproduced 1,068 frame/perception decisions and 15 prior
arm-session records exactly. The frozen replay took 27.83 s, including
25.14 s for shared original profile construction. No baseline, adaptive or
bounded-gap session passed the unchanged session gate.

| Source | Metric | A0 | A1 | A2 1 mm | A2 2 mm |
|---|---|---:|---:|---:|---:|
| Oracle | Brackets, records / deduplicated | 6/5 | 6/5 | 6/5 | 6/5 |
| Oracle | Joined profiles, unique | 0 | 2 | 4 | 4 |
| Oracle | Observed columns, deduplicated brackets | 55/159 | 141/159 | 141/159 | 145/159 |
| Oracle | Longest unsupported span, mm | 10.5 | 7.0 | 7.0 | 3.0 |
| Oracle | New echo columns, distinct | 0 | 88 | 88 | 90 |
| Oracle | Named L5 TP/SAP matches of new peaks | 0 | 14/88 | 14/88 | 14/90 |
| Oracle | Joined paths with verified competing vertebra match | 0 | 0/2 | 0/4 | 0/4 |
| Oracle | Candidate on negative control | 0/356 | 0/356 | 0/356 | 0/356 |
| Oracle | 2D regions, records / distinct | 0/0 | 3/2 | 5/4 | 5/4 |
| Oracle | Accepted sessions | 0/5 | 0/5 | 0/5 | 0/5 |
| Oracle | Largest available inter-view P95, mm | — | — | — | — |
| Oracle | Arm code time, s | 0.2244 | 0.0034 | 0.0050 | 0.0066 |
| U-Net | Brackets, records / deduplicated | 8/7 | 8/7 | 8/7 | 8/7 |
| U-Net | Joined profiles, unique | 0 | 2 | 4 | 7 |
| U-Net | Observed columns, deduplicated brackets | 91/196 | 174/196 | 187/196 | 182/196 |
| U-Net | Longest unsupported span, mm | 10.5 | 4.0 | 1.0 | 1.5 |
| U-Net | New echo columns, distinct | 0 | 84 | 94 | 90 |
| U-Net | Named L5 TP/SAP matches of new peaks | 0 | 11/84 | 11/94 | 11/90 |
| U-Net | Joined paths with verified competing vertebra match | 0 | 0/2 | 0/4 | 0/7 |
| U-Net | Candidate on negative control | 0/356 | 0/356 | 0/356 | 2/356 |
| U-Net | 2D regions, records / distinct | 0/0 | 3/2 | 5/4 | 7/6 |
| U-Net | Accepted sessions | 0/5 | 0/5 | 0/5 | 0/5 |
| U-Net | Largest available inter-view P95, mm | — | — | — | 6.29 |
| U-Net | Arm code time, s | 0.1740 | 0.0039 | 0.0058 | 0.0088 |

Arm code timing excludes shared tracking, peak extraction and projection.
The observed column count includes disconnected ends of unresolved
brackets. The learned 2 mm objective may prefer a cheaper path that
skips extra columns, so its supported fraction can be lower despite more
joined profiles. Zero confirmed competing vertebra matches leaves the
identity of unresolved subregions and unmatched echoes open.
The negative denominator is the union of distinct compatible transverse
images without both expected L5 anchors under the frozen Oracle support
rule. Positive negatives are 2D candidate violations of that evidence
rule before session acceptance. They are not independently validated
anatomically incorrect target locations.

### 1.4 Interpretation

- A1 fully joins two unique failure images with either perception
  source, R01 and R03. R08 is the repeated R01. In R03 all 20 originally
  missed columns are recovered. Three recovered samples match named
  L5 TP or SAP, six other L5 bone, nine first-visible L5 with uncertain
  subregion and two are unmatched.
- Across the two joined A1 profiles, each source recovers 38 new peaks,
  with seven named L5 TP matches, nine other L5 bone, 20 uncertain
  L5 subregion and two unmatched. Neither the named labels nor the new
  path establish the anatomical transition as a target.
- The other formerly discontinuous Oracle views stay unresolved under
  A1. The three prior span rejections R06, R07 and R09 produce no
  brackets. The frozen span limit is unchanged.
- Each source yields three A1 robot-base projected candidates across
  the five records, representing two unique images. R01 has only one
  transverse view per session. R03 lacks longitudinal context for
  U-Net and a distinct corroborating transverse view for both modes.
- The 2 mm diagnostic produces candidate regions for learned-only
  brackets in frames 2026121702/71 and 2026121900/42. Both have Oracle
  TP evidence without supported SAP. The associated learned sessions
  have pairwise bidirectional P95 of 4.99 and 6.29 mm, above 2 mm,
  and lack required longitudinal context. No gap-based arm passes.
- The review establishes correction of some fixed-window echo misses.
  It does not establish robust landmark localization or the missing
  anatomical reference. **TARGET_LOCALIZATION_NOT_EVALUABLE**.

### 1.5 Files

`metrics.csv`, `frames.csv`, `sessions.csv`, `brackets.json`,
`echo_evaluation.csv`, `bracket_evaluation.csv`,
`natural_negative_controls.csv`, `actor_seal.json` and evaluator
provenance are included in the review archive. Figures cover R01–R09
and both learned-only cases. Seven new focused checks and nine baseline
checks passed. Replay job `52856654` completed in 32 s and evaluator
job `52857130` in 5 s, both on two allocated CPUs and zero GPUs.
