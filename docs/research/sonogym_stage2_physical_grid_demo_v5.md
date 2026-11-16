# SonoGym Stage 2 physical-grid demo V5

## Result

Job `51641034` completed `0:0` on `anest-dept` with one NVIDIA L4, four CPU cores,
and 30 GB requested RAM. The actor completed a physically formalized search and returned
`ACCEPT`; the independent evaluator returned **`DEVELOPMENT_NO_GO`** because final orientation
error was 10.2871 degrees, above the frozen 10-degree gate. This result must not be rounded to a
pass.

The position grid contains 3x3 cells at 6 mm spacing in both translation axes. Each cell was
sampled at three executable orientations separated by 0.1 rad (5.7296 degrees), producing 27/27
fresh, technically valid, distinct ultrasound hashes. The ultrasound footprint is 75x100 mm, so
the 6 mm translation step preserves substantial field-of-view overlap while exceeding the 1 mm
coarse movement tolerance.

## Actor and evaluator

| measurement | result |
|---|---:|
| actions | 1,928 |
| search / total frames | 39 / 41 |
| distinct hashes | 41 |
| supporting hashes / viewpoints | 8 / 8 |
| selected validity probability | 0.997222 |
| selected predicted quality | 0.564428 |
| fresh confirmations | 2 / 2 |
| return error | 0.2366 mm / 0.0040 degrees |
| evaluator position error | 4.0998 mm |
| evaluator orientation error | **10.2871 degrees** |
| L4 visible fraction | 0.2579 |

The actor did not read the evaluator target pose, raw L4 identity, position/orientation error, or
L4 visibility. The model, historical V18 validity threshold, temporal decision rule, and evaluator
gates were unchanged.

## Why the earlier physical grids failed

- V4 job `51637731` reached the native ten-second episode horizon during the longer true-grid
  trajectory. Its failed artifacts are preserved.
- V4R1 job `51639443` established that the 3.00 rad coarse layer was mechanically ceiling-limited:
  position converged below 1 mm while orientation remained roughly 8-14 degrees from target.
- V5 retained 6 mm position cells but used executable coarse orientations 2.75, 2.85 and 2.95 rad,
  all below the 2.98 rad ceiling threshold.

These are development iterations, not independent confirmation attempts or V19.

## Resources and tests

The successful job ran for 2:28 on node `c1105a-s20`. Peak GPU memory was 7,317 MiB, peak sampled
GPU utilization was 59%, and peak RSS was 6,905,528 KiB. All 68 tests passed (11 active-pilot,
23 temporal-decision, and 34 Stage 2 regression tests). The run directory occupies 9,100,890
bytes. The two failed diagnostic runs occupy about 1.9 MB and 4.1 MB and remain preserved.

## Demo artifacts

The review ZIP contains exactly two MP4 demonstrations plus previews and provenance:

1. `01-ultrasound-spatial-grid-demo.mp4` (12.2 s, 1280x720) shows simulated ultrasound,
   commanded grid cells, realized serpentine trajectory, fine/micro search, selection, return and
   confirmation. It visibly reports `ACTOR ACCEPT` and `Evaluator NO_GO`.
2. `02-robot-probe-patient-camera.mp4` (21.53 s, 1280x720) shows the Isaac Sim robot arm and probe
   acting on the complete patient asset supplied by the task. That asset is a cropped torso, not a
   full-body anatomical model.

ZIP:
`/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/demo/sonogym_stage2_physical_grid_demo_20261116.zip`

SHA-256: `efaf0a4a6ec46a5fb62a50b6067d936bbc43a9817685d467d93008c128fb0c08`.

## Evidence boundary and next decision

This demonstrates an auditable physical grid, robot execution, image-based candidate selection,
return-to-best, and fresh confirmation on fixed `s0010`. It does **not** establish the frozen
geometric success gate, patient generalization, clinical validity, or MBB performance.

The next scientific decision is whether to freeze a new development protocol that targets the
remaining orientation error using a predeclared orientation-refinement rule. Do not tune a new
angle from the opened V5 evaluator result and call it independent confirmation.
