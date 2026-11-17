# SonoGym Stage 2 bounded orientation refinement V6R1

This record covers one fixed-`s0010` development execution of Stage 2 bounded orientation
refinement after the V5 physical-grid result. It covers physical execution and independent
post-actor evaluation; it does not cover independent confirmation, patient generalization, or
clinical performance.

## Result

**Job `51664534` is a numerical `DEVELOPMENT_PASS`, but its 0.004769470495040284-degree orientation margin is too small to establish robustness.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| bounded orientation refinement | actor acceptance, return, fresh confirmation, final geometry, and image visibility | actor `ACCEPT`; return <=5 mm / <=5 degrees; final <=10 mm / <=10 degrees; visibility >=0.20 | `ACCEPT`; 0.237496 mm / 0.006024 degrees; 4.085996 mm / 9.995231 degrees; 0.257033 visibility | `DEVELOPMENT_PASS` |
| technical retry | replay-subset identity after V6 job `51662670` | contiguous replay-local sequence with global frame identity preserved | V6R1 completed `0:0`; all 70 tests passed | pass |

The refinement changed the selected frame from 30 to 43 using only fresh ultrasound, frozen model
predictions, executed relative motion, and declared physical angle bounds. V16, V17, and V18 remain
`CONFIRMATION_NO_GO`; this opened fixed-anatomy development run does not replace them.

## 1. Bounded orientation refinement — can image-guided angular sampling clear the frozen development gate?

`source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/stage2_active_pilot.py`,
run by `hpg/jobs/two_stage_stage2_orientation_refinement_v6r1.sbatch`. The actor continues from
the physical 3x3 grid, samples bounded angular offsets around its preliminary candidate, selects
from those new images, returns physically, and obtains two fresh confirmations before evaluator
geometry is opened.

**Criterion.** The frozen protocol requires actor `ACCEPT`, at least three supporting viewpoints,
two fresh confirmations, return error <=5 mm and <=5 degrees, final position error <=10 mm, final
orientation error <=10 degrees, and L4 visible fraction >=0.20.

### 1.0 Quantities

| symbol | definition | source |
|---|---|---|
| Stage 2 bounded orientation refinement | protocol-bounded angular continuation around the actor-selected candidate | [`docs/notation.md`](../notation.md#stage-2-bounded-orientation-refinement) |
| Stage 2 fresh confirmation | post-return fresh-image evidence | [`docs/notation.md`](../notation.md#stage-2-fresh-confirmation) |
| Stage 2 multi-view evidence | support from protocol-separated executed viewpoints | [`docs/notation.md`](../notation.md#stage-2-multi-view-evidence) |

### 1.1 Method

- Preserve the V5 3x3 position grid, 6 mm translation spacing, and three coarse angles per cell.
- Form declared offsets `[-6, -4, -2, 0, 2, 4, 6]` degrees around the preliminary actor-selected
  angle, then retain only distinct targets inside `[2.70, 2.97]` rad.
- Acquire each retained target through the existing physical action interface and require a fresh,
  technically valid, distinct ultrasound observation.
- Apply the unchanged V18 probability threshold `0.5666666666666667`, frozen quality regressor,
  and Stage 2 multi-view evidence rule to the refinement observations.
- Freeze the selected actor result, physically return, acquire two Stage 2 fresh confirmations,
  hash the actor result, and only then run the evaluator.
- The actor may use the declared physical angle limits but may not read target coordinates,
  evaluator errors, raw label 28, or L4 visible fraction.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| run | Slurm `51664534`; execution commit `5dfc4df67dd647f3b20bb8527bd12ef9d47fdcaf` | §1 |
| protocol | `experiments/unity_two_stage_transfer/stage2_orientation_refinement_protocol_v6r1.yaml`; SHA-256 `a3c5b7d7b280f08950347f08acf2a35a4b3b062fb0dfc10b6ac96d5c1faa7bd8` | §1 |
| model | frozen V15 bundle; SHA-256 `7d3b113ba33b1b80448264a0d2e7f2c2cd5561b26021707648c4eef43d586672` | §1 |
| input handoff | Phase 1 `cf_00`; actor-result SHA-256 `9d3532f06f15443d7a0d3ac7c32d67430c93de45f1fe8d61bbdbe8ac7c58c13e` | §1 |
| task | `Isaac-robot-US-guidance-v0`, `conv`, one environment, patient `s0010` | §1 |
| random seed | `2026112100` | §1 |
| resources | `anest-dept`; one NVIDIA L4; four CPU cores; 30 GB; 20-minute limit | §1 |
| software | Python 3.10.22; NVIDIA driver 580.178.04; pinned project runtime | §1 |

```bash
sbatch --parsable --account=anest-dept --qos=anest-dept --partition=hpg-turin \
  --constraint=l4 --gres=gpu:l4:1 --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=30G --time=00:20:00 --job-name=sg-s2-orient-v6r1 \
  --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/sg-s2-orient-v6r1-%j.out \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage2_orientation_refinement_v6r1.sbatch
```

| | value |
|---|---:|
| attempted protocol-declared angular offsets | 7 |
| dropped, outside declared physical bounds or duplicate after bounding | 2 |
| **final physically acquired refinement observations** | **5** |

### 1.3 Results

| measurement | job `51664534` result |
|---|---:|
| action steps | 2,160 |
| search / total frames | 44 / 46 |
| distinct image hashes | 46 / 46 (100%) |
| preliminary / final selected frame | 30 / 43 |
| selected validity probability | 0.9916666666666667 |
| selected predicted quality | 0.5528182778921392 |
| supporting hashes / viewpoints | 3 / 3 |
| fresh confirmations | 2 / 2 (100%) |
| return error | 0.237496 mm / 0.006024 degrees |
| evaluator position error | 4.085996 mm |
| evaluator orientation error | 9.99523052950496 degrees |
| margin below orientation gate | **0.004769470495040284 degrees** |
| L4 visible fraction | 0.25703333333333334 |
| active / temporal / regression tests | 12 / 24 / 34 passed; 70 / 70 total |
| elapsed / peak RSS | 2:40 / 6,730,128 KiB |
| peak GPU utilization / memory | 57% / 7,317 MiB |

**Verdict: pass**, because every frozen development criterion was met; the orientation result is
only 0.004769470495040284 degrees below its gate and therefore does not demonstrate a robust
operating margin.

### 1.4 Reading it

- The angular continuation used image-model outputs and executed motion; evaluator target
  orientation remained unavailable until actor sealing.
- The selected refinement frame was fresh and supported by three distinct hashes and viewpoints.
- The final geometry is a valid numerical development pass on one opened anatomy.
- The orientation margin is smaller than a hundredth of a degree, so no independent or
  generalization claim follows.
- V16, V17, and V18 remain `CONFIRMATION_NO_GO`.
- Job `51662670` is preserved as a technical failure: refinement frames were acquired, but the
  replay subset retained global instead of contiguous replay-local sequence indices.
- V6R1 changed only replay-local indexing; it preserved global frame identity and every scientific,
  motion, perception, and evaluator parameter.

### 1.5 Files

| file | contents |
|---|---|
| `experiments/unity_two_stage_transfer/stage2_orientation_refinement_v6r1_result.json` | concise machine-readable result |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_orientation_refinement_v6r1/51664534/` | raw run, logs, actor seal, evaluator result, telemetry, and robot video |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_orientation_refinement_v6/51662670/` | preserved V6 technical failure |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/demo/sonogym_stage2_orientation_refinement_v6r1_20261116.zip` | review package with two videos and provenance; SHA-256 `1380a7790ab296e9e68e2a1c31a6ebcd3dbad4ae87c601922389523b0fa5793f` |
