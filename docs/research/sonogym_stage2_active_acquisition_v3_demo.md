# SonoGym Stage 2 continuous active-acquisition demo

## Result

**Development pilot V3 completed the fixed-`s0010` sequence coarse search → fine refinement →
micro refinement → multi-view candidate selection → physical return → two fresh confirmations →
independent evaluator, and received `DEVELOPMENT_PASS`.**

This establishes an engineering demo of the transferred Unity Stage 2 decision philosophy on one
fixed SonoGym anatomy. It is not V19, not an untouched confirmation set, and not evidence of
patient generalization, clinical validity, or MBB performance.

| component | result | verdict |
|---|---:|---|
| actor search | 39 observations; 3 supporting hashes/viewpoints | PASS |
| selected candidate | probability 0.92222; quality 0.46716 | recorded |
| confirmation 1 | 61 outbound + 44 return actions; producer 477; new hash | PASS |
| confirmation 2 | 32 outbound + 64 return actions; producer 573; new hash | PASS |
| return to best | 0.1432 mm; 0.0173 degrees | PASS |
| evaluator-only final geometry | 8.0565 mm; 9.5173 degrees | PASS at 10 mm / 10 degrees |
| evaluator-only L4 visibility | 0.25453 | PASS at 0.20 |
| actor outcome | `ACCEPT` | PASS |
| fixed-anatomy closed-loop development gate | `DEVELOPMENT_PASS` | PASS |

The evaluator values were read only after the actor result was sealed. The actor received clean
ultrasound, frozen image-only validity/quality predictions, executed relative motion, and its own
causal candidate memory; it did not receive L4 coordinates, segmentation labels, geometric error,
or visibility score.

## Frozen execution

| item | identity |
|---|---|
| Slurm | job `51611658`; `anest-dept`; `hpg-turin`; node `c0603a-s18` |
| resources | one NVIDIA L4; four CPUs; 30 GB requested RAM; 15-minute bound |
| execution commit | `40103ddd7df2c6f218077b963cbdbe6e5c9726ce` |
| implementation commit | `4f685ee14a79c6aa6bf6009be88b2156db896e5a` |
| protocol | SHA-256 `02934f11cf6544c534066046e6b3cfbf226df1acd99c4da3ad5f6a7ec5933d4a` |
| actor result | SHA-256 `bbba48fb1831e3a490c772cdff5aa98f91942efe26221ebf89ef90498fe0adb3` |
| evaluator report | SHA-256 `98728646d536bad9718cdb61cba5b20efd6d25df35f49c5a4a447605990c9ce4` |

The job completed `0:0` in 50 seconds. Actor wall time was 41.84 seconds with 3,655,844 KiB
peak RSS. L4 telemetry reached 5,005 MiB and 55% utilization. The run contains 572 physical action
steps, no position clamp, no termination/truncation, and 75 declared angle-ceiling-clamped steps.
All 66 tests passed: 9 active-pilot, 23 temporal-decision, and 34 existing Stage 2 regression
tests.

## Recovery history

V1 job `51607477` correctly returned `DEVELOPMENT_NO_GO`: its 0.75 mm confirmation dither was
smaller than the shared 5 mm movement tolerance, so no fresh frame was produced. V2 job
`51611052` demonstrated that physical confirmation worked but stopped on a technical total-action
budget: measured execution required 552 actions while the frozen limit was 550. V3 changed only
the maximum total action budget to 600 and repaired evaluator import/summary guards. It retained
the handoff, search poses, 0.75 mm dither, 0.25 mm confirmation tolerance, model, threshold,
decision rule, seed, and evaluator gates.

V16, V17, and V18 remain `CONFIRMATION_NO_GO`; the opened temporal replay remains
`EXPLORATORY_NO_GO`. V3 does not retroactively change those results.

## Demo

The 12.2-second MP4 shows the original simulated ultrasound actor input beside the causal
Home-relative search trajectory. It labels coarse/fine/micro phases, model scores, candidate
selection, return movements, producer advances, and the two fresh confirmations. Evaluator-only
geometry appears only on the final frozen actor frame.

| artifact | path |
|---|---|
| video | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/demo/stage2_active_pilot_v3_51611658/sonogym-stage2-active-acquisition-demo.mp4` |
| final preview | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/demo/stage2_active_pilot_v3_51611658/sonogym-stage2-active-acquisition-final.png` |
| selected candidate | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/demo/stage2_active_pilot_v3_51611658/sonogym-stage2-selected-candidate.png` |
| manifest | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/demo/stage2_active_pilot_v3_51611658/demo_manifest.json` |

Video SHA-256 is
`d5b6e628d1d475447e0ded7d9b3986fb1766eb2f018019bebf18a9db522aba1e`. The video is a
diagnostic rendering of observations and actor state; nonzero or visually plausible ultrasound
pixels do not independently prove anatomical correctness.

## Evidence boundary and next science

This is now demo-ready for the fixed-`s0010` engineering proof of concept. The next scientific
step is not another development retry. It is to freeze a new, separately reviewed confirmation
protocol using previously unused start states or sessions, preserving the V3 actor contract and
reporting acceptance, abstention, wrong-stop, geometry, image validity, movement count, and
resource cost. Multiple independent anatomies are still required before any generalization claim.
