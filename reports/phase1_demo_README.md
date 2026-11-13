# SonoGym Phase 1 engineering demo

This package shows preregistered confirmation session `cf_00` from Slurm job `51196094`. The actor physically scanned forward and reverse, accumulated image-derived evidence using executed relative motion, and emitted an L4 coordinate at -75.5 mm; the evaluator-only reference was -83.36209568080575 mm, giving 7.862095680805751 mm error.

The video is an engineering demo of one successful session. The complete four-session confirmation result is FAIL because `cf_03` had 10.362095680805751 mm error; do not treat this package as confirmation PASS, patient generalization, Phase 2 readiness, or clinical evidence.

## Contents

- `sonogym-phase1-confirmation.mp4`: 211 real clean SonoGym ultrasound frames with accumulated actor evidence and a separately marked evaluator-only reference.
- `sonogym-phase1-confirmation-preview.png`: preview frame.
- `actor_results.json` and `.sha256`: sealed actor output for all four v1 sessions.
- `evaluation_report.json` and `.sha256`: independent evaluator output for all four v1 sessions.
- `stage1_confirmation_actor_protocol.yaml` and `.sha256`: frozen actor protocol.
- `stage1_confirmation_evaluation_protocol.yaml` and `.sha256`: frozen evaluator protocol.
- `stage1_confirmation_v1_result.json`: concise repository result manifest.
- `SHA256SUMS`: package-file digests.

The clean ultrasound panel is the actor input. The evaluator-only L4 marker is a post-inference overlay and was not available to actor inference.
