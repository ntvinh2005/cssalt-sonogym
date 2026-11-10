# Phase 5A retry record

## Failed execution

- Slurm job: `50641608`
- Account/resources: `anest-dept`, one L4, four CPUs, 30 GB RAM
- Final state: `CANCELLED`, elapsed `00:20:44`
- Preserved output:
  `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/navigation_metric_validation/50641608/`
- Preserved Slurm log:
  `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/cssalt-sonogym-navigation_metrics-50641608.out`

The job completed the first four-start batch (1,200 recorded evaluation
transitions). The compressed step artifact stopped changing at
2026-11-10 11:12:57 EST. For more than 18 subsequent minutes, the evaluator's
main thread remained runnable at approximately 98--99% of one CPU, GPU memory
remained approximately 3.7 GiB, GPU utilization samples remained zero, and no
new step record or fatal error was emitted. This localized the failure to the
second direct wrapper reset after ending an episode at the protocol's 300-step
horizon; it was not an OOM, scheduler, asset, or GPU-allocation failure.

## Justified fix for the sole retry

The frozen starts, order, metrics, thresholds, controller seeds, 300-step
horizon, checkpoint, action semantics, reward, dynamics, and METRIC_GO gate are
unchanged. The evaluator now uses an explicit reset only for the first batch.
Between evaluated episodes it sets the native episode buffer to the already
implemented time limit, executes one zero-action transition, verifies a
synchronous native truncation, and discards that transition before recording
the next episode. This is the same native auto-reset path validated in the
Phase 4 smoke test. Batch-level flushed progress markers were added for
fail-fast diagnosis.

No measurement from job `50641608` will be combined with retry results.
