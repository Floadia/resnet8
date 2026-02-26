## Context

The existing PyTorch evaluation flow already supports simulated PTQ for weights
and activations, including calibration-aware activation quantization. This
change adds a paper-driven option from arXiv:1906.04721 (empirical bias
correction) after weight quantization to improve low-bit weight PTQ behavior
without changing default execution.

## Goals / Non-Goals

**Goals:**

- Add an explicit CLI switch for DFQ empirical bias correction.
- Keep default evaluation behavior unchanged when the option is not enabled.
- Implement correction in adapter logic with deterministic, testable behavior.
- Preserve shared evaluation/reporting pipeline semantics.

**Non-Goals:**

- Implement full DFQ pipeline (e.g., cross-layer equalization, analytic bias
  correction).
- Change ONNX evaluation workflows.
- Introduce new model export formats.

## Decisions

- Decision: Implement an opt-in CLI flag `--dfq-bias-corr`.
  Rationale: Keeps backward compatibility and enables direct A/B measurement.
  Alternative considered: always-on correction; rejected due to behavioral
  change risk.

- Decision: Require `--wq` when DFQ is enabled and load calibration images from
  `test_batch`.
  Rationale: DFQ correction targets weight quantization error and needs sample
  data for empirical mean estimates.
  Alternative considered: allow DFQ with activation-only quantization;
  rejected because method intent is weight-error compensation.

- Decision: Use sequential layer-by-layer empirical correction order.
  Rationale: Matches the paper's empirical workflow where each layer is
  corrected after prior layers are corrected.
  Alternative considered: one-pass correction for all layers; rejected after
  observed instability in evaluation accuracy.

- Decision: Reflect DFQ correction in quantization summary metadata by
  appending `+dfq-bias-corr` to the weight scheme and marking corrected rows as
  calibrated.
  Rationale: Makes execution mode auditable from standard CLI output.

## Risks / Trade-offs

- [Risk] Calibration data representativeness can affect correction quality.
  -> Mitigation: keep correction opt-in and retain baseline mode for A/B.

- [Risk] Additional calibration pass increases runtime.
  -> Mitigation: bounded by test-batch size and only runs when flag is enabled.

- [Risk] Method is partial DFQ (empirical correction only).
  -> Mitigation: document scope clearly in CLI help and change artifacts.
