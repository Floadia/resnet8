## Why

PyTorch evaluation supports weight and activation PTQ, but it does not expose the
paper-driven empirical bias correction flow from Data-Free Quantization
(arXiv:1906.04721). Adding an explicit option enables reproducible A/B
comparisons for low-bit weight PTQ and closes a practical gap in evaluation
workflows.

## What Changes

- Add a `--dfq-bias-corr` option to `scripts/evaluate_pytorch.py` for empirical
  DFQ weight-bias correction.
- Validate CLI contracts for DFQ mode (`--wq` required, calibration data loaded
  from `<data-dir>/test_batch`).
- Extend `PyTorchAdapter` to run empirical layer-wise bias correction after
  weight PTQ, using calibration images.
- Surface DFQ correction status in quantization summary metadata.
- Add adapter tests for DFQ option contract and correction effect.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `evaluation-workflow-standardization`: PyTorch PTQ evaluation supports an
  opt-in DFQ empirical bias-correction mode and preserves deterministic
  reporting behavior.

## Impact

- Affected code:
  - `scripts/evaluate_pytorch.py`
  - `resnet8/evaluation/adapters.py`
  - `tests/test_evaluation_adapters.py`
- No new external dependencies.
- New evaluation outputs for this change:
  - `logs/acc_w5_no_dfq_bias_corr.json`
  - `logs/acc_w5_dfq_bias_corr.json`
