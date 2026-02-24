## Why

PTQ simulation in `scripts/evaluate_pytorch.py` currently derives quantization scale from each tensor at inference time. This does not reflect calibration-driven deployment behavior where scales are fixed from representative data.

## What Changes

- Add a `--calib` option to `scripts/evaluate_pytorch.py` to enable calibration-driven PTQ for evaluation.
- Calibrate quantization parameters from CIFAR-10 `test_batch` (`<data-dir>/test_batch`) before inference.
- Keep existing behavior unchanged when `--calib` is not set.

## Capabilities

### Modified Capabilities
- `evaluation-workflow-standardization`: PyTorch evaluation supports optional calibration-driven PTQ parameter derivation using CIFAR-10 `test_batch`.

## Impact

- Affected code:
  - `scripts/evaluate_pytorch.py`
  - `resnet8/evaluation/adapters.py`
  - `tests/` (new calibration behavior coverage)
- No new external dependencies.
