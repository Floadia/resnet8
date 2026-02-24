## Why

Current PyTorch PTQ simulation quantizes activations with symmetric quantization only. For realistic PTQ comparison, we need activation asymmetric quantization (with zero-point), while keeping weight quantization symmetric.

## What Changes

- Add activation quantization scheme selection to PyTorch evaluation CLI.
- Implement asymmetric activation fake quantization in the PyTorch adapter.
- Support both dynamic and calibration-derived activation quant parameters for asymmetric mode.

## Capabilities

### Modified Capabilities
- `evaluation-workflow-standardization`: PyTorch evaluation supports selecting activation quantization scheme (`symmetric` or `asymmetric`) while preserving existing reporting pipeline.

## Impact

- Affected code:
  - `scripts/evaluate_pytorch.py`
  - `resnet8/evaluation/adapters.py`
  - `tests/test_evaluation_adapters.py`
- Backward compatibility:
  - Default behavior remains symmetric activation quantization.
