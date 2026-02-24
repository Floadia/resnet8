## Why

We need to compare accuracy sensitivity to activation quantization levels during PyTorch evaluation without requiring separate pre-quantized model artifacts for each bit setting.

## What Changes

- Add `--wq` and `--aq` options to `scripts/evaluate_pytorch.py`.
- Enable PTQ simulation during evaluation by applying configurable fake quantization to weights and activations.
- Keep existing evaluation workflow behavior unchanged when quantization bits are not specified.

## Capabilities

### Modified Capabilities
- `evaluation-workflow-standardization`: PyTorch evaluation accepts configurable PTQ bit-widths for weight and activation during inference-time evaluation.

## Impact

- Affected code:
  - `scripts/evaluate_pytorch.py`
  - `resnet8/evaluation/adapters.py`
- No new external dependencies.
