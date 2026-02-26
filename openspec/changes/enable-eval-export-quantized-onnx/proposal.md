## Why

PTQ evaluation in `scripts/evaluate_pytorch.py` can simulate quantization with
`--wq` and `--aq`, but there is no way to export that eval-time quantized graph
as ONNX for downstream runtime validation.

## What Changes

- Add an ONNX export option to `scripts/evaluate_pytorch.py` that writes the
  eval-time quantized graph to a user-provided `.onnx` path.
- Add optional parity validation that evaluates the exported ONNX file using
  the shared evaluation pipeline and compares accuracy against the PyTorch eval
  score.
- Keep default evaluation behavior unchanged when export is not requested.

## Capabilities

### Modified Capabilities
- `evaluation-workflow-standardization`: PyTorch PTQ evaluation can export an
  eval-quantized ONNX graph and validate ONNX score similarity.

## Impact

- Affected code:
  - `scripts/evaluate_pytorch.py`
  - `resnet8/evaluation/adapters.py`
  - `tests/test_evaluation_adapters.py`
- No new external dependencies.
