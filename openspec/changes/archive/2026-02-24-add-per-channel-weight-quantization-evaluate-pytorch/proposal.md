## Why

Weight PTQ in PyTorch evaluation currently uses per-tensor quantization only. We need a CLI switch to opt into per-channel weight quantization so users can compare accuracy and behavior without exporting separate artifacts.

## What Changes

- Add a `--pre-channel` CLI flag in `scripts/evaluate_pytorch.py` to enable per-channel weight quantization mode.
- Pass the flag through to `PyTorchAdapter` and surface the selected mode in run logs.
- Implement per-channel symmetric weight fake quantization in `resnet8/evaluation/adapters.py`.
- Include quantization metadata indicating per-tensor vs per-channel behavior.

## Capabilities

### Modified Capabilities
- `evaluation-workflow-standardization`: PyTorch evaluation can run weight PTQ in per-channel mode when requested from CLI.

## Impact

- Affected code:
  - `scripts/evaluate_pytorch.py`
  - `resnet8/evaluation/adapters.py`
  - `tests/test_evaluation_adapters.py`
- No new external dependencies.
