## Why

ResNet8 execution currently requires manual device handling. We want execution
to automatically use CUDA when available, while preserving CPU behavior when no
GPU is present.

## What Changes

- Add device auto-selection logic for ResNet8 run paths: prefer CUDA when
  available, otherwise use CPU.
- Ensure model and input tensors are moved to the selected device consistently.
- Keep existing CLI behavior backward compatible, with no required new flags.

## Capabilities

### Modified Capabilities
- `evaluation-workflow-standardization`: PyTorch-based ResNet8 execution
  automatically selects CUDA when available and falls back to CPU otherwise.

## Impact

- Affected specs:
  - `evaluation-workflow-standardization`
- Affected code:
  - `scripts/evaluate_pytorch.py`
  - `resnet8/evaluation/adapters.py`
  - Tests covering PyTorch evaluation device handling
- No new external dependencies.
