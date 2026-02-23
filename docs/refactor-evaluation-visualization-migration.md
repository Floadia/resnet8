# Evaluation + Weight Visualization Refactor Migration Note

## Scope

This migration refactors repository internals for two workflows:

- Evaluation (`scripts/evaluate.py`, `scripts/evaluate_pytorch.py`)
- Weight visualization (`playground/weight_visualizer.py`)

Behavioral goals are parity-focused: preserve existing user-facing entry points
while moving duplicated logic into shared modules.

## New Module Boundaries

### Evaluation

- Shared core: `resnet8/evaluation/`
  - `cifar10.py`: CIFAR-10 test loading + preprocessing source of truth
  - `metrics.py`: overall/per-class metric computation
  - `report.py`: deterministic output schema + text/json formatting
  - `adapters.py`: ONNX Runtime and PyTorch inference adapters
  - `pipeline.py`: backend-agnostic evaluation flow
- CLI wrappers:
  - `scripts/evaluate.py` (ONNX)
  - `scripts/evaluate_pytorch.py` (PyTorch)

### Weight visualization

- Shared extraction layer:
  - `playground/utils/tensor_schema.py`
  - `playground/utils/tensor_extractors.py`
- Presentation layer:
  - `playground/weight_visualizer.py` (Marimo UI cells)

## Compatibility

- Preserved:
  - Existing script paths and core flags (`--model`, `--data-dir`)
  - Marimo notebook interaction flow (model/layer/tensor selection, histogram)
- Added:
  - Evaluation `--max-samples`
  - Evaluation `--output-json`

## Rollback Approach

If regressions are detected:

1. Keep script entry points intact and revert only shared module wiring.
2. Temporarily restore script-local evaluation logic in wrappers.
3. Keep extractor modules isolated; notebook can be pointed back to in-cell
   extraction while issues are fixed.

Rollback is low risk because command interfaces are unchanged and module
boundaries are additive.
