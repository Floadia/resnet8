## Why

The repository currently mixes multiple experiment scripts, inconsistent entry points, and overlapping responsibilities, making evaluation results hard to trust and weight analysis hard to repeat. We should refactor now to establish a clear, testable workflow focused on model evaluation and weight visualization.

## What Changes

- Consolidate and standardize evaluation workflows (ONNX and PyTorch) around consistent data loading, preprocessing, metrics, and output format.
- Refactor evaluation-related scripts into clearer module boundaries to reduce duplicated logic and hidden coupling.
- Standardize weight visualization workflows and interfaces so layer/tensor inspection is consistent and reproducible.
- Define explicit CLI contracts and output artifacts for both evaluation and weight visualization paths.
- Add focused validation checks and documentation updates for the refactored workflows.

## Capabilities

### New Capabilities
- `evaluation-workflow-standardization`: Unified evaluation behavior, metric reporting, and CLI/output conventions across evaluation entry points.
- `weight-visualization-standardization`: Consistent weight visualization behavior for loading models, selecting tensors/layers, and presenting distribution/statistics outputs.

### Modified Capabilities
- None.

## Impact

- Affected code:
  - `scripts/evaluate.py`
  - `scripts/evaluate_pytorch.py`
  - `scripts/calibration_utils.py` (shared preprocessing/alignment impact)
  - `playground/weight_visualizer.py`
  - `playground/utils/model_loader.py`
  - `playground/utils/layer_inspector.py`
- Affected docs:
  - `README.md`
  - `playground/README.md`
  - related spec/docs for evaluation and visualization usage.
- APIs/interfaces:
  - CLI arguments and output schema for evaluation and visualization scripts (maintained or explicitly documented if changed).
- Dependencies/systems:
  - No new external runtime dependency is required; existing PyTorch/ONNX/Plotly tooling remains in use.
