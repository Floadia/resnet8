## 1. CLI and Export Wiring

- [x] 1.1 Add `--export-onnx` option to `scripts/evaluate_pytorch.py` and route
      export through `PyTorchAdapter`.
- [x] 1.2 Add `--verify-exported-onnx` and `--onnx-score-tol` options and wire
      ONNX re-evaluation through the shared evaluation pipeline.

## 2. Adapter Export Implementation

- [x] 2.1 Add `PyTorchAdapter.export_onnx(...)` for eval-time model export with
      configurable opset and dynamic batch axis.
- [x] 2.2 Ensure activation PTQ behavior is represented during export instead of
      being lost with hook-only execution.

## 3. Verification

- [x] 3.1 Add focused tests for export argument validation and ONNX file
      generation path.
- [x] 3.2 Run DoD command with `--wq=6 --aq=7`, export ONNX, evaluate exported
      ONNX, and confirm similar score.
