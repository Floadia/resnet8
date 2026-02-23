## 1. Evaluation Core Refactor

- [x] 1.1 Create a shared evaluation module for CIFAR-10 loading, shared preprocessing, and accuracy computation.
- [x] 1.2 Define a backend adapter interface and implement ONNX inference adapter.
- [x] 1.3 Implement PyTorch inference adapter and route both evaluation scripts through the shared pipeline.
- [x] 1.4 Add a canonical evaluation report schema and support optional machine-readable output.

## 2. Weight Visualization Refactor

- [x] 2.1 Extract ONNX tensor parsing and quantization metadata handling into reusable utilities.
- [x] 2.2 Extract PyTorch tensor parsing (including quantized packed params) into reusable utilities.
- [x] 2.3 Define a normalized tensor metadata schema consumed by visualizer rendering.
- [x] 2.4 Update `playground/weight_visualizer.py` to use shared extraction utilities while preserving current UI flow.

## 3. Validation and Regression Safety

- [x] 3.1 Add unit tests for shared evaluation functions (preprocessing, overall/per-class metrics, schema fields).
- [x] 3.2 Add parity checks to confirm ONNX and PyTorch evaluation paths use the same preprocessing and metric logic.
- [x] 3.3 Add fixture-based tests for weight tensor extraction across FP32 and quantized models.
- [x] 3.4 Run evaluation and visualization smoke checks and record baseline results for comparison.

## 4. Documentation and Migration

- [x] 4.1 Update `README.md` with standardized evaluation usage and output expectations.
- [x] 4.2 Update `playground/README.md` for weight visualization workflow and supported model behavior.
- [x] 4.3 Document compatibility notes for existing CLI entry points and any additive flags.
- [x] 4.4 Add a short migration note summarizing new module boundaries and rollback approach.
