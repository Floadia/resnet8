## Context

Current evaluation code is split across `scripts/evaluate.py` (ONNX) and `scripts/evaluate_pytorch.py` (PyTorch), with repeated CIFAR-10 loading, accuracy computation, and result printing logic. Weight visualization logic in `playground/weight_visualizer.py` mixes UI, model loading, tensor extraction, quantization interpretation, and plotting into one notebook flow, which makes behavior hard to test and hard to reuse.

This change targets cross-cutting refactoring across scripts and playground utilities without changing the core ResNet8 model architecture. Constraints:
- Keep existing toolchain (PyTorch, ONNX Runtime, marimo, Plotly).
- Preserve ability to run existing entry points during migration.
- Ensure evaluation and visualization outputs remain interpretable and comparable across runs.

## Goals / Non-Goals

**Goals:**
- Provide a single evaluation workflow contract shared by ONNX and PyTorch backends.
- Remove duplicated preprocessing/metric/output logic from evaluation scripts.
- Separate weight visualization core logic from marimo UI concerns.
- Standardize output schemas (terminal + optional machine-readable output) for reproducibility.
- Add targeted tests to guard metric regressions and tensor parsing regressions.

**Non-Goals:**
- Re-training or changing ResNet8 architecture.
- Introducing new model formats beyond ONNX/PT in this change.
- Rewriting the marimo notebook UI/UX from scratch.
- Large data pipeline changes outside evaluation/visualization scope.

## Decisions

### 1) Introduce shared evaluation core module
Decision: Create a reusable evaluation core (e.g., `resnet8/evaluation/`) for data loading, prediction-to-metrics conversion, and report generation.

Rationale:
- Eliminates duplicated logic in `scripts/evaluate.py` and `scripts/evaluate_pytorch.py`.
- Guarantees preprocessing and metric parity across runtimes.
- Improves testability of pure functions (loading, metrics, formatting).

Alternatives considered:
- Keep scripts independent and synchronize manually: rejected due to drift risk.
- Merge both scripts into a single monolithic script: rejected due to backend-specific complexity and lower maintainability.

### 2) Use backend adapter interface for inference
Decision: Define a small inference adapter contract (e.g., `predict(images) -> logits`) with separate ONNX and PyTorch implementations.

Rationale:
- Backend differences are isolated to loading/session details.
- Evaluation pipeline remains backend-agnostic.
- Future formats can be added without touching metric/report code.

Alternatives considered:
- Branch backend logic inline in one function: rejected because it couples runtime-specific concerns with generic evaluation flow.

### 3) Standardize evaluation output schema
Decision: Emit a canonical evaluation result structure (overall accuracy, per-class stats, metadata such as backend/model path), with stable key names.

Rationale:
- Enables consistent CLI output and optional JSON export.
- Makes downstream automation and comparisons deterministic.

Alternatives considered:
- Keep print-only free-form output: rejected because it is hard to diff and consume programmatically.

### 4) Split weight visualization into core extraction + presentation
Decision: Move model/tensor extraction and quantization metadata parsing into reusable utility modules; keep notebook cells mostly for interaction and rendering.

Rationale:
- Reduces cognitive load in `playground/weight_visualizer.py`.
- Allows unit testing tensor extraction for ONNX and Torch/PTQ models without UI runtime.
- Enables shared tensor metadata schema consumed by plots and potential CLI tools.

Alternatives considered:
- Keep all logic inside notebook: rejected due to poor testability and change risk.

### 5) Preserve CLI compatibility with additive flags
Decision: Keep current script entry points and primary arguments, adding optional flags only where needed (for example structured output path).

Rationale:
- Minimizes disruption for existing local workflows/docs.
- Supports incremental migration while validating parity.

Alternatives considered:
- Replace existing scripts with entirely new commands: rejected because it increases migration burden for immediate refactor value.

## Risks / Trade-offs

- [Risk] Hidden preprocessing mismatch between current scripts and shared module during migration. -> Mitigation: parity tests against known model/data fixtures and side-by-side output checks.
- [Risk] Refactoring notebook logic can break interactive controls. -> Mitigation: keep UI structure stable and move logic behind compatibility wrappers first.
- [Risk] Quantized tensor extraction edge cases (packed params/per-channel stats) regress. -> Mitigation: add fixture-based tests for representative quantized and FP32 models.
- [Risk] Added module boundaries increase short-term complexity. -> Mitigation: define minimal interfaces and document ownership of each module.

## Migration Plan

1. Introduce shared evaluation primitives (data loading, metrics, report schema) and unit tests.
2. Implement ONNX and PyTorch inference adapters and refit existing scripts to call shared core.
3. Validate output parity against current scripts on the same CIFAR-10 test set.
4. Extract weight/tensor parsing from notebook into reusable utilities with tests.
5. Update notebook to consume extracted utilities while preserving current interaction pattern.
6. Update README/playground docs with standardized commands and output expectations.

Rollback strategy:
- Keep legacy script paths and behavior behind lightweight wrappers until parity is validated.
- If regressions appear, temporarily route wrappers back to previous inline logic while keeping new modules isolated.

## Open Questions

- Should evaluation JSON output be enabled by default or behind an explicit `--output-json` flag?
- Do we need strict backward compatibility for exact console text formatting, or only for reported metrics/values?
- Should weight visualization support exporting summary stats to file in this change, or defer to a follow-up capability?
