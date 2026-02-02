# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-28)

**Core value:** Accurate model conversion across frameworks — converted models must produce equivalent results to the original Keras model (>85% accuracy on CIFAR-10)
**Current focus:** v1.3 Quantized Operations Documentation

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-02-02 — Milestone v1.3 started

Progress: [░░░░░░░░░░] 0% (0/? phases complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 8
- Average duration: 4.25min (v1.2 tracking started)
- Total execution time: 17min (v1.2 only)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Model Conversion | 1/1 | - | - |
| 2. Accuracy Evaluation | 1/1 | - | - |
| 3. PyTorch Conversion | 1/1 | - | - |
| 4. PyTorch Evaluation | 1/1 | - | - |
| 5. Calibration Infrastructure | 1/1 | 2min | 2min |
| 6. ONNX Runtime Quantization | 1/1 | 6min | 6min |
| 7. PyTorch Quantization | 1/1 | 6min | 6min |
| 8. Comparison Analysis | 1/1 | 3min | 3min |

**Recent Trend:**
- Last plan: 08-01 (3min)
- Analysis/documentation phases faster than implementation phases

## Accumulated Context

### Decisions

Recent decisions from PROJECT.md Key Decisions table:
- ONNX as intermediate format: Standard interchange format (Good)
- tf2onnx for Keras→ONNX: Standard tool, well-maintained (Good)
- onnx2torch for ONNX→PyTorch: Leverage existing ONNX model (Good - works with FX mode)
- Separate converter/eval scripts: Reusability and clarity (Good)
- Raw pixel values (0-255): Match Keras training preprocessing (Good)

From Phase 5 (Calibration Infrastructure):
- 1000 calibration samples (100 per class): Exceeds roadmap minimum (200) for better quantization quality
- Load from training batches: Prevents data leakage, maintains evaluation integrity
- Preprocessing matches evaluate.py exactly: Critical to avoid 10-40% accuracy drops from mismatches

From Phase 6 (ONNX Runtime Quantization):
- QDQ format for CPU inference: Recommended by ONNX Runtime for better tool support
- MinMax calibration method: Simpler and faster than Entropy/Percentile, good baseline
- Uint8 outperforms Int8: 86.75% vs 85.58% accuracy (both above 85% threshold)
- Per-channel quantization disabled: Start with per-tensor for initial validation
- Fresh CalibrationDataReader per quantization: Iterator consumed after first use

From Phase 7 (PyTorch Quantization):
- FX graph mode for onnx2torch models: Eager mode doesn't support custom ONNX ops
- JIT tracing for serialization: FX GraphModule has pickle issues, TorchScript works
- fbgemm uint8 limitation: PyTorch requires qint8 weights, uint8-only not supported
- PyTorch int8 matches ONNX Runtime int8: 85.68% vs 85.58% accuracy

From Phase 8 (Comparison Analysis):
- ONNX Runtime uint8 recommended for best accuracy retention (86.75%, -0.44% drop)
- All quantized models meet >85% accuracy and <5% drop requirements
- PyTorch uint8 documented as not supported (fbgemm limitation)

### Pending Todos

None - project complete.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Use uv instead of old python management system | 2026-01-28 | 62cd2a0 | [001-use-uv-instead-of-old-python-management-](./quick/001-use-uv-instead-of-old-python-management-/) |
| 002 | Add comprehensive README.md | 2026-01-28 | 8ca7d31 | [002-add-readme](./quick/002-add-readme/) |
| 003 | Add CI that lint by ruff | 2026-01-28 | cc40f18 | [003-add-ci-that-lint-by-ruff](./quick/003-add-ci-that-lint-by-ruff/) |

### Blockers/Concerns

**All risks resolved:**
- ✓ Calibration data quality: RESOLVED - Stratified sampling with 1000 samples (100 per class)
- ✓ Preprocessing mismatches: RESOLVED - Verified identical to evaluate.py (float32, 0-255, NHWC)
- ✓ ONNX Runtime quantization: RESOLVED - Both int8/uint8 achieve >85% accuracy (85.58%/86.75%)
- ✓ PyTorch FX mode: RESOLVED - FX graph mode works with onnx2torch models
- ✓ PyTorch serialization: RESOLVED - JIT tracing enables model serialization
- ✓ PyTorch uint8: DOCUMENTED - fbgemm requires qint8 weights, uint8-only not possible
- ✓ Comparison analysis: COMPLETE - Analysis document with recommendations

## Final Results Summary

**v1.2 PTQ Evaluation milestone complete.**

| Model | Accuracy | Delta | Size | Reduction |
|-------|----------|-------|------|-----------|
| FP32 baseline | 87.19% | - | 315KB | - |
| ONNX Runtime uint8 | 86.75% | -0.44% | 123KB | 61% |
| ONNX Runtime int8 | 85.58% | -1.61% | 123KB | 61% |
| PyTorch int8 | 85.68% | -1.51% | 165KB | 52% |
| PyTorch uint8 | N/A | N/A | N/A | Not supported |

**Recommendation:** ONNX Runtime uint8 for best accuracy-to-size ratio.

## Session Continuity

Last session: 2026-01-28
Stopped at: Completed quick task 003 (Add CI linting with ruff)
Resume file: None

**Project deliverables:**
- README.md - Comprehensive project documentation with usage examples
- docs/QUANTIZATION_ANALYSIS.md - Complete PTQ evaluation analysis
- models/resnet8_int8.onnx - ONNX int8 quantized model
- models/resnet8_uint8.onnx - ONNX uint8 quantized model
- models/resnet8_int8.pt - PyTorch int8 quantized model

---
*State initialized: 2026-01-27*
*Last updated: 2026-01-28 after completing Phase 8 (v1.2 milestone complete)*
