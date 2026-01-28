# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-28)

**Core value:** Accurate model conversion across frameworks — converted models must produce equivalent results to the original Keras model (>85% accuracy on CIFAR-10)
**Current focus:** Phase 6 - ONNX Runtime Quantization (v1.2 PTQ Evaluation)

## Current Position

Phase: 6 of 8 (ONNX Runtime Quantization)
Plan: 1 of 1 in current phase
Status: Phase complete
Last activity: 2026-01-28 — Completed 06-01-PLAN.md (ONNX static quantization)

Progress: [██████░░░░] 75.0% (6/8 phases complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 6
- Average duration: 4min (v1.2 tracking started)
- Total execution time: 8min (v1.2 only)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Model Conversion | 1/1 | - | - |
| 2. Accuracy Evaluation | 1/1 | - | - |
| 3. PyTorch Conversion | 1/1 | - | - |
| 4. PyTorch Evaluation | 1/1 | - | - |
| 5. Calibration Infrastructure | 1/1 | 2min | 2min |
| 6. ONNX Runtime Quantization | 1/1 | 6min | 6min |

**Recent Trend:**
- Last plan: 06-01 (6min)
- Trend: Quantization phase took 3x longer than calibration (dependency installation)

## Accumulated Context

### Decisions

Recent decisions from PROJECT.md Key Decisions table:
- ONNX as intermediate format: Standard interchange format (Good)
- tf2onnx for Keras→ONNX: Standard tool, well-maintained (Good)
- onnx2torch for ONNX→PyTorch: Leverage existing ONNX model (Pending v1.2 validation)
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

### Pending Todos

None yet.

### Blockers/Concerns

**v1.2 PTQ Evaluation risks (from research):**
- ✓ Calibration data quality: RESOLVED - Stratified sampling with 1000 samples (100 per class)
- ✓ Preprocessing mismatches: RESOLVED - Verified identical to evaluate.py (float32, 0-255, NHWC)
- ✓ ONNX Runtime quantization: RESOLVED - Both int8/uint8 achieve >85% accuracy (85.58%/86.75%)
- PyTorch PT2E export: onnx2torch-converted model compatibility unclear (Medium risk - will test in Phase 7)
- PyTorch data format: PT2E may require NCHW vs NHWC (Low risk - can transpose)

## Session Continuity

Last session: 2026-01-28
Stopped at: Completed Phase 6 (ONNX Runtime Quantization)
Resume file: None

**Next step:** Plan Phase 7 (PyTorch Quantization) via `/gsd:plan-phase 7`

---
*State initialized: 2026-01-27*
*Last updated: 2026-01-28 after completing Phase 6*
