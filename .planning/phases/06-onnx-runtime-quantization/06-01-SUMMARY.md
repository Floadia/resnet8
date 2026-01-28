---
phase: 06-onnx-runtime-quantization
plan: 01
subsystem: quantization
tags: [onnx, onnxruntime, ptq, int8, uint8, quantization, calibration]

# Dependency graph
requires:
  - phase: 05-calibration-infrastructure
    provides: Stratified calibration data loader with 1000 samples matching evaluate.py preprocessing
  - phase: 01-model-conversion
    provides: ResNet8 ONNX model at models/resnet8.onnx
provides:
  - Int8 quantized ONNX model (resnet8_int8.onnx) with 85.58% accuracy
  - Uint8 quantized ONNX model (resnet8_uint8.onnx) with 86.75% accuracy
  - ONNX quantization script (quantize_onnx.py) with CalibrationDataReader
  - MinMax calibration methodology validated on ResNet8
affects: [07-pytorch-quantization, quantization-comparison, deployment]

# Tech tracking
tech-stack:
  added: [onnxruntime.quantization, CalibrationDataReader, quantize_static]
  patterns: [static-quantization, qdq-format, calibration-reader-pattern]

key-files:
  created: [scripts/quantize_onnx.py]
  modified: []

key-decisions:
  - "QDQ format for CPU inference (recommended by ONNX Runtime)"
  - "MinMax calibration method for simplicity and speed"
  - "Per-channel quantization disabled for initial validation"
  - "Fresh CalibrationDataReader instance per quantization call (iterator exhaustion)"

patterns-established:
  - "CalibrationDataReader wraps calibration_utils for ONNX Runtime integration"
  - "Auto-convert base ONNX model if missing before quantization"
  - "Quantize to both int8 and uint8 for comparison"

# Metrics
duration: 6min
completed: 2026-01-28
---

# Phase 6 Plan 1: ONNX Runtime Quantization Summary

**Static quantization to int8/uint8 achieving 85.58%/86.75% accuracy (vs 87.19% FP32 baseline) with MinMax calibration**

## Performance

- **Duration:** 6 min
- **Started:** 2026-01-28T03:00:50Z
- **Completed:** 2026-01-28T03:06:30Z
- **Tasks:** 2
- **Files modified:** 1 created, 2 models generated

## Accomplishments
- Created quantize_onnx.py with CalibrationDataReader wrapping calibration_utils
- Generated int8 quantized model (123K) with 85.58% accuracy (-1.61% drop)
- Generated uint8 quantized model (123K) with 86.75% accuracy (-0.44% drop)
- Validated PTQ effectiveness: Both quantized models maintain >85% accuracy threshold
- Established ONNX Runtime quantization workflow for PyTorch comparison

## Task Commits

Each task was committed atomically:

1. **Task 1: Create ONNX quantization script with CalibrationDataReader** - `4bba1d8` (feat)
2. **Task 2: Run quantization and evaluate accuracy against baseline** - `294ab4f` (feat)

## Files Created/Modified
- `scripts/quantize_onnx.py` - ONNX quantization script with CIFARCalibrationDataReader class, quantize_static calls for int8/uint8, auto-conversion of base model
- `models/resnet8_int8.onnx` - Int8 quantized model (123K, QInt8 activations/weights)
- `models/resnet8_uint8.onnx` - Uint8 quantized model (123K, QUInt8 activations/weights)

## Decisions Made

**QDQ format selection:**
- Chose QuantFormat.QDQ over QOperator for CPU inference
- Rationale: QDQ is recommended by ONNX Runtime for CPU deployment and better tool support

**MinMax calibration method:**
- Selected CalibrationMethod.MinMax for initial validation
- Rationale: Simpler and faster than Entropy/Percentile methods, good baseline for comparison
- Future consideration: Could explore Entropy method if accuracy needs improvement

**Fresh CalibrationDataReader per quantization:**
- Create new instance for each quantize_static() call
- Rationale: Iterator is consumed after first use; reusing causes empty calibration
- Implementation: Separate reader instances for int8 and uint8 quantization

**Per-channel quantization disabled:**
- Set per_channel=False for both quantizations
- Rationale: Start with simpler per-tensor quantization for initial validation
- Future consideration: Enable per-channel for potential accuracy improvement

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed missing dependencies for quantization**
- **Found during:** Task 1 (Testing quantize_onnx.py script)
- **Issue:** onnxruntime and onnx packages not installed; ModuleNotFoundError on import
- **Fix:** Created virtual environment and installed onnxruntime, onnx, numpy
- **Files modified:** venv/ (gitignored)
- **Verification:** `python scripts/quantize_onnx.py --help` runs successfully
- **Committed in:** 4bba1d8 (Task 1 commit)

**2. [Rule 3 - Blocking] Resolved numpy/tf2onnx compatibility for ONNX conversion**
- **Found during:** Task 2 (Converting base ONNX model)
- **Issue:** tf2onnx 1.8.4 incompatible with numpy 2.x (np.bool deprecated); base ONNX model needed before quantization
- **Fix:** Downgraded numpy to 1.26.4 and installed tf2onnx from GitHub main branch
- **Files modified:** venv/ (gitignored)
- **Verification:** `python scripts/convert.py` succeeded, models/resnet8.onnx created
- **Committed in:** 294ab4f (Task 2 commit - empty commit documenting generated artifacts)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both fixes necessary to unblock execution. No scope creep - standard dependency resolution.

## Issues Encountered

**Dependency compatibility challenges:**
- tf2onnx PyPI release (1.8.4) incompatible with Python 3.12 and numpy 2.x
- Resolution: Used requirements.txt specification (GitHub main branch) and numpy 1.26.4
- Impact: Minimal - conversion worked correctly after dependency resolution

**Virtual environment needed:**
- System Python externally-managed, pip install requires venv
- Resolution: Created venv/ (gitignored) and installed dependencies
- Impact: None - standard Python workflow

## Quantization Results

### Accuracy Comparison

| Model | Accuracy | Delta | Size | Type |
|-------|----------|-------|------|------|
| FP32 (baseline) | 87.19% (8719/10000) | - | 315K | Float32 |
| Int8 quantized | 85.58% (8558/10000) | -1.61% | 123K | QInt8 |
| Uint8 quantized | 86.75% (8675/10000) | -0.44% | 123K | QUInt8 |

### Analysis

**Accuracy retention:**
- Both quantized models maintain >85% accuracy threshold
- Accuracy drops within typical PTQ range (0-3%)
- Uint8 performs better than Int8 (-0.44% vs -1.61%)

**Model size:**
- 61% size reduction (315K → 123K) from quantization
- Both int8 and uint8 same size (QDQ format adds Q/DQ nodes)

**Per-class impact:**
- Most classes maintain within 3% of baseline
- "dog" class shows largest drop in Int8 (74.2% → 68.0%, -6.2%)
- Uint8 maintains dog class better (74.2% → 74.4%, +0.2%)

**Recommendation:**
- Use Uint8 model for deployment (better accuracy, same size)
- Consider per-channel quantization if accuracy needs improvement
- MinMax calibration effective for this architecture

## Next Phase Readiness

**Ready for Phase 7 (PyTorch Quantization):**
- ONNX Runtime quantization baseline established (85.58% int8, 86.75% uint8)
- Calibration infrastructure validated with 1000 samples
- MinMax calibration methodology proven effective
- Target: Compare PyTorch PT2E quantization against ONNX Runtime results

**No blockers:**
- All ONNX quantization artifacts generated successfully
- Evaluation pipeline working correctly for quantized models
- Calibration data reusable for PyTorch quantization

**Considerations for Phase 7:**
- PyTorch PT2E may require different data format (NCHW vs NHWC)
- onnx2torch-converted model compatibility with PT2E unknown (Medium risk from research)
- May need to load PyTorch model directly if onnx2torch incompatible

---
*Phase: 06-onnx-runtime-quantization*
*Completed: 2026-01-28*
