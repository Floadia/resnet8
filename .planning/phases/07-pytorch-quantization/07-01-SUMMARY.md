---
phase: 07-pytorch-quantization
plan: 01
subsystem: quantization
tags: [pytorch, quantization, static-quantization, fx-mode, torchscript, fbgemm, cifar-10]

# Dependency graph
requires:
  - phase: 03-pytorch-conversion
    provides: PyTorch model (resnet8.pt) converted via onnx2torch
  - phase: 05-calibration-infrastructure
    provides: Calibration data loader (1000 stratified samples)
provides:
  - PyTorch int8 quantized model (resnet8_int8.pt)
  - FX graph mode quantization script with JIT serialization
  - Documentation of fbgemm uint8 limitation
affects: [08-comparison-analysis]

# Tech tracking
tech-stack:
  added: [onnx2torch]
  patterns: [FX graph mode quantization, JIT tracing for serialization]

key-files:
  created: [scripts/quantize_pytorch.py, models/resnet8_int8.pt]
  modified: [scripts/evaluate_pytorch.py]

key-decisions:
  - "FX graph mode for onnx2torch models (eager mode doesn't support custom ONNX ops)"
  - "JIT tracing for serialization (FX GraphModule has pickle issues)"
  - "fbgemm backend only - uint8-only quantization not supported"

patterns-established:
  - "Load TorchScript: Use torch.jit.load() for FX-quantized models"
  - "Quantization workflow: prepare_fx -> calibrate -> convert_fx -> jit.trace -> save"

# Metrics
duration: 6min
completed: 2026-01-28
---

# Phase 7 Plan 1: PyTorch Static Quantization Summary

**FX graph mode int8 quantization achieving 85.68% accuracy (-1.51% from baseline), uint8 documented as unsupported by fbgemm**

## Performance

- **Duration:** 6 min
- **Started:** 2026-01-28T06:10:57Z
- **Completed:** 2026-01-28T06:17:24Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Implemented PyTorch FX graph mode static quantization for onnx2torch model
- Achieved 85.68% accuracy with int8 quantized model (52% model size reduction)
- Documented fbgemm uint8 limitation (weights require qint8, not quint8)
- Updated evaluate_pytorch.py to support TorchScript model loading

## Task Commits

Each task was committed atomically:

1. **Task 1: Create PyTorch quantization script with model inspection mode** - `66a61a7` (feat)
2. **Task 2: Run quantization and evaluate accuracy against baseline** - `c610c01` (feat)

## Files Created/Modified

- `scripts/quantize_pytorch.py` - PyTorch static quantization with FX mode and JIT serialization
- `scripts/evaluate_pytorch.py` - Added TorchScript model loading support
- `models/resnet8_int8.pt` - Quantized int8 model (168KB, TorchScript format)

## Accuracy Results

| Model | Accuracy | Size | Delta vs Baseline |
|-------|----------|------|-------------------|
| FP32 baseline (resnet8.pt) | 87.19% | 353KB | - |
| PyTorch int8 (resnet8_int8.pt) | 85.68% | 168KB | -1.51% |
| ONNX Runtime int8 | 85.58% | 125KB | -1.61% |
| ONNX Runtime uint8 | 86.75% | 125KB | -0.44% |

**Key finding:** PyTorch int8 quantization is comparable to ONNX Runtime int8 (85.68% vs 85.58%).

## Decisions Made

1. **FX graph mode for onnx2torch models** - Eager mode quantization doesn't work with custom ONNX operations (OnnxBinaryMathOperation, OnnxMatMul, etc.). FX mode traces the computation graph instead of requiring specific module types.

2. **JIT tracing for serialization** - FX GraphModule has serialization issues (AttributeError on deserialization). JIT tracing produces a TorchScript model that serializes correctly.

3. **fbgemm uint8 limitation documented** - PyTorch's quantized convolution requires qint8 (signed 8-bit) weights. uint8-only quantization (like ONNX Runtime's U8U8) is not supported by fbgemm backend.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed onnx2torch dependency**
- **Found during:** Task 1 (model inspection)
- **Issue:** Model loading failed with ModuleNotFoundError: No module named 'onnx2torch'
- **Fix:** Installed onnx2torch package
- **Files modified:** (system package)
- **Verification:** Model loads successfully
- **Committed in:** 66a61a7 (Task 1 commit)

**2. [Rule 3 - Blocking] Switched to FX mode for quantization**
- **Found during:** Task 2 (quantization attempt)
- **Issue:** Eager mode warned "None of the submodule got qconfig applied" - onnx2torch custom ops don't support eager mode quantization
- **Fix:** Implemented FX graph mode quantization which traces computation graph
- **Files modified:** scripts/quantize_pytorch.py
- **Verification:** Quantization completes, model size reduced from 353KB to 168KB
- **Committed in:** c610c01 (Task 2 commit)

**3. [Rule 3 - Blocking] Added JIT tracing for serialization**
- **Found during:** Task 2 (model evaluation)
- **Issue:** FX GraphModule saved with torch.save could not be loaded (AttributeError: 'Conv2d' object has no attribute '_modules')
- **Fix:** Added JIT tracing step after quantization, save as TorchScript
- **Files modified:** scripts/quantize_pytorch.py, scripts/evaluate_pytorch.py
- **Verification:** Model loads and evaluates correctly
- **Committed in:** c610c01 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (all blocking issues)
**Impact on plan:** All auto-fixes necessary to complete quantization with onnx2torch model. No scope creep.

## Issues Encountered

- **onnx2torch model incompatibility with eager mode:** Expected based on research. Custom ONNX operations (OnnxBinaryMathOperation, etc.) don't support PyTorch's qconfig mechanism. Resolved by using FX graph mode.

- **FX GraphModule serialization:** Unexpected issue. The FX-converted model has internal structure that doesn't serialize properly with pickle. Resolved by JIT tracing which produces a more portable TorchScript model.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- PyTorch quantization complete, ready for Phase 8 comparison analysis
- All four quantized models now available:
  - ONNX Runtime int8: 85.58%
  - ONNX Runtime uint8: 86.75%
  - PyTorch int8: 85.68%
  - PyTorch uint8: Not supported (documented)

---
*Phase: 07-pytorch-quantization*
*Completed: 2026-01-28*
