---
phase: 05-calibration-infrastructure
plan: 01
subsystem: infra
tags: [ptq, quantization, calibration, cifar-10, onnx, pytorch, numpy]

# Dependency graph
requires:
  - phase: 02-accuracy-evaluation
    provides: "evaluate.py with CIFAR-10 preprocessing (float32, 0-255, NHWC)"
provides:
  - "Stratified calibration data loader (1000 samples, 100 per class)"
  - "Preprocessing pipeline identical to evaluation (float32, 0-255, NHWC)"
  - "Distribution verification utilities"
affects: [06-onnxruntime-ptq, 07-pytorch-ptq, 08-comparison-validation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Stratified sampling for balanced calibration datasets"
    - "Preprocessing verification with sanity checks"

key-files:
  created:
    - "scripts/calibration_utils.py"
  modified: []

key-decisions:
  - "Use 1000 calibration samples (100 per class) - exceeds roadmap minimum of 200 for better quantization quality"
  - "Load from training batches (data_batch_1-5) not test_batch - prevents data leakage"
  - "Match evaluate.py preprocessing exactly (float32, 0-255, NHWC) - critical for PTQ accuracy"

patterns-established:
  - "Calibration data must match evaluation preprocessing exactly to avoid 10-40% accuracy drops"
  - "Stratified sampling ensures balanced class representation in calibration"
  - "Built-in verification checks for dtype, shape, range, and distribution"

# Metrics
duration: 2min
completed: 2026-01-28
---

# Phase 5 Plan 1: Calibration Infrastructure Summary

**Stratified CIFAR-10 calibration loader with 1000 samples (100 per class), preprocessing identical to evaluation pipeline (float32, 0-255 range, NHWC format)**

## Performance

- **Duration:** 2 min 18 sec
- **Started:** 2026-01-27T23:57:34Z
- **Completed:** 2026-01-27T23:59:52Z
- **Tasks:** 2 (Task 2 verification built into Task 1 implementation)
- **Files modified:** 1

## Accomplishments
- Created `scripts/calibration_utils.py` with stratified sampling (100 samples per class)
- Implemented preprocessing identical to `evaluate.py` (float32, 0-255, NHWC)
- Built-in verification checks for dtype, shape, pixel range, and class distribution
- Exceeds roadmap minimum calibration size (1000 vs 200) for better quantization quality
- Ready for ONNX Runtime and PyTorch PTQ phases

## Task Commits

Each task was committed atomically:

1. **Task 1: Create calibration data loader with stratified sampling** - `b07da26` (feat)
2. **Task 2: Verify calibration data matches evaluation preprocessing** - Built into Task 1, no additional code needed
   - Additional commit: `49da1b9` (chore) - made script executable

**Plan metadata:** (will be committed after SUMMARY.md creation)

## Files Created/Modified
- `scripts/calibration_utils.py` - Provides `load_calibration_data()` for stratified CIFAR-10 sampling and `verify_distribution()` for class balance verification. Includes CLI for testing.

## Decisions Made

**1. Calibration dataset size: 1000 samples (100 per class)**
- Rationale: Exceeds roadmap minimum of 200 samples for better quantization parameter estimation
- Impact: Higher quality PTQ models, minimal overhead (1000 vs 50,000 training samples)

**2. Load from training batches, not test set**
- Rationale: Prevents data leakage - test set reserved for evaluation only
- Impact: Maintains evaluation integrity, follows best practices

**3. Preprocessing matches evaluate.py exactly**
- Rationale: Research shows preprocessing mismatches cause 10-40% accuracy drops
- Implementation:
  - Reshape: (N, 3072) → (N, 3, 32, 32) → (N, 32, 32, 3)
  - Type: float32
  - Range: 0-255 (NO normalization)
  - Format: NHWC

**4. Built-in verification checks**
- Rationale: Catch preprocessing errors early before PTQ
- Checks: dtype, shape, pixel range, class distribution
- Output: Clear visual confirmation of correct setup

## Deviations from Plan

None - plan executed exactly as written.

Task 2 verification requirements were already met by Task 1 implementation. The script includes comprehensive verification checks for dtype, shape, pixel range, and class distribution that Task 2 specified. No additional code changes were needed.

## Issues Encountered

None - straightforward implementation following established patterns from evaluate.py.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 6 (ONNX Runtime PTQ):**
- ✓ Calibration data loader available
- ✓ Preprocessing verified to match evaluation
- ✓ Class distribution balanced (100 per class)
- ✓ 1000 samples available (exceeds minimum requirement)

**Ready for Phase 7 (PyTorch PTQ):**
- ✓ Same calibration data can be used for PyTorch quantization
- ✓ Preprocessing matches evaluation pipeline

**No blockers or concerns.** The calibration infrastructure is complete and verified.

---
*Phase: 05-calibration-infrastructure*
*Completed: 2026-01-28*
