---
phase: 11-core-operations-documentation
plan: 01
subsystem: documentation
tags: [qlinearconv, quantization, int8, onnx, two-stage-computation, convolution]

# Dependency graph
requires:
  - phase: 10-boundary-operations-documentation
    provides: "Documentation pattern for quantization operations (QuantizeLinear/DequantizeLinear)"
  - phase: 09-operation-extraction-scripts
    provides: "Scripts for extracting ONNX operations and quantization parameters"
provides:
  - "QLinearConv documentation with two-stage computation pattern"
  - "INT32 accumulator overflow demonstration"
  - "Per-channel vs per-tensor quantization examples"
  - "Validation script for QLinearConv manual implementation"
affects: [12-architecture-documentation, 13-hardware-implementation-guide]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-stage computation documentation (Stage 1: INT8×INT8→INT32 MAC, Stage 2: requantization)"
    - "Overflow demonstration with runnable code"
    - "Synthetic examples when actual model format differs from spec"

key-files:
  created:
    - docs/quantization/02-qlinearconv.md
    - scripts/validate_qlinearconv.py
    - models/resnet8_int8_operations.json
  modified: []

key-decisions:
  - "Documented QLinearConv based on ONNX specification with synthetic examples (model uses QDQ format, not QLinearConv)"
  - "Demonstrated INT32 accumulator requirement with 283.5× overflow factor calculation"
  - "Used GitHub MathJax syntax for formulas (escape underscores in variable names)"
  - "Created validation script with 4 test cases covering simple, multichannel, asymmetric, and overflow scenarios"

patterns-established:
  - "Core operations documentation structure: Overview → Inputs → Two-stage formula → Per-tensor example → Per-channel example → Edge cases → Hardware pseudocode"
  - "Overflow demonstration pattern: Show worst-case MAC count, compute accumulator value, compare to INT16/INT32 max"
  - "Validation script pattern: Multiple test cases with verbose mode for intermediate value inspection"

# Metrics
duration: 5min
completed: 2026-02-03
---

# Phase 11 Plan 01: Core Operations Documentation Summary

**QLinearConv two-stage computation documented with INT32 overflow demonstration (283.5× INT16 max), per-channel quantization analysis, and validation script with 4 passing test cases**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-03T03:18:48Z
- **Completed:** 2026-02-03T03:24:45Z
- **Tasks:** 3
- **Files modified:** 3 created

## Accomplishments

- Comprehensive QLinearConv documentation (669 lines) covering all 9 inputs, two-stage computation, and per-channel quantization
- INT32 accumulator requirement proven with overflow demonstration showing 9,290,304 accumulator value (283.5× INT16 max)
- Validation script with 4 test cases (simple, multichannel, asymmetric, overflow) all passing
- Per-channel vs per-tensor storage overhead analysis (0.17% for typical layers)

## Task Commits

Each task was committed atomically:

1. **Task 1: Generate ONNX model and extract quantization parameters** - `c155638` (chore)
2. **Task 2: Create QLinearConv documentation with two-stage computation** - `47df9c3` (docs)
3. **Task 3: Create QLinearConv validation script** - `7d1f6f2` (feat)

## Files Created/Modified

- `docs/quantization/02-qlinearconv.md` - Complete QLinearConv reference with two-stage computation pattern, per-channel examples, overflow demonstration, edge cases, and hardware pseudocode
- `scripts/validate_qlinearconv.py` - Validation script demonstrating manual QLinearConv implementation with 4 test cases
- `models/resnet8_int8_operations.json` - Extracted quantization parameters from ONNX model (QDQ format)

## Decisions Made

**1. QDQ format adaptation**
- ResNet8 quantized model uses QDQ format (QuantizeLinear/DequantizeLinear pairs around Conv), not QLinearConv
- Decision: Document QLinearConv based on ONNX specification with synthetic examples
- Rationale: QLinearConv is the canonical quantized convolution operator; documentation is still valuable for hardware implementers

**2. INT32 accumulator demonstration**
- Created runnable code showing 64 channels × 3×3 kernel = 576 MACs
- Worst-case: 9,290,304 (283.5× INT16 max of 32,767)
- Proves INT32 requirement, not just states it

**3. Per-channel quantization analysis**
- Documented storage overhead: 2×M parameters vs tensor size
- Example: 256-channel layer = 0.17% overhead (negligible)
- Showed why per-channel improves accuracy (matches scale to weight distribution)

**4. GitHub MathJax syntax**
- Used `$$...$$` for display math, `$...$` for inline
- Escaped underscores in variable names: `y\_scale`, `x\_zero\_point`
- Validated rendering compatibility per Phase 10 decisions

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Model uses QDQ format instead of QLinearConv**
- **Found during:** Task 1 (ONNX model extraction)
- **Issue:** Generated model uses QDQ format (QuantizeLinear/DequantizeLinear pairs), not QLinearConv operations
- **Fix:** Documented QLinearConv based on ONNX specification with synthetic examples instead of actual model values
- **Files modified:** docs/quantization/02-qlinearconv.md (used specification-based examples)
- **Verification:** Documentation covers all ONNX QLinearConv inputs and computation stages per specification
- **Committed in:** 47df9c3 (Task 2 commit)
- **Rationale:** QLinearConv documentation is still essential for hardware implementers; ONNX spec is authoritative source

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Necessary adaptation when model format differs from documentation target. Documentation quality maintained by using authoritative ONNX specification instead of model-specific values.

## Issues Encountered

**Generated ONNX model format mismatch**
- Expected: QLinearConv operations for hardware documentation
- Actual: QDQ format (QuantizeLinear/DequantizeLinear pairs around standard Conv)
- Resolution: Used ONNX specification as authoritative source, created synthetic examples demonstrating two-stage computation pattern
- Impact: Documentation remains accurate and comprehensive; validation script confirms implementation correctness

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 12 (Architecture Documentation):**
- Core operations foundation established (boundary ops + QLinearConv)
- Two-stage computation pattern documented and validated
- Overflow handling and edge cases covered

**Ready for Phase 13 (Hardware Implementation Guide):**
- Critical pitfalls identified (INT32 accumulator, rounding, saturation)
- Hardware pseudocode patterns established
- Validation methodology demonstrated

**No blockers.** QDQ format vs QLinearConv difference doesn't affect architecture or hardware guide phases - both need quantized operation understanding which is now documented.

---
*Phase: 11-core-operations-documentation*
*Completed: 2026-02-03*
