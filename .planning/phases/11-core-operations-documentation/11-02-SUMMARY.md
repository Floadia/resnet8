---
phase: 11-core-operations-documentation
plan: 02
subsystem: documentation
tags: [qlinearmatmul, quantization, int8, onnx, two-stage-computation, matrix-multiplication]

# Dependency graph
requires:
  - phase: 11-01
    provides: "QLinearConv documentation establishing two-stage computation pattern"
  - phase: 10-boundary-operations-documentation
    provides: "Documentation pattern for quantization operations and GitHub MathJax syntax"
  - phase: 09-operation-extraction-scripts
    provides: "Scripts for extracting ONNX operations and quantization parameters"
provides:
  - "QLinearMatMul documentation with two-stage computation pattern"
  - "Explicit cross-reference links between QLinearConv and QLinearMatMul"
  - "ResNet8 FC layer worked example (64→10 classification)"
  - "INT32 accumulator requirement demonstration for matrix multiplication"
  - "Validation script for QLinearMatMul manual implementation"
affects: [12-architecture-documentation, 13-hardware-implementation-guide]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Cross-referencing between operation documentation (markdown links)"
    - "Comparison table showing operation similarities and differences"
    - "INT32 accumulator overflow analysis adapted to matrix multiplication"

key-files:
  created:
    - docs/quantization/03-qlinearmatmul.md
    - scripts/validate_qlinearmatmul.py
  modified: []

key-decisions:
  - "Documented explicit cross-reference links to QLinearConv to avoid duplicating two-stage computation explanation"
  - "Used comparison table to highlight structural differences vs arithmetic similarities"
  - "Created ResNet8 FC layer example (64→10) demonstrating classification use case"
  - "INT32 overflow demonstration shows 31.5× INT16 max for even small FC layers"

patterns-established:
  - "Cross-referencing pattern: Link to detailed explanation in related operation doc instead of duplicating"
  - "Comparison table pattern: Side-by-side showing 'Same' vs differences for related operations"
  - "Concise documentation: Focus on what's different, reference related docs for shared patterns"

# Metrics
duration: 4min
completed: 2026-02-03
---

# Phase 11 Plan 02: QLinearMatMul Documentation Summary

**QLinearMatMul two-stage computation documented with explicit cross-references to QLinearConv, ResNet8 FC layer example (64→10), and validation script with 4 passing test cases**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-03T03:28:59Z
- **Completed:** 2026-02-03T03:32:40Z
- **Tasks:** 2
- **Files modified:** 2 created

## Accomplishments

- Complete QLinearMatMul documentation (377 lines) with two-stage computation pattern and explicit cross-reference links
- Comparison table showing QLinearMatMul shares identical arithmetic pattern with QLinearConv (only structural differences)
- ResNet8 FC layer worked example (64 features → 10 classes) with actual quantization parameters
- INT32 accumulator requirement proven for matrix multiplication (31.5× INT16 max for 64 MACs)
- Validation script with 4 test cases (simple matmul, ResNet8 FC, asymmetric, overflow) all passing with exact match

## Task Commits

Each task was committed atomically:

1. **Task 1: Create QLinearMatMul documentation with explicit cross-references** - `b53653b` (docs)
2. **Task 2: Create QLinearMatMul validation script** - `d1545b2` (feat)

## Files Created/Modified

- `docs/quantization/03-qlinearmatmul.md` - Complete QLinearMatMul reference with two-stage computation, explicit cross-references to QLinearConv, comparison table, ResNet8 FC example, and hardware pseudocode
- `scripts/validate_qlinearmatmul.py` - Validation script demonstrating manual QLinearMatMul implementation with 4 test cases

## Decisions Made

**1. Explicit cross-referencing instead of duplication**
- QLinearMatMul shares identical two-stage computation pattern with QLinearConv
- Decision: Include explicit markdown links to QLinearConv documentation for detailed pattern explanation
- Rationale: Avoid duplication while ensuring readers understand the shared foundation
- Implementation: Two cross-reference links (two-stage computation section, INT32 accumulator section)

**2. Comparison table for operation relationships**
- Created side-by-side comparison showing "Same" vs different aspects
- Decision: Use table format to make structural vs arithmetic differences clear
- Rationale: Helps hardware implementers understand they need one implementation with different input handling
- Key insight documented: "If you understand QLinearConv, you understand QLinearMatMul"

**3. ResNet8 FC layer as worked example**
- Used final classification layer (64→10) instead of generic example
- Decision: Show actual CNN use case with realistic quantization parameters
- Rationale: Connects to ResNet8 model readers are familiar with, shows classification-specific context
- Parameters: a_scale=0.1903, a_zero_point=20 (asymmetric), b_scale=0.0245 (symmetric)

**4. Concise documentation leveraging cross-references**
- QLinearMatMul doc is more concise than QLinearConv (377 vs 669 lines)
- Decision: Focus on what's different (input names, no spatial dimensions, no bias)
- Rationale: Respect reader's time, leverage existing comprehensive QLinearConv documentation
- Must-have cross-references verified by grep pattern matching

## Deviations from Plan

None - plan executed exactly as written.

All verification criteria met:
- Two-stage computation documented with proper formulas ✓
- All 8 QLinearMatMul inputs documented ✓
- Explicit cross-reference links to QLinearConv present (2 links verified by grep) ✓
- ResNet8 FC layer example included with actual ONNX-based parameters ✓
- Validation script runs successfully with 4 passing test cases ✓
- Documentation is concise, avoiding QLinearConv content duplication ✓

## Issues Encountered

None - straightforward execution following established patterns from Plan 01.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 12 (Architecture Documentation):**
- Core operations fully documented (QLinearConv + QLinearMatMul)
- Two-stage computation pattern established and cross-referenced
- Both convolution and matrix multiplication covered (complete operation set for CNNs)
- QDQ format understanding in place (models use QuantizeLinear/DequantizeLinear, not QLinear* directly)

**Ready for Phase 13 (Hardware Implementation Guide):**
- INT32 accumulator requirement proven for both operations
- Validation methodology demonstrated (manual implementation vs reference)
- Critical pitfalls documented (overflow, rounding, saturation)

**No blockers.** Phase 11 complete with comprehensive documentation of both core quantized operations.

---
*Phase: 11-core-operations-documentation*
*Completed: 2026-02-03*
