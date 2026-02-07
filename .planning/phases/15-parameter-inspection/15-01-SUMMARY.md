---
phase: 15-parameter-inspection
plan: 01
subsystem: quantization-playground
tags: [onnx, quantization, parameter-extraction, marimo, visualization]

# Dependency graph
requires:
  - phase: 14-notebook-foundation
    provides: Marimo notebook infrastructure with model loading and layer selection
provides:
  - ONNX quantization parameter extraction utilities (scale, zero-point, weights)
  - Multi-model weight tensor extraction with automatic dequantization
  - Whole-model weight range computation for heatmap visualization
  - Layer dropdown enhanced with [Q] indicators showing which layers have quantization parameters
affects: [15-parameter-inspection-02, quantization-analysis, model-debugging]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - ONNX graph traversal using onnx.numpy_helper.to_array for tensor extraction
    - Dequantization formula: (raw - zero_point) * scale
    - Per-channel vs per-tensor quantization detection via scale.ndim
    - Graceful None handling for missing models/parameters

key-files:
  created:
    - playground/utils/parameter_inspector.py
  modified:
    - playground/utils/__init__.py
    - playground/quantization.py

key-decisions:
  - "Use onnx.numpy_helper.to_array for all tensor extraction (not hand-rolled raw_data parsing)"
  - "Return None for missing parameters instead of raising errors for graceful handling"
  - "Detect per-channel quantization via scale.ndim > 0 check"
  - "Use dict-based dropdown options mapping display strings to clean layer names"
  - "Extract layers_with_params from INT8 model (most comprehensive QDQ coverage)"

patterns-established:
  - "Parameter extraction: Build initializer dict, traverse Q/DQ nodes, extract scale/zero-point from inputs"
  - "Weight dequantization: Use formula (raw.astype(float32) - zero_point) * scale for INT8/UINT8"
  - "Layer identification: Match layer_name in node names, inputs, outputs for association"
  - "Dropdown enhancement: Dict options {\"Layer [Q]\": \"Layer\"} keeps .value clean while showing indicators"

# Metrics
duration: 3min
completed: 2026-02-07
---

# Phase 15 Plan 01: Parameter Inspection Summary

**ONNX quantization parameter extraction (scale/zero-point/weights) with dequantization support and layer dropdown [Q] indicators**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-07T07:22:01Z
- **Completed:** 2026-02-07T07:25:18Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Created parameter_inspector.py with 4 extraction functions for ONNX QDQ models
- All functions handle None models and missing parameters gracefully (return None or empty structures)
- Layer dropdown enhanced with [Q] indicators showing which layers have quantization parameters
- Dequantization formula correctly implemented: (raw - zero_point) * scale
- Per-channel and per-tensor quantization both supported with automatic detection

## Task Commits

Each task was committed atomically:

1. **Task 1: Create parameter_inspector.py with ONNX extraction functions** - `9bdd077` (feat)
2. **Task 2: Enhance layer dropdown with quantization parameter indicators** - `6316beb` (feat)

## Files Created/Modified

- `playground/utils/parameter_inspector.py` - ONNX quantization parameter extraction and weight tensor access with 4 exported functions
- `playground/utils/__init__.py` - Updated exports including extract_layer_params, extract_weight_tensors, compute_all_layer_ranges, get_layers_with_params
- `playground/quantization.py` - Enhanced layer dropdown with [Q] indicators for layers with QuantizeLinear/DequantizeLinear nodes

## Decisions Made

1. **Use onnx.numpy_helper.to_array for tensor extraction**
   - Rationale: Official ONNX API handles all dtypes, endianness, shape correctly; avoids hand-rolling raw_data parsing
   - Impact: Robust extraction across all ONNX model variants

2. **Return None instead of raising errors for missing data**
   - Rationale: Graceful handling needed for interactive notebook; not all layers have Q/DQ nodes
   - Impact: UI can show "No parameters available" message instead of crashing

3. **Detect per-channel via scale.ndim > 0**
   - Rationale: Per-channel scales are 1D arrays (one per channel); per-tensor is scalar (ndim=0)
   - Impact: Functions can return summary stats (min/max/mean) for per-channel instead of full arrays

4. **Use dict-based dropdown options for [Q] indicators**
   - Rationale: Keeps layer_selector.value clean (returns "Conv_0" not "Conv_0 [Q]") while showing visual indicators
   - Impact: Downstream cells don't need to strip [Q] markers from layer names

5. **Extract layers_with_params from INT8 model**
   - Rationale: INT8 model has most comprehensive QDQ coverage; FP32 model has no Q/DQ nodes
   - Impact: [Q] indicators show accurately for quantized layers

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for Plan 02 (Parameter Visualization):
- All extraction functions implemented and tested
- Layer dropdown enhanced with indicators
- Functions handle edge cases (None models, missing params, per-channel scales)
- Next plan can use extract_layer_params() for table display and extract_weight_tensors() for histograms

No blockers or concerns.

---
*Phase: 15-parameter-inspection*
*Completed: 2026-02-07*
