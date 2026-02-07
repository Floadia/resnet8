---
phase: 15-parameter-inspection
plan: 02
subsystem: quantization-playground
tags: [marimo, matplotlib, heatmap, histogram, visualization, quantization]

# Dependency graph
requires:
  - phase: 15-parameter-inspection-01
    provides: Parameter extraction utilities (extract_layer_params, extract_weight_tensors, compute_all_layer_ranges)
provides:
  - Whole-model weight range heatmap (FP32 vs INT8 side-by-side)
  - Per-layer quantization parameter table (scale, zero-point, shape, dtype)
  - FP32 vs INT8/UINT8 weight distribution histograms with consistent binning
affects: [quantization-analysis, model-debugging]

# Tech tracking
tech-stack:
  added: [matplotlib]
  patterns:
    - Marimo cell local variables prefixed with _ to avoid redefinition errors
    - plt.close('all') before creating figures to prevent memory leaks
    - Return figure objects (not plt.show()) for Marimo rendering
    - Consistent histogram binning across model variants for valid comparison

key-files:
  created: []
  modified:
    - playground/quantization.py
    - pyproject.toml

key-decisions:
  - "Prefix all local variables in Marimo cells with _ to avoid variable redefinition errors"
  - "Use matplotlib (not plotly) for heatmap and histogram visualizations"
  - "Consistent bin range across FP32/INT8/UINT8 histograms for valid visual comparison"
  - "Color-code heatmap bars by range magnitude using viridis colormap"
  - "Show informative callout messages for layers without quantization parameters"

patterns-established:
  - "Marimo figure cells: plt.close('all') -> create figure -> return figure (never plt.show())"
  - "Marimo local vars: prefix with _ to prevent cell output registration"
  - "Reactive visualization: cells take layer_selector as parameter for automatic updates"

# Metrics
duration: 8min
completed: 2026-02-07
---

# Phase 15 Plan 02: Parameter Visualization Summary

**Whole-model heatmap, per-layer parameter table, and weight distribution histograms for ONNX quantization inspection**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-07
- **Completed:** 2026-02-07
- **Tasks:** 2 (1 auto + 1 checkpoint)
- **Files modified:** 2

## Accomplishments

- Added whole-model heatmap showing FP32 and INT8 weight ranges side-by-side with viridis color coding
- Added per-layer parameter table displaying scale, zero-point, weight shape, dtype, and per-channel detection
- Added FP32 vs INT8 (and optionally UINT8) weight distribution histograms with consistent binning
- Added matplotlib to project dependencies
- Fixed Marimo variable redefinition errors by prefixing all local variables with _
- All visualization cells handle missing data gracefully with informative callouts

## Task Commits

Each task was committed atomically:

1. **Task 1: Add whole-model heatmap and per-layer detail cells** - `589ee08` (feat)
2. **Fix: Add matplotlib dep and prefix local vars for Marimo compatibility** - `a2cbd71` (fix)

## Files Created/Modified

- `playground/quantization.py` - Added heatmap overview, parameter table, and histogram cells with reactive updates
- `pyproject.toml` - Added matplotlib dependency

## Decisions Made

1. **Prefix local variables with _ in Marimo cells**
   - Rationale: Marimo treats all non-prefixed variables as cell outputs; redefinitions across cells cause errors
   - Impact: All cells load without variable conflicts

2. **Use matplotlib for visualizations**
   - Rationale: Better suited for side-by-side bar charts and histograms; consistent with scientific visualization norms
   - Impact: Added matplotlib as project dependency

3. **Consistent histogram binning**
   - Rationale: Using same bin range across FP32/INT8/UINT8 enables valid visual comparison of distributions
   - Impact: Users can directly compare weight distributions across model variants

4. **Color-coded heatmap by range magnitude**
   - Rationale: Viridis colormap highlights layers with widest ranges (potential quantization trouble spots)
   - Impact: Quick visual identification of problematic layers

## Deviations from Plan

1. **Added matplotlib dependency** - Not in original plan but required for visualization (auto-fixed)
2. **Variable prefixing** - Marimo compatibility issue discovered during browser testing; fixed with _ prefixes

## Issues Encountered

1. **ModuleNotFoundError: matplotlib** - Not in project dependencies; fixed by adding via `uv add matplotlib`
2. **Marimo variable redefinition** - Local variables like `fig`, `layer_names`, `onnx_int8` caused cell conflicts; fixed with _ prefix convention

## Next Phase Readiness

Phase 15 visualization complete. Ready for verification.

No blockers or concerns.

---
*Phase: 15-parameter-inspection*
*Completed: 2026-02-07*
