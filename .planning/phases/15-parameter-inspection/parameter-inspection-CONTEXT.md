# Phase 15: Parameter Inspection - Context

**Gathered:** 2026-02-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Add parameter inspection to the existing Marimo notebook. Users can explore quantization parameters (scales, zero-points, weights) per layer, compare FP32 vs quantized weight distributions, and see a whole-model range heatmap. ONNX models only. Builds on Phase 14's model loading, layer dropdown, and extraction utilities.

</domain>

<decisions>
## Implementation Decisions

### Parameter display format
- Raw table (no styled cards or expandable sections)
- Shows data for the currently-selected layer from the existing dropdown (not all layers at once)
- Single table containing: scale, zero-point, weight tensor shape, dtype — all in one view
- Per-channel scales shown as summary stats (min, max, mean) not full value lists
- Rounded number formatting (not full precision)

### Model structure navigation
- ONNX only — no PyTorch layer navigation needed
- Reuse the existing layer dropdown from Phase 14 (no new navigation widget)
- Show both operation nodes (Conv, Relu) and initializers (weights/biases) in dropdown
- Add summary info per layer in the dropdown (e.g., badge/indicator showing "has quantization params" vs "no params")

### FP32 vs quantized weight comparison
- Two separate horizontal histograms placed side-by-side (not overlaid)
- FP32 on left, INT8 on right
- Include uint8 as a third histogram when the uint8 model variant is available
- Histograms show actual weight values (not error/difference)
- Bars go left-to-right (x-axis = value bins, y-axis = count)

### Layer range heatmap
- Bar-range style: each row = one layer, bar shows min-to-max weight range
- Whole-model overview displayed as a separate section ABOVE the per-layer detail (stacked layout, Option A)
- FP32 and INT8 (dequantized) ranges shown side-by-side (two heatmaps next to each other)
- Clicking a layer row in the heatmap drives the layer dropdown (detail view updates to that layer)
- Sequential color scheme (light-to-dark showing magnitude)

### Claude's Discretion
- Exact matplotlib chart styling (colors, fonts, spacing)
- Histogram bin count and axis scaling
- Heatmap color palette choice within "sequential" constraint
- Table column widths and alignment
- How to handle layers with no quantization parameters in the heatmap

</decisions>

<specifics>
## Specific Ideas

- The notebook is a single scrolling page — heatmap overview at top, then layer dropdown, then detail (table + histograms)
- Heatmap should make it visually obvious which layers have the widest ranges (potential quantization trouble spots)
- Keep it simple: matplotlib is already available, no need for Plotly or other interactive libraries

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 15-parameter-inspection*
*Context gathered: 2026-02-07*
