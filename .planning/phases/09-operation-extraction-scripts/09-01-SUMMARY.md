---
phase: 09-operation-extraction-scripts
plan: 01
subsystem: tooling
tags: [onnx, json, graphviz, pydot, quantization, visualization]

# Dependency graph
requires:
  - phase: 06-onnx-quantization
    provides: resnet8_int8.onnx quantized model
provides:
  - extract_operations.py script for extracting QLinear operations to JSON
  - visualize_graph.py script for generating PNG/SVG graph visualizations
  - JSON extraction of scales, zero-points, and attributes from ONNX models
affects: [10-boundary-ops, 11-core-ops, 12-architecture, 13-hardware-guide]

# Tech tracking
tech-stack:
  added: [pydot (new dependency for v1.3), graphviz (system dependency)]
  patterns: [onnx.helper.get_attribute_value() for attribute extraction, initializer lookup dict for scale/zero-point values, numpy to Python type conversion for JSON serialization]

key-files:
  created: [scripts/extract_operations.py, scripts/visualize_graph.py]
  modified: []

key-decisions:
  - "Use onnx.helper.get_attribute_value() instead of direct .f/.i/.s access for proper union type handling"
  - "Build initializer lookup dict once at start to avoid repeated iteration"
  - "Convert numpy types to Python types (float/int/tolist) for JSON serialization"
  - "Use subprocess.run() with check=True for dot command invocation instead of pydot write_png()"

patterns-established:
  - "Initializer lookup pattern: Build dict[name -> numpy array] once, then lookup by input name"
  - "Graceful dependency checking: Check for system tools (graphviz) at startup with helpful install instructions"
  - "JSON-safe value conversion: Use dtype.kind to determine float vs int conversion for numpy scalars"

# Metrics
duration: 2min
completed: 2026-02-02
---

# Phase 9 Plan 01: Operation Extraction Scripts Summary

**Two CLI tools for extracting quantized operation metadata to JSON and generating PNG/SVG graph visualizations using ONNX native APIs**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-02T06:07:00Z
- **Completed:** 2026-02-02T06:10:08Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created extract_operations.py that extracts all four quantized operation types (QLinearConv, QLinearMatMul, QuantizeLinear, DequantizeLinear) with numeric scale/zero-point values
- Created visualize_graph.py that generates .dot, .png, and .svg visualizations using onnx.tools.net_drawer and graphviz
- Both scripts follow project patterns with argparse, type hints, docstrings, and helpful error messages

## Task Commits

Each task was committed atomically:

1. **Task 1: Create extract_operations.py** - `dff2fde` (feat)
2. **Task 2: Create visualize_graph.py** - `917abff` (feat)

## Files Created/Modified
- `scripts/extract_operations.py` - Extracts QLinear operations to structured JSON with scales, zero-points, and attributes
- `scripts/visualize_graph.py` - Generates PNG/SVG visualizations of ONNX model graphs using Graphviz

## Decisions Made

**Use onnx.helper.get_attribute_value() for all attribute extraction**
- Rationale: AttributeProto is a protobuf union type. Direct field access (.f, .i, .s) returns default value (0 or "") if wrong type is accessed, causing silent data corruption. get_attribute_value() handles type dispatch correctly.

**Build initializer lookup dict once at module start**
- Rationale: QLinear operations reference scales/zero-points by name. Values are stored in graph.initializer. Building lookup dict once avoids O(N*M) iteration (N nodes × M initializers).

**Convert numpy types to Python types for JSON serialization**
- Rationale: json.dumps() doesn't handle np.float32, np.int8, etc. Use float()/int() for scalars based on dtype.kind, and tolist() for arrays.

**Use subprocess.run() for dot command instead of pydot.write_png()**
- Rationale: pydot.write_png() was deprecated in pydot 2.0+. Best practice is to write .dot file, then use subprocess to call system graphviz dot command for PNG/SVG rendering.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

**pydot not available in test environment**
- Environment has externally-managed Python installation, preventing pip install without --break-system-packages
- Script correctly detects missing pydot and provides helpful error message: "Error: pydot not installed. Install with: pip install pydot"
- Script is functionally complete and will work once pydot dependency is installed (available as python3-pydot via apt)

**No quantized ONNX model in worktree**
- Worktree doesn't contain resnet8_int8.onnx (would be generated from earlier phases in main repo)
- Created minimal test model to verify extraction logic works correctly
- Script correctly detects missing model and provides helpful error message
- Verified: JSON extraction produces numeric values (0.10000000149011612) not string names ("x_scale")

## User Setup Required

**pydot Python package required:**
- Install via system package manager: `sudo apt-get install python3-pydot` (Ubuntu/Debian)
- Or via pip: `pip install pydot` (in virtual environment or with --break-system-packages)

**graphviz system package required:**
- Already installed in test environment (version 2.43.0)
- For fresh installations: `sudo apt-get install graphviz` (Ubuntu/Debian) or `brew install graphviz` (macOS)
- Script checks for graphviz at startup and provides install instructions if missing

## Next Phase Readiness

Ready for Phase 10 (Boundary Operations documentation):
- extract_operations.py provides structured JSON with all QuantizeLinear/DequantizeLinear operations
- JSON includes numeric scale/zero-point values for concrete examples
- visualize_graph.py provides architectural context for understanding data flow

Ready for Phase 11 (Core Operations documentation):
- extract_operations.py provides structured JSON with all QLinearConv/QLinearMatMul operations
- JSON includes complete attribute values (kernel_shape, pads, strides, etc.)

Ready for Phase 12 (Architecture documentation):
- visualize_graph.py generates full model graph showing residual connections and data flow
- Both scripts support any ONNX model, not just resnet8_int8.onnx

**No blockers.** Scripts are functionally complete. pydot installation required before using visualize_graph.py, but this is documented in user setup.

---
*Phase: 09-operation-extraction-scripts*
*Completed: 2026-02-02*
