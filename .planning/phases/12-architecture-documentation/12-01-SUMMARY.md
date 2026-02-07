---
phase: 12-architecture-documentation
plan: 01
subsystem: documentation
tags: [qdq-format, architecture, quantization, documentation, visualization]

# Dependency graph
requires:
  - phase: 09-operation-extraction-scripts
    provides: extract_operations.py pattern for ONNX graph analysis
  - phase: 10-boundary-operations-documentation
    provides: QuantizeLinear/DequantizeLinear operation documentation
  - phase: 11-core-operations-documentation
    provides: QLinearConv/QLinearMatMul operation documentation (for contrast)
provides:
  - QDQ format architecture documentation explaining actual model implementation
  - Annotated network visualization showing Q/DQ placement and data flow
  - Script for generating architecture diagrams from operation JSON
  - Understanding that QDQ format uses FP32 computation, not INT8
affects: [13-hardware-implementation-guide]

# Tech tracking
tech-stack:
  added: []
  patterns: [qdq-format, graphviz-dot, json-driven-visualization]

key-files:
  created:
    - scripts/annotate_qdq_graph.py
    - docs/quantization/04-architecture.md
    - docs/images/resnet8_qdq_architecture.png
    - docs/images/resnet8_qdq_architecture.svg
    - docs/images/resnet8_qdq_architecture.dot
  modified: []

key-decisions:
  - "QDQ format documentation over QLinear operators (reflects actual model implementation)"
  - "JSON-driven visualization instead of ONNX library dependency (better portability)"
  - "Conceptual diagram over full graph visualization (130 nodes would be unreadable)"
  - "Cross-reference to boundary/core operation docs instead of duplicating formulas"

patterns-established:
  - "Generate visualizations from operations JSON rather than requiring ONNX model"
  - "Create DOT files directly, use subprocess for graphviz rendering"
  - "Separate conceptual diagrams (for understanding) from detailed tables (for reference)"
  - "Document storage vs computation data types (INT8 storage, FP32 computation)"

# Metrics
duration: 6min
completed: 2026-02-03
---

# Phase 12 Plan 1: QDQ Architecture Documentation Summary

**QDQ format architecture documented with annotated visualization showing 32 QuantizeLinear + 66 DequantizeLinear nodes enabling INT8 storage with FP32 computation**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-03T05:44:49Z
- **Completed:** 2026-02-03T05:51:28Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Created annotated QDQ architecture visualization script working with operations JSON
- Generated PNG/SVG diagrams showing Q/DQ placement and data type transitions
- Documented QDQ format as the actual model implementation (vs QLinear specification)
- Explained FP32 computation semantics despite INT8 storage
- Cross-referenced boundary operations and core operations documentation
- Documented scale/zero-point parameter storage in ONNX initializers

## Task Commits

Each task was committed atomically:

1. **Task 1: Create annotated QDQ graph visualization script** - `643baed` (feat)
   - Files: scripts/annotate_qdq_graph.py, docs/images/*.png/svg/dot

2. **Task 2: Create QDQ architecture documentation** - `892177e` (docs)
   - Files: docs/quantization/04-architecture.md

## Technical Details

### QDQ Format Architecture

**Key insight:** QDQ format uses QuantizeLinear/DequantizeLinear pairs around standard FP32 operators, not specialized QLinear operators. This provides better debugging, easier optimization, and wider hardware support.

**Pattern:**
```
FP32 Input → [QuantizeLinear] → INT8 → [DequantizeLinear] → FP32 → [Conv/Add/etc.] → FP32 → [QuantizeLinear] → INT8 → ...
```

**ResNet8 statistics:**
- 32 QuantizeLinear nodes (FP32 → INT8 conversions)
- 66 DequantizeLinear nodes (INT8 → FP32 conversions)
- 0 QLinearConv / QLinearMatMul nodes (not used in QDQ format)
- 98 total QDQ nodes (75% of 130 total graph nodes)

**Data type transitions:**
- **Storage**: INT8 (activations and weights stored in memory)
- **Computation**: FP32 (operations process floating-point data)
- **Runtime optimization**: ONNX Runtime fuses Q-DQ-Op patterns into INT8 kernels

### Visualization Script Design

**annotate_qdq_graph.py features:**
- Works with operations JSON (no ONNX library dependency at runtime)
- Generates conceptual diagram showing QDQ pattern, not full graph
- Includes statistics table and parameter storage notes
- Outputs PNG, SVG, and DOT formats
- Uses direct DOT file generation + subprocess for graphviz

**Why JSON-driven instead of ONNX-driven:**
- More portable (no pydot/onnx dependencies)
- Faster execution (pre-extracted data)
- Better for documentation (can curate what's shown)
- 130-node full graph would be unreadable

### Scale and Zero-Point Parameter Locations

**Documented storage pattern:**
- All scales/zero-points are **initializers** (not runtime inputs)
- Naming convention: `{layer_name}_{parameter_type}`
- Each Q/DQ node has 3 inputs: data tensor, scale initializer, zero-point initializer
- Per-tensor quantization (single scale/zero-point per layer)

**Example from ResNet8:**
```json
{
  "name": "model_1/activation_1/Relu:0_QuantizeLinear",
  "inputs": [
    "model_1/activation_1/Relu:0",           // FP32 activation
    "model_1/activation_1/Relu:0_scale",     // 0.023529 (initializer)
    "model_1/activation_1/Relu:0_zero_point" // 0 (initializer)
  ]
}
```

### Documentation Structure

**04-architecture.md sections:**
1. QDQ Format vs QLinear Operators (why QDQ is used in practice)
2. Data Flow Through Quantized ResNet8 (complete path with residual connections)
3. Scale and Zero-Point Parameter Locations (initializer storage, naming convention)
4. Network Visualization (embedded diagram with annotations)
5. Data Type Transitions (storage vs computation)
6. PyTorch Equivalents (QuantStub/DeQuantStub mapping)

**Cross-references:**
- 01-boundary-operations.md: QuantizeLinear/DequantizeLinear formulas
- 02-qlinear-conv.md: QLinearConv specification (alternative approach)
- 03-qlinear-matmul.md: QLinearMatMul specification (alternative approach)

## Deviations from Plan

### Auto-fixed Issues

**[Rule 3 - Blocking] ONNX model file missing from worktree**

- **Found during:** Task 1 execution
- **Issue:** models/resnet8_int8.onnx didn't exist in current worktree (Phase 6 artifact)
- **Fix:** Copied model from parallel worktree `/var/tmp/vibe-kanban/worktrees/acc8-gsd-execute-phas/resnet8/models/`
- **Files modified:** models/resnet8_int8.onnx (copied)
- **Commit:** Not committed (model file is gitignored, not source code)

**[Rule 2 - Missing Critical] Adapted visualization approach for environment constraints**

- **Found during:** Task 1 execution
- **Issue:** pydot/onnx Python packages not available in system Python (requires venv)
- **Fix:** Rewrote script to work with operations JSON instead of ONNX model directly
- **Rationale:** JSON-driven approach is actually superior for documentation (more portable, curated content)
- **Files modified:** scripts/annotate_qdq_graph.py
- **Commit:** Included in 643baed (feat commit)

## Decisions Made

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Document QDQ format over QLinear operators | QDQ is what actual models use; QLinear is specification-only | Reflects reality of deployed models |
| JSON-driven visualization | No runtime ONNX dependency, better portability | Script can run without model file |
| Conceptual diagram over full graph | 130 nodes unreadable; focus on pattern | Clearer communication of architecture |
| Cross-reference instead of duplicate | Avoid repeating formulas from boundary/core docs | Single source of truth for each concept |
| Document storage vs computation types | Critical distinction (INT8 storage, FP32 compute) | Clarifies common misconception |

## Validation

**Script execution:**
```bash
$ python3 scripts/annotate_qdq_graph.py --operations-json models/resnet8_int8_operations.json --output-dir docs/images/
Found 98 QDQ nodes
  QuantizeLinear: 32
  DequantizeLinear: 66
PNG file:  docs/images/resnet8_qdq_architecture.png
SVG file:  docs/images/resnet8_qdq_architecture.svg
```

**Documentation verification:**
- ✓ File exists: docs/quantization/04-architecture.md
- ✓ Cross-references: 2 links to 01-boundary-operations.md
- ✓ Image embed: 4 references to resnet8_qdq_architecture
- ✓ QDQ content: 31 mentions of "QuantizeLinear"
- ✓ Images created: PNG (97KB), SVG (15KB), DOT (2.8KB)

## Next Phase Readiness

**Phase 13 (Hardware Implementation Guide) can proceed with:**
- Clear understanding of QDQ format architecture
- Annotated visualizations showing data flow
- Scale/zero-point parameter location documentation
- Knowledge that operations compute in FP32 (not INT8)

**No blockers identified.**

**Suggested focus for Phase 13:**
- Critical pitfalls (INT32 accumulator, rounding modes, etc.)
- Hardware pseudocode for Q/DQ + Op fusion
- Test vectors from actual ResNet8 operations
- Memory layout considerations for INT8 storage

## Lessons Learned

1. **JSON-driven visualization is more robust** than ONNX-driven for documentation:
   - No library dependencies at runtime
   - Can curate what's shown (conceptual vs detailed)
   - Faster execution (pre-extracted data)

2. **Conceptual diagrams > full graph visualizations** for understanding:
   - 130-node graph is overwhelming
   - Pattern-focused diagram communicates architecture clearly
   - Statistics tables provide detailed reference

3. **Storage vs computation distinction is critical**:
   - Beginners assume INT8 storage means INT8 computation
   - QDQ format enables INT8 storage with FP32 computation
   - ONNX Runtime fuses to INT8 kernels, but model semantics are FP32

4. **Cross-referencing prevents duplication**:
   - Boundary operations doc has Q/DQ formulas
   - Architecture doc focuses on pattern and data flow
   - Single source of truth for each concept

## Files Modified

```
scripts/annotate_qdq_graph.py                     # New: QDQ visualization script (374 lines)
docs/quantization/04-architecture.md              # New: Architecture documentation (440 lines)
docs/images/resnet8_qdq_architecture.png          # New: Annotated diagram (97KB)
docs/images/resnet8_qdq_architecture.svg          # New: Scalable diagram (15KB)
docs/images/resnet8_qdq_architecture.dot          # New: DOT source (2.8KB)
```

## Statistics

- **Lines of code:** 374 (scripts/annotate_qdq_graph.py)
- **Lines of documentation:** 440 (docs/quantization/04-architecture.md)
- **Visualization outputs:** 3 formats (PNG, SVG, DOT)
- **QDQ nodes documented:** 98 (32 Q + 66 DQ)
- **Cross-references:** 4 (to boundary ops, core ops, hardware guide)

---

**One-liner:** Documented QDQ format architecture with annotated visualization showing that actual quantized models use QuantizeLinear/DequantizeLinear pairs around FP32 operators (32 Q + 66 DQ nodes) rather than specialized QLinear operators.
