---
phase: 12-architecture-documentation
plan: 02
subsystem: documentation
tags: [residual-connections, pytorch-equivalents, scale-mismatch, quantization, documentation]

# Dependency graph
requires:
  - phase: 12-architecture-documentation
    plan: 01
    provides: Base QDQ architecture documentation with data flow and visualization
  - phase: 11-core-operations-documentation
    provides: QLinearConv/QLinearMatMul two-stage computation pattern for cross-reference
  - phase: 10-boundary-operations-documentation
    provides: QuantizeLinear/DequantizeLinear operation documentation
provides:
  - Residual connection scale mismatch problem explained with concrete ResNet8 examples
  - QDQ dequant-add-quant solution documented showing FP32 addition pattern
  - Alternative approaches compared (scale matching, rescaling, PyTorch FloatFunctional)
  - PyTorch → ONNX quantized operation mapping table with 6 operations
  - Conversion limitations documented with recommended workflow
affects: [13-hardware-implementation-guide]

# Tech tracking
tech-stack:
  added: []
  patterns: [residual-connection-quantization, pytorch-onnx-mapping, float-functional]

key-files:
  created: []
  modified:
    - docs/quantization/04-architecture.md

key-decisions:
  - "Document all Add operations in ResNet8 (11 total) showing scale mismatches up to 3.32×"
  - "Provide concrete scale values from actual model for 3 primary residual connections"
  - "Document recommended PyTorch→ONNX workflow: export FP32 then quantize with ONNX Runtime"
  - "Cross-reference QLinear operators while clarifying QDQ format is actual implementation"

patterns-established:
  - "Extract actual scale values from quantized model for documentation examples"
  - "Compare multiple solution approaches with pros/cons tables"
  - "Provide both conceptual explanation and concrete code examples"
  - "Link PyTorch patterns to ONNX equivalents with conversion guidance"

# Metrics
duration: 3min
completed: 2026-02-03
---

# Phase 12 Plan 2: Residual Connections and PyTorch Equivalents Summary

**Residual connection scale mismatch problem documented with ResNet8 examples showing 2.65×-3.32× scale ratios requiring QDQ dequant-add-quant pattern, plus PyTorch→ONNX operation mapping table with conversion limitations and recommended workflow**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-03T05:56:28Z
- **Completed:** 2026-02-03T06:00:02Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Documented scale mismatch problem in residual connections with concrete examples
- Extracted actual scale values from ResNet8 quantized model (3 primary residual connections)
- Explained QDQ dequant-add-quant solution pattern (FP32 addition)
- Compared alternative approaches: scale matching, INT8 rescaling, PyTorch FloatFunctional
- Created PyTorch → ONNX operation mapping table (6 operations)
- Documented PyTorch export limitations and recommended workflow
- Provided FloatFunctional code examples for residual connection handling
- Cross-referenced to QLinearConv/QLinearMatMul documentation

## Task Commits

Each task was committed atomically:

1. **Task 1: Document residual connection handling** - `29338c1` (docs)
   - Files: docs/quantization/04-architecture.md

2. **Task 2: Document PyTorch quantized operation equivalents** - `e1b9030` (docs)
   - Files: docs/quantization/04-architecture.md

## Technical Details

### Residual Connection Scale Mismatch Problem

**Critical finding:** ResNet8 has 3 primary residual connections where different computational paths merge, each with significant scale mismatches:

| Add Node | Branch 1 Scale | Branch 2 Scale | Scale Ratio |
|----------|----------------|----------------|-------------|
| `model_1/add_1/Add` | 0.046150 | 0.122343 | 2.65× |
| `model_1/add_1_2/Add` | 0.045742 | 0.151687 | 3.32× |
| `model_1/add_2_1/Add` | 0.086567 | 0.239572 | 2.77× |

**Why this matters:** Direct INT8 addition with different scales produces incorrect results. The same INT8 value (e.g., 100) represents different FP32 magnitudes in each branch:
- Branch 1: 100 × 0.046150 = 4.615
- Branch 2: 100 × 0.122343 = 12.234
- Incorrect: 100 + 100 = 200
- Correct: 4.615 + 12.234 = 16.849

### QDQ Solution Pattern

ResNet8 uses the **dequant-add-quant pattern** for all residual connections:

```
Branch 1: INT8 → DequantizeLinear(scale₁) → FP32 → \
                                                      Add(FP32) → QuantizeLinear → INT8
Branch 2: INT8 → DequantizeLinear(scale₂) → FP32 → /
```

**Benefits:**
- Mathematically correct (addition in FP32)
- Handles arbitrary scale mismatches
- Standard pattern in ONNX Runtime quantization

**Trade-off:** Requires FP32 arithmetic for Add operation

### Alternative Approaches Documented

**1. Scale Matching** (hardware optimization):
- Force both branches to use identical scales during calibration
- Enables pure INT8 addition
- Constrains calibration, may reduce accuracy

**2. INT8 Rescaling** (no FP32 required):
- Rescale one branch to match the other before adding
- Additional rounding error
- Complex implementation

**3. PyTorch FloatFunctional** (framework-specific):
- Wraps add operation to track quantization statistics
- Automatically handles scale/zero-point mismatches
- Exports to ONNX as QDQ pattern

All three approaches are documented with code examples and pros/cons.

### PyTorch → ONNX Operation Mapping

Created comprehensive mapping table showing 6 operations:

| PyTorch Op | ONNX QDQ Pattern |
|-----------|------------------|
| `torch.nn.quantized.Conv2d` | Q → DQ → Conv → Q → DQ |
| `torch.nn.quantized.Linear` | Q → DQ → MatMul → Q → DQ |
| `torch.nn.quantized.ReLU` | Q → DQ → Relu → Q → DQ |
| `torch.ao.nn.quantized.FloatFunctional.add` | DQ (×2) → Add → Q |
| `torch.quantization.QuantStub` | QuantizeLinear |
| `torch.quantization.DeQuantStub` | DequantizeLinear |

**Key insight:** All PyTorch quantized operations follow the same QDQ pattern - operations compute in FP32 while activations are stored as INT8.

### Conversion Limitations and Recommended Workflow

**Known limitation:** `aten::quantize_per_channel` operator not supported in ONNX export for most opset versions.

**Recommended workflow (documented with code examples):**
1. Export FP32 PyTorch model to ONNX
2. Quantize ONNX model with ONNX Runtime tools (`quantize_dynamic` or `quantize_static`)

This two-step approach avoids PyTorch export limitations and leverages ONNX Runtime's mature quantization infrastructure.

### FloatFunctional Code Examples

Provided complete code examples showing:
- How to define FloatFunctional in ResNet block
- Usage in forward pass for residual addition
- How it tracks quantization statistics during calibration
- How it exports to ONNX QDQ pattern
- Alternative manual approach with QuantStub/DeQuantStub

### Cross-References to Core Operations

**Critical clarification added:** The two-stage computation pattern (INT8×INT8→INT32 accumulation, then requantization) documented in Phases 10-11 **still applies in QDQ format**:

- **QLinear operator approach**: Fused operator contains both stages (specification)
- **QDQ format approach**: Runtime fuses Q-DQ-Op pattern into equivalent INT8 kernel (actual models)

Both approaches execute the same underlying mathematics, just with different graph representations. This clarification bridges the conceptual gap between Phase 11 documentation (QLinear operators) and Phase 12 reality (QDQ format).

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Extract actual scale values from ResNet8 model | Concrete examples more valuable than hypothetical | Documentation grounded in real model behavior |
| Document all 3 solution approaches | Readers may have different hardware constraints | Comprehensive reference for different use cases |
| Include scale ratio calculations | Shows magnitude of mismatch problem | Quantifies why direct INT8 addition fails |
| Recommend FP32 export + ONNX Runtime quantization | Avoids PyTorch export limitations | Practical workflow that actually works |
| Cross-reference QLinear operators | Readers may wonder about Phase 11 docs | Clarifies relationship between spec and implementation |

## Validation

**Verification checks (all passed):**
```bash
✓ Residual connection section exists (1 occurrence)
✓ Scale mismatch problem documented (1 occurrence)
✓ QDQ solution documented (1 occurrence)
✓ PyTorch mapping table present (13 FloatFunctional mentions)
✓ Conversion limitations documented (aten::quantize_per_channel mentioned)
✓ Cross-references to QLinearConv/QLinearMatMul (2 links)
```

**Documentation structure verification:**
```
## Residual Connections in Quantized Networks
  ### 5.1 The Scale Mismatch Problem
  ### 5.2 QDQ Solution (Used in ResNet8)
  ### 5.3 Alternative Approaches (For Reference)
  ### 5.4 ResNet8 Specific Analysis

## PyTorch Quantized Operation Equivalents
  ### 6.1 Mapping Table
  ### 6.2 Important Notes on Conversion
  ### 6.3 FloatFunctional for Residual Connections
  ### 6.4 Cross-Reference to Core Operations
```

**Must-haves verification:**
- ✓ Residual connection handling documented with scale mismatch problem explained
- ✓ Solution approaches compared (QDQ, scale matching, FloatFunctional)
- ✓ PyTorch quantized operation equivalents mapped to ONNX QDQ patterns
- ✓ Known PyTorch→ONNX conversion limitations documented with recommended workflow

## Next Phase Readiness

**Phase 13 (Hardware Implementation Guide) can proceed with:**
- Understanding of residual connection scale mismatch problem (critical for hardware design)
- Knowledge that QDQ format requires FP32 addition or scale matching
- Awareness that PyTorch FloatFunctional exports to same QDQ pattern
- Complete picture of quantized network architecture (QDQ format + residual handling)

**No blockers identified.**

**Suggested focus for Phase 13:**
- Hardware-specific solutions for residual connections (scale matching vs FP32 support)
- INT32 accumulator requirements (still apply in QDQ format)
- Rounding modes and numerical precision requirements
- Test vectors from ResNet8 operations for validation

## Lessons Learned

1. **Extract real data from actual models for documentation examples:**
   - Concrete scale values (0.046150, 0.122343) more impactful than hypothetical examples
   - Scale ratios (2.65×-3.32×) quantify the problem magnitude
   - Shows that scale mismatch is not edge case but normal in quantized ResNets

2. **Document all viable approaches, not just one:**
   - Hardware constraints vary (FP32 support, memory bandwidth, accuracy requirements)
   - Comparing approaches (QDQ, scale matching, rescaling) helps readers choose
   - Each approach has valid use cases

3. **Bridge specification and implementation reality:**
   - Phase 11 documented QLinear operators (specification)
   - Phase 12 documents QDQ format (actual implementation)
   - Explicit clarification prevents confusion about "where are the QLinear nodes?"

4. **Provide working code examples for framework conversion:**
   - PyTorch export limitations are well-known pain point
   - Recommended workflow (FP32 export + ONNX Runtime quantization) saves readers time
   - FloatFunctional code example shows practical usage, not just theory

5. **Cross-reference to avoid duplication:**
   - Linked to Phases 10-11 for detailed operation math
   - Focused Phase 12 on architecture-level patterns
   - Single source of truth for each concept

## Files Modified

```
docs/quantization/04-architecture.md    # +336 lines
  - Added Section 5: Residual Connections in Quantized Networks (156 lines)
  - Added Section 6: PyTorch Quantized Operation Equivalents (180 lines)
```

## Statistics

- **Lines of documentation added:** 336
- **Code examples:** 4 (Python code blocks)
- **Tables created:** 3 (residual connections, PyTorch mapping, comparison)
- **Residual connections documented:** 3 primary + 8 batch norm Add operations
- **PyTorch operations mapped:** 6 (Conv2d, Linear, ReLU, FloatFunctional, QuantStub, DeQuantStub)
- **Cross-references:** 2 (to QLinearConv and QLinearMatMul docs)

---

**One-liner:** Documented residual connection scale mismatch problem (2.65×-3.32× ratios in ResNet8 requiring QDQ dequant-add-quant pattern) and PyTorch→ONNX operation mapping (6 operations) with conversion limitations and recommended FP32 export + ONNX Runtime quantization workflow.
