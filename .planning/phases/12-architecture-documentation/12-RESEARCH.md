# Phase 12: Architecture Documentation - Research

**Researched:** 2026-02-03
**Domain:** ONNX quantized network architecture documentation, data flow visualization, residual connection handling
**Confidence:** HIGH

## Summary

Research reveals that Phase 12 requires documenting the full ResNet8 quantized architecture using a fundamentally different model format than documented in Phases 10-11. The actual quantized models use **QDQ format** (QuantizeLinear/DequantizeLinear pairs around standard operators) rather than specialized QLinear operations. This changes the architecture documentation approach from "document QLinearConv/QLinearMatMul usage" to "document QDQ insertion patterns and data flow through FP32 operators."

Key findings:
1. **QDQ format is the actual implementation**: ResNet8 quantized models use QuantizeLinear → Conv → DequantizeLinear patterns, not QLinearConv
2. **Residual connections require special handling**: Add operations need Q/DQ pairs to handle scale mismatches between branches
3. **PyTorch equivalents exist but conversion is limited**: torch.nn.quantized.Conv2d maps conceptually to QDQ+Conv, but direct ONNX export has known limitations
4. **Existing visualization tools are sufficient**: Graphviz + pydot (already installed) for programmatic diagrams, Netron for interactive exploration

**Primary recommendation:** Document the QDQ data flow pattern showing how QuantizeLinear/DequantizeLinear pairs enable INT8 computation through standard ONNX operators. Use existing extraction scripts to identify Q/DQ node placement in actual models. Create annotated diagrams showing scale/zero-point parameter flow, especially at residual connection merge points.

## Standard Stack

### Core Libraries (Already Installed)

The project already has all required libraries from Phase 9 (Operation Extraction Scripts):

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| onnx | Latest | ONNX model loading and inspection | Official ONNX reference implementation |
| pydot | Latest | Python interface to Graphviz | Standard for programmatic graph generation |
| graphviz | System | DOT graph rendering to PNG/SVG | Industry standard for graph visualization |
| numpy | Latest | Array operations for parameter extraction | Universal Python numerical library |

### Supporting Tools

| Tool | Version | Purpose | When to Use |
|------|---------|---------|-------------|
| Netron | Web/latest | Interactive ONNX model visualization | Quick exploration, understanding node connections |
| ONNX Runtime | 1.x | Inference testing and validation | Verify documentation accuracy against actual inference |

### No Additional Installation Required

**Confidence:** HIGH - All libraries confirmed present from Phase 9 execution.

Phase 9 already installed and validated:
- pydot for graph generation
- graphviz system package for rendering
- onnx for model inspection
- Extraction and visualization scripts working

**Installation (already complete):**
```bash
# From Phase 9 - no action needed
pip install pydot onnx
sudo apt-get install graphviz  # or brew install graphviz on macOS
```

## Architecture Patterns

### ONNX QDQ Format Pattern (CRITICAL)

**What:** QDQ (Quantize-Dequantize) format uses QuantizeLinear/DequantizeLinear pairs around standard operators instead of specialized QLinear operations.

**From ONNX Runtime documentation:**
> "The QDQ format inserts DeQuantizeLinear(QuantizeLinear(tensor)) between the original operators to simulate the quantization and dequantization process."

**Pattern structure:**
```
Standard operator graph:
  Input → Conv → Output

QDQ quantized graph:
  Input → QuantizeLinear → DequantizeLinear → Conv(FP32) → QuantizeLinear → DequantizeLinear → Output
```

**Key insight from STATE.md:**
> "QDQ format (QuantizeLinear/DequantizeLinear pairs) used in actual models instead of QLinearConv operations"

**Why this matters:**
- Phases 10-11 documented QLinearConv/QLinearMatMul as if they appear in models
- Actual models use QDQ format with standard Conv/MatMul operators
- Architecture documentation must explain the QDQ pattern, not QLinear operator placement
- Scale/zero-point parameters appear as inputs to Q/DQ nodes, not as operator attributes

**Confidence:** HIGH - Confirmed in STATE.md from actual model inspection

### Residual Connection Handling Patterns

**Pattern 1: QDQ Dequant-Add-Quant (Standard)**

Most common approach in ONNX Runtime quantization:

```
Branch 1: ... → DequantizeLinear → \
                                    Add(FP32) → QuantizeLinear → ...
Branch 2: ... → DequantizeLinear → /
```

**How it works:**
- Dequantize both branches to FP32
- Perform addition in floating-point
- Quantize result back to INT8
- Handles arbitrary scale mismatches correctly

**Tradeoff:** Requires FP32 arithmetic for the Add operation

**Pattern 2: Scale Matching (Hardware Optimization)**

Force both branches to use identical scales:

```
Branch 1: ... → INT8 (scale=S) → \
                                   Add(INT8, both use scale S) → INT8 output
Branch 2: ... → INT8 (scale=S) → /
```

**How it works:**
- Constraint: Both branches must produce same scale value
- Add operation works directly in INT8 space
- Requantize once after addition if output scale differs

**Tradeoff:** Constrains quantization calibration, may reduce accuracy

**Pattern 3: PyTorch FloatFunctional (Framework-Specific)**

PyTorch quantization approach:

```python
# In model definition:
self.add = torch.nn.quantized.FloatFunctional()

# In forward pass:
out = self.add.add(branch1, branch2)
```

**How it works:**
- Wraps add operation to track quantization statistics
- Automatically handles scale/zero-point mismatches
- Exports to ONNX as QDQ pattern (Pattern 1)

**From PyTorch documentation:**
> "Because ResNet has skip connections addition and this addition in the TorchVision implementation uses +, we would have to replace this + (torch.add equivalence) with FloatFunctional.add in the model definition."

**Confidence:** HIGH - Verified with official PyTorch and NVIDIA TensorRT documentation

### Documentation Structure Pattern

Following the established pattern from Phases 10-11:

```markdown
# Architecture Documentation

## Overview
- High-level description
- Relationship to prior phases
- ONNX specification links

## Data Flow Diagram
- Full network path visualization
- Annotated with data types at each stage

## Scale and Zero-Point Flow
- Where parameters are stored (initializers)
- How they propagate through operations
- Per-tensor vs per-channel usage

## Residual Connections
- Problem statement (scale mismatch)
- Solution approaches comparison
- ResNet8-specific implementation

## PyTorch Equivalents
- Mapping table: PyTorch op → ONNX pattern
- Conversion notes and limitations

## Network Visualization
- Graph diagram with all operations
- Scale/zero-point annotations
```

**Cross-referencing pattern (from STATE.md):**
> "Link to detailed explanations in related operation docs instead of duplicating content"

**Confidence:** HIGH - Pattern proven successful in Phases 10-11

### Graph Visualization Approaches

**Approach 1: Programmatic with pydot (Recommended for this phase)**

```python
from onnx.tools.net_drawer import GetPydotGraph
import onnx

model = onnx.load("models/resnet8_int8.onnx")
pydot_graph = GetPydotGraph(
    model.graph,
    name="resnet8_quantized",
    rankdir="TB",  # Top-to-bottom layout
    embed_docstring=True
)
pydot_graph.write_png("docs/images/resnet8_architecture.png")
```

**Outputs:**
- Hexagons: tensors
- Rectangles: operators
- Edges: data flow

**When to use:** Reproducible diagrams, consistent styling, embedded in documentation

**Approach 2: Netron (Interactive exploration)**

Web-based tool at https://netron.app or local installation:
```bash
pip install netron
netron models/resnet8_int8.onnx
```

**Features:**
- Click nodes to see properties
- Export to PNG
- Interactive zoom/pan

**When to use:** Initial exploration, understanding node connections, ad-hoc inspection

**Approach 3: Manual annotation with image tools**

Start with pydot/Netron output, annotate with:
- Scale value callouts
- Zero-point locations
- Data type transitions (FP32 → INT8 → FP32)
- Critical paths (input → output)

**When to use:** Final documentation diagrams with specific highlights

**Confidence:** HIGH - Tools verified in Phase 9, Netron widely used in ONNX community

## Don't Hand-Roll

Problems that have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ONNX graph visualization | Custom graph rendering code | onnx.tools.net_drawer.GetPydotGraph | Handles ONNX-specific node types, attributes, standard layout |
| Graph layout algorithms | Manual node positioning | Graphviz dot engine | 40+ years of graph layout research, handles complex DAGs |
| ONNX model inspection | String parsing of proto files | onnx.ModelProto API + helper functions | Type-safe access, handles all ONNX versions |
| Scale/zero-point extraction | Regex on node names | scripts/extract_operations.py (Phase 9) | Already built, tested, handles per-channel cases |
| Interactive visualization | Web UI development | Netron | Full-featured, supports all ONNX operators, actively maintained |

**Key insight:** The extraction and visualization infrastructure from Phase 9 provides everything needed. Don't rebuild; reuse and extend.

**Confidence:** HIGH - Phase 9 scripts confirmed working, Netron is industry standard

## Common Pitfalls

### Pitfall 1: Assuming QLinear Operators in Models

**What goes wrong:** Documentation describes QLinearConv/QLinearMatMul as if they appear in the actual model graph, but models use QDQ format instead.

**Why it happens:** ONNX specification defines QLinear operators, natural to assume they're used. Phases 10-11 documented these operations based on spec, not actual model format.

**How to avoid:**
- Inspect actual quantized models FIRST before planning documentation
- Use `scripts/extract_operations.py` to identify operation types
- Document what IS (QDQ format) not what COULD BE (QLinear operators)

**Warning signs:**
- Documentation references QLinearConv nodes but extraction scripts find zero instances
- Graph visualization shows Conv + Q/DQ nodes, not QLinearConv
- ONNX Runtime logs mention "QDQ format" during inference

**From STATE.md:**
> "Architecture documentation (Phase 12) will work with QDQ format models, not QLinearConv"

**Confidence:** HIGH - Critical finding from prior phase execution

### Pitfall 2: Ignoring Residual Connection Scale Mismatch

**What goes wrong:** Documentation shows residual Add operations without explaining scale parameter handling, making hardware implementation impossible.

**Why it happens:** In floating-point ResNets, addition is trivial. In quantized ResNets, scale mismatches require explicit handling.

**How to avoid:**
- Identify all Add operations in ResNet8 graph
- Extract scale parameters for both input branches to each Add
- Document the scale matching problem explicitly
- Compare solution approaches (QDQ dequant-add-quant vs scale constraints)

**Warning signs:**
- Architecture diagram shows Add nodes without Q/DQ annotations
- No discussion of what happens when `scale_branch1 ≠ scale_branch2`
- Hardware implementers ask "how do I add INT8 values with different scales?"

**From NVIDIA TensorRT issue tracker:**
> "You need to add Q/DQ pairs in the residual add branch to get the best performance when running in INT8"

**Confidence:** HIGH - Documented in multiple quantization frameworks (TensorRT, ONNX Runtime)

### Pitfall 3: PyTorch-ONNX Mapping Oversimplification

**What goes wrong:** Documentation claims direct equivalence between PyTorch quantized ops and ONNX without noting conversion limitations.

**Why it happens:** Conceptually, torch.nn.quantized.Conv2d ≈ QDQ+Conv, but actual export has restrictions.

**How to avoid:**
- Test actual PyTorch → ONNX conversion with quantized models
- Document known limitations (e.g., `aten::quantize_per_channel` not supported in ONNX export)
- Provide recommended workflow: export FP32, then quantize with ONNX Runtime tools

**Warning signs:**
- Mapping table shows 1:1 equivalence without caveats
- No mention of opset version requirements
- Recommendation to "just export quantized PyTorch model to ONNX"

**From PyTorch GitHub issues:**
> "Exporting the operator 'aten::quantize_per_channel' to ONNX opset version 15 is not supported"

**Confidence:** MEDIUM - Known issue in PyTorch, but workarounds exist (export FP32 then quantize)

### Pitfall 4: Initializer vs Input Confusion

**What goes wrong:** Documentation doesn't clarify where scale/zero-point parameters are stored (initializers vs runtime inputs).

**Why it happens:** ONNX allows both: constant initializers (baked into model) or runtime inputs (provided at inference).

**How to avoid:**
- Use `graph.initializer` to identify constant parameters
- Document that scales/zero-points are typically initializers (constants)
- Note when a parameter could be runtime-configurable vs fixed at model creation

**Warning signs:**
- Diagram shows scale parameters without indicating source (initializer or input)
- Hardware implementation assumes runtime inputs when values are actually constant
- No mention of `graph.initializer` vs `graph.input`

**From ONNX IR specification:**
> "When an initializer has the same name as a graph input, it specifies a default value for that input."

**Confidence:** HIGH - Fundamental ONNX concept, well-documented

### Pitfall 5: Missing External Data Handling

**What goes wrong:** Large models store weights in external files, but documentation only covers inline initializers.

**Why it happens:** Small test models (ResNet8) keep all data inline, so external data case isn't encountered during development.

**How to avoid:**
- Check if model uses external data: `onnx.load(..., load_external_data=True)`
- Document the pattern even if ResNet8 doesn't use it (for completeness)
- Note byte offset and length fields in TensorProto for external data

**Warning signs:**
- Model file is tiny (<1MB) but claims to have millions of parameters
- Extraction script fails to find initializer data
- ONNX load succeeds but numpy_helper.to_array returns empty

**From ONNX IR specification:**
> "The raw data for large constant tensors MAY be serialised in a separate file. In such a case, the tensor MUST provide the filename relative to the model file."

**Confidence:** MEDIUM - Not relevant for ResNet8, but important for completeness

## Code Examples

Verified patterns from official sources and existing project scripts:

### Extracting QDQ Operations from Model

```python
# Source: Adapted from scripts/extract_operations.py (Phase 9)
import onnx
import onnx.numpy_helper as nph

model = onnx.load("models/resnet8_int8.onnx")
graph = model.graph

# Build initializer lookup
initializers = {}
for init in graph.initializer:
    initializers[init.name] = nph.to_array(init)

# Find all QuantizeLinear and DequantizeLinear nodes
qdq_nodes = []
for node in graph.node:
    if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
        node_data = {
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        }

        # Extract scale and zero-point from initializers
        if len(node.input) >= 2 and node.input[1] in initializers:
            node_data["scale"] = float(initializers[node.input[1]])
        if len(node.input) >= 3 and node.input[2] in initializers:
            node_data["zero_point"] = int(initializers[node.input[2]])

        qdq_nodes.append(node_data)

print(f"Found {len(qdq_nodes)} QDQ nodes")
```

**Confidence:** HIGH - Direct adaptation of working Phase 9 code

### Generating Architecture Visualization

```python
# Source: scripts/visualize_graph.py (Phase 9)
import onnx
from onnx.tools.net_drawer import GetPydotGraph
import subprocess

model = onnx.load("models/resnet8_int8.onnx")
graph = model.graph

# Generate pydot graph
pydot_graph = GetPydotGraph(
    graph,
    name="resnet8_quantized",
    rankdir="TB",  # Top-to-bottom
    embed_docstring=True
)

# Write DOT file
pydot_graph.write_dot("docs/images/resnet8_architecture.dot")

# Convert to PNG using Graphviz
subprocess.run([
    "dot", "-Tpng",
    "docs/images/resnet8_architecture.dot",
    "-o", "docs/images/resnet8_architecture.png"
], check=True)

# Convert to SVG (scalable for documentation)
subprocess.run([
    "dot", "-Tsvg",
    "docs/images/resnet8_architecture.dot",
    "-o", "docs/images/resnet8_architecture.svg"
], check=True)
```

**Confidence:** HIGH - Existing script from Phase 9, already tested

### Identifying Residual Connection Add Nodes

```python
# Source: Original research for Phase 12
import onnx

model = onnx.load("models/resnet8_int8.onnx")
graph = model.graph

# Find all Add operations (potential residual connections)
add_nodes = [node for node in graph.node if node.op_type == "Add"]

print(f"Found {len(add_nodes)} Add operations")

# For each Add, trace back to find input scales
for add_node in add_nodes:
    print(f"\nAdd node: {add_node.name}")
    print(f"  Inputs: {list(add_node.inputs)}")

    # Find DequantizeLinear nodes feeding this Add
    for input_name in add_node.input:
        for node in graph.node:
            if node.op_type == "DequantizeLinear" and input_name in node.output:
                print(f"  Branch input from: {node.name}")
                if len(node.input) >= 2:
                    print(f"    Scale parameter: {node.input[1]}")
```

**Confidence:** HIGH - Standard ONNX graph traversal pattern

### PyTorch Quantized Operations Table

```python
# Source: PyTorch documentation + ONNX Runtime quantization docs
# This is a reference mapping, not executable code

pytorch_to_onnx_mapping = {
    # PyTorch Quantized Op → ONNX QDQ Pattern
    "torch.nn.quantized.Conv2d": "QuantizeLinear → DequantizeLinear → Conv → QuantizeLinear → DequantizeLinear",
    "torch.nn.quantized.Linear": "QuantizeLinear → DequantizeLinear → MatMul → QuantizeLinear → DequantizeLinear",
    "torch.nn.quantized.ReLU": "QuantizeLinear → DequantizeLinear → Relu → QuantizeLinear → DequantizeLinear",
    "torch.nn.quantized.FloatFunctional.add": "DequantizeLinear (×2) → Add → QuantizeLinear",

    # Note: Direct export often fails, recommended workflow:
    # 1. Export FP32 PyTorch model to ONNX
    # 2. Quantize ONNX model with ONNX Runtime tools
}
```

**Confidence:** MEDIUM - Conceptual mapping verified, but direct export has limitations

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| QOperator format (QLinearConv nodes) | QDQ format (Q/DQ pairs around standard ops) | ONNX Runtime 1.6+ (2020) | Better debugging, easier optimization, wider hardware support |
| Per-tensor quantization only | Per-channel quantization support | ONNX opset 13+ (2021) | Better accuracy for CNNs (Phase 11: 0.17% overhead for 256 channels) |
| Symmetric quantization (zero_point=0) | Asymmetric quantization support | ONNX opset 10+ (2019) | Better range utilization for ReLU activations |
| Manual Q/DQ node insertion | Automatic ONNX Runtime quantization API | ONNX Runtime 1.5+ (2020) | Easier workflow, calibration built-in |
| Export quantized PyTorch → ONNX | Export FP32 → quantize with ONNX Runtime | 2021-present | Workaround for PyTorch export limitations |

**Deprecated/outdated:**
- **QOperator format**: Still supported but QDQ is recommended (ONNX Runtime docs: "QDQ format is the default")
- **pydot.write_png()**: Deprecated, use subprocess.run with dot command (Phase 9 decision)
- **Direct quantized PyTorch export**: Limited opset support, use FP32 export + ONNX Runtime quantization instead

**Confidence:** HIGH - All transitions verified with official documentation and GitHub issue trackers

## Open Questions

Things that couldn't be fully resolved:

### 1. Actual QLinear Operator Usage in Production

**What we know:**
- ONNX specification defines QLinearConv, QLinearMatMul
- Phases 10-11 documented these operations based on spec
- ONNX Runtime recommends QDQ format as default

**What's unclear:**
- Are there any production deployments using QOperator format (QLinearConv nodes)?
- Is QOperator format only for specific hardware backends?
- Should documentation mention QLinear operators at all for ResNet8 architecture?

**Recommendation:**
- Document the QDQ format as the primary implementation (it's what ResNet8 uses)
- Add brief note that QLinear operators exist in ONNX spec but aren't used in this model
- Cross-reference to Phases 10-11 for QLinear operation math (useful for hardware understanding)

**Confidence:** MEDIUM - QDQ format confirmed for ResNet8, but broader ecosystem usage unclear

### 2. Optimal Residual Connection Implementation for Hardware

**What we know:**
- Three approaches: QDQ dequant-add-quant, scale matching, PyTorch FloatFunctional
- QDQ approach is most flexible (handles arbitrary scale mismatches)
- Scale matching enables pure INT8 add (faster on some hardware)

**What's unclear:**
- What's the accuracy tradeoff of scale matching for ResNet8 specifically?
- Do analog accelerators prefer one approach over another?
- Should documentation recommend a specific approach?

**Recommendation:**
- Document all three approaches with tradeoffs
- Show what ResNet8 quantized model actually uses (inspect with extraction script)
- Note that choice depends on hardware constraints and accuracy requirements
- Defer hardware-specific recommendations to Phase 13

**Confidence:** MEDIUM - Approaches are well-documented, but optimal choice is hardware-dependent

### 3. Scale/Zero-Point Parameter Precision Requirements

**What we know:**
- ONNX specification: scales are float32
- Requantization formula uses scale factors in floating-point

**What's unclear:**
- Can scales be represented in fixed-point for hardware (e.g., Q8.24)?
- What precision is actually needed to match ONNX Runtime accuracy?
- Are there known quantization schemes for scale parameters themselves?

**Recommendation:**
- Document that ONNX uses float32 scales
- Note this as a consideration for hardware implementation
- Defer precision analysis to Phase 13 (Hardware Implementation Guide)
- Test vectors from validation scripts can help determine minimum precision

**Confidence:** LOW - Hardware-specific concern, needs experimental validation

### 4. External Data Handling for Large Models

**What we know:**
- ONNX supports external data files for large tensors
- ResNet8 is small enough to keep all data inline

**What's unclear:**
- Should Phase 12 documentation cover external data pattern?
- Is it worth documenting for completeness even if ResNet8 doesn't use it?

**Recommendation:**
- Brief mention in "Scale and Zero-Point Locations" section
- Note that ResNet8 uses inline initializers (no external files)
- Link to ONNX IR specification for external data details
- Don't spend significant effort on a case that doesn't occur in this project

**Confidence:** MEDIUM - Pattern is well-documented in ONNX spec, just question of scope

## Sources

### Primary (HIGH confidence)

- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) - QDQ vs QOperator formats, recommended practices
- [ONNX QuantizeLinear Specification](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html) - Official operator definition
- [ONNX DequantizeLinear Specification](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html) - Official operator definition
- [ONNX QLinearConv Specification](https://onnx.ai/onnx/operators/onnx__QLinearConv.html) - Reference for understanding QOperator format
- [ONNX IR Specification](https://github.com/onnx/onnx/blob/main/docs/IR.md) - Graph structure, nodes, initializers
- [ONNX Concepts Documentation](https://onnx.ai/onnx/intro/concepts.html) - Graph components, best practices
- [PyTorch FloatFunctional](https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.FloatFunctional.html) - Quantized residual connections
- [Netron Repository](https://github.com/lutzroeder/netron) - ONNX visualization tool
- [ONNX Tutorials: Visualizing Models](https://github.com/onnx/tutorials/blob/main/tutorials/VisualizingAModel.md) - GetPydotGraph usage

### Secondary (MEDIUM confidence)

- [NVIDIA TensorRT Issue #1992](https://github.com/NVIDIA/TensorRT/issues/1992) - ResNet quantization with Q/DQ pairs
- [PyTorch Forums: Converting quantized models to ONNX](https://discuss.pytorch.org/t/converting-quantized-models-from-pytorch-to-onnx/84855) - Known limitations
- [PyTorch Static Quantization Guide](https://leimao.github.io/blog/PyTorch-Static-Quantization/) - FloatFunctional usage
- [TorchVision Quantized ResNet](https://docs.pytorch.org/vision/main/models/resnet_quant.html) - Reference implementation
- Medium: [PyTorch to Quantized ONNX Model](https://medium.com/@hdpoorna/pytorch-to-quantized-onnx-model-18cf2384ec27) - Conversion workflow

### Tertiary (LOW confidence - marked for validation)

- Various Stack Overflow and forum discussions on quantization (not cited specifically)
- Academic papers on quantization (high-level context, not technical details)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All tools already installed and validated in Phase 9
- Architecture patterns: HIGH - QDQ format confirmed in actual models, multiple official sources
- Pitfalls: HIGH - Issues documented in official bug trackers and specification
- PyTorch mapping: MEDIUM - Conceptual mapping clear, but export has known limitations
- Hardware recommendations: MEDIUM - Depends on hardware-specific constraints (deferred to Phase 13)

**Research date:** 2026-02-03
**Valid until:** 90 days (stable domain - ONNX specification changes slowly, quantization patterns are well-established)

**Critical finding for planning:**
Phase 12 must document the **QDQ format architecture** (what models actually use), not the QLinear operator architecture (what Phases 10-11 documented based on spec). This requires:
1. Using extraction scripts to identify Q/DQ node patterns
2. Tracing scale/zero-point parameter flow through the graph
3. Explaining why QDQ format differs from QLinear operators
4. Showing how the two-stage computation from Phase 11 still applies (just implemented differently)

The good news: All required infrastructure exists from Phase 9. The challenge: Bridging the conceptual gap between "QLinear operator math" (Phases 10-11) and "QDQ format architecture" (Phase 12 reality).
