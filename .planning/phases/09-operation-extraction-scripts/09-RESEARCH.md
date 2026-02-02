# Phase 9: Operation Extraction Scripts - Research

**Researched:** 2026-02-02
**Domain:** ONNX model analysis and graph visualization
**Confidence:** HIGH

## Summary

Phase 9 creates programmatic tools to extract quantized operation details from ONNX models and generate visual representations of model graphs. This enables data-driven documentation in subsequent phases by providing structured JSON data about QLinear operations and PNG/SVG visualizations.

The standard approach uses the official ONNX Python library for model parsing and graph traversal, with `onnx.helper.get_attribute_value()` for extracting node attributes. For visualization, the official `onnx.tools.net_drawer` module generates Graphviz `.dot` files, which are then rendered to PNG/SVG using the `dot` command-line utility. No third-party graph manipulation libraries are needed - the built-in ONNX API provides all necessary functionality.

The research confirms that quantized ONNX models store scales and zero-points as initializers (constants) in the graph, making them accessible through `model.graph.initializer`. The four target operation types (QLinearConv, QLinearMatMul, QuantizeLinear, DequantizeLinear) all follow standard ONNX operator patterns with well-documented input/output structures.

**Primary recommendation:** Use native ONNX library APIs for extraction, `net_drawer` for visualization, and `json` module for structured output. Add only `pydot` and `graphviz` as new dependencies.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| onnx | >=1.17.0 | ONNX model loading and graph traversal | Official ONNX implementation, already in project |
| json | stdlib | JSON serialization of extracted data | Python standard library, zero dependencies |
| pydot | >=2.0.0 | Python interface to Graphviz | Required by net_drawer for graph generation |
| graphviz | system | Graph rendering engine (dot command) | Industry-standard graph visualization tool |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| onnx.helper | (part of onnx) | Attribute value extraction | Use `get_attribute_value()` for all node attributes |
| onnx.tools.net_drawer | (part of onnx) | Graph visualization generation | Use `GetPydotGraph()` to create .dot files |
| argparse | stdlib | CLI argument parsing | Consistent with existing project scripts |
| logging | stdlib | Progress and error reporting | Consistent with existing project scripts |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| onnx.tools.net_drawer | Netron (GUI tool) | Netron is excellent for interactive exploration but doesn't support programmatic batch generation |
| pydot | pygraphviz | pygraphviz requires C extensions and more complex installation; pydot is pure Python wrapper |
| Custom JSON | google.protobuf.json_format.MessageToDict | MessageToDict includes protobuf metadata clutter; custom extraction gives cleaner output |

**Installation:**
```bash
# Already in project:
# onnx>=1.17.0

# New dependencies for v1.3:
pip install pydot>=2.0.0

# System package (varies by OS):
# Ubuntu/Debian: sudo apt-get install graphviz
# macOS: brew install graphviz
# Windows: Download from https://graphviz.org/download/
```

## Architecture Patterns

### Recommended Project Structure
```
scripts/
├── extract_operations.py    # Extract QLinear nodes to JSON
├── visualize_graph.py        # Generate PNG/SVG from ONNX models
└── (existing scripts)        # convert.py, evaluate.py, quantize_*.py
```

### Pattern 1: Graph Traversal for Node Extraction
**What:** Load ONNX model, iterate through graph nodes, filter by op_type, extract attributes and inputs
**When to use:** Extracting specific operation types with their metadata
**Example:**
```python
# Source: https://onnx.ai/onnx/intro/python.html
import onnx
from onnx.helper import get_attribute_value

model = onnx.load("model.onnx")
graph = model.graph

# Extract all QLinear operations
qlinear_ops = []
for node in graph.node:
    if node.op_type in ["QLinearConv", "QLinearMatMul",
                         "QuantizeLinear", "DequantizeLinear"]:
        # Extract node metadata
        node_data = {
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": {}
        }

        # Extract attributes with proper type handling
        for attr in node.attribute:
            node_data["attributes"][attr.name] = get_attribute_value(attr)

        qlinear_ops.append(node_data)
```

### Pattern 2: Initializer Lookup for Scales and Zero-Points
**What:** Access constant tensors (initializers) to retrieve quantization parameters
**When to use:** Getting scale/zero-point values referenced by QLinear node inputs
**Example:**
```python
# Source: https://onnx.ai/onnx/intro/python.html
import numpy as np

# Build initializer lookup dictionary
initializers = {}
for init in graph.initializer:
    # Convert TensorProto to numpy array
    initializers[init.name] = onnx.numpy_helper.to_array(init)

# For each QLinearConv node, get its scale values
for node in graph.node:
    if node.op_type == "QLinearConv":
        # QLinearConv inputs: [x, x_scale, x_zero_point, w, w_scale, ...]
        x_scale_name = node.input[1]
        if x_scale_name in initializers:
            x_scale_value = initializers[x_scale_name]
            print(f"x_scale: {x_scale_value}")
```

### Pattern 3: Graph Visualization with net_drawer
**What:** Generate Graphviz .dot file, convert to PNG/SVG with dot command
**When to use:** Creating visual diagrams of ONNX model structure
**Example:**
```python
# Source: https://github.com/onnx/tutorials/blob/main/tutorials/VisualizingAModel.md
import onnx
from onnx.tools.net_drawer import GetPydotGraph
import subprocess

model = onnx.load("resnet8_int8.onnx")

# Generate .dot file
pydot_graph = GetPydotGraph(
    model.graph,
    name="ResNet8_Quantized",
    rankdir="TB",  # Top-to-bottom layout
    embed_docstring=True  # Include operator docs as tooltips
)
pydot_graph.write_dot("resnet8_int8.dot")

# Convert .dot to PNG and SVG using system dot command
subprocess.run(["dot", "-Tpng", "resnet8_int8.dot", "-o", "resnet8_int8.png"])
subprocess.run(["dot", "-Tsvg", "resnet8_int8.dot", "-o", "resnet8_int8.svg"])
```

### Pattern 4: JSON Output Structure
**What:** Serialize extracted node data to structured JSON for documentation consumption
**When to use:** Creating machine-readable data for subsequent documentation phases
**Example:**
```python
import json

# Structure: model-level metadata + list of operations
output_data = {
    "model_path": "resnet8_int8.onnx",
    "opset_version": model.opset_import[0].version,
    "graph_name": model.graph.name,
    "operations": qlinear_ops,  # From Pattern 1
    "summary": {
        "total_nodes": len(graph.node),
        "qlinear_nodes": len(qlinear_ops),
        "op_type_counts": {
            "QLinearConv": sum(1 for op in qlinear_ops if op["op_type"] == "QLinearConv"),
            "QLinearMatMul": sum(1 for op in qlinear_ops if op["op_type"] == "QLinearMatMul"),
            "QuantizeLinear": sum(1 for op in qlinear_ops if op["op_type"] == "QuantizeLinear"),
            "DequantizeLinear": sum(1 for op in qlinear_ops if op["op_type"] == "DequantizeLinear"),
        }
    }
}

with open("resnet8_int8_operations.json", "w") as f:
    json.dump(output_data, f, indent=2)
```

### Anti-Patterns to Avoid
- **Using MessageToDict on entire model:** Produces enormous JSON with protobuf metadata clutter. Extract only needed fields instead.
- **Iterating initializers repeatedly:** Build single lookup dict at start (Pattern 2), don't search initializers for each node.
- **Generating PNG/SVG directly in Python:** net_drawer only outputs .dot format. Use subprocess to call system `dot` command.
- **Skipping attribute type checking:** Use `onnx.helper.get_attribute_value()` which handles type dispatch automatically. Don't access `.f`, `.i`, `.s` fields directly.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Graph visualization | Custom graph renderer | onnx.tools.net_drawer + graphviz | Handles subgraphs, control flow, proper layout. Already tested on millions of models. |
| Attribute value extraction | Direct protobuf field access | onnx.helper.get_attribute_value() | Handles union types (float/int/string/tensor/graph/repeated), avoids type checking boilerplate |
| Tensor data extraction | Manual protobuf parsing | onnx.numpy_helper.to_array() | Handles all tensor data types, endianness, raw_data vs list formats |
| ONNX model validation | Try/except on load | onnx.checker.check_model() | Validates protobuf structure, type consistency, shape inference, opset compatibility |
| Graph layout algorithms | Custom graph positioning | graphviz dot engine | Professional layout algorithms (hierarchical, orthogonal, force-directed). Decades of optimization. |

**Key insight:** ONNX protobuf structures are more complex than they appear. Nested union types, optional fields, and multiple serialization formats create edge cases. Official helper functions handle these correctly.

## Common Pitfalls

### Pitfall 1: Missing System Graphviz Installation
**What goes wrong:** Script imports pydot successfully but crashes when calling `write_png()` or system `dot` command fails
**Why it happens:** pydot is just a Python wrapper. The actual rendering requires Graphviz system binaries (dot, neato, etc.)
**How to avoid:** Document system dependency clearly. Check for `dot` command availability with `subprocess.run(["dot", "-V"])` and provide helpful error message.
**Warning signs:** ImportError doesn't occur, but FileNotFoundError or "graphviz executables not found" at runtime

### Pitfall 2: Scale/Zero-Point Value Misidentification
**What goes wrong:** Script extracts node input names (strings like "x_scale") but not the actual numeric values
**Why it happens:** QLinear operations reference scales/zero-points by name. The values are stored in `graph.initializer`, not in the node itself.
**How to avoid:** Use Pattern 2 (Initializer Lookup). Build dict mapping initializer names to numpy arrays before processing nodes.
**Warning signs:** JSON output contains strings like "Conv_0_x_scale" instead of numeric values like `[0.00784314]`

### Pitfall 3: Incomplete Quantized Operation Coverage
**What goes wrong:** Script only extracts QLinearConv/QLinearMatMul but misses QuantizeLinear/DequantizeLinear boundary operations
**Why it happens:** Focusing only on "QLinear*" prefix pattern. QuantizeLinear and DequantizeLinear don't follow naming convention.
**How to avoid:** Explicitly list all four operation types: `["QLinearConv", "QLinearMatMul", "QuantizeLinear", "DequantizeLinear"]`
**Warning signs:** ResNet8 has ~20 quantized nodes but extraction only finds ~5

### Pitfall 4: Attribute Access Without Type Checking
**What goes wrong:** Code like `attr.f` or `attr.i` returns 0 when attribute is different type, causing silent data corruption
**Why it happens:** AttributeProto is protobuf union type. Accessing wrong field returns default value (0 for numbers, "" for string).
**How to avoid:** Always use `onnx.helper.get_attribute_value(attr)` which automatically checks type and returns correct field.
**Warning signs:** All extracted attribute values are 0 or empty string

### Pitfall 5: Graph Visualization Output Size
**What goes wrong:** PNG/SVG of full ResNet8 quantized model is huge or unreadable due to layout density
**Why it happens:** Quantized graphs have many more nodes than FP32 (each Conv becomes QLinearConv + 5 scale/zero-point inputs). Default layout packs everything tightly.
**How to avoid:** Test visualization output early. Consider per-layer subgraph extraction if full graph is too cluttered. Use rankdir="TB" for vertical layout which often works better for CNNs.
**Warning signs:** Generated PNG is >10MB or all text is illegible due to size

### Pitfall 6: JSON Serialization of NumPy Types
**What goes wrong:** `json.dumps()` crashes with "Object of type float32 is not JSON serializable"
**Why it happens:** ONNX initializers convert to NumPy arrays with NumPy dtypes. JSON encoder doesn't handle `np.float32`, `np.int8`, etc.
**How to avoid:** Convert NumPy values to Python types: `float(value)` for scalars, `value.tolist()` for arrays
**Warning signs:** Script runs fine until trying to serialize extracted scales/zero-points

## Code Examples

Verified patterns from official sources:

### Loading ONNX Model and Accessing Graph
```python
# Source: https://onnx.ai/onnx/intro/python.html
import onnx

model = onnx.load("resnet8_int8.onnx")
graph = model.graph

print(f"Graph name: {graph.name}")
print(f"Nodes: {len(graph.node)}")
print(f"Initializers: {len(graph.initializer)}")
```

### Safe Attribute Extraction
```python
# Source: https://onnx.ai/onnx/api/helper.html
from onnx.helper import get_attribute_value

for node in graph.node:
    print(f"Node: {node.name} ({node.op_type})")
    for attr in node.attribute:
        value = get_attribute_value(attr)
        print(f"  {attr.name}: {value}")
```

### Converting Initializers to NumPy
```python
# Source: https://onnx.ai/onnx/intro/python.html
import onnx.numpy_helper as nph

for init in graph.initializer:
    array = nph.to_array(init)
    print(f"{init.name}: shape={array.shape}, dtype={array.dtype}")
```

### Generating Graph Visualization
```python
# Source: https://github.com/onnx/tutorials/blob/main/tutorials/VisualizingAModel.md
from onnx.tools.net_drawer import GetPydotGraph
import subprocess

# Load model
model = onnx.load("resnet8_int8.onnx")

# Create pydot graph
pydot_graph = GetPydotGraph(
    model.graph,
    name=model.graph.name or "Model",
    rankdir="TB",  # TB=top-to-bottom, LR=left-to-right
    embed_docstring=True
)

# Write .dot file
dot_path = "resnet8_int8.dot"
pydot_graph.write_dot(dot_path)

# Convert to PNG and SVG using graphviz dot command
subprocess.run(["dot", "-Tpng", dot_path, "-o", "resnet8_int8.png"], check=True)
subprocess.run(["dot", "-Tsvg", dot_path, "-o", "resnet8_int8.svg"], check=True)
```

### Complete Extraction Script Pattern
```python
# Source: Synthesis from https://onnx.ai/onnx/intro/python.html
import json
import onnx
import onnx.numpy_helper as nph
from onnx.helper import get_attribute_value

def extract_qlinear_operations(model_path):
    """Extract all quantized operations from ONNX model."""
    model = onnx.load(model_path)
    graph = model.graph

    # Build initializer lookup
    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = nph.to_array(init)

    # Extract quantized operations
    qlinear_ops = []
    target_ops = ["QLinearConv", "QLinearMatMul",
                  "QuantizeLinear", "DequantizeLinear"]

    for node in graph.node:
        if node.op_type not in target_ops:
            continue

        # Extract node metadata
        node_data = {
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": {},
            "scales": {},
            "zero_points": {}
        }

        # Extract attributes
        for attr in node.attribute:
            value = get_attribute_value(attr)
            node_data["attributes"][attr.name] = value

        # Extract scales and zero-points from initializers
        for input_name in node.input:
            if input_name in initializers:
                value = initializers[input_name]
                # Convert numpy to Python types for JSON serialization
                if value.ndim == 0:  # scalar
                    py_value = float(value) if value.dtype.kind == 'f' else int(value)
                else:  # array
                    py_value = value.tolist()

                # Categorize by naming convention
                if "scale" in input_name.lower():
                    node_data["scales"][input_name] = py_value
                elif "zero" in input_name.lower():
                    node_data["zero_points"][input_name] = py_value

        qlinear_ops.append(node_data)

    return {
        "model_path": model_path,
        "operations": qlinear_ops,
        "summary": {
            "total_nodes": len(graph.node),
            "qlinear_nodes": len(qlinear_ops)
        }
    }

# Usage
data = extract_qlinear_operations("resnet8_int8.onnx")
with open("operations.json", "w") as f:
    json.dump(data, indent=2, fp=f)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| onnx.utils.extract_model() | onnx_ir RecursiveGraphIterator | 2024+ (onnx_ir release) | New ir-py provides modern API, but original approach still valid for simple traversal |
| Manual protobuf field access | onnx.helper.get_attribute_value() | Always recommended | Helper function exists since early ONNX versions, but still commonly bypassed |
| Netron GUI for all visualization | net_drawer for programmatic + Netron for interactive | Ongoing | Use both: net_drawer for CI/docs generation, Netron for manual exploration |
| ONNX 1.10 opset support | ONNX 1.17+ with opset 15-25 | 2023-2025 | QuantizeLinear v25 adds blocked quantization, DequantizeLinear v21 adds output_dtype |

**Deprecated/outdated:**
- **onnx.utils.Extractor:** Use onnx.utils.extract_model() for subgraph extraction (renamed in ONNX 1.9+)
- **Direct .f/.i/.s attribute access:** Use onnx.helper.get_attribute_value() instead (handles type unions correctly)
- **pydot.write_png():** Deprecated in pydot 2.0+. Use write_dot() + subprocess call to graphviz dot command

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal graph visualization layout for quantized ResNet8**
   - What we know: net_drawer supports rankdir="TB" (top-to-bottom) and rankdir="LR" (left-right). Full graph will be dense.
   - What's unclear: Whether full graph is legible or if per-residual-block subgraphs are needed
   - Recommendation: Generate full graph first, assess readability. If cluttered, Phase 12 (Architecture Documentation) can extract subgraphs using onnx.utils.extract_model()

2. **Per-channel quantization representation in JSON**
   - What we know: QLinearConv w_scale can be 1-D tensor for per-channel quantization
   - What's unclear: Best JSON structure for representing per-channel scales (flat list vs. nested structure)
   - Recommendation: Use flat list with metadata: `{"w_scale": {"shape": [64], "values": [0.1, 0.11, ...], "per_channel": true}}`

3. **Graphviz installation verification strategy**
   - What we know: dot command must be in PATH. Installation varies by OS.
   - What's unclear: Best way to check for Graphviz at script runtime (subprocess vs. shutil.which vs. try/except)
   - Recommendation: Use `subprocess.run(["dot", "-V"], capture_output=True)` at script start with clear error message pointing to https://graphviz.org/download/

## Sources

### Primary (HIGH confidence)
- [ONNX Python API Documentation](https://onnx.ai/onnx/intro/python.html) - Loading models, graph traversal, node access
- [ONNX Helper API](https://onnx.ai/onnx/api/helper.html) - get_attribute_value() usage
- [ONNX Protobuf Classes](https://onnx.ai/onnx/api/classes.html) - AttributeProto, NodeProto structure
- [ONNX Operators - QLinearConv](https://onnx.ai/onnx/operators/onnx__QLinearConv.html) - Complete specification
- [ONNX Operators - QLinearMatMul](https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html) - Complete specification
- [ONNX Operators - QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html) - Complete specification
- [ONNX Operators - DequantizeLinear](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html) - Complete specification
- [ONNX Tutorial - Visualizing Models](https://github.com/onnx/tutorials/blob/main/tutorials/VisualizingAModel.md) - net_drawer usage
- [ONNX net_drawer source](https://github.com/onnx/onnx/blob/main/onnx/tools/net_drawer.py) - GetPydotGraph API
- [Graphviz Command Line](https://graphviz.org/doc/info/command.html) - dot command usage
- [Graphviz Output Formats](https://graphviz.org/docs/outputs/) - PNG, SVG format options
- [pydot PyPI](https://pypi.org/project/pydot/) - Python Graphviz wrapper, version 2.0.0+

### Secondary (MEDIUM confidence)
- [ONNX Runtime Quantization Guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) - Quantization concepts, initializer storage
- [Google Protocol Buffers JSON Format](https://googleapis.dev/python/protobuf/latest/google/protobuf/json_format.html) - MessageToDict (not recommended for this use case)

### Tertiary (LOW confidence)
- None - all findings verified with official documentation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries from official ONNX ecosystem or Python stdlib, versions verified
- Architecture: HIGH - Patterns extracted from official tutorials and API documentation with working examples
- Pitfalls: MEDIUM - Based on common ONNX usage patterns and web search findings, not all tested on ResNet8 specifically

**Research date:** 2026-02-02
**Valid until:** 2026-03-02 (30 days - ONNX is stable, infrequent breaking changes)
