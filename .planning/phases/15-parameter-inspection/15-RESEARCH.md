# Phase 15: Parameter Inspection - Research

**Researched:** 2026-02-07
**Domain:** ONNX quantization parameter extraction and visualization with Marimo/matplotlib
**Confidence:** HIGH

## Summary

Phase 15 adds parameter inspection capabilities to the existing Marimo notebook (Phase 14). Users can explore quantization parameters (scales, zero-points, weight tensors) per layer, compare FP32 vs quantized weight distributions via histograms, and view a whole-model range heatmap to identify quantization trouble spots.

The standard approach uses ONNX's `numpy_helper.to_array()` to extract initializer data, matplotlib for side-by-side histograms and horizontal bar charts, and Marimo's reactive cell system to automatically update visualizations when layer selection changes. All operations work with ONNX models only (QDQ format), not PyTorch models.

Key technical insight: ONNX QDQ quantized models store scales/zero-points as initializers (constant tensors) in the graph, not as runtime inputs. Extraction involves traversing `graph.initializer` and matching names from QuantizeLinear/DequantizeLinear node inputs. Per-channel quantization stores scales as 1D arrays; per-tensor uses scalars.

**Primary recommendation:** Extract parameters via `onnx.numpy_helper.to_array()`, use matplotlib's `fig, axes = plt.subplots(1, N)` for side-by-side histograms with consistent binning, return Axes objects directly in Marimo cells for automatic rendering.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| onnx | >=1.17.0 | ONNX model loading and graph traversal | Official ONNX Python SDK, provides numpy_helper for tensor extraction |
| numpy | >=1.26.4 | Array operations and statistics | Universal standard for numerical computing, histogram binning, array math |
| matplotlib | >=3.0 | Histogram and heatmap visualization | Industry standard for scientific plotting, excellent subplot support |
| marimo | >=0.15.5 | Reactive notebook environment | Already in Phase 14, provides reactive cell updates and UI widgets |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | >=1.0 | Tabular data display | Optional for formatting parameter tables with better control |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| matplotlib | plotly | User explicitly requested matplotlib; Plotly adds interactive features but requires additional library and learning curve |
| matplotlib | seaborn | Seaborn adds statistical styling but is overkill for basic histograms and heatmaps; matplotlib sufficient |
| Manual extraction | onnxruntime inference | onnxruntime provides inference, not parameter inspection; direct ONNX graph traversal more appropriate |

**Installation:**
Already in project dependencies (pyproject.toml). No additional libraries needed.

## Architecture Patterns

### Recommended Notebook Structure
```
playground/quantization.py (existing Marimo notebook)
├── Cell: Model loading (Phase 14 - existing)
├── Cell: Layer dropdown (Phase 14 - existing)
├── Cell: **NEW - Whole-model heatmap overview**
│   └── Output: matplotlib figure with two side-by-side bar charts (FP32, INT8 ranges)
├── Cell: **NEW - Extract parameters for selected layer**
│   └── Reactive to layer_selector.value
├── Cell: **NEW - Parameter table display**
│   └── Output: Formatted table with scale, zero-point, shape, dtype
├── Cell: **NEW - Weight histograms (FP32 vs INT8)**
│   └── Output: matplotlib figure with 2-3 side-by-side histograms
└── playground/utils/parameter_inspector.py (NEW utility module)
    ├── extract_layer_parameters()
    ├── extract_weight_tensors()
    ├── compute_layer_ranges()
    └── create_summary_stats()
```

### Pattern 1: ONNX Parameter Extraction
**What:** Extract scales, zero-points, and weight tensors from ONNX QDQ models
**When to use:** Whenever displaying or analyzing quantization parameters
**Example:**
```python
# Source: ONNX documentation + existing extract_operations.py script
import onnx
from onnx import numpy_helper as nph

def extract_layer_parameters(model: onnx.ModelProto, layer_name: str):
    """Extract quantization parameters for a specific layer."""
    # Build initializer lookup dict
    initializers = {}
    for init in model.graph.initializer:
        initializers[init.name] = nph.to_array(init)

    # Find QuantizeLinear/DequantizeLinear nodes for this layer
    for node in model.graph.node:
        if node.name == layer_name or layer_name in node.output:
            if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
                # Extract scale and zero-point from inputs
                scale_name = node.input[1]  # Input 1 is scale
                zp_name = node.input[2]     # Input 2 is zero-point

                scale = initializers.get(scale_name)
                zero_point = initializers.get(zp_name)

                return {
                    "scale": scale,
                    "zero_point": zero_point,
                    "node_type": node.op_type
                }

    return None
```

### Pattern 2: Marimo Reactive Cell Dependencies
**What:** Cells automatically re-execute when dependencies change
**When to use:** For all interactive visualizations tied to dropdown selection
**Example:**
```python
# Cell 1: Layer selector (existing from Phase 14)
layer_selector = mo.ui.dropdown(
    options=layer_names,
    value=None,
    label="Layer to analyze"
)

# Cell 2: Extract parameters (NEW - reactive to layer_selector)
layer_params = None
if layer_selector.value and models:
    layer_params = extract_layer_parameters(
        models["onnx_int8"],
        layer_selector.value
    )

# Cell 3: Display table (NEW - reactive to layer_params)
if layer_params:
    # Create and display table
    display_parameter_table(layer_params)
```

**Key insight:** No explicit callbacks needed. Marimo's dependency graph automatically detects that Cell 2 uses `layer_selector.value` and Cell 3 uses `layer_params`, triggering re-execution when layer_selector changes.

### Pattern 3: Side-by-Side Histograms with Consistent Binning
**What:** Display multiple histograms with aligned bins for comparison
**When to use:** For FP32 vs INT8 weight distribution comparison
**Example:**
```python
# Source: Official matplotlib gallery - multiple_histograms_side_by_side.html
import matplotlib.pyplot as plt
import numpy as np

def plot_weight_histograms(fp32_weights, int8_weights, uint8_weights=None):
    """Create side-by-side histograms with consistent binning."""
    num_plots = 3 if uint8_weights is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(12, 4))

    # Determine global bin range for consistency
    all_data = [fp32_weights]
    if int8_weights is not None:
        all_data.append(int8_weights)
    if uint8_weights is not None:
        all_data.append(uint8_weights)

    bin_range = (np.min(all_data), np.max(all_data))
    num_bins = 50

    # Plot FP32
    axes[0].hist(fp32_weights.flatten(), bins=num_bins, range=bin_range)
    axes[0].set_title("FP32 Weights")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Count")

    # Plot INT8 (dequantized)
    axes[1].hist(int8_weights.flatten(), bins=num_bins, range=bin_range)
    axes[1].set_title("INT8 (Dequantized)")
    axes[1].set_xlabel("Value")

    # Plot UINT8 if available
    if uint8_weights is not None:
        axes[2].hist(uint8_weights.flatten(), bins=num_bins, range=bin_range)
        axes[2].set_title("UINT8 (Dequantized)")
        axes[2].set_xlabel("Value")

    plt.tight_layout()
    return axes  # Return axes object for Marimo rendering
```

### Pattern 4: Horizontal Bar Chart Heatmap
**What:** Range visualization showing min-to-max weight range per layer
**When to use:** For whole-model overview to identify quantization trouble spots
**Example:**
```python
# Source: matplotlib imshow + horizontal bar concepts
import matplotlib.pyplot as plt
import numpy as np

def plot_layer_range_heatmap(layer_ranges):
    """Create horizontal bar chart showing weight ranges per layer.

    Args:
        layer_ranges: List of dicts with keys: 'name', 'min', 'max', 'mean'
    """
    fig, (ax_fp32, ax_int8) = plt.subplots(1, 2, figsize=(14, 8))

    layer_names = [r['name'] for r in layer_ranges]
    y_positions = np.arange(len(layer_names))

    # FP32 ranges
    fp32_mins = [r['fp32_min'] for r in layer_ranges]
    fp32_maxs = [r['fp32_max'] for r in layer_ranges]
    fp32_ranges = np.array(fp32_maxs) - np.array(fp32_mins)

    # Use sequential colormap based on range magnitude
    colors_fp32 = plt.cm.viridis(fp32_ranges / np.max(fp32_ranges))

    ax_fp32.barh(y_positions, fp32_ranges, left=fp32_mins, color=colors_fp32)
    ax_fp32.set_yticks(y_positions)
    ax_fp32.set_yticklabels(layer_names)
    ax_fp32.set_xlabel("Weight Value")
    ax_fp32.set_title("FP32 Weight Ranges")

    # INT8 ranges (similar structure)
    int8_mins = [r['int8_min'] for r in layer_ranges]
    int8_maxs = [r['int8_max'] for r in layer_ranges]
    int8_ranges = np.array(int8_maxs) - np.array(int8_mins)

    colors_int8 = plt.cm.viridis(int8_ranges / np.max(int8_ranges))

    ax_int8.barh(y_positions, int8_ranges, left=int8_mins, color=colors_int8)
    ax_int8.set_yticks(y_positions)
    ax_int8.set_yticklabels(layer_names)
    ax_int8.set_xlabel("Weight Value")
    ax_int8.set_title("INT8 Weight Ranges (Dequantized)")

    plt.tight_layout()
    return (ax_fp32, ax_int8)  # Return axes for Marimo rendering
```

### Anti-Patterns to Avoid
- **Overlaid histograms instead of side-by-side:** Hard to compare when distributions overlap significantly; user explicitly requested side-by-side layout
- **Inconsistent binning across histograms:** Makes visual comparison invalid; must use same `range` and `bins` parameters
- **Using plt.show() in Marimo:** Returns None instead of figure object; Marimo cannot display it in cell output. Return axes or figure object directly
- **Creating new figure without closing previous:** Memory leak in interactive notebooks; Marimo's @mo.cache helps but explicit `plt.close()` better for long sessions
- **Extracting weights via ONNX Runtime inference:** Wrong tool; use direct graph traversal with `graph.initializer` instead

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ONNX tensor to numpy conversion | Custom parser for raw_data bytes | `onnx.numpy_helper.to_array()` | Handles all dtypes, endianness, shape correctly; official API |
| Histogram bin selection | Fixed bin count like 20 or 50 | Computed via Astropy's algorithms or use numpy defaults | Optimal bin count depends on data distribution; Astropy provides robust algorithms |
| Sequential colormap creation | Custom RGB interpolation | `plt.cm.viridis`, `plt.cm.plasma`, etc. | Perceptually uniform, scientifically validated, grayscale-printable |
| Parameter table formatting | String concatenation for alignment | pandas DataFrame with `pd.set_option('display.float_format')` | Handles alignment, precision, scientific notation automatically |
| Summary statistics calculation | Manual min/max/mean loops | `np.min()`, `np.max()`, `np.mean()`, `np.std()` | Vectorized, optimized C implementation, handles edge cases |

**Key insight:** ONNX provides official APIs for tensor extraction (numpy_helper), matplotlib provides scientific visualization primitives (hist, barh, colormaps), and NumPy provides statistical functions. Building custom solutions introduces bugs and loses optimization.

## Common Pitfalls

### Pitfall 1: Assuming All Layers Have Quantization Parameters
**What goes wrong:** Not all ONNX nodes have associated scale/zero-point initializers. Operations like ReLU, Add, MaxPool may only have QuantizeLinear/DequantizeLinear around them, not within them. Attempting to extract parameters for nodes without them causes KeyError or returns None.

**Why it happens:** QDQ format places Q/DQ nodes as boundaries, not inside every operation. Only nodes that produce or consume quantized tensors have scale/zero-point parameters.

**How to avoid:**
- Check if node type is QuantizeLinear/DequantizeLinear before extracting
- Filter layer dropdown to show only layers with quantization parameters
- Handle None returns gracefully with "No parameters available" message

**Warning signs:** KeyError when accessing initializers dict, None values in parameter extraction, empty tables displayed

### Pitfall 2: Forgetting to Dequantize INT8 Weights for Histograms
**What goes wrong:** Displaying raw INT8 weight values (range -128 to 127) instead of dequantized FP32 equivalents. User wants to compare actual weight magnitudes, not quantized representations.

**Why it happens:** ONNX stores weights as INT8 in initializers. Must apply dequantization formula: `fp32_value = (int8_value - zero_point) * scale`

**How to avoid:**
```python
# WRONG: Display raw INT8 values
int8_weights = initializers["weight_quantized"]  # Range: -128 to 127

# CORRECT: Dequantize first
int8_weights_raw = initializers["weight_quantized"]
scale = initializers["weight_scale"]
zero_point = initializers["weight_zero_point"]
int8_weights_dequantized = (int8_weights_raw.astype(np.float32) - zero_point) * scale
```

**Warning signs:** Histograms show values between -128 and 127 for "INT8" weights; FP32 and INT8 histograms have completely different x-axis ranges (should be similar)

### Pitfall 3: Per-Channel vs Per-Tensor Scale Handling
**What goes wrong:** Assuming scale is always a scalar. Per-channel quantization stores scales as 1D arrays (one per output channel). Displaying array of 64 values in table cell instead of summary statistics.

**Why it happens:** ONNX supports both per-tensor (scalar scale) and per-channel (array scale). User decisions specify showing summary stats (min, max, mean) for per-channel, not full arrays.

**How to avoid:**
```python
# Check if per-channel or per-tensor
if scale.ndim == 0:
    # Per-tensor: scalar
    display_value = f"{float(scale):.6f}"
else:
    # Per-channel: array
    display_value = f"min={scale.min():.6f}, max={scale.max():.6f}, mean={scale.mean():.6f}"
```

**Warning signs:** Table cell shows array like "[0.012, 0.015, 0.011, ...]" instead of summary; table becomes unreadably wide

### Pitfall 4: Matplotlib Figure Memory Leaks in Notebooks
**What goes wrong:** Creating multiple figures without closing previous ones accumulates memory. In interactive notebook with repeated cell execution, memory usage grows unbounded.

**Why it happens:** Matplotlib keeps references to all created figures. Interactive notebooks re-execute cells frequently. Marimo's `@mo.cache` helps but doesn't solve repeated manual re-runs.

**How to avoid:**
```python
# Option 1: Explicit close (if creating new figure)
plt.close('all')  # Close all figures before creating new ones

# Option 2: Reuse figure/axes objects
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# ... plotting code ...
return axes  # Marimo displays, manages lifecycle

# Option 3: Use Marimo's @mo.cache on plotting functions
@mo.cache
def plot_histograms(weights_fp32, weights_int8):
    # Cached based on input arrays
    fig, axes = plt.subplots(1, 2)
    # ... plotting ...
    return axes
```

**Warning signs:** Notebook memory usage increases with each layer selection change; kernel eventually crashes or slows down significantly

### Pitfall 5: Inconsistent Histogram Binning
**What goes wrong:** Using different bin counts or ranges for FP32 vs INT8 histograms makes visual comparison invalid. Bins don't align, distribution shapes appear artificially different.

**Why it happens:** Default `plt.hist()` auto-selects bins based on each dataset independently. FP32 and quantized weights may have slightly different ranges.

**How to avoid:**
```python
# WRONG: Independent binning
axes[0].hist(fp32_weights.flatten())  # Auto-bins for FP32
axes[1].hist(int8_weights.flatten())  # Auto-bins for INT8 (different!)

# CORRECT: Consistent binning
all_data = [fp32_weights.flatten(), int8_weights.flatten()]
bin_range = (np.min(all_data), np.max(all_data))
num_bins = 50

axes[0].hist(fp32_weights.flatten(), bins=num_bins, range=bin_range)
axes[1].hist(int8_weights.flatten(), bins=num_bins, range=bin_range)
```

**Warning signs:** Histogram x-axes show different ranges; bin widths look different between subplots; distributions don't visually align

### Pitfall 6: Heatmap Click Handler Complexity
**What goes wrong:** Attempting to add onclick handlers to matplotlib bar chart for interactive layer selection. Matplotlib's event system is complex and doesn't integrate cleanly with Marimo's reactive model.

**Why it happens:** User wants clicking heatmap row to update layer dropdown. Matplotlib provides low-level event API but requires manual coordinate-to-layer mapping.

**How to avoid:**
- **Simpler approach:** Display heatmap as read-only overview; users select layers via existing dropdown
- **If click needed:** Use `mo.ui.table()` with selectable rows instead of matplotlib heatmap; Marimo's UI elements integrate natively with reactive cells
- **Defer to post-MVP:** User explicitly marked this as "Claude's discretion" - implement basic heatmap first, add interactivity if time permits

**Warning signs:** Complex event handler code in plotting functions; coordinate math to map clicks to layer indices; manual state management between matplotlib and Marimo

## Code Examples

Verified patterns from official sources:

### Extract Weight Tensors from ONNX Initializers
```python
# Source: Existing scripts/extract_operations.py + ONNX documentation
import onnx
from onnx import numpy_helper as nph

def extract_weight_tensors(model: onnx.ModelProto, layer_name: str):
    """Extract weight tensors for a layer from ONNX model.

    Returns dict with 'fp32', 'int8', 'uint8' keys (values may be None).
    """
    initializers = {}
    for init in model.graph.initializer:
        initializers[init.name] = nph.to_array(init)

    # Search for weight initializers associated with this layer
    # Naming convention: layer_name appears in initializer name
    weights = {"fp32": None, "int8": None, "uint8": None}

    for init_name, init_array in initializers.items():
        if layer_name in init_name:
            # Determine dtype
            if init_array.dtype == np.float32:
                weights["fp32"] = init_array
            elif init_array.dtype == np.int8:
                weights["int8"] = init_array
            elif init_array.dtype == np.uint8:
                weights["uint8"] = init_array

    return weights
```

### Create Summary Statistics for Per-Channel Scales
```python
# Source: NumPy documentation - statistical functions
import numpy as np

def create_summary_stats(scale_array):
    """Create summary statistics for per-channel scales.

    Args:
        scale_array: Scalar (per-tensor) or 1D array (per-channel)

    Returns:
        String formatted for table display
    """
    if np.ndim(scale_array) == 0:
        # Per-tensor: single value
        return f"{float(scale_array):.6f}"
    else:
        # Per-channel: summary stats
        return (f"min={scale_array.min():.6f}, "
                f"max={scale_array.max():.6f}, "
                f"mean={scale_array.mean():.6f}")
```

### Format Parameter Table with Pandas
```python
# Source: Pandas documentation - display formatting
import pandas as pd

def display_parameter_table(params):
    """Display quantization parameters as formatted table.

    Args:
        params: Dict with keys: scale, zero_point, shape, dtype
    """
    # Set float display format globally
    pd.set_option('display.float_format', '{:.6f}'.format)

    data = {
        "Parameter": ["Scale", "Zero-Point", "Shape", "Data Type"],
        "Value": [
            create_summary_stats(params["scale"]),
            str(params["zero_point"]),
            str(params["shape"]),
            str(params["dtype"])
        ]
    }

    df = pd.DataFrame(data)
    return df  # Marimo displays DataFrames natively
```

### Marimo Reactive Cell Pattern
```python
# Source: marimo documentation - reactive cells
import marimo as mo

# Cell 1: UI widget
layer_selector = mo.ui.dropdown(
    options=layer_names,
    value=None,
    label="Select layer"
)

# Cell 2: Reactive data extraction
# Automatically re-runs when layer_selector.value changes
layer_params = None
if layer_selector.value and models:
    layer_params = extract_layer_parameters(
        models["onnx_int8"],
        layer_selector.value
    )

# Cell 3: Reactive visualization
# Automatically re-runs when layer_params changes
if layer_params:
    fig, axes = plot_weight_histograms(
        layer_params["weights_fp32"],
        layer_params["weights_int8"]
    )
    axes  # Return for display
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| QLinear operators (QLinearConv, QLinearMatMul) | QDQ format (QuantizeLinear/DequantizeLinear pairs) | ONNX Runtime 1.10+ | QDQ is now standard; easier debugging, wider hardware support. QLinear operators rarely used in practice. |
| Jupyter notebooks | Marimo reactive notebooks | 2024-2025 | Reactive cells eliminate manual re-run; cleaner Python files (not JSON); already adopted in Phase 14 |
| Plotly for interactivity | Matplotlib with simpler UX | Still evolving | Matplotlib sufficient for static visualizations; Plotly adds complexity without clear benefit for this use case |
| Per-tensor quantization only | Per-channel quantization standard for CNNs | ~2020 | Better accuracy with negligible overhead; per-channel now default for conv layers in most frameworks |

**Deprecated/outdated:**
- **QLinear operators:** Still in ONNX spec but models use QDQ format instead
- **Manual bin selection:** Astropy provides optimal bin count algorithms; matplotlib defaults often sufficient

## Open Questions

1. **Optimal bin count for histograms**
   - What we know: Matplotlib defaults work; Astropy has algorithms based on data distribution
   - What's unclear: Whether user prefers fixed bin count (e.g., 50) or adaptive based on data
   - Recommendation: Start with fixed bins=50, adjust if histograms look over/under-binned during testing

2. **Heatmap click interactivity feasibility**
   - What we know: User wants clicking heatmap row to drive layer dropdown (marked as "Claude's discretion")
   - What's unclear: Effort vs benefit; matplotlib onclick requires complex coordinate mapping
   - Recommendation: Implement read-only heatmap first, defer click interactivity to post-MVP unless trivial

3. **Handling layers with missing quantization parameters**
   - What we know: Not all layers have Q/DQ nodes (e.g., ReLU, MaxPool)
   - What's unclear: Should these appear in dropdown with indicator, or be filtered out?
   - Recommendation: Show all layers (existing behavior from Phase 14) but display "No quantization parameters" message when selected

## Sources

### Primary (HIGH confidence)
- [ONNX QuantizeLinear Specification](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html) - Quantization formula and parameter structure
- [ONNX DequantizeLinear Specification](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html) - Dequantization formula
- [Matplotlib Multiple Histograms Side by Side](https://matplotlib.org/stable/gallery/statistics/multiple_histograms_side_by_side.html) - Official example for consistent binning
- [Matplotlib Choosing Colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html) - Sequential colormap recommendations (viridis, plasma, inferno)
- [Marimo Plotting Guide](https://docs.marimo.io/guides/working_with_data/plotting/) - How to render matplotlib in marimo cells
- [NumPy Statistics Functions](https://numpy.org/doc/stable/reference/routines.statistics.html) - Official statistical function documentation
- Existing codebase: `scripts/extract_operations.py` (ONNX parameter extraction patterns), `docs/quantization/04-architecture.md` (QDQ format structure)

### Secondary (MEDIUM confidence)
- [ONNX Runtime Quantization Documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) - Verified with official docs
- [Marimo Reactivity Documentation](https://docs.marimo.io/guides/reactivity/) - Verified with official docs
- [Matplotlib Colorbar Documentation](https://matplotlib.org/stable/gallery/color/colorbar_basics.html) - Verified for range control with vmin/vmax
- [Pandas Float Formatting](https://medium.com/@anala007/float-display-in-pandas-no-more-scientific-notation-80e3dd28eabe) - Common pattern, not official but widely verified

### Tertiary (LOW confidence)
- WebSearch results on "ONNX numpy_helper to_array" - Multiple sources agree but not directly from official docs
- WebSearch results on "matplotlib memory leak plt.close" - Community discussions, not definitive official guidance

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in project dependencies, well-documented official APIs
- Architecture: HIGH - Patterns verified from official matplotlib gallery and existing codebase (scripts/extract_operations.py)
- Pitfalls: HIGH - Derived from official ONNX spec (per-channel vs per-tensor) and matplotlib documentation (memory leaks, binning)

**Research date:** 2026-02-07
**Valid until:** 90 days (stable domain - ONNX spec and matplotlib APIs change slowly; marimo evolving but backwards compatible)
