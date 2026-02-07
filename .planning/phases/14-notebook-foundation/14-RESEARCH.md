# Phase 14: Notebook Foundation - Research

**Researched:** 2026-02-05
**Domain:** Marimo reactive notebooks, ONNX/PyTorch model introspection, distribution visualization
**Confidence:** HIGH

## Summary

Marimo is a reactive Python notebook framework where cells automatically re-run when dependencies change. For this phase, the standard approach is to use Marimo's built-in UI components (`mo.ui.file_browser` for folder selection, `mo.ui.dropdown` for layer selection, `mo.status.spinner` for loading indicators) combined with direct ONNX/PyTorch model introspection. The core architecture pattern is: (1) file picker triggers model loading with caching, (2) layer list extracted via `model.graph.node` (ONNX) or `model.named_modules()` (PyTorch), (3) dropdown selection triggers immediate reactive plot updates.

Key findings show that ONNX models expose layers through `model.graph.node` with weights in `model.graph.initializer`, while PyTorch models use `named_modules()` for hierarchical layer paths. Marimo's reactivity is automatic but doesn't track object mutations, so use `@mo.cache` decorators for expensive model loads. For visualization, matplotlib with `plt.subplots()` provides side-by-side comparison plots, or Altair for declarative histograms with consistent binning.

Critical pitfall: ONNX Runtime has known memory leak issues with repeated session creation. Solution: load models once with `@mo.cache` and reuse sessions. Don't reload models on every cell re-run.

**Primary recommendation:** Use `@mo.cache` for model loading functions, `mo.ui.file_browser(selection_mode="directory")` for folder selection, extract full layer paths via framework-specific APIs, and create comparison plots with `plt.subplots(1, 3)` for horizontal layout.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| marimo | 0.15.5+ | Reactive notebook framework | Official framework for phase requirements, built-in UI components |
| onnx | 1.21.0 | ONNX model loading/introspection | Standard ONNX format parser, exposes graph structure |
| onnxruntime | Latest | ONNX model inference (if needed) | Official Microsoft runtime, though may have memory issues |
| torch | 2.10+ | PyTorch model loading | Standard deep learning framework for PyTorch models |
| matplotlib | 3.10+ | Histogram plotting | Industry standard for scientific visualization |
| numpy | 2.4+ | Tensor statistics | Foundation for numerical computing |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| altair | 6.0+ | Declarative visualization | Alternative to matplotlib for cleaner histogram syntax |
| pathlib | stdlib | File path handling | Type-safe path operations for file browser |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| matplotlib | altair | Altair is more declarative but less flexible for custom layouts |
| matplotlib | plotly | Plotly adds interactivity but heavier for static comparisons |
| functools.lru_cache | mo.cache | mo.cache integrates better with Marimo's reactivity |

**Installation:**
```bash
pip install marimo onnx onnxruntime torch matplotlib numpy
```

## Architecture Patterns

### Recommended Project Structure
```
notebooks/
├── quantization_compare.py    # Main marimo notebook
└── utils/
    ├── model_loader.py         # Cached model loading functions
    ├── layer_inspector.py      # Layer name extraction
    └── viz.py                  # Plotting utilities
```

### Pattern 1: Reactive Model Loading with Caching
**What:** Use `@mo.cache` decorator on model loading functions to prevent memory leaks and re-loading on cell re-runs.
**When to use:** Any expensive computation in Marimo, especially model loading.
**Example:**
```python
# Source: https://docs.marimo.io/api/caching/
import marimo as mo

@mo.cache
def load_model_set(folder_path):
    """Load all three model variants from folder."""
    models = {
        'float': onnx.load(folder_path / 'model_float.onnx'),
        'int8': onnx.load(folder_path / 'model_int8.onnx'),
        'uint8': onnx.load(folder_path / 'model_uint8.onnx'),
    }
    return models

# In notebook cell
models = load_model_set(selected_folder.path(0)) if selected_folder.value else None
```

### Pattern 2: File Browser with Directory Selection
**What:** Use `mo.ui.file_browser` with `selection_mode="directory"` for folder picking.
**When to use:** When users need to select model folders containing multiple files.
**Example:**
```python
# Source: https://docs.marimo.io/api/inputs/file_browser/
import marimo as mo
from pathlib import Path

folder_picker = mo.ui.file_browser(
    initial_path=Path("./models"),
    selection_mode="directory",
    multiple=False,
    label="Select model folder"
)
folder_picker

# Access selected path
if folder_picker.value:
    selected_path = folder_picker.path(index=0)
```

### Pattern 3: ONNX Layer Extraction
**What:** Extract layer names and initializer names from ONNX model graph.
**When to use:** Building layer dropdown for ONNX models.
**Example:**
```python
# Source: https://onnx.ai/onnx/intro/concepts.html
import onnx

model = onnx.load("model.onnx")

# Get node names (operations)
node_names = [node.name for node in model.graph.node]

# Get initializer names (weights/biases)
weight_names = [init.name for init in model.graph.initializer]

# Combine for full layer list
layer_options = node_names + weight_names
```

### Pattern 4: PyTorch Layer Extraction with Full Paths
**What:** Use `named_modules()` to get hierarchical layer paths like `layer1.conv1.weight`.
**When to use:** Building layer dropdown for PyTorch models.
**Example:**
```python
# Source: https://discuss.pytorch.org/t/how-to-get-layer-names-in-a-network/134238
import torch

model = torch.load("model.pth")

# Get all layers with full paths
layer_names = []
for name, module in model.named_modules():
    if name:  # Skip root module (empty name)
        layer_names.append(name)

# Filter to specific layer types if needed
conv_layers = [
    name for name, module in model.named_modules()
    if isinstance(module, torch.nn.Conv2d)
]
```

### Pattern 5: Reactive Dropdown with Immediate Update
**What:** Use `mo.ui.dropdown` with `.value` access; Marimo auto-triggers dependent cells.
**When to use:** Layer selection that immediately updates plots.
**Example:**
```python
# Source: https://docs.marimo.io/api/inputs/dropdown/
import marimo as mo

# Create dropdown (in one cell)
layer_selector = mo.ui.dropdown(
    options=layer_names if layer_names else ["Select a layer..."],
    value=None,
    label="Layer to analyze",
    allow_select_none=True
)
layer_selector

# Use selected value (in another cell - auto-runs when selection changes)
selected_layer = layer_selector.value
if selected_layer:
    # Extract layer data and plot
    layer_data = extract_layer_data(models, selected_layer)
    plot_distributions(layer_data)
```

### Pattern 6: Loading Indicator with Context Manager
**What:** Use `mo.status.spinner()` as context manager during model loading.
**When to use:** Long-running operations like model loading.
**Example:**
```python
# Source: https://docs.marimo.io/api/status/
import marimo as mo

with mo.status.spinner(title="Loading models", subtitle="Reading model files...") as spinner:
    models = load_all_models(folder_path)
    spinner.update(subtitle="Extracting layer information...")
    layers = extract_layers(models)
    spinner.update(subtitle="Complete!")
```

### Pattern 7: Side-by-Side Histogram Comparison
**What:** Use matplotlib subplots with shared bins for consistent comparison.
**When to use:** Comparing distributions across precision variants.
**Example:**
```python
# Source: https://matplotlib.org/stable/gallery/statistics/multiple_histograms_side_by_side.html
import matplotlib.pyplot as plt
import numpy as np

def plot_precision_comparison(float_data, int8_data, uint8_data, layer_name):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Use consistent bins across all plots
    all_data = np.concatenate([float_data, int8_data, uint8_data])
    bins = np.histogram_bin_edges(all_data, bins=30)

    ax1.hist(float_data, bins=bins, alpha=0.7, color='blue')
    ax1.set_title(f'Float32 - {layer_name}')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Count')

    ax2.hist(int8_data, bins=bins, alpha=0.7, color='green')
    ax2.set_title(f'Int8 - {layer_name}')
    ax2.set_xlabel('Value')

    ax3.hist(uint8_data, bins=bins, alpha=0.7, color='red')
    ax3.set_title(f'Uint8 - {layer_name}')
    ax3.set_xlabel('Value')

    plt.tight_layout()
    return fig
```

### Pattern 8: Horizontal Layout with mo.hstack
**What:** Use `mo.hstack()` to arrange plots or statistics side-by-side.
**When to use:** Displaying summary statistics horizontally next to plots.
**Example:**
```python
# Source: https://docs.marimo.io/api/layouts/stacks/
import marimo as mo

# Create statistics cards
float_stats = mo.md(f"""
**Float32 Stats**
- Mean: {float_mean:.4f}
- Std: {float_std:.4f}
- Min/Max: {float_min:.4f} / {float_max:.4f}
""")

int8_stats = mo.md(f"""
**Int8 Stats**
- Mean: {int8_mean:.4f}
- Std: {int8_std:.4f}
- Min/Max: {int8_min:.4f} / {int8_max:.4f}
""")

uint8_stats = mo.md(f"""
**Uint8 Stats**
- Mean: {uint8_mean:.4f}
- Std: {uint8_std:.4f}
- Min/Max: {uint8_min:.4f} / {uint8_max:.4f}
""")

# Display horizontally
mo.hstack([float_stats, int8_stats, uint8_stats], justify='space-between')
```

### Anti-Patterns to Avoid
- **Mutating objects across cells:** Marimo doesn't track mutations. Create new objects instead of modifying existing ones.
- **Reloading models without caching:** Causes memory leaks (ONNX) and poor performance. Always use `@mo.cache`.
- **Using functools.lru_cache instead of mo.cache:** mo.cache integrates with Marimo's persistence system.
- **Inconsistent histogram bins:** Makes visual comparison meaningless. Always compute shared bin edges.
- **Creating ONNX Runtime sessions repeatedly:** Known memory leak. Create once and reuse.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| File/folder picker UI | Custom file input widget | `mo.ui.file_browser(selection_mode="directory")` | Built-in, handles local/cloud paths, consistent UX |
| Loading spinner | Manual "Loading..." text | `mo.status.spinner()` with context manager | Standard Marimo pattern, auto-cleans on exit |
| Layer name extraction | String parsing ONNX/PyTorch files | `model.graph.node` (ONNX), `named_modules()` (PyTorch) | Official APIs, handle edge cases, framework-maintained |
| Histogram binning | Manual np.arange calculations | `np.histogram_bin_edges()` for shared bins | Handles edge cases, consistent algorithm |
| Tensor statistics | Manual mean/std calculations | `np.mean()`, `np.std()`, `np.percentile()` | Optimized, handles edge cases (NaN, Inf) |
| Model caching | Manual dict/global variables | `@mo.cache` decorator | Thread-safe, integrates with Marimo persistence |

**Key insight:** Marimo provides built-in UI components specifically designed for its reactive model. Custom implementations break reactivity patterns and introduce bugs.

## Common Pitfalls

### Pitfall 1: ONNX Runtime Memory Leak
**What goes wrong:** Memory continuously grows when loading ONNX models repeatedly, even after deleting sessions. Production servers can reach 100GB RAM usage.
**Why it happens:** Known issue in ONNX Runtime where memory isn't properly deallocated when sessions are created and deleted in succession. Python garbage collection doesn't trigger C++ memory cleanup.
**How to avoid:** Use `@mo.cache` decorator on model loading functions. Load once per model path, reuse the loaded model object across cell re-runs.
**Warning signs:** Steadily increasing memory usage in notebook, system slowing down after multiple re-runs.

**Example:**
```python
# BAD: Loads model on every cell re-run
def get_model(path):
    return onnx.load(path)

# GOOD: Caches model, loads only once per unique path
@mo.cache
def get_model(path):
    return onnx.load(path)
```

### Pitfall 2: Marimo Doesn't Track Object Mutations
**What goes wrong:** Modifying an object's attributes in one cell doesn't trigger re-runs in cells that reference the object.
**Why it happens:** Marimo tracks variable assignments, not mutations. `foo.bar = 10` doesn't trigger cells that reference `foo`.
**How to avoid:** Create new objects instead of mutating existing ones. Use immutable patterns or reassign the entire object.
**Warning signs:** Cells not updating when you expect them to, stale values persisting.

**Example:**
```python
# BAD: Mutation doesn't trigger reactivity
selected_layer.name = "conv1"  # Other cells won't re-run

# GOOD: Reassignment triggers reactivity
selected_layer = {"name": "conv1"}  # Other cells will re-run
```

### Pitfall 3: Inconsistent Histogram Bins Across Comparisons
**What goes wrong:** Each precision variant histogram uses different bin edges, making visual comparison meaningless. Peaks appear in different locations due to binning, not actual data differences.
**Why it happens:** Default `plt.hist()` or `ax.hist()` computes bins independently for each dataset based on its range.
**How to avoid:** Compute shared bin edges using `np.histogram_bin_edges()` across all datasets before plotting. Pass `bins` parameter explicitly to all `hist()` calls.
**Warning signs:** Histograms with noticeably different x-axis ranges, peaks that don't align visually.

**Example:**
```python
# BAD: Independent bins per histogram
ax1.hist(float_data, bins=30)  # Bins computed from float_data range
ax2.hist(int8_data, bins=30)   # Bins computed from int8_data range

# GOOD: Shared bins across all histograms
all_data = np.concatenate([float_data, int8_data, uint8_data])
bins = np.histogram_bin_edges(all_data, bins=30)
ax1.hist(float_data, bins=bins)
ax2.hist(int8_data, bins=bins)
ax3.hist(uint8_data, bins=bins)
```

### Pitfall 4: ONNX Nodes vs Initializers Confusion
**What goes wrong:** Expecting `model.graph.node` to contain all layers including weights, but weights are stored separately in `model.graph.initializer`. Layer dropdown is incomplete or shows wrong structure.
**Why it happens:** ONNX separates computation nodes (operations) from constant data (weights). A Conv node references an initializer by name, but they're in different lists.
**How to avoid:** Extract both `model.graph.node` names and `model.graph.initializer` names. Combine them for a complete layer list. Understand the relationship: nodes perform operations, initializers provide data.
**Warning signs:** Layer dropdown only shows operation nodes, weight tensors are missing from selection options.

**Example:**
```python
# BAD: Only getting operation nodes
layer_names = [node.name for node in model.graph.node]

# GOOD: Getting both operations and weights
node_names = [node.name for node in model.graph.node]
weight_names = [init.name for init in model.graph.initializer]
layer_names = node_names + weight_names
```

### Pitfall 5: Empty Dropdown Before Model Load
**What goes wrong:** Creating `mo.ui.dropdown()` with empty options list causes errors or shows broken UI.
**Why it happens:** Marimo UI elements require valid options to render. Trying to create dropdown before models are loaded results in empty list.
**How to avoid:** Use conditional rendering or provide default placeholder option like `["Select a layer..."]`. Check if model is loaded before creating dropdown with actual layer names.
**Warning signs:** Dropdown doesn't render, errors about empty options, UI breaks on initial load.

**Example:**
```python
# BAD: Empty options if models not loaded
layer_selector = mo.ui.dropdown(options=layer_names)

# GOOD: Fallback to placeholder
layer_selector = mo.ui.dropdown(
    options=layer_names if layer_names else ["Select a layer..."],
    value=None,
    allow_select_none=True
)
```

### Pitfall 6: PyTorch named_modules() Includes Root Module
**What goes wrong:** First entry in layer list is empty string (root module), confusing users or breaking layer extraction logic.
**Why it happens:** `named_modules()` returns the model itself with name=`""` as first entry, then all child modules.
**How to avoid:** Filter out empty names with `if name:` when iterating. Or skip first entry with slicing.
**Warning signs:** Dropdown shows blank entry, layer extraction fails for first selection.

**Example:**
```python
# BAD: Includes root module with empty name
layer_names = [name for name, _ in model.named_modules()]

# GOOD: Filter out root module
layer_names = [name for name, _ in model.named_modules() if name]
```

## Code Examples

Verified patterns from official sources:

### Loading Three Model Variants with Caching
```python
# Source: https://docs.marimo.io/api/caching/
import marimo as mo
import onnx
from pathlib import Path

@mo.cache
def load_model_variants(folder_path):
    """Load float, int8, uint8 variants from folder. Cached per folder path."""
    folder = Path(folder_path)
    variants = {}

    for precision in ['float', 'int8', 'uint8']:
        model_file = folder / f'model_{precision}.onnx'
        if model_file.exists():
            variants[precision] = onnx.load(str(model_file))
        else:
            raise FileNotFoundError(f"Missing {model_file.name}")

    return variants

# Usage in cell
if folder_picker.value:
    with mo.status.spinner(title="Loading models") as spinner:
        try:
            models = load_model_variants(folder_picker.path(0))
            mo.md(f"✓ Loaded {len(models)} model variants").callout(kind="success")
        except FileNotFoundError as e:
            mo.md(f"Error: {e}").callout(kind="danger")
else:
    mo.md("Select a model folder to begin")
```

### Extracting ONNX Layer Names
```python
# Source: https://onnx.ai/onnx/intro/concepts.html
import onnx

def get_onnx_layer_names(model):
    """Extract all layer names from ONNX model (nodes + initializers)."""
    # Get operation nodes
    node_names = [node.name for node in model.graph.node if node.name]

    # Get weight/bias initializers
    init_names = [init.name for init in model.graph.initializer]

    # Combine and deduplicate
    all_names = list(set(node_names + init_names))
    all_names.sort()

    return all_names

# Usage
layer_options = get_onnx_layer_names(models['float']) if models else []
```

### Extracting PyTorch Layer Names with Full Paths
```python
# Source: https://discuss.pytorch.org/t/how-to-get-layer-names-in-a-network/134238
import torch

def get_pytorch_layer_names(model):
    """Extract hierarchical layer paths from PyTorch model."""
    layer_names = []

    for name, module in model.named_modules():
        if name:  # Skip root module (empty name)
            layer_names.append(name)

    return layer_names

# Usage
layer_options = get_pytorch_layer_names(models['float']) if models else []
```

### Computing Summary Statistics
```python
# Source: https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
import numpy as np

def compute_distribution_stats(tensor_data):
    """Compute comprehensive statistics for a tensor."""
    flat_data = tensor_data.flatten()

    stats = {
        'mean': np.mean(flat_data),
        'std': np.std(flat_data),
        'min': np.min(flat_data),
        'max': np.max(flat_data),
        'q25': np.percentile(flat_data, 25),
        'q50': np.percentile(flat_data, 50),  # median
        'q75': np.percentile(flat_data, 75),
    }

    return stats
```

### Creating Side-by-Side Comparison Plots
```python
# Source: https://matplotlib.org/stable/gallery/statistics/multiple_histograms_side_by_side.html
import matplotlib.pyplot as plt
import numpy as np

def plot_precision_comparison(layer_data_dict, layer_name):
    """
    Plot side-by-side histograms for float, int8, uint8 variants.

    Args:
        layer_data_dict: {'float': array, 'int8': array, 'uint8': array}
        layer_name: str, name of layer for title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Compute shared bins for consistent comparison
    all_data = np.concatenate([
        layer_data_dict['float'].flatten(),
        layer_data_dict['int8'].flatten(),
        layer_data_dict['uint8'].flatten()
    ])
    bins = np.histogram_bin_edges(all_data, bins=30)

    # Plot each variant
    colors = {'float': 'blue', 'int8': 'green', 'uint8': 'red'}
    for idx, (precision, ax) in enumerate(zip(['float', 'int8', 'uint8'], axes)):
        data = layer_data_dict[precision].flatten()
        ax.hist(data, bins=bins, alpha=0.7, color=colors[precision])
        ax.set_title(f'{precision.upper()} - {layer_name}')
        ax.set_xlabel('Value')
        if idx == 0:
            ax.set_ylabel('Count')

    plt.tight_layout()
    return fig

# Usage in Marimo cell
if selected_layer:
    layer_data = {
        precision: extract_layer_tensor(model, selected_layer)
        for precision, model in models.items()
    }
    fig = plot_precision_comparison(layer_data, selected_layer)
    fig  # Return figure to display in Marimo
```

### Displaying Statistics with mo.hstack
```python
# Source: https://docs.marimo.io/api/layouts/stacks/
import marimo as mo

def display_comparison_stats(stats_dict):
    """Display statistics for all precision variants side-by-side."""
    cards = []

    for precision in ['float', 'int8', 'uint8']:
        stats = stats_dict[precision]
        card = mo.md(f"""
**{precision.upper()} Statistics**
- Mean: {stats['mean']:.6f}
- Std Dev: {stats['std']:.6f}
- Min: {stats['min']:.6f}
- Max: {stats['max']:.6f}
- Q25/Q50/Q75: {stats['q25']:.4f} / {stats['q50']:.4f} / {stats['q75']:.4f}
        """)
        cards.append(card)

    return mo.hstack(cards, justify='space-between', gap=1.0)

# Usage
if selected_layer:
    stats = {
        precision: compute_distribution_stats(layer_data[precision])
        for precision in ['float', 'int8', 'uint8']
    }
    display_comparison_stats(stats)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Jupyter notebooks with manual cell execution | Marimo reactive notebooks with auto-execution | 2023-2024 | Eliminates hidden state, ensures reproducibility |
| functools.lru_cache for caching | mo.cache with persistence | Marimo 0.15+ | Cache survives notebook restarts, better integration |
| Custom file upload widgets | mo.ui.file_browser with cloud support | Marimo 0.14+ | Unified API for local/S3/GCS/Azure |
| String parsing for ONNX structure | model.graph.node/initializer API | ONNX 1.0+ | Official API, handles all edge cases |
| Loading full state_dict for inspection | named_modules() for layer enumeration | PyTorch 1.0+ | More efficient, provides hierarchy |

**Deprecated/outdated:**
- **Manual matplotlib.pyplot.show()**: In Marimo, return figure object directly instead of using `plt.show()`. Marimo auto-renders returned figures.
- **mo.ui.file() for folders**: Use `mo.ui.file_browser(selection_mode="directory")` instead. `mo.ui.file()` is for file uploads, not browsing.
- **Global variables for model caching**: Use `@mo.cache` decorator. Global variables break reactivity and persistence.

## Open Questions

Things that couldn't be fully resolved:

1. **ONNX Runtime memory leak severity in latest versions**
   - What we know: Multiple GitHub issues (2024-2025) report memory leaks with session creation/deletion
   - What's unclear: Whether latest ONNX Runtime versions (post-1.16) have mitigations, severity with simple load/inference patterns
   - Recommendation: Use `@mo.cache` to load once regardless. Monitor memory during testing. Consider onnx library (just load, no inference) if Runtime issues persist.

2. **Altair vs Matplotlib for this specific use case**
   - What we know: Altair is more declarative, Matplotlib more flexible. Both work in Marimo.
   - What's unclear: Whether Altair's faceting would simplify side-by-side comparison vs matplotlib subplots
   - Recommendation: Start with matplotlib (shown in examples). Matplotlib's subplot control better matches "separate plots for input/weight/output" requirement. Altair alternative if team prefers declarative syntax.

3. **Layer data extraction performance for large models**
   - What we know: ONNX initializers and PyTorch state_dict contain raw tensor data
   - What's unclear: Whether extracting full tensors for each layer causes performance issues with large models (>100 layers, GB-size weights)
   - Recommendation: Profile first implementation. If slow, add layer-level caching or lazy loading. `@mo.cache` can cache per-layer extraction.

## Sources

### Primary (HIGH confidence)
- [marimo official documentation](https://docs.marimo.io) - UI components, reactivity model, caching API
- [marimo file_browser API](https://docs.marimo.io/api/inputs/file_browser/) - Directory selection patterns
- [marimo dropdown API](https://docs.marimo.io/api/inputs/dropdown/) - Layer selection UI
- [marimo status API](https://docs.marimo.io/api/status/) - Spinner/loading indicators
- [marimo caching API](https://docs.marimo.io/api/caching/) - mo.cache decorator
- [marimo layouts API](https://docs.marimo.io/api/layouts/stacks/) - mo.hstack for horizontal layout
- [ONNX concepts documentation](https://onnx.ai/onnx/intro/concepts.html) - Model structure, nodes, initializers
- [NumPy percentile documentation](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html) - Statistical functions
- [Matplotlib histogram gallery](https://matplotlib.org/stable/gallery/statistics/multiple_histograms_side_by_side.html) - Side-by-side plotting

### Secondary (MEDIUM confidence)
- [PyTorch named_modules discussion](https://discuss.pytorch.org/t/how-to-get-layer-names-in-a-network/134238) - Layer name extraction patterns (community-verified)
- [ONNX Runtime memory leak issue #22271](https://github.com/microsoft/onnxruntime/issues/22271) - Recent memory leak reports (2025)
- [marimo reactivity guide](https://docs.marimo.io/guides/reactivity/) - Mutation tracking behavior
- [PyTorch quantized model state_dict](https://docs.pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html) - Model loading patterns

### Tertiary (LOW confidence)
- WebSearch results for "marimo best practices 2026" - General usage patterns (not deeply validated)
- Community discussions on histogram comparison techniques - Various approaches, not all tested

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries verified via official documentation, current versions checked
- Architecture: HIGH - Patterns sourced from official docs with working examples
- Pitfalls: HIGH - ONNX memory leak documented in GitHub issues, Marimo reactivity constraints in official docs, histogram binning verified in matplotlib docs

**Research date:** 2026-02-05
**Valid until:** 2026-03-05 (30 days - stable domain, but Marimo is fast-moving so verify version compatibility)

**Notes:**
- Marimo version 0.15.5 referenced in docs (as of research date). Check for updates.
- ONNX Runtime memory leak issues persist as of 2025 GitHub issues. Caching pattern is essential.
- All code examples verified against official documentation sources listed above.
