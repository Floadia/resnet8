# Playground - Marimo Notebooks

Interactive Marimo notebooks for exploring and analyzing ResNet8 quantization.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `quantization.py` | Quantization parameter explorer |
| `weight_visualizer.py` | Weight distribution visualizer |

## Weight Visualizer Workflow

`weight_visualizer.py` now uses shared extraction utilities from
`playground/utils/tensor_extractors.py`. The notebook remains the presentation
layer and consumes normalized metadata with this structure:

- `format`: `onnx` / `pytorch` / `none`
- `layers`: layer-to-tensor-name mapping
- `is_quantized`: model-level quantization signal
- `tensor_data`: per-layer tensor entries with:
  - `values`, `shape`, `is_quantized`
  - optional quantization fields: `int_values`, `scale`, `zero_point`,
    `scale_values`, `zero_point_values`

### Supported model behavior

- `.pt` models: FP32 checkpoints and quantized TorchScript/packed-params models
  are both supported.
- `.onnx` models: initializer-driven weight/bias extraction is supported; when
  quantization ops are present, raw integer values and dequantized values are
  both available for rendering.
- Missing files or unsupported extensions resolve to an empty index (`format:
  none`) instead of failing notebook startup.

## Usage

```bash
# Edit mode (interactive)
marimo edit playground/<notebook>.py

# Run mode (read-only)
marimo run playground/<notebook>.py
```

## Marimo Development Notes

Marimo notebooks have unique scoping rules that differ from standard Python. Keep these in mind when editing notebooks:

### Cell-private functions

`def _xxx()` functions (prefixed with `_`) are **cell-private** -- they cannot be called from other cells. If a helper function is needed by a cell, define it **inside** that cell.

```python
# OK: helper defined inside the cell that uses it
@app.cell
def _(mo):
    def _my_helper():
        return 42
    result = _my_helper()
    return (result,)
```

### Variable scoping

Variables defined without `_` prefix are **exported** across cells and must be unique. Use `_` prefix for all cell-local variables to avoid "redefines variables from other cells" errors.

```python
# BAD: _data in multiple cells causes conflict if not prefixed
data = tensor["values"]

# GOOD: cell-local variable
_data = tensor["values"]
```

### `mo.ui.radio` value parameter

The `value` parameter takes an **option key (label)**, not the option value.

```python
# GOOD
mo.ui.radio(
    options={"dequantized (FP32)": "fp32", "int8 raw": "int"},
    value="dequantized (FP32)",  # key, not "fp32"
)
```

### Formatting

Run `marimo check --fix <file>` to auto-format notebooks to match Marimo's expected style.
