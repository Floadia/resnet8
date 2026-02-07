"""Quantization Playground - Interactive notebook for model quantization experiments.

Usage:
    marimo edit playground/quantization.py
"""

import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    return


@app.cell
def _():
    """Import dependencies."""
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # Add project root to path for imports
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from playground.utils import (
        compute_all_layer_ranges,
        extract_layer_params,
        extract_weight_tensors,
        get_all_layer_names,
        get_layer_type,
        get_layers_with_params,
        get_model_summary,
        load_model_variants,
    )

    return (
        Path,
        compute_all_layer_ranges,
        extract_layer_params,
        extract_weight_tensors,
        get_all_layer_names,
        get_layer_type,
        get_layers_with_params,
        get_model_summary,
        load_model_variants,
        mo,
        np,
        plt,
        project_root,
    )


@app.cell
def _():
    """Display notebook header and instructions."""
    return


@app.cell
def _(mo, project_root):
    """File picker for selecting model file (folder derived from selection)."""
    folder_picker = mo.ui.file_browser(
        initial_path=str(project_root / "models"),
        selection_mode="file",
        multiple=False,
        label="Select any model file to load the folder",
        filetypes=[".onnx", ".pt"],
    )
    return (folder_picker,)


@app.cell
def _(folder_picker):
    """Display folder picker."""
    folder_picker
    return


@app.cell
def _(Path, folder_picker, get_model_summary, load_model_variants, mo):
    """Load models from selected folder with spinner."""
    models = None
    models_summary = None
    load_error = None
    selected_folder = None

    if folder_picker.value:
        try:
            selected_folder = Path(folder_picker.path(0)).parent
            with mo.status.spinner(title="Loading models..."):
                models = load_model_variants(selected_folder)
                models_summary = get_model_summary(models)
        except Exception as e:
            load_error = str(e)
    return load_error, models, models_summary, selected_folder


@app.cell
def _(folder_picker, load_error, mo, models_summary, selected_folder):
    """Display model loading status and summary."""
    if not folder_picker.value:
        # Initial state - show instructions
        display = mo.md("**Select a model file to load the folder**").callout(
            kind="info"
        )
    elif load_error:
        # Error state - show error message
        display = mo.md(f"**Error loading models:** {load_error}").callout(
            kind="danger"
        )
    elif models_summary and models_summary["total_loaded"] > 0:
        # Success state - show summary
        onnx_avail = models_summary["onnx_available"]
        pytorch_avail = models_summary["pytorch_available"]
        onnx_list = ", ".join(onnx_avail) if onnx_avail else "None"
        pytorch_list = ", ".join(pytorch_avail) if pytorch_avail else "None"

        summary_text = f"""
        **Models loaded successfully!**

        **Folder:** `{selected_folder}`

        **Total models:** {models_summary["total_loaded"]}

        **ONNX variants:** {onnx_list}

        **PyTorch variants:** {pytorch_list}
        """
        display = mo.md(summary_text).callout(kind="success")
    else:
        # No models found
        display = mo.md(f"**No models found in:** `{selected_folder}`").callout(
            kind="warn"
        )
    return (display,)


@app.cell
def _(display):
    """Render the display."""
    display
    return


@app.cell
def _(compute_all_layer_ranges, models):
    """Compute weight ranges for all layers (for heatmap visualization)."""
    layer_ranges = None

    if models:
        _onnx_float = models.get("onnx_float")
        _onnx_int8 = models.get("onnx_int8")

        # Only compute if both models are available
        if _onnx_float is not None and _onnx_int8 is not None:
            layer_ranges = compute_all_layer_ranges(_onnx_float, _onnx_int8)

    return (layer_ranges,)


@app.cell
def _(layer_ranges, mo, np, plt):
    """Visualize weight ranges as heatmap for whole-model overview."""
    heatmap_fig = None

    if layer_ranges is not None and len(layer_ranges) > 0:
        # Close any previous figures to prevent memory leaks
        plt.close("all")

        # Create subplots for FP32 and INT8 side-by-side
        _num_layers = len(layer_ranges)
        _fig_height = max(6, _num_layers * 0.35)
        _fig, (_ax_fp32, _ax_int8) = plt.subplots(1, 2, figsize=(14, _fig_height))

        # Extract data for plotting
        _layer_names = [lr["name"] for lr in layer_ranges]
        _fp32_mins = [lr["fp32_min"] for lr in layer_ranges]
        _fp32_maxs = [lr["fp32_max"] for lr in layer_ranges]
        _fp32_ranges = [lr["fp32_range"] for lr in layer_ranges]
        _int8_mins = [lr["int8_min"] for lr in layer_ranges]
        _int8_maxs = [lr["int8_max"] for lr in layer_ranges]
        _int8_ranges = [lr["int8_range"] for lr in layer_ranges]

        # Y positions for layers (reversed so first layer is on top)
        _y_positions = np.arange(_num_layers)[::-1]

        # Normalize range magnitudes for color mapping
        _all_ranges = _fp32_ranges + _int8_ranges
        _range_min = min(_all_ranges) if _all_ranges else 0
        _range_max = max(_all_ranges) if _all_ranges else 1

        # FP32 subplot - horizontal bars from min to max
        for _i, (_y_pos, _fp32_min, _fp32_max, _fp32_range) in enumerate(
            zip(_y_positions, _fp32_mins, _fp32_maxs, _fp32_ranges)
        ):
            # Color based on range magnitude
            _color_val = (_fp32_range - _range_min) / (
                _range_max - _range_min + 1e-10
            )
            _color = plt.cm.viridis(_color_val)

            # Horizontal bar from min to max
            _ax_fp32.barh(
                _y_pos,
                width=_fp32_max - _fp32_min,
                left=_fp32_min,
                height=0.8,
                color=_color,
            )

        _ax_fp32.set_yticks(_y_positions)
        _ax_fp32.set_yticklabels(_layer_names, fontsize=8)
        _ax_fp32.set_xlabel("Weight Value")
        _ax_fp32.set_title("FP32 Weight Ranges")
        _ax_fp32.grid(axis="x", alpha=0.3)

        # INT8 subplot - same layout
        for _i, (_y_pos, _int8_min, _int8_max, _int8_range) in enumerate(
            zip(_y_positions, _int8_mins, _int8_maxs, _int8_ranges)
        ):
            # Color based on range magnitude
            _color_val = (_int8_range - _range_min) / (
                _range_max - _range_min + 1e-10
            )
            _color = plt.cm.viridis(_color_val)

            # Horizontal bar from min to max
            _ax_int8.barh(
                _y_pos,
                width=_int8_max - _int8_min,
                left=_int8_min,
                height=0.8,
                color=_color,
            )

        _ax_int8.set_yticks(_y_positions)
        _ax_int8.set_yticklabels(_layer_names, fontsize=8)
        _ax_int8.set_xlabel("Weight Value")
        _ax_int8.set_title("INT8 Weight Ranges (Dequantized)")
        _ax_int8.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        heatmap_fig = _fig

    elif layer_ranges is None:
        # Models not loaded yet
        heatmap_fig = mo.md(
            "**Load ONNX float and int8 models to see weight range heatmap.**"
        ).callout(kind="info")

    return (heatmap_fig,)


@app.cell
def _(heatmap_fig):
    """Render heatmap figure."""
    heatmap_fig
    return


@app.cell
def _(get_all_layer_names, get_layers_with_params, models):
    """Extract layer names from loaded models and identify layers with params."""
    layer_data = None
    layer_names = []
    layer_source = None
    layers_with_params = set()

    if models:
        layer_data = get_all_layer_names(models)
        layer_names = layer_data.get("layer_names", [])
        layer_source = layer_data.get("source")

        # Get layers with quantization parameters (from INT8 model if available)
        if models.get("onnx_int8") is not None:
            layers_with_params = get_layers_with_params(models["onnx_int8"])

    return layer_names, layer_source, layers_with_params


@app.cell
def _(layer_names, layers_with_params, mo):
    """Layer selection dropdown with quantization parameter indicators."""
    # Create options dict mapping display strings to raw layer names
    # Add [Q] indicator for layers with quantization parameters
    _dropdown_options = {}

    if layer_names:
        for _layer in layer_names:
            # Check if layer or any substring matches layers_with_params
            _has_params = any(
                _layer in _param_layer or _param_layer in _layer
                for _param_layer in layers_with_params
            )

            if _has_params:
                _display_name = f"{_layer} [Q]"
            else:
                _display_name = _layer

            _dropdown_options[_display_name] = _layer
    else:
        _dropdown_options = {"Select a layer...": None}

    layer_selector = mo.ui.dropdown(
        options=_dropdown_options,
        value=None,
        allow_select_none=True,
        label="Layer to analyze",
    )
    return (layer_selector,)


@app.cell
def _(layer_selector):
    """Display layer selector."""
    layer_selector
    return


@app.cell
def _(get_layer_type, layer_selector, layer_source, mo, models):
    """Display layer information (reactive to dropdown selection)."""
    layer_info_display = None

    if not layer_selector.value:
        # No selection - show placeholder
        layer_info_display = mo.md(
            "**Select a layer from the dropdown above to view details.**"
        ).callout(kind="neutral")
    else:
        # Layer selected - show info
        selected_layer = layer_selector.value

        # Get layer type
        layer_type = None
        if models and layer_source:
            valid = ["onnx", "pytorch"]
            model_key = f"{layer_source}_float" if layer_source in valid else None
            if model_key and models.get(model_key):
                layer_type = get_layer_type(models[model_key], selected_layer)

        type_text = f" ({layer_type})" if layer_type else ""

        layer_info_text = f"""
        **Layer:** `{selected_layer}`{type_text}

        **Source:** {layer_source.upper() if layer_source else "Unknown"}
        """

        layer_info_display = mo.md(layer_info_text).callout(kind="info")
    return (layer_info_display,)


@app.cell
def _(layer_info_display):
    """Render layer info display."""
    layer_info_display
    return


@app.cell
def _(extract_layer_params, layer_selector, models):
    """Extract quantization parameters for selected layer (reactive to dropdown)."""
    layer_params = None

    if layer_selector.value and models:
        _onnx_int8 = models.get("onnx_int8")
        if _onnx_int8 is not None:
            layer_params = extract_layer_params(_onnx_int8, layer_selector.value)

    return (layer_params,)


@app.cell
def _(layer_params, layer_selector, mo, np):
    """Display parameter table for selected layer."""
    param_table_display = None

    if not layer_selector.value:
        # No layer selected - show nothing
        param_table_display = None
    elif layer_params is None:
        # Layer selected but no parameters found
        param_table_display = mo.md(
            "**No quantization parameters found for this layer.**"
        ).callout(kind="neutral")
    else:
        # Parameters found - create formatted table
        _scale = layer_params["scale"]
        _zero_point = layer_params["zero_point"]
        _weight_shape = layer_params["weight_shape"]
        _weight_dtype = layer_params["weight_dtype"]
        _node_type = layer_params["node_type"]
        _is_per_channel = layer_params["is_per_channel"]

        # Format scale value
        if _is_per_channel:
            _scale_str = (
                f"min={np.min(_scale):.6f}, max={np.max(_scale):.6f}, "
                f"mean={np.mean(_scale):.6f}"
            )
        else:
            _scale_str = f"{float(_scale):.6f}"

        # Format zero-point value
        if _is_per_channel and _zero_point.ndim > 0:
            _zp_str = (
                f"min={int(np.min(_zero_point))}, max={int(np.max(_zero_point))}, "
                f"mean={float(np.mean(_zero_point)):.1f}"
            )
        else:
            _zp_str = str(int(_zero_point))

        # Format shape and dtype
        _shape_str = str(_weight_shape) if _weight_shape else "N/A"
        _dtype_str = _weight_dtype if _weight_dtype else "N/A"

        # Create markdown table
        _table_md = f"""
**Quantization Parameters**

| Parameter | Value |
|-----------|-------|
| **Node Type** | {_node_type} |
| **Scale** | {_scale_str} |
| **Zero-point** | {_zp_str} |
| **Weight Shape** | {_shape_str} |
| **Weight Dtype** | {_dtype_str} |
| **Per-channel** | {_is_per_channel} |
        """

        param_table_display = mo.md(_table_md.strip()).callout(kind="info")

    return (param_table_display,)


@app.cell
def _(param_table_display):
    """Render parameter table."""
    param_table_display
    return


@app.cell
def _(extract_weight_tensors, layer_selector, models):
    """Extract weight tensors for histogram visualization (reactive to dropdown)."""
    weight_tensors = None

    if layer_selector.value and models:
        _fp32_model = models.get("onnx_float")
        _int8_model = models.get("onnx_int8")
        _uint8_model = models.get("onnx_uint8")

        weight_tensors = extract_weight_tensors(
            _fp32_model, _int8_model, layer_selector.value, _uint8_model
        )

    return (weight_tensors,)


@app.cell
def _(layer_selector, mo, np, plt, weight_tensors):
    """Display weight distribution histograms for selected layer."""
    histogram_fig = None

    if not layer_selector.value:
        # No layer selected - show nothing
        histogram_fig = None
    elif weight_tensors is None:
        # Extraction failed
        histogram_fig = mo.md(
            "**No weight tensors found for this layer.**"
        ).callout(kind="neutral")
    else:
        # Check which variants are available
        _fp32 = weight_tensors.get("fp32")
        _int8_dq = weight_tensors.get("int8_dequantized")
        _uint8_dq = weight_tensors.get("uint8_dequantized")

        _available = []
        if _fp32 is not None:
            _available.append(("FP32 Weights", _fp32.flatten()))
        if _int8_dq is not None:
            _available.append(("INT8 (Dequantized)", _int8_dq.flatten()))
        if _uint8_dq is not None:
            _available.append(("UINT8 (Dequantized)", _uint8_dq.flatten()))

        if len(_available) == 0:
            # No weights found
            histogram_fig = mo.md(
                "**No weight tensors found for this layer.**"
            ).callout(kind="neutral")
        else:
            # Close any previous figures to prevent memory leaks
            plt.close("all")

            # Create subplots
            _num_variants = len(_available)
            _fig, _axes = plt.subplots(
                1, _num_variants, figsize=(5 * _num_variants, 4)
            )

            # Ensure axes is always a list
            if _num_variants == 1:
                _axes = [_axes]

            # Determine consistent bin range across all variants
            _all_weights = np.concatenate([_w for _, _w in _available])
            _bin_range = (
                float(np.min(_all_weights)), float(np.max(_all_weights))
            )

            # Plot each variant
            for _ax, (_label, _weights) in zip(_axes, _available):
                _ax.hist(
                    _weights, bins=50, range=_bin_range, alpha=0.7,
                    edgecolor="black"
                )
                _ax.set_xlabel("Weight Value")
                _ax.set_ylabel("Count")
                _ax.set_title(_label)
                _ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            histogram_fig = _fig

    return (histogram_fig,)


@app.cell
def _(histogram_fig):
    """Render histogram figure."""
    histogram_fig
    return


if __name__ == "__main__":
    app.run()
