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
        onnx_float = models.get("onnx_float")
        onnx_int8 = models.get("onnx_int8")

        # Only compute if both models are available
        if onnx_float is not None and onnx_int8 is not None:
            layer_ranges = compute_all_layer_ranges(onnx_float, onnx_int8)

    return (layer_ranges,)


@app.cell
def _(layer_ranges, mo, np, plt):
    """Visualize weight ranges as heatmap for whole-model overview."""
    heatmap_fig = None

    if layer_ranges is not None and len(layer_ranges) > 0:
        # Close any previous figures to prevent memory leaks
        plt.close("all")

        # Create subplots for FP32 and INT8 side-by-side
        num_layers = len(layer_ranges)
        fig_height = max(6, num_layers * 0.35)
        fig, (ax_fp32, ax_int8) = plt.subplots(1, 2, figsize=(14, fig_height))

        # Extract data for plotting
        layer_names = [lr["name"] for lr in layer_ranges]
        fp32_mins = [lr["fp32_min"] for lr in layer_ranges]
        fp32_maxs = [lr["fp32_max"] for lr in layer_ranges]
        fp32_ranges = [lr["fp32_range"] for lr in layer_ranges]
        int8_mins = [lr["int8_min"] for lr in layer_ranges]
        int8_maxs = [lr["int8_max"] for lr in layer_ranges]
        int8_ranges = [lr["int8_range"] for lr in layer_ranges]

        # Y positions for layers (reversed so first layer is on top)
        y_positions = np.arange(num_layers)[::-1]

        # Normalize range magnitudes for color mapping
        all_ranges = fp32_ranges + int8_ranges
        range_min = min(all_ranges) if all_ranges else 0
        range_max = max(all_ranges) if all_ranges else 1

        # FP32 subplot - horizontal bars from min to max
        for i, (y_pos, fp32_min, fp32_max, fp32_range) in enumerate(
            zip(y_positions, fp32_mins, fp32_maxs, fp32_ranges)
        ):
            # Color based on range magnitude
            color_val = (fp32_range - range_min) / (
                range_max - range_min + 1e-10
            )
            color = plt.cm.viridis(color_val)

            # Horizontal bar from min to max
            ax_fp32.barh(
                y_pos,
                width=fp32_max - fp32_min,
                left=fp32_min,
                height=0.8,
                color=color,
            )

        ax_fp32.set_yticks(y_positions)
        ax_fp32.set_yticklabels(layer_names, fontsize=8)
        ax_fp32.set_xlabel("Weight Value")
        ax_fp32.set_title("FP32 Weight Ranges")
        ax_fp32.grid(axis="x", alpha=0.3)

        # INT8 subplot - same layout
        for i, (y_pos, int8_min, int8_max, int8_range) in enumerate(
            zip(y_positions, int8_mins, int8_maxs, int8_ranges)
        ):
            # Color based on range magnitude
            color_val = (int8_range - range_min) / (
                range_max - range_min + 1e-10
            )
            color = plt.cm.viridis(color_val)

            # Horizontal bar from min to max
            ax_int8.barh(
                y_pos,
                width=int8_max - int8_min,
                left=int8_min,
                height=0.8,
                color=color,
            )

        ax_int8.set_yticks(y_positions)
        ax_int8.set_yticklabels(layer_names, fontsize=8)
        ax_int8.set_xlabel("Weight Value")
        ax_int8.set_title("INT8 Weight Ranges (Dequantized)")
        ax_int8.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        heatmap_fig = fig

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
    dropdown_options = {}

    if layer_names:
        for layer in layer_names:
            # Check if layer or any substring matches layers_with_params
            has_params = any(layer in param_layer or param_layer in layer
                           for param_layer in layers_with_params)

            if has_params:
                display_name = f"{layer} [Q]"
            else:
                display_name = layer

            dropdown_options[display_name] = layer
    else:
        dropdown_options = {"Select a layer...": None}

    layer_selector = mo.ui.dropdown(
        options=dropdown_options,
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
        onnx_int8 = models.get("onnx_int8")
        if onnx_int8 is not None:
            layer_params = extract_layer_params(onnx_int8, layer_selector.value)

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
        scale = layer_params["scale"]
        zero_point = layer_params["zero_point"]
        weight_shape = layer_params["weight_shape"]
        weight_dtype = layer_params["weight_dtype"]
        node_type = layer_params["node_type"]
        is_per_channel = layer_params["is_per_channel"]

        # Format scale value
        if is_per_channel:
            scale_min = np.min(scale)
            scale_max = np.max(scale)
            scale_mean = np.mean(scale)
            scale_str = (
                f"min={scale_min:.6f}, max={scale_max:.6f}, "
                f"mean={scale_mean:.6f}"
            )
        else:
            scale_str = f"{float(scale):.6f}"

        # Format zero-point value
        if is_per_channel and zero_point.ndim > 0:
            zp_min = int(np.min(zero_point))
            zp_max = int(np.max(zero_point))
            zp_mean = float(np.mean(zero_point))
            zp_str = f"min={zp_min}, max={zp_max}, mean={zp_mean:.1f}"
        else:
            zp_str = str(int(zero_point))

        # Format shape and dtype
        shape_str = str(weight_shape) if weight_shape else "N/A"
        dtype_str = weight_dtype if weight_dtype else "N/A"

        # Create markdown table
        table_md = f"""
**Quantization Parameters**

| Parameter | Value |
|-----------|-------|
| **Node Type** | {node_type} |
| **Scale** | {scale_str} |
| **Zero-point** | {zp_str} |
| **Weight Shape** | {shape_str} |
| **Weight Dtype** | {dtype_str} |
| **Per-channel** | {is_per_channel} |
        """

        param_table_display = mo.md(table_md.strip()).callout(kind="info")

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
        fp32_model = models.get("onnx_float")
        int8_model = models.get("onnx_int8")
        uint8_model = models.get("onnx_uint8")

        weight_tensors = extract_weight_tensors(
            fp32_model, int8_model, layer_selector.value, uint8_model
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
        fp32 = weight_tensors.get("fp32")
        int8_dq = weight_tensors.get("int8_dequantized")
        uint8_dq = weight_tensors.get("uint8_dequantized")

        available = []
        if fp32 is not None:
            available.append(("FP32 Weights", fp32.flatten()))
        if int8_dq is not None:
            available.append(("INT8 (Dequantized)", int8_dq.flatten()))
        if uint8_dq is not None:
            available.append(("UINT8 (Dequantized)", uint8_dq.flatten()))

        if len(available) == 0:
            # No weights found
            histogram_fig = mo.md(
                "**No weight tensors found for this layer.**"
            ).callout(kind="neutral")
        else:
            # Close any previous figures to prevent memory leaks
            plt.close("all")

            # Create subplots
            num_variants = len(available)
            fig, axes = plt.subplots(1, num_variants, figsize=(5 * num_variants, 4))

            # Ensure axes is always a list
            if num_variants == 1:
                axes = [axes]

            # Determine consistent bin range across all variants
            all_weights = np.concatenate([w for _, w in available])
            bin_range = (float(np.min(all_weights)), float(np.max(all_weights)))
            bins = 50

            # Plot each variant
            for ax, (label, weights) in zip(axes, available):
                ax.hist(
                    weights, bins=bins, range=bin_range, alpha=0.7,
                    edgecolor="black"
                )
                ax.set_xlabel("Weight Value")
                ax.set_ylabel("Count")
                ax.set_title(label)
                ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            histogram_fig = fig

    return (histogram_fig,)


@app.cell
def _(histogram_fig):
    """Render histogram figure."""
    histogram_fig
    return


if __name__ == "__main__":
    app.run()
