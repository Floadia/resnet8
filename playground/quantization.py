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

    # Add project root to path for imports
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from playground.utils import (
        get_all_layer_names,
        get_layer_type,
        get_model_summary,
        load_model_variants,
    )

    return (
        Path,
        get_all_layer_names,
        get_layer_type,
        get_model_summary,
        load_model_variants,
        mo,
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
def _(get_all_layer_names, models):
    """Extract layer names from loaded models."""
    layer_data = None
    layer_names = []
    layer_source = None

    if models:
        layer_data = get_all_layer_names(models)
        layer_names = layer_data.get("layer_names", [])
        layer_source = layer_data.get("source")
    return layer_names, layer_source


@app.cell
def _(layer_names, mo):
    """Layer selection dropdown."""
    layer_selector = mo.ui.dropdown(
        options=layer_names if layer_names else ["Select a layer..."],
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


if __name__ == "__main__":
    app.run()
