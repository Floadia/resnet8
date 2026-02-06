"""Quantization Playground - Interactive notebook for model quantization experiments.

Usage:
    marimo edit playground/quantization.py
"""

import marimo

__generated_with = "0.15.5"
app = marimo.App()


@app.cell
def __():
    """Import dependencies."""
    import marimo as mo
    from pathlib import Path
    import sys

    # Add project root to path for imports
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from playground.utils import load_model_variants, get_model_summary, get_all_layer_names, get_layer_type

    return mo, Path, load_model_variants, get_model_summary, get_all_layer_names, get_layer_type


@app.cell
def __(mo):
    """Display notebook header and instructions."""
    return mo.md(
        """
        # Quantization Playground

        Interactive notebook for exploring ONNX and PyTorch quantization.
        """
    )


@app.cell
def __(mo):
    """File picker for selecting model folder."""
    folder_picker = mo.ui.file_browser(
        initial_path="../models",
        selection_mode="directory",
        multiple=False,
        label="Select model folder"
    )

    return folder_picker,


@app.cell
def __(mo, folder_picker):
    """Display folder picker."""
    folder_picker


@app.cell
def __(mo, folder_picker, load_model_variants, get_model_summary):
    """Load models from selected folder with spinner."""
    models = None
    models_summary = None
    load_error = None

    if folder_picker.value:
        try:
            with mo.status.spinner(title="Loading models..."):
                models = load_model_variants(folder_picker.path(0))
                models_summary = get_model_summary(models)
        except Exception as e:
            load_error = str(e)

    return models, models_summary, load_error


@app.cell
def __(mo, folder_picker, models_summary, load_error):
    """Display model loading status and summary."""
    if not folder_picker.value:
        # Initial state - show instructions
        display = mo.md("**Select a model folder to begin**").callout(kind="info")
    elif load_error:
        # Error state - show error message
        display = mo.md(f"**Error loading models:** {load_error}").callout(kind="danger")
    elif models_summary and models_summary['total_loaded'] > 0:
        # Success state - show summary
        onnx_list = ", ".join(models_summary['onnx_available']) if models_summary['onnx_available'] else "None"
        pytorch_list = ", ".join(models_summary['pytorch_available']) if models_summary['pytorch_available'] else "None"

        summary_text = f"""
        **Models loaded successfully!**

        **Folder:** `{folder_picker.path(0)}`

        **Total models:** {models_summary['total_loaded']}

        **ONNX variants:** {onnx_list}

        **PyTorch variants:** {pytorch_list}
        """
        display = mo.md(summary_text).callout(kind="success")
    else:
        # No models found
        display = mo.md(f"**No models found in:** `{folder_picker.path(0)}`").callout(kind="warn")

    return display,


@app.cell
def __(display):
    """Render the display."""
    display


@app.cell
def __(models, get_all_layer_names):
    """Extract layer names from loaded models."""
    layer_data = None
    layer_names = []
    layer_source = None

    if models:
        layer_data = get_all_layer_names(models)
        layer_names = layer_data.get('layer_names', [])
        layer_source = layer_data.get('source')

    return layer_data, layer_names, layer_source


@app.cell
def __(mo, layer_names):
    """Layer selection dropdown."""
    layer_selector = mo.ui.dropdown(
        options=layer_names if layer_names else ["Select a layer..."],
        value=None,
        allow_select_none=True,
        label="Layer to analyze"
    )

    return layer_selector,


@app.cell
def __(mo, layer_selector):
    """Display layer selector."""
    layer_selector


@app.cell
def __(mo, layer_selector, models, layer_source, get_layer_type):
    """Display layer information (reactive to dropdown selection)."""
    layer_info_display = None

    if not layer_selector.value:
        # No selection - show placeholder
        layer_info_display = mo.md("**Select a layer from the dropdown above to view details.**").callout(kind="neutral")
    else:
        # Layer selected - show info
        selected_layer = layer_selector.value

        # Get layer type
        layer_type = None
        if models and layer_source:
            model_key = f"{layer_source}_float" if layer_source in ['onnx', 'pytorch'] else None
            if model_key and models.get(model_key):
                layer_type = get_layer_type(models[model_key], selected_layer)

        type_text = f" ({layer_type})" if layer_type else ""

        layer_info_text = f"""
        **Layer:** `{selected_layer}`{type_text}

        **Source:** {layer_source.upper() if layer_source else 'Unknown'}
        """

        layer_info_display = mo.md(layer_info_text).callout(kind="info")

    return layer_info_display,


@app.cell
def __(layer_info_display):
    """Render layer info display."""
    layer_info_display


if __name__ == "__main__":
    app.run()
