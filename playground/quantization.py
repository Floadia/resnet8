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
    from playground.utils import load_model_variants, get_model_summary

    return mo, Path, load_model_variants, get_model_summary


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
        initial_path="./models",
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


if __name__ == "__main__":
    app.run()
