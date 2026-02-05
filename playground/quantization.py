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
    """Display folder picker and instructions."""
    if not folder_picker.value:
        content = mo.vstack([
            folder_picker,
            mo.md("**Select a model folder to begin**").callout(kind="info")
        ])
    else:
        content = mo.vstack([
            folder_picker,
            mo.md(f"**Selected folder:** `{folder_picker.path(0)}`").callout(kind="success")
        ])

    return content,


@app.cell
def __(content):
    """Render the content."""
    content


if __name__ == "__main__":
    app.run()
