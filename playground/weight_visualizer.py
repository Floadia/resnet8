"""Neural Network Weight Visualizer - Interactive weight distribution analysis.

Usage:
    marimo edit playground/weight_visualizer.py
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell
def _():
    """Import dependencies."""
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    from playground.utils.notebook import ensure_project_root
    from playground.utils.weight_visualizer_data import (
        collect_model_files,
        load_model_data,
    )

    project_root = ensure_project_root(__file__)

    MODELS_DIR = project_root / "models"
    return (
        MODELS_DIR,
        collect_model_files,
        go,
        load_model_data,
        mo,
        np,
    )


@app.cell
def _(mo):
    """Display notebook title."""
    mo.md("# Neural Network Weight Visualizer")
    return


@app.cell
def _(MODELS_DIR):
    """Detect model files in models/ directory."""
    model_files = {}
    if MODELS_DIR.exists():
        for f in sorted(MODELS_DIR.iterdir()):
            if f.suffix in (".onnx", ".pt"):
                model_files[f.name] = str(f)
    return (model_files,)


@app.cell
def _(mo, model_files):
    """Model selection dropdown."""
    model_selector = mo.ui.dropdown(
        options=model_files,
        value=None,
        allow_select_none=True,
        label="Model",
    )
    model_selector
    return (model_selector,)


@app.cell
def _(collect_model_files, MODELS_DIR):
    """Detect model files in models/ directory."""
    model_files = collect_model_files(MODELS_DIR)
    return (model_files,)


@app.cell
def _(load_model_data, mo, model_selector):
    """Load selected model and extract layer/tensor info."""
    model_data = None
    load_error = None

    if model_selector.value:
        try:
            with mo.status.spinner(title="Loading model..."):
                model_data = load_model_data(model_selector.value)
        except Exception as e:
            load_error = str(e)

    return load_error, model_data


@app.cell
def _(load_error, mo):
    """Display load error if any."""
    mo.md(f"**Error loading model:** {load_error}").callout(
        kind="danger"
    ) if load_error else None
    return


@app.cell
def _(mo, model_data):
    """Layer selection dropdown."""
    _layer_options = {}
    if model_data:
        _layer_options = {name: name for name in sorted(model_data["layers"].keys())}

    layer_selector = mo.ui.dropdown(
        options=_layer_options,
        value=None,
        allow_select_none=True,
        label="Layer",
    )
    return (layer_selector,)


@app.cell
def _(layer_selector, mo, model_data):
    """Tensor type selection dropdown."""
    _tensor_options = {}
    if model_data and layer_selector.value:
        _layer_info = model_data["layers"].get(layer_selector.value, {})
        for _t in ["weight", "bias"]:
            if _t in _layer_info:
                _tensor_options[_t] = _t

    tensor_selector = mo.ui.dropdown(
        options=_tensor_options,
        value="weight" if "weight" in _tensor_options else None,
        allow_select_none=True,
        label="Tensor",
    )
    mo.hstack([layer_selector, tensor_selector])
    return (tensor_selector,)


@app.cell
def _(layer_selector, model_data, tensor_selector):
    """Look up pre-extracted tensor data."""
    tensor_entry = None

    if model_data and layer_selector.value and tensor_selector.value:
        _td = model_data.get("tensor_data", {})
        _layer_td = _td.get(layer_selector.value, {})
        tensor_entry = _layer_td.get(tensor_selector.value)
    return (tensor_entry,)


@app.cell
def _(mo, tensor_entry):
    """Quantization view toggle (only shown for quantized tensors)."""
    _is_quant = tensor_entry is not None and tensor_entry.get("is_quantized", False)
    quant_view = mo.ui.radio(
        options={"int8 raw values": "int", "dequantized (FP32)": "fp32"},
        value="dequantized (FP32)",
        label="View",
    )
    mo.md(f"""**Quantization View**

    {quant_view}""").callout(kind="neutral") if _is_quant else None
    return (quant_view,)


@app.cell
def _(mo, tensor_entry):
    """Bins slider for histogram."""
    bins_slider = mo.ui.slider(start=10, stop=200, step=5, value=50, label="Bins")
    bins_slider if tensor_entry is not None else None
    return (bins_slider,)


@app.cell
def _(bins_slider, go, mo, quant_view, tensor_entry):
    """Plotly histogram of weight distribution."""
    _fig = None

    if tensor_entry is not None:
        _is_q = tensor_entry.get("is_quantized", False)
        if _is_q and quant_view.value == "int":
            _data = tensor_entry["int_values"]
            _x_title = "Value (int8/uint8)"
            _title = "Weight Distribution (Quantized Integer Values)"
        else:
            _data = tensor_entry["values"]
            _x_title = "Value"
            _title = "Weight Distribution"

        _fig = go.Figure()
        _fig.add_trace(
            go.Histogram(
                x=_data,
                nbinsx=bins_slider.value,
                marker_color="steelblue",
                opacity=0.85,
                name="all",
                histnorm="percent",
                hovertemplate="Range: %{x}<br>Percent: %{y:.2f}%<extra></extra>",
            )
        )
        _fig.update_layout(
            title=_title,
            xaxis_title=_x_title,
            yaxis_title="Percentage (%)",
            bargap=0.05,
            height=450,
            template="plotly_white",
        )

    histogram_fig = _fig
    mo.ui.plotly(histogram_fig) if histogram_fig else None
    return


@app.cell
def _(mo, np, quant_view, tensor_entry):
    """Value range analysis inputs."""
    _default_min = ""
    _default_max = ""

    if tensor_entry is not None:
        _is_q = tensor_entry.get("is_quantized", False)
        if _is_q and quant_view.value == "int":
            _data = tensor_entry["int_values"]
        else:
            _data = tensor_entry["values"]
        _std = float(np.std(_data))
        _mean = float(np.mean(_data))
        _default_min = f"{_mean - _std:.6f}"
        _default_max = f"{_mean + _std:.6f}"

    range_min_input = mo.ui.text(value=_default_min, label="Min")
    range_max_input = mo.ui.text(value=_default_max, label="Max")
    apply_button = mo.ui.run_button(label="Apply Range")
    mo.hstack(
        [range_min_input, range_max_input, apply_button], justify="start", gap=1
    ) if tensor_entry is not None else None
    return apply_button, range_max_input, range_min_input


@app.cell
def _(
    apply_button,
    bins_slider,
    go,
    mo,
    np,
    quant_view,
    range_max_input,
    range_min_input,
    tensor_entry,
):
    """Value range analysis results and highlighted histogram."""
    _range_display = None

    if tensor_entry is not None and apply_button.value:
        try:
            _r_min = float(range_min_input.value)
            _r_max = float(range_max_input.value)
        except (ValueError, TypeError):
            _range_display = mo.md("**Invalid min/max values.**").callout(kind="danger")
        else:
            _is_q = tensor_entry.get("is_quantized", False)
            if _is_q and quant_view.value == "int":
                _data = tensor_entry["int_values"]
                _x_title = "Value (int8/uint8)"
            else:
                _data = tensor_entry["values"]
                _x_title = "Value"

            _in_range = (_data >= _r_min) & (_data <= _r_max)
            _count_in = int(np.sum(_in_range))
            _total = len(_data)
            _pct = _count_in / _total * 100 if _total > 0 else 0.0

            _nbins = bins_slider.value
            _counts_all, _bin_edges = np.histogram(_data, bins=_nbins)
            _counts_in, _ = np.histogram(_data[_in_range], bins=_bin_edges)
            _counts_out = _counts_all - _counts_in
            _pct_in = _counts_in / _total * 100
            _pct_out = _counts_out / _total * 100
            _bin_centers = (_bin_edges[:-1] + _bin_edges[1:]) / 2
            _bin_width = _bin_edges[1] - _bin_edges[0]

            _fig = go.Figure()
            _fig.add_trace(
                go.Bar(
                    x=_bin_centers,
                    y=_pct_out,
                    width=_bin_width * 0.95,
                    marker_color="lightgray",
                    opacity=0.7,
                    name="outside range",
                    hovertemplate=("Range: %{x}<br>Percent: %{y:.2f}%<extra></extra>"),
                )
            )
            _fig.add_trace(
                go.Bar(
                    x=_bin_centers,
                    y=_pct_in,
                    width=_bin_width * 0.95,
                    marker_color="orange",
                    opacity=0.85,
                    name="in range",
                    hovertemplate=("Range: %{x}<br>Percent: %{y:.2f}%<extra></extra>"),
                )
            )
            _fig.update_layout(
                title="Weight Distribution (Range Highlighted)",
                xaxis_title=_x_title,
                yaxis_title="Percentage (%)",
                barmode="stack",
                bargap=0.05,
                height=450,
                template="plotly_white",
            )

            _range_md = f"""**Selected Range:** [{_r_min:.6f}, {_r_max:.6f}]

    **Count in range:** {_count_in} / {_total}

    **Percentage:** {_pct:.2f}%"""

            _range_display = mo.vstack(
                [
                    mo.ui.plotly(_fig),
                    mo.md(_range_md).callout(kind="info"),
                ]
            )

    _range_display
    return


@app.cell
def _(mo, np, tensor_entry):
    """Statistics panel."""
    if tensor_entry is not None:
        _vals = tensor_entry["values"]
        _shape = tensor_entry["shape"]
        _min = float(np.min(_vals))
        _max = float(np.max(_vals))
        _mean = float(np.mean(_vals))
        _std = float(np.std(_vals))
        _total = int(np.prod(_shape))

        _stats_lines = [
            f"**Min:** {_min:.6f} &nbsp;&nbsp; **Max:** {_max:.6f}",
            f"**Mean:** {_mean:.6f} &nbsp;&nbsp; **Std:** {_std:.6f}",
            f"**Shape:** {_shape} &nbsp;&nbsp; **Total params:** {_total:,}",
        ]
        if tensor_entry.get("is_quantized", False):
            _scale = tensor_entry.get("scale")
            _zp = tensor_entry.get("zero_point")
            _scale_str = f"{_scale:.6f}" if _scale is not None else "N/A"
            _zp_str = f"{_zp:.1f}" if _zp is not None else "N/A"
            _stats_lines.append(
                f"**Scale:** {_scale_str} &nbsp;&nbsp; **Zero Point:** {_zp_str}"
            )

        mo.md("\n\n".join(_stats_lines)).callout(kind="neutral")
    else:
        None
    return


if __name__ == "__main__":
    app.run()
