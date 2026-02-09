"""Neural Network Weight Visualizer - Interactive weight distribution analysis.

Usage:
    marimo edit playground/weight_visualizer.py
"""

import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    """Import dependencies."""
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    MODELS_DIR = project_root / "models"

    return MODELS_DIR, go, mo, np


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
def _(model_selector, mo, np):
    """Load selected model and extract layer/tensor info."""
    import torch

    _model_data = None
    _load_error = None

    if model_selector.value:
        model_path = model_selector.value
        try:
            with mo.status.spinner(title="Loading model..."):
                if model_path.endswith(".onnx"):
                    import onnx
                    from onnx import numpy_helper

                    _model_data = _load_onnx(model_path, onnx, numpy_helper, np)
                elif model_path.endswith(".pt"):
                    _model_data = _load_pytorch(model_path, torch, np)
        except Exception as e:
            _load_error = str(e)

    model_data = _model_data
    load_error = _load_error
    return load_error, model_data, torch


def _load_onnx(path, onnx, numpy_helper, np):
    """Load ONNX model and extract tensor metadata."""
    model = onnx.load(str(path))
    initializers = {}
    for init in model.graph.initializer:
        if init.name:
            initializers[init.name] = init

    # Detect quantization
    node_op_types = {n.op_type for n in model.graph.node}
    is_quantized = bool(
        node_op_types & {"QLinearConv", "QLinearMatMul", "QuantizeLinear"}
    )

    # Build layer list from initializers
    layers = {}
    for name in initializers:
        if name.endswith("_scale") or name.endswith("_zero_point"):
            continue
        if name.endswith("_quantized"):
            base = name.rsplit("_quantized", 1)[0]
        else:
            base = name

        if "bias" in name.lower() or name.endswith(".bias"):
            tensor_type = "bias"
            layer_name = base.replace(".bias", "").replace("/bias", "")
        elif "weight" in name.lower() or name.endswith(".weight"):
            tensor_type = "weight"
            layer_name = base.replace(".weight", "").replace("/weight", "")
        else:
            tensor_type = "weight"
            layer_name = base

        if layer_name not in layers:
            layers[layer_name] = {}
        layers[layer_name][tensor_type] = name

    # Pre-extract all tensor data for reactivity
    tensor_data = {}
    for layer_name, tensors in layers.items():
        tensor_data[layer_name] = {}
        for tensor_type, init_name in tensors.items():
            init = initializers[init_name]
            arr = numpy_helper.to_array(init)
            entry = {"values": arr.flatten().astype(np.float64), "shape": arr.shape}

            if arr.dtype in (np.int8, np.uint8) and is_quantized:
                entry["is_quantized"] = True
                entry["int_values"] = arr.flatten().astype(np.float64)

                # Find scale/zero_point
                scale_name = init_name + "_scale"
                zp_name = init_name + "_zero_point"
                if scale_name not in initializers:
                    base_name = init_name.replace("_quantized", "")
                    scale_name = base_name + "_scale"
                    zp_name = base_name + "_zero_point"

                scale = None
                zp = 0.0
                if scale_name in initializers:
                    scale = float(
                        numpy_helper.to_array(initializers[scale_name]).flat[0]
                    )
                if zp_name in initializers:
                    zp = float(
                        numpy_helper.to_array(initializers[zp_name]).flat[0]
                    )

                entry["scale"] = scale
                entry["zero_point"] = zp
                if scale is not None:
                    entry["values"] = scale * (entry["int_values"] - zp)
            else:
                entry["is_quantized"] = False

            tensor_data[layer_name][tensor_type] = entry

    return {
        "format": "onnx",
        "layers": layers,
        "is_quantized": is_quantized,
        "tensor_data": tensor_data,
    }


def _load_pytorch(path, torch, np):
    """Load PyTorch model and extract tensor metadata."""
    loaded = torch.load(str(path), weights_only=False, map_location="cpu")
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
    else:
        model = loaded
    model.eval()

    is_quantized = False
    layers = {}
    tensor_data = {}

    # Detect TorchScript quantized modules with packed params
    packed_modules = {}
    for name, mod in model.named_modules():
        if not name:
            continue
        try:
            if mod._c.hasattr("_packed_params"):
                packed_modules[name] = mod
        except (AttributeError, RuntimeError):
            pass

    if packed_modules:
        is_quantized = True
        for name, mod in packed_modules.items():
            packed = mod._c.__getattr__("_packed_params")
            w, b = None, None

            # Try conv2d unpack first, then linear unpack
            for unpack_fn in [
                torch.ops.quantized.conv2d_unpack,
                torch.ops.quantized.linear_unpack,
            ]:
                try:
                    w, b = unpack_fn(packed)
                    break
                except (RuntimeError, TypeError):
                    continue

            if w is None:
                continue

            layers[name] = {"weight": name + ".weight"}
            tensor_data[name] = {}

            # Weight
            if hasattr(w, "int_repr"):
                int_arr = w.int_repr().cpu().numpy()
                deq_arr = w.dequantize().cpu().numpy()
                entry = {
                    "values": deq_arr.flatten().astype(np.float64),
                    "shape": tuple(int_arr.shape),
                    "is_quantized": True,
                    "int_values": int_arr.flatten().astype(np.float64),
                    "scale": float(w.q_per_channel_scales().mean()),
                    "zero_point": float(
                        w.q_per_channel_zero_points().float().mean()
                    ),
                }
            else:
                arr = w.detach().cpu().numpy()
                entry = {
                    "values": arr.flatten().astype(np.float64),
                    "shape": arr.shape,
                    "is_quantized": False,
                }
            tensor_data[name]["weight"] = entry

            # Bias
            if b is not None:
                layers[name]["bias"] = name + ".bias"
                arr = b.detach().cpu().numpy()
                tensor_data[name]["bias"] = {
                    "values": arr.flatten().astype(np.float64),
                    "shape": arr.shape,
                    "is_quantized": False,
                }

    # Also include FP32 params from state_dict
    state_dict = model.state_dict()
    for pname, tensor in state_dict.items():
        parts = pname.rsplit(".", 1)
        if len(parts) != 2:
            continue
        layer_name, tensor_type = parts
        if tensor_type not in ("weight", "bias"):
            continue
        # Skip if already handled by quantized extraction
        if layer_name in tensor_data and tensor_type in tensor_data[layer_name]:
            continue

        if layer_name not in layers:
            layers[layer_name] = {}
        layers[layer_name][tensor_type] = pname

        if layer_name not in tensor_data:
            tensor_data[layer_name] = {}

        arr = tensor.cpu().numpy()
        tensor_data[layer_name][tensor_type] = {
            "values": arr.flatten().astype(np.float64),
            "shape": arr.shape,
            "is_quantized": False,
        }

    return {
        "format": "pytorch",
        "layers": layers,
        "is_quantized": is_quantized,
        "tensor_data": tensor_data,
    }


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
        layer_info = model_data["layers"].get(layer_selector.value, {})
        for t in ["weight", "bias"]:
            if t in layer_info:
                _tensor_options[t] = t

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
        td = model_data.get("tensor_data", {})
        layer_td = td.get(layer_selector.value, {})
        tensor_entry = layer_td.get(tensor_selector.value)

    return (tensor_entry,)


@app.cell
def _(mo, tensor_entry):
    """Quantization view toggle (only shown for quantized tensors)."""
    _is_quant = tensor_entry is not None and tensor_entry.get("is_quantized", False)
    quant_view = mo.ui.radio(
        options={"int8 raw values": "int", "dequantized (FP32)": "fp32"},
        value="fp32",
        label="View",
    )
    mo.md(f"""**Quantization View**

{quant_view}""").callout(kind="neutral") if _is_quant else None
    return (quant_view,)


@app.cell
def _(mo, tensor_entry):
    """Bins slider for histogram."""
    bins_slider = mo.ui.slider(
        start=10, stop=200, step=5, value=50, label="Bins"
    )
    bins_slider if tensor_entry is not None else None
    return (bins_slider,)


@app.cell
def _(bins_slider, go, mo, quant_view, tensor_entry):
    """Plotly histogram of weight distribution."""
    _fig = None

    if tensor_entry is not None:
        is_q = tensor_entry.get("is_quantized", False)
        if is_q and quant_view.value == "int":
            data = tensor_entry["int_values"]
            x_title = "Value (int8/uint8)"
            title = "Weight Distribution (Quantized Integer Values)"
        else:
            data = tensor_entry["values"]
            x_title = "Value"
            title = "Weight Distribution"

        _fig = go.Figure()
        _fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=bins_slider.value,
                marker_color="steelblue",
                opacity=0.85,
                name="all",
                histnorm="percent",
                hovertemplate="Range: %{x}<br>Percent: %{y:.2f}%<extra></extra>",
            )
        )
        _fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title="Percentage (%)",
            bargap=0.05,
            height=450,
            template="plotly_white",
        )

    histogram_fig = _fig
    mo.ui.plotly(histogram_fig) if histogram_fig else None
    return (histogram_fig,)


@app.cell
def _(mo, np, quant_view, tensor_entry):
    """Value range analysis inputs."""
    _default_min = ""
    _default_max = ""

    if tensor_entry is not None:
        is_q = tensor_entry.get("is_quantized", False)
        if is_q and quant_view.value == "int":
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
            r_min = float(range_min_input.value)
            r_max = float(range_max_input.value)
        except (ValueError, TypeError):
            _range_display = mo.md("**Invalid min/max values.**").callout(
                kind="danger"
            )
        else:
            is_q = tensor_entry.get("is_quantized", False)
            if is_q and quant_view.value == "int":
                data = tensor_entry["int_values"]
                x_title = "Value (int8/uint8)"
            else:
                data = tensor_entry["values"]
                x_title = "Value"

            in_range = (data >= r_min) & (data <= r_max)
            count_in = int(np.sum(in_range))
            total = len(data)
            pct = count_in / total * 100 if total > 0 else 0.0

            data_in = data[in_range]
            data_out = data[~in_range]
            nbins = bins_slider.value

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=data_out,
                    nbinsx=nbins,
                    marker_color="lightgray",
                    opacity=0.7,
                    name="outside range",
                    histnorm="percent",
                    hovertemplate=(
                        "Range: %{x}<br>Percent: %{y:.2f}%<extra></extra>"
                    ),
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=data_in,
                    nbinsx=nbins,
                    marker_color="orange",
                    opacity=0.85,
                    name="in range",
                    histnorm="percent",
                    hovertemplate=(
                        "Range: %{x}<br>Percent: %{y:.2f}%<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                title="Weight Distribution (Range Highlighted)",
                xaxis_title=x_title,
                yaxis_title="Percentage (%)",
                barmode="overlay",
                bargap=0.05,
                height=450,
                template="plotly_white",
            )

            range_md = f"""**Selected Range:** [{r_min:.6f}, {r_max:.6f}]

**Count in range:** {count_in} / {total}

**Percentage:** {pct:.2f}%"""

            _range_display = mo.vstack(
                [
                    mo.ui.plotly(fig),
                    mo.md(range_md).callout(kind="info"),
                ]
            )

    _range_display
    return


@app.cell
def _(mo, np, tensor_entry):
    """Statistics panel."""
    if tensor_entry is not None:
        vals = tensor_entry["values"]
        shape = tensor_entry["shape"]
        _min = float(np.min(vals))
        _max = float(np.max(vals))
        _mean = float(np.mean(vals))
        _std = float(np.std(vals))
        _total = int(np.prod(shape))

        stats_lines = [
            f"**Min:** {_min:.6f} &nbsp;&nbsp; **Max:** {_max:.6f}",
            f"**Mean:** {_mean:.6f} &nbsp;&nbsp; **Std:** {_std:.6f}",
            f"**Shape:** {shape} &nbsp;&nbsp; **Total params:** {_total:,}",
        ]
        if tensor_entry.get("is_quantized", False):
            scale = tensor_entry.get("scale")
            zp = tensor_entry.get("zero_point")
            scale_str = f"{scale:.6f}" if scale is not None else "N/A"
            zp_str = f"{zp:.1f}" if zp is not None else "N/A"
            stats_lines.append(
                f"**Scale:** {scale_str} &nbsp;&nbsp; **Zero Point:** {zp_str}"
            )

        mo.md("\n\n".join(stats_lines)).callout(kind="neutral")
    else:
        None
    return


if __name__ == "__main__":
    app.run()
