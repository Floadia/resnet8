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
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    scripts_dir = project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from get_resnet8_intermediate import (
        load_cifar10_test_sample,
        normalize_input,
        run_with_hook,
        collect_named_tensors,
        get_model_layers,
        load_model as load_intermediate_model,
    )

    MODELS_DIR = project_root / "models"
    return (
        MODELS_DIR,
        collect_named_tensors,
        get_model_layers,
        go,
        load_cifar10_test_sample,
        load_intermediate_model,
        mo,
        normalize_input,
        np,
        run_with_hook,
    )


@app.cell
def _(mo):
    """Detect script vs interactive mode."""
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


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
        for _f in sorted(MODELS_DIR.iterdir()):
            if _f.suffix in (".onnx", ".pt"):
                model_files[_f.name] = str(_f)
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
def _(is_script_mode, mo, model_files, model_selector, np):
    """Load selected model and extract layer/tensor info."""
    import torch

    def _load_onnx_model(path):
        """Load ONNX model and extract tensor metadata."""
        import onnx
        from onnx import numpy_helper as nh

        model = onnx.load(str(path))
        initializers = {}
        for init in model.graph.initializer:
            if init.name:
                initializers[init.name] = init

        node_op_types = {n.op_type for n in model.graph.node}
        is_quantized = bool(
            node_op_types & {"QLinearConv", "QLinearMatMul", "QuantizeLinear"}
        )

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

        tensor_data = {}
        for ln, tensors in layers.items():
            tensor_data[ln] = {}
            for tt, init_name in tensors.items():
                init = initializers[init_name]
                arr = nh.to_array(init)
                entry = {
                    "values": arr.flatten().astype(np.float64),
                    "shape": arr.shape,
                }

                if arr.dtype in (np.int8, np.uint8) and is_quantized:
                    entry["is_quantized"] = True
                    entry["int_values"] = arr.flatten().astype(np.float64)

                    scale_name = init_name + "_scale"
                    zp_name = init_name + "_zero_point"
                    if scale_name not in initializers:
                        base_name = init_name.replace("_quantized", "")
                        scale_name = base_name + "_scale"
                        zp_name = base_name + "_zero_point"

                    scale = None
                    zp = 0.0
                    if scale_name in initializers:
                        scale = float(nh.to_array(initializers[scale_name]).flat[0])
                    if zp_name in initializers:
                        zp = float(nh.to_array(initializers[zp_name]).flat[0])

                    entry["scale"] = scale
                    entry["zero_point"] = zp
                    if scale is not None:
                        entry["values"] = scale * (entry["int_values"] - zp)
                else:
                    entry["is_quantized"] = False

                tensor_data[ln][tt] = entry

        return {
            "format": "onnx",
            "layers": layers,
            "is_quantized": is_quantized,
            "tensor_data": tensor_data,
        }

    def _load_pytorch_model(path):
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

    if is_script_mode:
        # Use first available model file in script mode
        _path = next(iter(model_files.values()), None)
    else:
        _path = model_selector.value

    with mo.status.spinner(title="Loading model..."):
        if _path and _path.endswith(".onnx"):
            model_data = _load_onnx_model(_path)
        elif _path and _path.endswith(".pt"):
            model_data = _load_pytorch_model(_path)
        else:
            model_data = {
                "format": "none",
                "layers": {},
                "is_quantized": False,
                "tensor_data": {},
            }
    return (model_data,)


@app.cell
def _(mo, model_data):
    """Input source selection for activation capture (PyTorch only)."""
    _is_pytorch = model_data.get("format") == "pytorch"

    input_source = mo.ui.radio(
        options={"CIFAR-10 sample": "cifar10", "Random input": "random"},
        value="CIFAR-10 sample",
        label="Input Source",
    )
    sample_index = mo.ui.number(
        start=0, stop=9999, step=1, value=0, label="Sample Index"
    )
    run_button = mo.ui.run_button(label="Run Inference")

    if _is_pytorch:
        mo.vstack([
            mo.md("### Intermediate Activations"),
            mo.hstack([input_source, sample_index, run_button], justify="start", gap=1),
        ])
    else:
        mo.md("_Activation capture requires a PyTorch model (.pt)_").callout(kind="neutral") if model_data.get("format") != "none" else None

    return input_source, run_button, sample_index


@app.cell
def _(
    collect_named_tensors,
    get_model_layers,
    input_source,
    is_script_mode,
    load_cifar10_test_sample,
    load_intermediate_model,
    model_data,
    model_selector,
    mo,
    normalize_input,
    np,
    run_button,
    run_with_hook,
    sample_index,
):
    """Capture intermediate activations via forward hooks (PyTorch only)."""
    import torch as _torch

    # ALWAYS define defaults so downstream cells can reference activation_data
    activation_data = {}
    activation_status = ""
    _output = mo.md("")

    _is_pytorch = model_data.get("format") == "pytorch"
    if _is_pytorch and run_button.value:
        _path = model_selector.value
        _device = _torch.device("cpu")
        _data_dir = "/mnt/ext1/references/tiny/benchmark/training/image_classification/cifar-10-batches-py"

        with mo.status.spinner(title="Capturing activations..."):
            try:
                _model = load_intermediate_model(_path, _device)
                _model.eval()

                _sample = None
                if input_source.value == "random":
                    _sample = np.random.random((1, 32, 32, 3)).astype(np.float32)
                    activation_status = "Input: random (1, 32, 32, 3)"
                else:
                    try:
                        _sample, _label = load_cifar10_test_sample(
                            _data_dir, int(sample_index.value)
                        )
                        activation_status = f"Input: CIFAR-10 test[{int(sample_index.value)}] (label={_label})"
                    except FileNotFoundError:
                        _output = mo.md(
                            "**CIFAR-10 data not found.** Use 'Random input' instead, "
                            f"or ensure data exists at `{_data_dir}`."
                        ).callout(kind="danger")

                if _sample is not None:
                    _x = normalize_input(_sample, _device)
                    _layers = get_model_layers(_model)

                    for _ln in _layers:
                        try:
                            _value = run_with_hook(_model, _ln, _x)
                            _named = list(collect_named_tensors(_value, _ln))
                            for _tn, _arr in _named:
                                activation_data[_tn] = {
                                    "values": _arr.flatten().astype(np.float64),
                                    "shape": _arr.shape,
                                }
                        except Exception:
                            pass  # Skip layers where hooks fail

                    _output = mo.md(
                        f"**Activations captured:** {len(activation_data)} tensors | {activation_status}"
                    ).callout(kind="success")

            except Exception as _e:
                _output = mo.md(
                    f"**Activation capture failed:** {_e}"
                ).callout(kind="danger")

    _output
    return activation_data, activation_status


@app.cell
def _(mo, model_data):
    """Layer selection dropdown."""
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
    _layer_info = model_data["layers"].get(layer_selector.value, {})
    _tensor_options = {_t: _t for _t in ["weight", "bias"] if _t in _layer_info}

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
    _td = model_data.get("tensor_data", {})
    _layer_td = _td.get(layer_selector.value, {})
    tensor_entry = _layer_td.get(tensor_selector.value)
    return (tensor_entry,)


@app.cell
def _(activation_data, mo, model_data):
    """Toggle between weight and activation views."""
    _has_activations = bool(activation_data) if 'activation_data' in dir() else False
    _is_pytorch = model_data.get("format") == "pytorch"

    view_toggle = mo.ui.radio(
        options={"Weights": "weights", "Activations": "activations"},
        value="Weights",
        label="View",
    )
    view_toggle if _is_pytorch and _has_activations else None
    return (view_toggle,)


@app.cell
def _(activation_data, layer_selector, mo, model_data, tensor_entry, view_toggle):
    """Resolve which data to display based on view toggle."""
    activation_mismatch_msg = None
    if view_toggle.value == "activations" and activation_data:
        # Look for activation matching current layer
        _layer = layer_selector.value
        _act = activation_data.get(_layer)
        if _act is None:
            # Try prefix match (activation keys may have suffixes from collect_named_tensors)
            for _k, _v in activation_data.items():
                if _k.startswith(_layer + "_") or _k == _layer:
                    _act = _v
                    break
        if _act is None:
            activation_mismatch_msg = mo.md(
                f"**No activation captured for layer '{_layer}'.** "
                "Weight layer names and activation layer names may not align. "
                "Showing weight view instead."
            ).callout(kind="warn")
        display_entry = _act if _act is not None else tensor_entry
        display_mode = "activations" if _act is not None else "weights"
    else:
        display_entry = tensor_entry
        display_mode = "weights"
    activation_mismatch_msg
    return activation_mismatch_msg, display_entry, display_mode


@app.cell
def _(display_entry, display_mode, mo):
    """Quantization view toggle (only shown for quantized weights)."""
    _is_weight_mode = display_mode == "weights" if 'display_mode' in dir() else True
    _is_quant = (display_entry is not None and
                 display_entry.get("is_quantized", False) and
                 _is_weight_mode)
    quant_view = mo.ui.radio(
        options={"int8 raw values": "int", "dequantized (FP32)": "fp32"},
        value="dequantized (FP32)",
        label="View",
    )
    mo.md(f"""**Quantization View**

    {quant_view}""").callout(kind="neutral") if _is_quant else None
    return (quant_view,)


@app.cell
def _(display_entry, mo):
    """Bins slider for histogram."""
    bins_slider = mo.ui.slider(start=10, stop=200, step=5, value=50, label="Bins")
    bins_slider if display_entry is not None else None
    return (bins_slider,)


@app.cell
def _(bins_slider, display_entry, display_mode, go, mo, quant_view):
    """Plotly histogram of weight or activation distribution."""
    mo.stop(display_entry is None)

    if display_mode == "activations":
        _data = display_entry["values"]
        _x_title = "Value"
        _title = "Activation Distribution"
        _color = "darkorange"
    else:
        _is_q = display_entry.get("is_quantized", False)
        if _is_q and quant_view.value == "int":
            _data = display_entry["int_values"]
            _x_title = "Value (int8/uint8)"
            _title = "Weight Distribution (Quantized Integer Values)"
        else:
            _data = display_entry["values"]
            _x_title = "Value"
            _title = "Weight Distribution"
        _color = "steelblue"

    _fig = go.Figure()
    _fig.add_trace(
        go.Histogram(
            x=_data,
            nbinsx=bins_slider.value,
            marker_color=_color,
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

    mo.ui.plotly(_fig)
    return


@app.cell
def _(display_entry, mo, np, quant_view):
    """Value range analysis inputs."""
    _is_q = display_entry.get("is_quantized", False) if display_entry is not None else False
    if _is_q and quant_view.value == "int":
        _data = display_entry["int_values"]
    else:
        _data = display_entry["values"] if display_entry is not None else np.array([])
    _std = float(np.std(_data)) if len(_data) > 0 else 0.0
    _mean = float(np.mean(_data)) if len(_data) > 0 else 0.0
    _default_min = f"{_mean - _std:.6f}"
    _default_max = f"{_mean + _std:.6f}"

    range_min_input = mo.ui.text(value=_default_min, label="Min")
    range_max_input = mo.ui.text(value=_default_max, label="Max")
    apply_button = mo.ui.run_button(label="Apply Range")
    mo.hstack([range_min_input, range_max_input, apply_button], justify="start", gap=1) if display_entry is not None else None
    return apply_button, range_max_input, range_min_input


@app.cell
def _(
    apply_button,
    bins_slider,
    display_entry,
    go,
    mo,
    np,
    quant_view,
    range_max_input,
    range_min_input,
):
    """Value range analysis results and highlighted histogram."""
    mo.stop(not apply_button.value)

    _r_min = float(range_min_input.value)
    _r_max = float(range_max_input.value)

    _is_q = display_entry.get("is_quantized", False) if display_entry is not None else False
    if _is_q and quant_view.value == "int":
        _data = display_entry["int_values"]
        _x_title = "Value (int8/uint8)"
    else:
        _data = display_entry["values"]
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
            hovertemplate="Range: %{x}<br>Percent: %{y:.2f}%<extra></extra>",
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
            hovertemplate="Range: %{x}<br>Percent: %{y:.2f}%<extra></extra>",
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

    mo.vstack(
        [
            mo.ui.plotly(_fig),
            mo.md(_range_md).callout(kind="info"),
        ]
    )
    return


@app.cell
def _(display_entry, display_mode, mo, np):
    """Statistics panel."""
    mo.stop(display_entry is None)

    _vals = display_entry["values"]
    _shape = display_entry["shape"]
    _min = float(np.min(_vals))
    _max = float(np.max(_vals))
    _mean = float(np.mean(_vals))
    _std = float(np.std(_vals))
    _total = int(np.prod(_shape))

    _mode_label = "Activation" if display_mode == "activations" else "Weight"
    _stats_lines = [
        f"**{_mode_label} Statistics**",
        f"**Min:** {_min:.6f} &nbsp;&nbsp; **Max:** {_max:.6f}",
        f"**Mean:** {_mean:.6f} &nbsp;&nbsp; **Std:** {_std:.6f}",
        f"**Shape:** {_shape} &nbsp;&nbsp; **Total elements:** {_total:,}",
    ]

    if display_mode == "weights" and display_entry.get("is_quantized", False):
        _scale = display_entry.get("scale")
        _zp = display_entry.get("zero_point")
        _scale_str = f"{_scale:.6f}" if _scale is not None else "N/A"
        _zp_str = f"{_zp:.1f}" if _zp is not None else "N/A"
        _stats_lines.append(
            f"**Scale:** {_scale_str} &nbsp;&nbsp; **Zero Point:** {_zp_str}"
        )

    mo.md("\n\n".join(_stats_lines)).callout(kind="neutral")
    return


if __name__ == "__main__":
    app.run()
