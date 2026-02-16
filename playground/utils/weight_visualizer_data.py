"""Utilities for extracting tensor-level data for the weight visualizer notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnx
from onnx import numpy_helper as nh
import torch

from .model_loader import load_onnx_model, load_pytorch_model


_SUPPORTED_QUANTIZED_OPS = {"QLinearConv", "QLinearMatMul", "QuantizeLinear"}


def collect_model_files(models_dir: Path | str) -> Dict[str, str]:
    """Return a name-to-path map for available model files."""
    models_path = Path(models_dir)
    model_files: Dict[str, str] = {}

    if not models_path.exists():
        return model_files

    for model_file in sorted(models_path.iterdir()):
        if model_file.suffix in (".onnx", ".pt"):
            model_files[model_file.name] = str(model_file)

    return model_files


def load_model_data(model_path: Path | str) -> Dict[str, Any]:
    """Load either ONNX or PyTorch model and normalize to notebook format."""
    path = Path(model_path)
    if path.suffix == ".onnx":
        return _load_onnx_model_data(path)
    if path.suffix == ".pt":
        return _load_pytorch_model_data(path)
    raise ValueError("Unsupported model format. Use .onnx or .pt")


def _load_onnx_model_data(model_path: Path) -> Dict[str, Any]:
    """Extract tensor metadata from an ONNX model."""
    model = load_onnx_model(model_path)

    initializers = {
        init.name: init for init in model.graph.initializer if init.name
    }
    node_op_types = {node.op_type for node in model.graph.node}
    is_quantized = bool(node_op_types & _SUPPORTED_QUANTIZED_OPS)

    layers: Dict[str, Dict[str, str]] = {}
    for init_name in initializers:
        if init_name.endswith("_scale") or init_name.endswith("_zero_point"):
            continue

        layer_name, tensor_type = _split_tensor_name(init_name)
        layers.setdefault(layer_name, {})[tensor_type] = init_name

    tensor_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for layer_name, layer_tensors in layers.items():
        tensor_data[layer_name] = {}
        for tensor_type, init_name in layer_tensors.items():
            init_tensor = initializers[init_name]
            array = nh.to_array(init_tensor)
            values = array.flatten().astype(np.float64)
            layer_entry: Dict[str, Any] = {
                "values": values,
                "shape": array.shape,
            }

            if array.dtype in (np.int8, np.uint8) and is_quantized:
                int_values = values
                scale, zero_point = _load_onnx_quantization_metadata(
                    initializers, init_name
                )
                layer_entry.update(
                    {
                        "is_quantized": True,
                        "int_values": int_values,
                        "scale": scale,
                        "zero_point": zero_point,
                    }
                )
                if scale is not None:
                    layer_entry["values"] = scale * (int_values - zero_point)
            else:
                layer_entry["is_quantized"] = False

            tensor_data[layer_name][tensor_type] = layer_entry

    return {
        "format": "onnx",
        "layers": layers,
        "is_quantized": is_quantized,
        "tensor_data": tensor_data,
    }


def _load_onnx_quantization_metadata(
    initializers: Dict[str, onnx.TensorProto],
    tensor_name: str,
) -> tuple[float | None, float]:
    """Load int scale/zero-point metadata for a tensor if present."""
    direct_scale_name = f"{tensor_name}_scale"
    direct_zero_point_name = f"{tensor_name}_zero_point"

    scale_name = direct_scale_name
    zero_point_name = direct_zero_point_name
    if scale_name not in initializers and zero_point_name not in initializers:
        base_name = tensor_name.removesuffix("_quantized")
        scale_name = f"{base_name}_scale"
        zero_point_name = f"{base_name}_zero_point"

    scale = None
    zero_point = 0.0

    if scale_name in initializers:
        scale = float(nh.to_array(initializers[scale_name]).flat[0])
    if zero_point_name in initializers:
        zero_point = float(nh.to_array(initializers[zero_point_name]).flat[0])

    return scale, zero_point


def _split_tensor_name(init_name: str) -> tuple[str, str]:
    """Infer tensor type and layer name from initializer naming convention."""
    base_name = init_name.removesuffix("_quantized")
    lower_name = base_name.lower()

    if "bias" in lower_name or base_name.endswith(".bias"):
        layer_name = base_name.replace(".bias", "").replace("/bias", "")
        return layer_name, "bias"
    if "weight" in lower_name or base_name.endswith(".weight"):
        layer_name = base_name.replace(".weight", "").replace("/weight", "")
        return layer_name, "weight"
    return base_name, "weight"


def _load_pytorch_model_data(model_path: Path) -> Dict[str, Any]:
    """Extract tensor metadata from a PyTorch model."""
    model = load_pytorch_model(model_path)

    layers: Dict[str, Dict[str, str]] = {}
    tensor_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    is_quantized = False

    packed_modules: Dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        try:
            if module._c.hasattr("_packed_params"):
                packed_modules[name] = module
        except (AttributeError, RuntimeError):
            continue

    if packed_modules:
        is_quantized = True
        for name, module in packed_modules.items():
            packed_params = module._c.__getattr__("_packed_params")
            weight, bias = None, None
            for unpack_fn in (
                torch.ops.quantized.conv2d_unpack,
                torch.ops.quantized.linear_unpack,
            ):
                try:
                    weight, bias = unpack_fn(packed_params)
                    break
                except (RuntimeError, TypeError):
                    continue
            if weight is None:
                continue

            layers[name] = {"weight": f"{name}.weight"}
            tensor_data[name] = {}

            if hasattr(weight, "int_repr"):
                int_values = weight.int_repr().cpu().numpy().astype(np.float64)
                dequant_values = weight.dequantize().cpu().numpy().flatten().astype(
                    np.float64
                )
                tensor_data[name]["weight"] = {
                    "values": dequant_values,
                    "shape": tuple(int_values.shape),
                    "is_quantized": True,
                    "int_values": int_values,
                    "scale": float(weight.q_per_channel_scales().mean()),
                    "zero_point": float(weight.q_per_channel_zero_points().float().mean()),
                }
            else:
                values = weight.detach().cpu().numpy().flatten().astype(np.float64)
                tensor_data[name]["weight"] = {
                    "values": values,
                    "shape": weight.shape,
                    "is_quantized": False,
                }

            if bias is not None:
                layers[name]["bias"] = f"{name}.bias"
                bias_values = bias.detach().cpu().numpy().flatten().astype(np.float64)
                tensor_data[name]["bias"] = {
                    "values": bias_values,
                    "shape": bias.shape,
                    "is_quantized": False,
                }

    # Include standard FP32 params for completeness.
    state_dict = model.state_dict()
    for param_name, tensor in state_dict.items():
        if "." not in param_name:
            continue
        layer_name, tensor_type = param_name.rsplit(".", 1)
        if tensor_type not in ("weight", "bias"):
            continue
        if layer_name in tensor_data and tensor_type in tensor_data[layer_name]:
            continue

        layers.setdefault(layer_name, {})[tensor_type] = param_name
        array = tensor.detach().cpu().numpy().astype(np.float64).flatten()
        tensor_data.setdefault(layer_name, {})[tensor_type] = {
            "values": array,
            "shape": tensor.shape,
            "is_quantized": False,
        }

    return {
        "format": "pytorch",
        "layers": layers,
        "is_quantized": is_quantized,
        "tensor_data": tensor_data,
    }

