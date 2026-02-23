"""Reusable tensor extraction utilities for ONNX and PyTorch models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper as onnx_numpy_helper

from playground.utils.tensor_schema import ModelTensorIndex, TensorMetadata

_QUANT_OP_TYPES = {"QLinearConv", "QLinearMatMul", "QuantizeLinear"}


def _empty_index() -> ModelTensorIndex:
    return {
        "format": "none",
        "layers": {},
        "is_quantized": False,
        "tensor_data": {},
    }


def _infer_layer_and_tensor_type(name: str) -> tuple[str, str]:
    if name.endswith("_quantized"):
        base_name = name.rsplit("_quantized", 1)[0]
    else:
        base_name = name

    if "bias" in name.lower() or name.endswith(".bias"):
        layer_name = base_name.replace(".bias", "").replace("/bias", "")
        return layer_name, "bias"

    if "weight" in name.lower() or name.endswith(".weight"):
        layer_name = base_name.replace(".weight", "").replace("/weight", "")
        return layer_name, "weight"

    return base_name, "weight"


def _as_float_vector(values: np.ndarray) -> np.ndarray:
    return values.flatten().astype(np.float64, copy=False)


def _quant_param_names(initializer_name: str) -> tuple[str, str]:
    return initializer_name + "_scale", initializer_name + "_zero_point"


def _lookup_quant_params(
    initializers: dict[str, onnx.TensorProto], initializer_name: str
) -> tuple[np.ndarray | None, np.ndarray | None]:
    scale_name, zero_point_name = _quant_param_names(initializer_name)

    if scale_name not in initializers and initializer_name.endswith("_quantized"):
        base_name = initializer_name.replace("_quantized", "")
        scale_name = base_name + "_scale"
        zero_point_name = base_name + "_zero_point"

    scale_values: np.ndarray | None = None
    zero_point_values: np.ndarray | None = None

    if scale_name in initializers:
        scale_values = onnx_numpy_helper.to_array(initializers[scale_name]).astype(
            np.float64
        )
    if zero_point_name in initializers:
        zero_point_values = onnx_numpy_helper.to_array(initializers[zero_point_name])
        zero_point_values = zero_point_values.astype(np.float64)

    return scale_values, zero_point_values


def extract_onnx_tensor_index(path: str | Path) -> ModelTensorIndex:
    """Load ONNX model and normalize tensor metadata for visualization."""
    model = onnx.load(str(path))
    initializers = {init.name: init for init in model.graph.initializer if init.name}
    node_op_types = {node.op_type for node in model.graph.node}
    is_quantized_model = bool(node_op_types & _QUANT_OP_TYPES)

    layers: dict[str, dict[str, str]] = {}
    tensor_data: dict[str, dict[str, TensorMetadata]] = {}

    for initializer_name, initializer in initializers.items():
        if initializer_name.endswith("_scale") or initializer_name.endswith(
            "_zero_point"
        ):
            continue

        layer_name, tensor_type = _infer_layer_and_tensor_type(initializer_name)
        layers.setdefault(layer_name, {})[tensor_type] = initializer_name
        tensor_data.setdefault(layer_name, {})

        raw_values = onnx_numpy_helper.to_array(initializer)
        entry: TensorMetadata = {
            "values": _as_float_vector(raw_values),
            "shape": tuple(raw_values.shape),
            "is_quantized": False,
        }

        if raw_values.dtype in (np.int8, np.uint8) and is_quantized_model:
            int_values = _as_float_vector(raw_values)
            scale_values, zero_point_values = _lookup_quant_params(
                initializers, initializer_name
            )

            scale_scalar = (
                float(scale_values.reshape(-1)[0]) if scale_values is not None else None
            )
            zero_point_scalar = (
                float(zero_point_values.reshape(-1)[0])
                if zero_point_values is not None
                else 0.0
            )

            entry.update(
                {
                    "is_quantized": True,
                    "int_values": int_values,
                    "scale": scale_scalar,
                    "zero_point": zero_point_scalar,
                    "scale_values": (
                        _as_float_vector(scale_values)
                        if scale_values is not None
                        else None
                    ),
                    "zero_point_values": (
                        _as_float_vector(zero_point_values)
                        if zero_point_values is not None
                        else None
                    ),
                }
            )

            if scale_scalar is not None:
                entry["values"] = scale_scalar * (int_values - zero_point_scalar)

        tensor_data[layer_name][tensor_type] = entry

    return {
        "format": "onnx",
        "layers": layers,
        "is_quantized": is_quantized_model,
        "tensor_data": tensor_data,
    }


def _load_pytorch_model_for_inspection(path: str | Path) -> torch.nn.Module:
    model_path = str(path)
    try:
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
        return model
    except RuntimeError:
        loaded = torch.load(model_path, weights_only=False, map_location="cpu")

    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
    else:
        model = loaded

    model.eval()
    return model


def _quant_params_from_torch_tensor(
    tensor: torch.Tensor,
) -> tuple[float | None, float | None, np.ndarray | None, np.ndarray | None]:
    if not tensor.is_quantized:
        return None, None, None, None

    qscheme = tensor.qscheme()
    if qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric):
        scale_scalar = float(tensor.q_scale())
        zero_point_scalar = float(tensor.q_zero_point())
        scale_values = np.asarray([scale_scalar], dtype=np.float64)
        zero_point_values = np.asarray([zero_point_scalar], dtype=np.float64)
        return scale_scalar, zero_point_scalar, scale_values, zero_point_values

    scale_values = tensor.q_per_channel_scales().cpu().numpy().astype(np.float64)
    zero_point_values = (
        tensor.q_per_channel_zero_points().cpu().numpy().astype(np.float64)
    )
    scale_scalar = float(np.mean(scale_values))
    zero_point_scalar = float(np.mean(zero_point_values))
    return scale_scalar, zero_point_scalar, scale_values, zero_point_values


def _torch_tensor_entry(tensor: torch.Tensor) -> TensorMetadata:
    if tensor.is_quantized:
        int_values = tensor.int_repr().cpu().numpy()
        values = tensor.dequantize().cpu().numpy()
        scale_scalar, zero_point_scalar, scale_values, zero_point_values = (
            _quant_params_from_torch_tensor(tensor)
        )
        return {
            "values": _as_float_vector(values),
            "shape": tuple(int_values.shape),
            "is_quantized": True,
            "int_values": _as_float_vector(int_values),
            "scale": scale_scalar,
            "zero_point": zero_point_scalar,
            "scale_values": (
                _as_float_vector(scale_values) if scale_values is not None else None
            ),
            "zero_point_values": (
                _as_float_vector(zero_point_values)
                if zero_point_values is not None
                else None
            ),
        }

    values = tensor.detach().cpu().numpy()
    return {
        "values": _as_float_vector(values),
        "shape": tuple(values.shape),
        "is_quantized": False,
    }


def extract_pytorch_tensor_index(path: str | Path) -> ModelTensorIndex:
    """Load PyTorch model and normalize tensor metadata for visualization."""
    model = _load_pytorch_model_for_inspection(path)

    layers: dict[str, dict[str, str]] = {}
    tensor_data: dict[str, dict[str, TensorMetadata]] = {}
    is_quantized_model = False

    packed_modules: dict[str, torch.nn.Module] = {}
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        try:
            if module._c.hasattr("_packed_params"):
                packed_modules[module_name] = module
        except (AttributeError, RuntimeError):
            continue

    if packed_modules:
        is_quantized_model = True
        for module_name, module in packed_modules.items():
            packed = module._c.__getattr__("_packed_params")
            weight_tensor = None
            bias_tensor = None
            for unpack_fn in (
                torch.ops.quantized.conv2d_unpack,
                torch.ops.quantized.linear_unpack,
            ):
                try:
                    weight_tensor, bias_tensor = unpack_fn(packed)
                    break
                except (RuntimeError, TypeError):
                    continue

            if weight_tensor is None:
                continue

            layers.setdefault(module_name, {})["weight"] = module_name + ".weight"
            tensor_data.setdefault(module_name, {})
            tensor_data[module_name]["weight"] = _torch_tensor_entry(weight_tensor)

            if bias_tensor is not None:
                layers[module_name]["bias"] = module_name + ".bias"
                tensor_data[module_name]["bias"] = {
                    "values": _as_float_vector(bias_tensor.detach().cpu().numpy()),
                    "shape": tuple(bias_tensor.shape),
                    "is_quantized": False,
                }

    for param_name, tensor in model.state_dict().items():
        parts = param_name.rsplit(".", 1)
        if len(parts) != 2:
            continue
        layer_name, tensor_type = parts
        if tensor_type not in {"weight", "bias"}:
            continue

        if layer_name in tensor_data and tensor_type in tensor_data[layer_name]:
            continue

        layers.setdefault(layer_name, {})[tensor_type] = param_name
        tensor_data.setdefault(layer_name, {})

        if tensor.is_quantized:
            is_quantized_model = True
        tensor_data[layer_name][tensor_type] = _torch_tensor_entry(tensor)

    return {
        "format": "pytorch",
        "layers": layers,
        "is_quantized": is_quantized_model,
        "tensor_data": tensor_data,
    }


def load_tensor_index(path: str | Path | None) -> ModelTensorIndex:
    """Load tensor index for a supported model file."""
    if path is None:
        return _empty_index()

    suffix = Path(path).suffix.lower()
    if suffix == ".onnx":
        return extract_onnx_tensor_index(path)
    if suffix == ".pt":
        return extract_pytorch_tensor_index(path)
    return _empty_index()
