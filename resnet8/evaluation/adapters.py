"""Inference adapters for evaluation backends."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import onnxruntime as ort
import torch


class InferenceAdapter(Protocol):
    """Adapter interface for backend-specific inference."""

    backend: str
    model_path: str

    def predict_logits(self, images: np.ndarray) -> np.ndarray:
        """Run inference and return logits with shape [N, C]."""


class OnnxRuntimeAdapter:
    """ONNX Runtime adapter."""

    backend = "onnx"

    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self._session = ort.InferenceSession(self.model_path)
        input_info = self._session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = input_info.shape

    def predict_logits(self, images: np.ndarray) -> np.ndarray:
        outputs = self._session.run(None, {self.input_name: images})
        return np.asarray(outputs[0])


class PyTorchAdapter:
    """PyTorch adapter supporting checkpoint and TorchScript formats."""

    backend = "pytorch"

    def __init__(
        self,
        model_path: str | Path,
        *,
        weight_bits: int | None = None,
        activation_bits: int | None = None,
        activation_scheme: Literal["symmetric", "asymmetric"] = "symmetric",
        calibrate: bool = False,
        calibration_images: np.ndarray | None = None,
    ):
        self.model_path = str(model_path)
        self._weight_bits = weight_bits
        self._activation_bits = activation_bits
        self._activation_scheme = activation_scheme
        self._calibrate = calibrate
        _validate_bit_width(weight_bits, "weight_bits")
        _validate_bit_width(activation_bits, "activation_bits")
        _validate_activation_scheme(activation_scheme)
        if calibrate and activation_bits is not None and calibration_images is None:
            msg = (
                "calibration_images must be provided when "
                "calibrate=True and activation_bits is set"
            )
            raise ValueError(msg)

        base_model = load_pytorch_model(self.model_path)
        self._model = base_model
        self._activation_quantizer: _ActivationQuantizer | None = None
        self._weight_params_by_module_id: dict[int, _QuantizationParams] = {}
        self._activation_params_by_module_id: dict[int, _QuantizationParams] = {}

        if weight_bits is not None or activation_bits is not None:
            self._model = copy.deepcopy(base_model)
            if weight_bits is not None:
                self._weight_params_by_module_id = _apply_weight_ptq(
                    self._model,
                    weight_bits,
                )
            if activation_bits is not None:
                params_by_module_id: dict[int, _QuantizationParams] | None = None
                if calibrate and calibration_images is not None:
                    params_by_module_id = _collect_activation_params(
                        self._model,
                        calibration_images,
                        _ACTIVATION_TARGET_TYPES,
                        bits=activation_bits,
                        scheme=activation_scheme,
                    )
                self._activation_params_by_module_id = params_by_module_id or {}
                self._activation_quantizer = _ActivationQuantizer(
                    self._model,
                    activation_bits,
                    scheme=activation_scheme,
                    params_by_module_id=params_by_module_id,
                )

    def predict_logits(self, images: np.ndarray) -> np.ndarray:
        images_tensor = torch.from_numpy(images)
        with torch.no_grad():
            outputs = self._model(images_tensor)

        if isinstance(outputs, torch.Tensor):
            return outputs.detach().cpu().numpy()

        msg = "PyTorch model output is not a Tensor"
        raise TypeError(msg)

    def describe_quantization(self) -> list[dict[str, object]]:
        if self._weight_bits is None and self._activation_bits is None:
            return []

        rows: list[dict[str, object]] = []
        module_index = 0
        for module_name, module in self._model.named_modules():
            display_name = module_name if module_name else "<root>"
            layer_type = type(module).__name__

            if self._weight_bits is not None:
                weight = getattr(module, "weight", None)
                if isinstance(weight, torch.nn.Parameter):
                    params = self._weight_params_by_module_id.get(id(module))
                    rows.append(
                        {
                            "layer": display_name,
                            "layer_type": layer_type,
                            "tensor": "weight",
                            "bits": self._weight_bits,
                            "scheme": "symmetric",
                            "scale": params.scale if params is not None else None,
                            "zero_point": (
                                params.zero_point if params is not None else None
                            ),
                            "qmin": params.qmin if params is not None else None,
                            "qmax": params.qmax if params is not None else None,
                            "calibrated": False,
                            "order": module_index,
                        }
                    )

            if (
                self._activation_bits is not None
                and layer_type in _ACTIVATION_TARGET_TYPES
            ):
                params = self._activation_params_by_module_id.get(id(module))
                rows.append(
                    {
                        "layer": display_name,
                        "layer_type": layer_type,
                        "tensor": "activation",
                        "bits": self._activation_bits,
                        "scheme": self._activation_scheme,
                        "scale": params.scale if params is not None else None,
                        "zero_point": params.zero_point if params is not None else None,
                        "qmin": params.qmin if params is not None else None,
                        "qmax": params.qmax if params is not None else None,
                        "calibrated": bool(params is not None and self._calibrate),
                        "order": module_index,
                    }
                )

            module_index += 1

        return rows


def load_pytorch_model(model_path: str | Path) -> torch.nn.Module:
    """Load PyTorch model from checkpoint dict or TorchScript."""
    model_path = str(model_path)

    try:
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
        return model
    except RuntimeError:
        pass

    loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
    else:
        model = loaded

    model.eval()
    return model


def _validate_bit_width(value: int | None, label: str) -> None:
    if value is None:
        return
    if not 2 <= value <= 16:
        msg = f"{label} must be in [2, 16], got {value}"
        raise ValueError(msg)


def _validate_activation_scheme(value: str) -> None:
    if value not in {"symmetric", "asymmetric"}:
        msg = f"activation_scheme must be 'symmetric' or 'asymmetric', got {value!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class _QuantizationParams:
    scale: float
    zero_point: int
    qmin: int
    qmax: int


def _symmetric_fake_quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if not torch.is_floating_point(tensor):
        return tensor

    max_abs = tensor.detach().abs().max()
    if float(max_abs.item()) == 0.0:
        return tensor

    params = _symmetric_params_from_max_abs(max_abs=float(max_abs.item()), bits=bits)
    return _fake_quantize_tensor_with_params(tensor, params)


def _symmetric_params_from_max_abs(*, max_abs: float, bits: int) -> _QuantizationParams:
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    if qmax <= 0:
        msg = f"Invalid bit-width for symmetric quantization: {bits}"
        raise ValueError(msg)
    return _QuantizationParams(
        scale=max_abs / float(qmax),
        zero_point=0,
        qmin=qmin,
        qmax=qmax,
    )


def _asymmetric_params_from_min_max(
    *, min_value: float, max_value: float, bits: int
) -> _QuantizationParams:
    qmin = 0
    qmax = (1 << bits) - 1
    if qmax <= qmin:
        msg = f"Invalid bit-width for asymmetric quantization: {bits}"
        raise ValueError(msg)
    scale = (max_value - min_value) / float(qmax - qmin)
    if scale <= 0.0:
        return _QuantizationParams(scale=0.0, zero_point=0, qmin=qmin, qmax=qmax)
    zero_point = int(round(qmin - (min_value / scale)))
    zero_point = max(qmin, min(qmax, zero_point))
    return _QuantizationParams(
        scale=scale,
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
    )


def _fake_quantize_tensor_with_params(
    tensor: torch.Tensor,
    params: _QuantizationParams,
) -> torch.Tensor:
    if not torch.is_floating_point(tensor):
        return tensor
    if params.scale <= 0.0:
        return tensor
    quantized = torch.round(tensor / params.scale + params.zero_point).clamp(
        params.qmin, params.qmax
    )
    return (quantized - params.zero_point) * params.scale


def _apply_weight_ptq(
    model: torch.nn.Module,
    bits: int,
) -> dict[int, _QuantizationParams]:
    params_by_module_id: dict[int, _QuantizationParams] = {}
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.nn.Parameter):
            detached = weight.detach()
            if detached.numel() == 0:
                continue
            max_abs = float(detached.abs().max().item())
            params = _symmetric_params_from_max_abs(max_abs=max_abs, bits=bits)
            params_by_module_id[id(module)] = params
            with torch.no_grad():
                weight.copy_(_fake_quantize_tensor_with_params(weight, params))
    return params_by_module_id


def _quantize_output(output: object, bits: int) -> object:
    if isinstance(output, torch.Tensor):
        return _symmetric_fake_quantize_tensor(output, bits)
    if isinstance(output, tuple):
        return tuple(_quantize_output(item, bits) for item in output)
    if isinstance(output, list):
        return [_quantize_output(item, bits) for item in output]
    return output


def _min_max_from_output(output: object) -> tuple[float, float] | None:
    if isinstance(output, torch.Tensor):
        if not torch.is_floating_point(output):
            return None
        if output.numel() == 0:
            return None
        tensor = output.detach()
        return float(tensor.min().item()), float(tensor.max().item())
    if isinstance(output, tuple):
        values = [_min_max_from_output(item) for item in output]
        values = [value for value in values if value is not None]
        if not values:
            return None
        min_value = min(value[0] for value in values)
        max_value = max(value[1] for value in values)
        return min_value, max_value
    if isinstance(output, list):
        values = [_min_max_from_output(item) for item in output]
        values = [value for value in values if value is not None]
        if not values:
            return None
        min_value = min(value[0] for value in values)
        max_value = max(value[1] for value in values)
        return min_value, max_value
    return None


_ACTIVATION_TARGET_TYPES = {
    "Conv2d",
    "Linear",
    "OnnxMatMul",
    "ReLU",
    "AvgPool2d",
    "AdaptiveAvgPool2d",
}


def _collect_activation_params(
    model: torch.nn.Module,
    calibration_images: np.ndarray,
    target_types: set[str],
    *,
    bits: int,
    scheme: Literal["symmetric", "asymmetric"],
    batch_size: int = 256,
) -> dict[int, _QuantizationParams]:
    min_max_by_module_id: dict[int, tuple[float, float]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _collector(
        module: torch.nn.Module, _inputs: tuple[object, ...], output: object
    ):
        module_id = id(module)
        observed = _min_max_from_output(output)
        if observed is None:
            return output
        prev = min_max_by_module_id.get(module_id)
        if prev is None:
            min_max_by_module_id[module_id] = observed
            return output
        min_max_by_module_id[module_id] = (
            min(prev[0], observed[0]),
            max(prev[1], observed[1]),
        )
        return output

    for module in model.modules():
        if type(module).__name__ not in target_types:
            continue
        handles.append(module.register_forward_hook(_collector))

    try:
        with torch.no_grad():
            for start in range(0, calibration_images.shape[0], batch_size):
                stop = min(start + batch_size, calibration_images.shape[0])
                batch = torch.from_numpy(calibration_images[start:stop])
                model(batch)
    finally:
        for handle in handles:
            handle.remove()

    params_by_module_id: dict[int, _QuantizationParams] = {}
    for module_id, (min_value, max_value) in min_max_by_module_id.items():
        if max_value <= min_value:
            continue
        if scheme == "asymmetric":
            params_by_module_id[module_id] = _asymmetric_params_from_min_max(
                min_value=min_value,
                max_value=max_value,
                bits=bits,
            )
        else:
            max_abs = max(abs(min_value), abs(max_value))
            params_by_module_id[module_id] = _symmetric_params_from_max_abs(
                max_abs=max_abs,
                bits=bits,
            )
    return params_by_module_id


class _ActivationQuantizer:
    def __init__(
        self,
        model: torch.nn.Module,
        bits: int,
        *,
        scheme: Literal["symmetric", "asymmetric"] = "symmetric",
        params_by_module_id: dict[int, _QuantizationParams] | None = None,
    ):
        self._bits = bits
        self._scheme = scheme
        self._params_by_module_id = params_by_module_id or {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for module in model.modules():
            if type(module).__name__ not in _ACTIVATION_TARGET_TYPES:
                continue
            handle = module.register_forward_hook(self._hook)
            self._handles.append(handle)

    def _hook(
        self, module: torch.nn.Module, _inputs: tuple[object, ...], output: object
    ) -> object:
        params = self._params_by_module_id.get(id(module))
        if params is not None:
            return _quantize_output_with_params(output, params)
        return _quantize_output_with_scheme(output, self._bits, self._scheme)


def _quantize_output_with_params(output: object, params: _QuantizationParams) -> object:
    if isinstance(output, torch.Tensor):
        return _fake_quantize_tensor_with_params(output, params)
    if isinstance(output, tuple):
        return tuple(_quantize_output_with_params(item, params) for item in output)
    if isinstance(output, list):
        return [_quantize_output_with_params(item, params) for item in output]
    return output


def _quantize_output_with_scheme(
    output: object, bits: int, scheme: Literal["symmetric", "asymmetric"]
) -> object:
    if isinstance(output, torch.Tensor):
        if not torch.is_floating_point(output):
            return output
        if output.numel() == 0:
            return output
        if scheme == "asymmetric":
            min_value = float(output.detach().min().item())
            max_value = float(output.detach().max().item())
            if max_value <= min_value:
                return output
            params = _asymmetric_params_from_min_max(
                min_value=min_value,
                max_value=max_value,
                bits=bits,
            )
            return _fake_quantize_tensor_with_params(output, params)
        return _symmetric_fake_quantize_tensor(output, bits)
    if isinstance(output, tuple):
        return tuple(
            _quantize_output_with_scheme(item, bits, scheme) for item in output
        )
    if isinstance(output, list):
        return [_quantize_output_with_scheme(item, bits, scheme) for item in output]
    return output
