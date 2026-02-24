"""Inference adapters for evaluation backends."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Protocol

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
        calibrate: bool = False,
        calibration_images: np.ndarray | None = None,
    ):
        self.model_path = str(model_path)
        _validate_bit_width(weight_bits, "weight_bits")
        _validate_bit_width(activation_bits, "activation_bits")
        if calibrate and activation_bits is not None and calibration_images is None:
            msg = (
                "calibration_images must be provided when "
                "calibrate=True and activation_bits is set"
            )
            raise ValueError(msg)

        base_model = load_pytorch_model(self.model_path)
        self._model = base_model
        self._activation_quantizer: _ActivationQuantizer | None = None

        if weight_bits is not None or activation_bits is not None:
            self._model = copy.deepcopy(base_model)
            if weight_bits is not None:
                _apply_weight_ptq(self._model, weight_bits)
            if activation_bits is not None:
                scales_by_module_id: dict[int, float] | None = None
                if calibrate and calibration_images is not None:
                    scales_by_module_id = _collect_activation_scales(
                        self._model,
                        calibration_images,
                        _ACTIVATION_TARGET_TYPES,
                        bits=activation_bits,
                    )
                self._activation_quantizer = _ActivationQuantizer(
                    self._model,
                    activation_bits,
                    scales_by_module_id=scales_by_module_id,
                )

    def predict_logits(self, images: np.ndarray) -> np.ndarray:
        images_tensor = torch.from_numpy(images)
        with torch.no_grad():
            outputs = self._model(images_tensor)

        if isinstance(outputs, torch.Tensor):
            return outputs.detach().cpu().numpy()

        msg = "PyTorch model output is not a Tensor"
        raise TypeError(msg)


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


def _symmetric_fake_quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if not torch.is_floating_point(tensor):
        return tensor

    max_abs = tensor.detach().abs().max()
    if float(max_abs.item()) == 0.0:
        return tensor

    scale = _symmetric_scale(max_abs=float(max_abs.item()), bits=bits)
    return _symmetric_fake_quantize_tensor_with_scale(tensor, bits, scale)


def _symmetric_scale(*, max_abs: float, bits: int) -> float:
    qmax = (1 << (bits - 1)) - 1
    if qmax <= 0:
        msg = f"Invalid bit-width for symmetric quantization: {bits}"
        raise ValueError(msg)
    return max_abs / float(qmax)


def _symmetric_fake_quantize_tensor_with_scale(
    tensor: torch.Tensor, bits: int, scale: float
) -> torch.Tensor:
    if not torch.is_floating_point(tensor):
        return tensor
    if scale <= 0.0:
        return tensor

    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    quantized = torch.round(tensor / scale).clamp(qmin, qmax)
    return quantized * scale


def _apply_weight_ptq(model: torch.nn.Module, bits: int) -> None:
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.nn.Parameter):
            with torch.no_grad():
                weight.copy_(_symmetric_fake_quantize_tensor(weight, bits))


def _quantize_output(output: object, bits: int) -> object:
    if isinstance(output, torch.Tensor):
        return _symmetric_fake_quantize_tensor(output, bits)
    if isinstance(output, tuple):
        return tuple(_quantize_output(item, bits) for item in output)
    if isinstance(output, list):
        return [_quantize_output(item, bits) for item in output]
    return output


def _max_abs_from_output(output: object) -> float:
    if isinstance(output, torch.Tensor):
        if not torch.is_floating_point(output):
            return 0.0
        if output.numel() == 0:
            return 0.0
        return float(output.detach().abs().max().item())
    if isinstance(output, tuple):
        return max((_max_abs_from_output(item) for item in output), default=0.0)
    if isinstance(output, list):
        return max((_max_abs_from_output(item) for item in output), default=0.0)
    return 0.0


_ACTIVATION_TARGET_TYPES = {
    "Conv2d",
    "Linear",
    "OnnxMatMul",
    "ReLU",
    "AvgPool2d",
    "AdaptiveAvgPool2d",
}


def _collect_activation_scales(
    model: torch.nn.Module,
    calibration_images: np.ndarray,
    target_types: set[str],
    *,
    bits: int,
    batch_size: int = 256,
) -> dict[int, float]:
    max_abs_by_module_id: dict[int, float] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _collector(
        module: torch.nn.Module, _inputs: tuple[object, ...], output: object
    ):
        module_id = id(module)
        observed = _max_abs_from_output(output)
        prev = max_abs_by_module_id.get(module_id, 0.0)
        if observed > prev:
            max_abs_by_module_id[module_id] = observed
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

    scales_by_module_id: dict[int, float] = {}
    for module_id, max_abs in max_abs_by_module_id.items():
        if max_abs <= 0.0:
            continue
        scales_by_module_id[module_id] = _symmetric_scale(max_abs=max_abs, bits=bits)
    return scales_by_module_id


class _ActivationQuantizer:
    def __init__(
        self,
        model: torch.nn.Module,
        bits: int,
        *,
        scales_by_module_id: dict[int, float] | None = None,
    ):
        self._bits = bits
        self._scales_by_module_id = scales_by_module_id or {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for module in model.modules():
            if type(module).__name__ not in _ACTIVATION_TARGET_TYPES:
                continue
            handle = module.register_forward_hook(self._hook)
            self._handles.append(handle)

    def _hook(
        self, module: torch.nn.Module, _inputs: tuple[object, ...], output: object
    ) -> object:
        scale = self._scales_by_module_id.get(id(module))
        if scale is not None:
            return _quantize_output_with_scale(output, self._bits, scale)
        return _quantize_output(output, self._bits)


def _quantize_output_with_scale(output: object, bits: int, scale: float) -> object:
    if isinstance(output, torch.Tensor):
        return _symmetric_fake_quantize_tensor_with_scale(output, bits, scale)
    if isinstance(output, tuple):
        return tuple(_quantize_output_with_scale(item, bits, scale) for item in output)
    if isinstance(output, list):
        return [_quantize_output_with_scale(item, bits, scale) for item in output]
    return output
