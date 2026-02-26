"""Inference adapters for evaluation backends."""

from __future__ import annotations

import copy
import warnings
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
        if self._uses_static_batch_one() and images.shape[0] != 1:
            outputs: list[np.ndarray] = []
            for index in range(images.shape[0]):
                sample = images[index : index + 1]
                sample_outputs = self._session.run(None, {self.input_name: sample})
                outputs.append(np.asarray(sample_outputs[0]))
            return np.concatenate(outputs, axis=0)

        outputs = self._session.run(None, {self.input_name: images})
        return np.asarray(outputs[0])

    def _uses_static_batch_one(self) -> bool:
        if not self.input_shape:
            return False
        batch_dim = self.input_shape[0]
        return isinstance(batch_dim, int) and batch_dim == 1


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
        device: Literal["auto", "cpu", "cuda"] = "auto",
        per_channel: bool = False,
        weight_bias_correction: bool = False,
    ):
        self.model_path = str(model_path)
        self._weight_bits = weight_bits
        self._activation_bits = activation_bits
        self._activation_scheme = activation_scheme
        self._calibrate = calibrate
        self._per_channel = per_channel
        self._weight_bias_correction = weight_bias_correction
        self.device = _resolve_torch_device(device)
        _validate_bit_width(weight_bits, "weight_bits")
        _validate_bit_width(activation_bits, "activation_bits")
        _validate_activation_scheme(activation_scheme)
        if calibrate and activation_bits is not None and calibration_images is None:
            msg = (
                "calibration_images must be provided when "
                "calibrate=True and activation_bits is set"
            )
            raise ValueError(msg)
        if weight_bias_correction and weight_bits is None:
            msg = "weight_bias_correction=True requires weight_bits to be set"
            raise ValueError(msg)
        if weight_bias_correction and calibration_images is None:
            msg = "calibration_images must be provided when weight_bias_correction=True"
            raise ValueError(msg)

        base_model = load_pytorch_model(self.model_path)
        self._model = base_model.to(self.device)
        self._activation_quantizer: _ActivationQuantizer | None = None
        self._weight_params_by_module_id: dict[int, _WeightQuantizationRecord] = {}
        self._activation_params_by_module_id: dict[int, _QuantizationParams] = {}
        self._bias_corrected_module_ids: set[int] = set()

        if weight_bits is not None or activation_bits is not None:
            self._model = copy.deepcopy(base_model)
            if weight_bits is not None:
                self._weight_params_by_module_id = _apply_weight_ptq(
                    self._model,
                    weight_bits,
                    per_channel=per_channel,
                )
                if weight_bias_correction and calibration_images is not None:
                    self._bias_corrected_module_ids = (
                        _apply_empirical_weight_bias_correction(
                            fp32_model=base_model,
                            quantized_model=self._model,
                            calibration_images=calibration_images,
                            device=self.device,
                        )
                    )
            if activation_bits is not None:
                params_by_module_id: dict[int, _QuantizationParams] | None = None
                if calibrate and calibration_images is not None:
                    params_by_module_id = _collect_activation_params(
                        self._model,
                        calibration_images,
                        _ACTIVATION_TARGET_TYPES,
                        device=self.device,
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
        images_tensor = torch.from_numpy(images).to(self.device)
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
                    scheme = (
                        params.scheme
                        if params is not None
                        else (
                            "symmetric-per-channel"
                            if self._per_channel
                            else "symmetric-per-tensor"
                        )
                    )
                    if id(module) in self._bias_corrected_module_ids:
                        scheme = f"{scheme}+dfq-bias-corr"
                    rows.append(
                        {
                            "layer": display_name,
                            "layer_type": layer_type,
                            "tensor": "weight",
                            "bits": self._weight_bits,
                            "scheme": scheme,
                            "scale": params.scale if params is not None else None,
                            "zero_point": (
                                params.zero_point if params is not None else None
                            ),
                            "qmin": params.qmin if params is not None else None,
                            "qmax": params.qmax if params is not None else None,
                            "calibrated": id(module) in self._bias_corrected_module_ids,
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

    def export_onnx(
        self,
        output_path: str | Path,
        *,
        opset_version: int = 17,
        input_shape: tuple[int, int, int, int] = (1, 32, 32, 3),
        dynamic_batch: bool = False,
        simplify: bool = True,
        constant_propagation: bool = True,
    ) -> str:
        """Export the current eval model (including PTQ behavior) to ONNX."""
        if len(input_shape) != 4:
            msg = f"input_shape must be rank-4 NHWC, got {input_shape}"
            raise ValueError(msg)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model = self._model
        was_training = model.training
        model.eval()
        first_param = next(model.parameters(), None)
        model_device = first_param.device if first_param is not None else torch.device(
            self.device
        )
        example_input = torch.zeros(
            input_shape,
            dtype=torch.float32,
            device=model_device,
        )
        dynamic_axes = (
            {"input": {0: "batch"}, "logits": {0: "batch"}}
            if dynamic_batch
            else None
        )
        export_model: torch.nn.Module | torch.jit.ScriptModule = model
        if constant_propagation:
            export_model = _prepare_model_for_constant_propagation(
                model=model,
                example_input=example_input,
            )

        try:
            with torch.no_grad():
                torch.onnx.export(
                    export_model,
                    example_input,
                    str(out_path),
                    export_params=True,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["logits"],
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                    dynamo=False,
                )
        finally:
            if was_training:
                model.train()

        if simplify:
            _try_simplify_onnx_model(out_path)

        return str(out_path)


def _try_simplify_onnx_model(path: Path) -> None:
    try:
        import onnx
        from onnxsim import simplify
    except Exception:
        warnings.warn(
            "onnxsim is unavailable; skipping ONNX simplification",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    try:
        model = onnx.load(str(path))
        simplified_model, ok = simplify(model)
        if not ok:
            warnings.warn(
                "onnxsim simplify check failed; keeping original ONNX export",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        onnx.save(simplified_model, str(path))
    except Exception as exc:
        warnings.warn(
            f"ONNX simplification failed; keeping original export ({exc})",
            RuntimeWarning,
            stacklevel=2,
        )


def _prepare_model_for_constant_propagation(
    *,
    model: torch.nn.Module,
    example_input: torch.Tensor,
) -> torch.nn.Module | torch.jit.ScriptModule:
    try:
        with torch.no_grad():
            traced = torch.jit.trace(model, example_input, strict=False)
            frozen = torch.jit.freeze(traced.eval())
            torch._C._jit_pass_inline(frozen.graph)
            torch._C._jit_pass_constant_propagation(frozen.graph)
            return frozen
    except Exception as exc:
        warnings.warn(
            "Constant propagation before ONNX export failed; "
            f"falling back to direct export ({exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return model


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


def _resolve_torch_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            msg = "CUDA device requested but torch.cuda.is_available() is false"
            raise ValueError(msg)
        return "cuda"
    msg = f"device must be 'auto', 'cpu', or 'cuda', got {requested!r}"
    raise ValueError(msg)


@dataclass(frozen=True)
class _QuantizationParams:
    scale: float
    zero_point: int
    qmin: int
    qmax: int


@dataclass(frozen=True)
class _WeightQuantizationRecord:
    scale: float | list[float]
    zero_point: int | list[int]
    qmin: int
    qmax: int
    scheme: Literal["symmetric-per-tensor", "symmetric-per-channel"]


def _symmetric_fake_quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if not torch.is_floating_point(tensor):
        return tensor

    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    max_abs = tensor.detach().abs().amax()
    scale = max_abs / float(qmax)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(tensor / safe_scale).clamp(qmin, qmax)
    dequantized = quantized * safe_scale
    return torch.where(scale > 0, dequantized, tensor)


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
    *,
    per_channel: bool = False,
) -> dict[int, _WeightQuantizationRecord]:
    params_by_module_id: dict[int, _WeightQuantizationRecord] = {}
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.nn.Parameter):
            detached = weight.detach()
            if detached.numel() == 0:
                continue
            with torch.no_grad():
                if per_channel and detached.dim() >= 2:
                    quantized, scales, qmin, qmax = (
                        _symmetric_per_channel_fake_quantize(
                            weight,
                            bits=bits,
                        )
                    )
                    params_by_module_id[id(module)] = _WeightQuantizationRecord(
                        scale=scales,
                        zero_point=[0] * len(scales),
                        qmin=qmin,
                        qmax=qmax,
                        scheme="symmetric-per-channel",
                    )
                    weight.copy_(quantized)
                else:
                    max_abs = float(detached.abs().max().item())
                    params = _symmetric_params_from_max_abs(max_abs=max_abs, bits=bits)
                    params_by_module_id[id(module)] = _WeightQuantizationRecord(
                        scale=params.scale,
                        zero_point=params.zero_point,
                        qmin=params.qmin,
                        qmax=params.qmax,
                        scheme="symmetric-per-tensor",
                    )
                    weight.copy_(_fake_quantize_tensor_with_params(weight, params))
    return params_by_module_id


def _apply_empirical_weight_bias_correction(
    *,
    fp32_model: torch.nn.Module,
    quantized_model: torch.nn.Module,
    calibration_images: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> set[int]:
    fp32_modules = _named_weight_modules(fp32_model)
    quantized_modules = _named_weight_modules(quantized_model)

    candidate_names: list[str] = []
    for name, quantized_module in quantized_modules.items():
        if name not in fp32_modules:
            continue
        bias = getattr(quantized_module, "bias", None)
        if not isinstance(bias, torch.nn.Parameter):
            continue
        if bias.numel() == 0:
            continue
        candidate_names.append(name)

    if not candidate_names:
        return set()

    fp32_means = _collect_module_channel_means(
        fp32_model,
        set(candidate_names),
        calibration_images,
        device=device,
        batch_size=batch_size,
    )

    corrected_module_ids: set[int] = set()
    with torch.no_grad():
        for name in candidate_names:
            fp32_mean = fp32_means.get(name)
            quantized_mean = _collect_module_channel_means(
                quantized_model,
                {name},
                calibration_images,
                device=device,
                batch_size=batch_size,
            ).get(name)
            if fp32_mean is None or quantized_mean is None:
                continue

            module = quantized_modules[name]
            bias = getattr(module, "bias", None)
            if not isinstance(bias, torch.nn.Parameter):
                continue
            if (
                fp32_mean.numel() != bias.numel()
                or quantized_mean.numel() != bias.numel()
            ):
                continue

            mean_error = quantized_mean - fp32_mean
            bias.sub_(mean_error.to(device=bias.device, dtype=bias.dtype))
            corrected_module_ids.add(id(module))

    return corrected_module_ids


def _named_weight_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    modules: dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.nn.Parameter):
            modules[name] = module
    return modules


def _collect_module_channel_means(
    model: torch.nn.Module,
    module_names: set[str],
    calibration_images: np.ndarray,
    *,
    device: str,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    if not module_names:
        return {}

    output_sums: dict[str, torch.Tensor] = {}
    output_counts: dict[str, int] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    named_modules = dict(model.named_modules())

    def _make_collector(module_name: str):
        def _collector(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: object,
        ) -> object:
            tensor_output = _first_tensor_output(output)
            if tensor_output is None:
                return output
            if not torch.is_floating_point(tensor_output):
                return output
            if tensor_output.dim() < 2 or tensor_output.numel() == 0:
                return output

            reduce_dims = tuple(dim for dim in range(tensor_output.dim()) if dim != 1)
            channel_sum = (
                tensor_output.detach().sum(dim=reduce_dims).to(torch.float64).cpu()
            )
            channel_count = tensor_output.numel() // tensor_output.shape[1]

            if module_name in output_sums:
                output_sums[module_name] += channel_sum
            else:
                output_sums[module_name] = channel_sum
            output_counts[module_name] = (
                output_counts.get(module_name, 0) + channel_count
            )
            return output

        return _collector

    for module_name in module_names:
        module = named_modules.get(module_name)
        if module is None:
            continue
        handles.append(module.register_forward_hook(_make_collector(module_name)))

    try:
        with torch.no_grad():
            for start in range(0, calibration_images.shape[0], batch_size):
                stop = min(start + batch_size, calibration_images.shape[0])
                batch = torch.from_numpy(calibration_images[start:stop]).to(device)
                model(batch)
    finally:
        for handle in handles:
            handle.remove()

    means: dict[str, torch.Tensor] = {}
    for module_name, output_sum in output_sums.items():
        count = output_counts.get(module_name, 0)
        if count <= 0:
            continue
        means[module_name] = output_sum / float(count)
    return means


def _first_tensor_output(output: object) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        for item in output:
            tensor = _first_tensor_output(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(output, list):
        for item in output:
            tensor = _first_tensor_output(item)
            if tensor is not None:
                return tensor
        return None
    return None


def _symmetric_per_channel_fake_quantize(
    tensor: torch.Tensor,
    *,
    bits: int,
    axis: int = 0,
) -> tuple[torch.Tensor, list[float], int, int]:
    if not torch.is_floating_point(tensor):
        return tensor, [], 0, 0
    if tensor.numel() == 0:
        return tensor, [], 0, 0

    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    moved = tensor.movedim(axis, 0)
    reshaped = moved.detach().abs().reshape(moved.shape[0], -1)
    max_abs = reshaped.max(dim=1).values
    scales = max_abs / float(qmax)
    scale_shape = [moved.shape[0]] + [1] * (moved.dim() - 1)
    scale_broadcast = scales.reshape(scale_shape)
    safe_scale = torch.where(
        scale_broadcast > 0,
        scale_broadcast,
        torch.ones_like(scale_broadcast),
    )
    quantized = torch.round(moved / safe_scale).clamp(qmin, qmax) * safe_scale
    quantized = torch.where(scale_broadcast > 0, quantized, moved)
    return quantized.movedim(0, axis), scales.tolist(), qmin, qmax


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
    device: str,
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
                batch = torch.from_numpy(calibration_images[start:stop]).to(device)
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
            qmin = 0
            qmax = (1 << bits) - 1
            min_value = output.detach().amin()
            max_value = output.detach().amax()
            value_range = max_value - min_value
            scale = value_range / float(qmax - qmin)
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            zero_point = torch.round(
                torch.tensor(
                    float(qmin),
                    device=output.device,
                    dtype=output.dtype,
                )
                - (min_value / safe_scale)
            )
            zero_point = zero_point.clamp(float(qmin), float(qmax))
            quantized = torch.round(output / safe_scale + zero_point).clamp(qmin, qmax)
            dequantized = (quantized - zero_point) * safe_scale
            return torch.where(scale > 0, dequantized, output)
        return _symmetric_fake_quantize_tensor(output, bits)
    if isinstance(output, tuple):
        return tuple(
            _quantize_output_with_scheme(item, bits, scheme) for item in output
        )
    if isinstance(output, list):
        return [_quantize_output_with_scheme(item, bits, scheme) for item in output]
    return output
