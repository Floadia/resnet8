from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from resnet8.evaluation.adapters import (
    _ACTIVATION_TARGET_TYPES,
    OnnxRuntimeAdapter,
    PyTorchAdapter,
    _ActivationQuantizer,
    _apply_empirical_weight_bias_correction,
    _apply_weight_ptq,
    _asymmetric_params_from_min_max,
    _collect_activation_params,
    _fake_quantize_tensor_with_params,
    _resolve_torch_device,
    _symmetric_fake_quantize_tensor,
)


def test_asymmetric_quantization_uses_zero_point():
    tensor = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float32)
    params = _asymmetric_params_from_min_max(min_value=-1.0, max_value=1.0, bits=2)
    quantized = _fake_quantize_tensor_with_params(tensor, params)

    assert params.zero_point == 2
    assert torch.allclose(
        quantized,
        torch.tensor([[-1.3333, 0.0, 0.6667]], dtype=torch.float32),
        atol=1e-4,
    )


def test_collect_asymmetric_activation_params_from_calibration_data():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.ReLU(),
    )
    with torch.no_grad():
        linear = model[0]
        assert isinstance(linear, torch.nn.Linear)
        linear.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float32))

    calibration_images = np.array(
        [[1.0, 2.0], [-1.0, -2.0], [0.5, 1.5]],
        dtype=np.float32,
    )
    params = _collect_activation_params(
        model,
        calibration_images,
        _ACTIVATION_TARGET_TYPES,
        device="cpu",
        bits=2,
        scheme="asymmetric",
        batch_size=1,
    )

    assert len(params) >= 2
    assert any(value.zero_point > 0 for value in params.values())


def test_activation_quantizer_uses_calibrated_fixed_params():
    linear = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        linear.weight.fill_(1.0)
    model = torch.nn.Sequential(linear)
    params = _asymmetric_params_from_min_max(min_value=-1.0, max_value=1.0, bits=2)

    _ActivationQuantizer(
        model,
        bits=2,
        scheme="asymmetric",
        params_by_module_id={id(linear): params},
    )
    outputs = model(torch.tensor([[0.2], [0.7]], dtype=torch.float32))

    assert torch.allclose(
        outputs,
        torch.tensor([[0.0], [0.6667]], dtype=torch.float32),
        atol=1e-4,
    )


def test_symmetric_quantization_is_unchanged():
    tensor = torch.tensor([[0.2, 0.7]], dtype=torch.float32)
    dynamic = _symmetric_fake_quantize_tensor(tensor, bits=2)
    assert torch.allclose(dynamic, torch.tensor([[0.0, 0.7]], dtype=torch.float32))


def test_pytorch_adapter_describe_quantization_includes_scales_and_zero_points(
    tmp_path,
):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.ReLU(),
    )
    model_path = tmp_path / "ptq-model.pt"
    torch.save({"model": model}, model_path)

    calibration_images = np.array(
        [[1.0, 2.0], [-1.0, -2.0], [0.5, 1.5]],
        dtype=np.float32,
    )
    adapter = PyTorchAdapter(
        model_path,
        weight_bits=8,
        activation_bits=6,
        activation_scheme="asymmetric",
        calibrate=True,
        calibration_images=calibration_images,
    )

    rows = adapter.describe_quantization()
    assert rows

    weight_rows = [row for row in rows if row["tensor"] == "weight"]
    activation_rows = [row for row in rows if row["tensor"] == "activation"]

    assert weight_rows
    assert any(row["layer"] == "0" for row in weight_rows)
    assert all(row["scale"] is not None for row in weight_rows)
    assert all(row["zero_point"] == 0 for row in weight_rows)

    assert activation_rows
    assert any(row["layer"] == "0" for row in activation_rows)
    assert any(row["layer"] == "1" for row in activation_rows)
    assert any(row["scale"] is not None for row in activation_rows)
    assert any(row["zero_point"] is not None for row in activation_rows)


def test_resolve_torch_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_torch_device("auto") == "cpu"


def test_resolve_torch_device_explicit_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _resolve_torch_device("cpu") == "cpu"


def test_resolve_torch_device_explicit_cuda_without_cuda_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA device requested"):
        _resolve_torch_device("cuda")


def test_pytorch_adapter_weight_quantization_per_channel_metadata(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        linear = model[0]
        assert isinstance(linear, torch.nn.Linear)
        linear.weight.copy_(
            torch.tensor([[10.0, 0.0], [0.5, 0.0]], dtype=torch.float32)
        )

    model_path = tmp_path / "ptq-model-per-channel.pt"
    torch.save({"model": model}, model_path)

    adapter = PyTorchAdapter(
        model_path,
        weight_bits=8,
        per_channel=True,
    )
    rows = adapter.describe_quantization()
    weight_rows = [row for row in rows if row["tensor"] == "weight"]

    assert weight_rows
    row = weight_rows[0]
    assert row["scheme"] == "symmetric-per-channel"
    assert isinstance(row["scale"], list)
    assert len(row["scale"]) == 2
    assert row["scale"][0] != row["scale"][1]
    assert row["zero_point"] == [0, 0]


def test_pytorch_adapter_weight_bias_correction_requires_calibration_images(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=True))
    model_path = tmp_path / "ptq-model-bias-corr.pt"
    torch.save({"model": model}, model_path)

    with pytest.raises(
        ValueError,
        match="calibration_images must be provided when weight_bias_correction=True",
    ):
        PyTorchAdapter(
            model_path,
            weight_bits=8,
            weight_bias_correction=True,
        )


def test_empirical_weight_bias_correction_reduces_output_mean_shift():
    fp32_model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=True))
    with torch.no_grad():
        layer = fp32_model[0]
        assert isinstance(layer, torch.nn.Linear)
        layer.weight.copy_(torch.tensor([[0.6, -0.4]], dtype=torch.float32))
        layer.bias.copy_(torch.tensor([0.3], dtype=torch.float32))

    quantized_model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=True))
    quantized_model.load_state_dict(fp32_model.state_dict())
    _apply_weight_ptq(quantized_model, bits=2)

    calibration_images = np.array(
        [
            [1.0, 2.0],
            [2.0, 2.5],
            [0.5, 1.5],
            [1.5, 2.0],
        ],
        dtype=np.float32,
    )
    calibration_tensor = torch.from_numpy(calibration_images)
    with torch.no_grad():
        fp32_mean = fp32_model(calibration_tensor).mean(dim=0)
        quantized_mean_before = quantized_model(calibration_tensor).mean(dim=0)
    error_before = torch.abs(quantized_mean_before - fp32_mean).item()

    corrected_module_ids = _apply_empirical_weight_bias_correction(
        fp32_model=fp32_model,
        quantized_model=quantized_model,
        calibration_images=calibration_images,
        device="cpu",
        batch_size=2,
    )

    with torch.no_grad():
        quantized_mean_after = quantized_model(calibration_tensor).mean(dim=0)
    error_after = torch.abs(quantized_mean_after - fp32_mean).item()

    assert corrected_module_ids
    assert error_after < error_before


def test_pytorch_adapter_can_export_eval_graph_to_onnx(tmp_path):
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 32 * 3, 10),
    )
    model_path = tmp_path / "ptq-export-model.pt"
    torch.save({"model": model}, model_path)

    calibration_images = np.random.default_rng(7).random(
        (4, 32, 32, 3),
        dtype=np.float32,
    )
    adapter = PyTorchAdapter(
        model_path,
        weight_bits=6,
        activation_bits=7,
        calibrate=True,
        calibration_images=calibration_images,
    )

    export_path = tmp_path / "export" / "eval_quantized.onnx"
    written_path = adapter.export_onnx(export_path)
    assert Path(written_path).exists()

    test_images = np.random.default_rng(9).random(
        (3, 32, 32, 3),
        dtype=np.float32,
    )
    pytorch_logits = adapter.predict_logits(test_images)

    session = ort.InferenceSession(written_path)
    input_shape = session.get_inputs()[0].shape
    assert input_shape[0] == 1

    onnx_adapter = OnnxRuntimeAdapter(written_path)
    onnx_logits = onnx_adapter.predict_logits(test_images)

    assert onnx_logits.shape == pytorch_logits.shape
    np.testing.assert_allclose(onnx_logits, pytorch_logits, rtol=1e-4, atol=1e-4)
