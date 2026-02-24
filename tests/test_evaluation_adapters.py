from __future__ import annotations

import numpy as np
import torch

from resnet8.evaluation.adapters import (
    _ACTIVATION_TARGET_TYPES,
    _ActivationQuantizer,
    _asymmetric_params_from_min_max,
    _collect_activation_params,
    _fake_quantize_tensor_with_params,
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
