from __future__ import annotations

import numpy as np
import torch

from resnet8.evaluation.adapters import (
    _ACTIVATION_TARGET_TYPES,
    _ActivationQuantizer,
    _collect_activation_scales,
    _symmetric_fake_quantize_tensor,
    _symmetric_fake_quantize_tensor_with_scale,
)


def test_fixed_scale_quantization_differs_from_dynamic():
    tensor = torch.tensor([[0.2, 0.7]], dtype=torch.float32)

    dynamic = _symmetric_fake_quantize_tensor(tensor, bits=2)
    fixed = _symmetric_fake_quantize_tensor_with_scale(tensor, bits=2, scale=1.0)

    assert torch.allclose(dynamic, torch.tensor([[0.0, 0.7]], dtype=torch.float32))
    assert torch.allclose(fixed, torch.tensor([[0.0, 1.0]], dtype=torch.float32))


def test_collect_activation_scales_from_calibration_data():
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
    scales = _collect_activation_scales(
        model,
        calibration_images,
        _ACTIVATION_TARGET_TYPES,
        bits=2,
        batch_size=1,
    )

    assert len(scales) >= 2
    assert any(abs(scale - 6.0) < 1e-6 for scale in scales.values())


def test_activation_quantizer_uses_calibrated_fixed_scale():
    linear = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        linear.weight.fill_(1.0)
    model = torch.nn.Sequential(linear)

    _ActivationQuantizer(
        model,
        bits=2,
        scales_by_module_id={id(linear): 1.0},
    )
    outputs = model(torch.tensor([[0.2], [0.7]], dtype=torch.float32))

    assert torch.allclose(outputs, torch.tensor([[0.0], [1.0]], dtype=torch.float32))
