from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from playground.utils.tensor_extractors import (
    extract_onnx_tensor_index,
    extract_pytorch_tensor_index,
)


def _write_quantized_onnx_fixture(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.UINT8, [1])

    node = helper.make_node(
        "QuantizeLinear",
        inputs=["x", "x_scale", "x_zero_point"],
        outputs=["y"],
        name="quant",
    )

    initializers = [
        helper.make_tensor(
            "layer.weight_quantized",
            TensorProto.INT8,
            [2, 2],
            [-2, 0, 1, 3],
        ),
        helper.make_tensor(
            "layer.weight_scale",
            TensorProto.FLOAT,
            [1],
            [0.25],
        ),
        helper.make_tensor(
            "layer.weight_zero_point",
            TensorProto.INT8,
            [1],
            [0],
        ),
        helper.make_tensor(
            "layer.bias",
            TensorProto.FLOAT,
            [2],
            [0.1, -0.2],
        ),
        helper.make_tensor("x_scale", TensorProto.FLOAT, [1], [0.1]),
        helper.make_tensor("x_zero_point", TensorProto.UINT8, [1], [128]),
    ]

    graph = helper.make_graph([node], "quant_graph", [x], [y], initializer=initializers)
    model = helper.make_model(graph)
    onnx.save(model, str(path))


def test_extract_onnx_tensor_index_quantized(tmp_path):
    model_path = tmp_path / "quantized.onnx"
    _write_quantized_onnx_fixture(model_path)

    index = extract_onnx_tensor_index(model_path)

    assert index["format"] == "onnx"
    assert index["is_quantized"] is True
    assert "layer" in index["layers"]

    weight = index["tensor_data"]["layer"]["weight"]
    assert weight["is_quantized"] is True
    assert np.allclose(weight["int_values"], np.array([-2.0, 0.0, 1.0, 3.0]))
    assert weight["scale"] == pytest.approx(0.25)
    assert weight["zero_point"] == pytest.approx(0.0)
    assert np.allclose(weight["values"], np.array([-0.5, 0.0, 0.25, 0.75]))


@pytest.mark.skipif(
    not Path("models/resnet8.pt").exists(), reason="missing FP32 fixture"
)
def test_extract_pytorch_tensor_index_fp32_fixture():
    index = extract_pytorch_tensor_index("models/resnet8.pt")

    assert index["format"] == "pytorch"
    assert index["layers"]
    assert index["tensor_data"]
    found_weight = False
    for layer_tensors in index["tensor_data"].values():
        if "weight" in layer_tensors:
            found_weight = True
            assert "values" in layer_tensors["weight"]
            assert "shape" in layer_tensors["weight"]
            break
    assert found_weight is True


@pytest.mark.skipif(
    not Path("models/resnet8_int8.pt").exists(), reason="missing quantized fixture"
)
def test_extract_pytorch_tensor_index_quantized_fixture():
    index = extract_pytorch_tensor_index("models/resnet8_int8.pt")

    assert index["format"] == "pytorch"
    assert index["layers"]
    assert index["tensor_data"]
    quantized_entries = [
        entry
        for layer_tensors in index["tensor_data"].values()
        for entry in layer_tensors.values()
        if entry.get("is_quantized", False)
    ]
    assert quantized_entries, "expected at least one quantized tensor entry"
