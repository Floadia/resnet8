"""Normalized tensor metadata schema for visualization workflows."""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray


class TensorMetadata(TypedDict, total=False):
    """Normalized metadata for one tensor."""

    values: NDArray[np.float64]
    shape: tuple[int, ...]
    is_quantized: bool
    int_values: NDArray[np.float64]
    scale: float | None
    zero_point: float | None
    scale_values: NDArray[np.float64] | None
    zero_point_values: NDArray[np.float64] | None


LayerTensorMap = dict[str, dict[str, str]]
TensorDataMap = dict[str, dict[str, TensorMetadata]]


class ModelTensorIndex(TypedDict):
    """Top-level tensor index returned to visualization frontends."""

    format: Literal["onnx", "pytorch", "none"]
    layers: LayerTensorMap
    is_quantized: bool
    tensor_data: TensorDataMap
