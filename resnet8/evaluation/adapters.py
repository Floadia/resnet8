"""Inference adapters for evaluation backends."""

from __future__ import annotations

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

    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self._model = load_pytorch_model(self.model_path)

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
