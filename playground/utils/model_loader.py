"""Cached model loading utilities for quantization playground.

Uses @mo.cache decorator to prevent ONNX Runtime memory leaks on cell re-execution.
"""

import marimo as mo
import onnx
import torch
from pathlib import Path
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


@mo.cache
def load_onnx_model(path: Path | str) -> onnx.ModelProto:
    """Load ONNX model with caching to prevent memory leaks.

    Args:
        path: Path to ONNX model file (.onnx)

    Returns:
        Loaded ONNX ModelProto

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    logger.info(f"Loading ONNX model from {path}")
    return onnx.load(str(path))


@mo.cache
def load_pytorch_model(path: Path | str) -> torch.nn.Module:
    """Load PyTorch model with caching to prevent memory leaks.

    Args:
        path: Path to PyTorch model file (.pt)

    Returns:
        Loaded PyTorch model in eval mode

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PyTorch model not found: {path}")

    logger.info(f"Loading PyTorch model from {path}")
    # weights_only=False needed for quantized models
    loaded = torch.load(str(path), weights_only=False)

    # Handle dict format (e.g., {'model': ..., 'state_dict': ..., ...})
    if isinstance(loaded, dict) and 'model' in loaded:
        model = loaded['model']
    else:
        model = loaded

    model.eval()
    return model


@mo.cache
def load_model_variants(folder_path: Path | str) -> Dict[str, Optional[Any]]:
    """Load all model variants from a folder with caching.

    Looks for standard ResNet8 model files:
    - ONNX: resnet8.onnx, resnet8_int8.onnx, resnet8_uint8.onnx
    - PyTorch: resnet8.pt, resnet8_int8.pt

    Args:
        folder_path: Path to folder containing model variants

    Returns:
        Dictionary with keys: 'onnx_float', 'onnx_int8', 'onnx_uint8',
        'pytorch_float', 'pytorch_int8'. Value is None for missing files.
    """
    folder = Path(folder_path)

    models = {
        'onnx_float': None,
        'onnx_int8': None,
        'onnx_uint8': None,
        'pytorch_float': None,
        'pytorch_int8': None,
    }

    # ONNX variants
    onnx_files = {
        'onnx_float': folder / 'resnet8.onnx',
        'onnx_int8': folder / 'resnet8_int8.onnx',
        'onnx_uint8': folder / 'resnet8_uint8.onnx',
    }

    for key, path in onnx_files.items():
        if path.exists():
            try:
                models[key] = load_onnx_model(path)
                logger.info(f"Loaded {key}: {path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {key} from {path}: {e}")

    # PyTorch variants
    pytorch_files = {
        'pytorch_float': folder / 'resnet8.pt',
        'pytorch_int8': folder / 'resnet8_int8.pt',
    }

    for key, path in pytorch_files.items():
        if path.exists():
            try:
                models[key] = load_pytorch_model(path)
                logger.info(f"Loaded {key}: {path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {key} from {path}: {e}")

    return models


def get_model_summary(models_dict: Dict[str, Optional[Any]]) -> Dict[str, Any]:
    """Get summary of loaded models.

    Args:
        models_dict: Dictionary from load_model_variants

    Returns:
        Dictionary with:
        - 'total_loaded': count of non-None models
        - 'onnx_available': list of available ONNX variants
        - 'pytorch_available': list of available PyTorch variants
    """
    onnx_keys = ['onnx_float', 'onnx_int8', 'onnx_uint8']
    pytorch_keys = ['pytorch_float', 'pytorch_int8']

    onnx_available = [
        key.replace('onnx_', '')
        for key in onnx_keys
        if models_dict.get(key) is not None
    ]

    pytorch_available = [
        key.replace('pytorch_', '')
        for key in pytorch_keys
        if models_dict.get(key) is not None
    ]

    total_loaded = len(onnx_available) + len(pytorch_available)

    return {
        'total_loaded': total_loaded,
        'onnx_available': onnx_available,
        'pytorch_available': pytorch_available,
    }
