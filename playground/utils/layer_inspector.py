"""Layer inspection utilities for extracting layer names."""

from typing import Any, Dict, List, Optional

import onnx
import torch


def get_onnx_layer_names(model: onnx.ModelProto) -> List[str]:
    """Extract layer names from ONNX model.

    Extracts both operation names (from nodes) and parameter names (from initializers).

    Args:
        model: ONNX ModelProto

    Returns:
        Sorted, deduplicated list of layer names
    """
    layer_names = []

    # Extract node names (operations)
    for node in model.graph.node:
        if node.name:  # Filter out empty names
            layer_names.append(node.name)

    # Extract initializer names (weights, biases)
    for initializer in model.graph.initializer:
        if initializer.name:  # Filter out empty names
            layer_names.append(initializer.name)

    # Deduplicate and sort
    return sorted(set(layer_names))


def get_pytorch_layer_names(model: torch.nn.Module) -> List[str]:
    """Extract layer names from PyTorch model.

    Uses named_modules() to get hierarchical layer paths.

    Args:
        model: PyTorch model (nn.Module)

    Returns:
        List of layer names (e.g., 'layer1.conv1', 'layer2.0.bn1')
    """
    layer_names = []

    for name, module in model.named_modules():
        # Filter out root module (empty name)
        if name:
            layer_names.append(name)

    return layer_names


def get_all_layer_names(models_dict: Dict[str, Optional[Any]]) -> Dict[str, Any]:
    """Extract layer names from available models.

    Prioritizes ONNX float model if available, else PyTorch float model.

    Args:
        models_dict: Dictionary from load_model_variants with keys like
                     'onnx_float', 'pytorch_float', etc.

    Returns:
        Dictionary with:
        - 'layer_names': List of layer names (or empty list if no models)
        - 'source': 'onnx' or 'pytorch' indicating source model type
    """
    # Try ONNX float first
    if models_dict.get("onnx_float") is not None:
        return {
            "layer_names": get_onnx_layer_names(models_dict["onnx_float"]),
            "source": "onnx",
        }

    # Fall back to PyTorch float
    if models_dict.get("pytorch_float") is not None:
        return {
            "layer_names": get_pytorch_layer_names(models_dict["pytorch_float"]),
            "source": "pytorch",
        }

    # Try any ONNX model
    for key in ["onnx_int8", "onnx_uint8"]:
        if models_dict.get(key) is not None:
            return {
                "layer_names": get_onnx_layer_names(models_dict[key]),
                "source": "onnx",
            }

    # Try any PyTorch model
    if models_dict.get("pytorch_int8") is not None:
        return {
            "layer_names": get_pytorch_layer_names(models_dict["pytorch_int8"]),
            "source": "pytorch",
        }

    # No models available
    return {"layer_names": [], "source": None}


def get_layer_type(model: Any, layer_name: str) -> Optional[str]:
    """Get the type of a specific layer.

    Args:
        model: ONNX ModelProto or PyTorch nn.Module
        layer_name: Name of the layer to look up

    Returns:
        Layer type as string (e.g., "Conv", "BatchNormalization", "Linear")
        or None if not found
    """
    if isinstance(model, onnx.ModelProto):
        # Search in nodes
        for node in model.graph.node:
            if node.name == layer_name:
                return node.op_type

        # Not found in nodes - might be an initializer (just return "Parameter")
        for initializer in model.graph.initializer:
            if initializer.name == layer_name:
                return "Parameter"

        return None

    elif isinstance(model, torch.nn.Module):
        # Search in named_modules
        for name, module in model.named_modules():
            if name == layer_name:
                return module.__class__.__name__

        return None

    return None
