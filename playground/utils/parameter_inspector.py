"""Parameter inspection utilities for ONNX quantized models.

Extracts quantization parameters (scales, zero-points) and weight tensors
from ONNX QDQ format models for visualization and analysis.
"""

from typing import Any, Dict, List, Optional, Set

import numpy as np
import onnx
import onnx.numpy_helper as nph


def extract_layer_params(
    model: Optional[onnx.ModelProto], layer_name: str
) -> Optional[Dict[str, Any]]:
    """Extract quantization parameters for a specific ONNX layer.

    Searches for QuantizeLinear/DequantizeLinear nodes associated with the
    given layer name and extracts their scale and zero-point parameters.

    Args:
        model: ONNX ModelProto (can be None for graceful handling)
        layer_name: Name of the layer to extract parameters for

    Returns:
        Dictionary with keys:
        - scale: numpy array (scalar or 1D for per-channel)
        - zero_point: numpy array (scalar or 1D)
        - weight_shape: tuple or None
        - weight_dtype: str (e.g., "int8", "uint8", "float32") or None
        - node_type: str (e.g., "QuantizeLinear", "DequantizeLinear")
        - is_per_channel: bool (True if scale.ndim > 0)

        Returns None if no quantization parameters found or model is None.
    """
    if model is None:
        return None

    # Build initializer lookup dict
    initializers = {}
    for init in model.graph.initializer:
        initializers[init.name] = nph.to_array(init)

    # Search for QuantizeLinear/DequantizeLinear nodes for this layer
    for node in model.graph.node:
        if node.op_type not in ["QuantizeLinear", "DequantizeLinear"]:
            continue

        # Check if this node is associated with the layer
        # Match by node name or if layer_name appears in inputs/outputs
        is_match = (
            node.name == layer_name
            or layer_name in node.name
            or any(layer_name in inp for inp in node.input)
            or any(layer_name in out for out in node.output)
        )

        if not is_match:
            continue

        # Extract scale and zero-point from node inputs
        # QuantizeLinear/DequantizeLinear have inputs: [x, scale, zero_point]
        if len(node.input) < 3:
            continue  # Not enough inputs

        scale_name = node.input[1]
        zp_name = node.input[2]

        scale = initializers.get(scale_name)
        zero_point = initializers.get(zp_name)

        if scale is None:
            continue  # Scale not found in initializers

        # Try to find associated weight tensor
        weight_shape = None
        weight_dtype = None

        # Look for weight initializers that contain layer_name
        for init_name, init_array in initializers.items():
            if layer_name in init_name and init_array.ndim >= 2:
                # Likely a weight tensor (2D+ dimensions)
                weight_shape = init_array.shape
                weight_dtype = str(init_array.dtype)
                break

        # Determine if per-channel quantization
        is_per_channel = scale.ndim > 0

        return {
            "scale": scale,
            "zero_point": zero_point if zero_point is not None else np.array(0),
            "weight_shape": weight_shape,
            "weight_dtype": weight_dtype,
            "node_type": node.op_type,
            "is_per_channel": is_per_channel,
        }

    # No quantization parameters found
    return None


def extract_weight_tensors(
    fp32_model: Optional[onnx.ModelProto],
    int8_model: Optional[onnx.ModelProto],
    layer_name: str,
    uint8_model: Optional[onnx.ModelProto] = None,
) -> Dict[str, Optional[np.ndarray]]:
    """Extract weight tensors from multiple model variants for comparison.

    For quantized models (INT8/UINT8), dequantizes the weights using the
    formula: (raw - zero_point) * scale

    Args:
        fp32_model: FP32 ONNX model (can be None)
        int8_model: INT8 quantized ONNX model (can be None)
        layer_name: Name of the layer to extract weights for
        uint8_model: UINT8 quantized ONNX model (can be None)

    Returns:
        Dictionary with keys:
        - fp32: numpy array or None
        - int8_dequantized: numpy array or None
        - uint8_dequantized: numpy array or None
        - int8_raw: numpy array or None
        - uint8_raw: numpy array or None
    """
    result = {
        "fp32": None,
        "int8_dequantized": None,
        "uint8_dequantized": None,
        "int8_raw": None,
        "uint8_raw": None,
    }

    # Extract FP32 weights
    if fp32_model is not None:
        for init in fp32_model.graph.initializer:
            if layer_name in init.name and init.name.endswith(
                ("weight", "kernel", "W")
            ):
                arr = nph.to_array(init)
                if arr.dtype == np.float32 and arr.ndim >= 2:
                    result["fp32"] = arr
                    break

    # Helper function to extract and dequantize quantized weights
    def extract_quantized(model: Optional[onnx.ModelProto], dtype_name: str):
        if model is None:
            return None, None

        # Build initializer dict
        initializers = {}
        for init in model.graph.initializer:
            initializers[init.name] = nph.to_array(init)

        # Find weight tensor
        weight_raw = None
        for init_name, init_array in initializers.items():
            if layer_name in init_name and init_array.ndim >= 2:
                if init_array.dtype in [np.int8, np.uint8]:
                    weight_raw = init_array
                    break

        if weight_raw is None:
            return None, None

        # Extract scale and zero-point for dequantization
        params = extract_layer_params(model, layer_name)
        if params is None:
            return weight_raw, None

        scale = params["scale"]
        zero_point = params["zero_point"]

        # Dequantize: (raw - zero_point) * scale
        weight_dequantized = (weight_raw.astype(np.float32) - zero_point) * scale

        return weight_raw, weight_dequantized

    # Extract INT8 weights
    int8_raw, int8_dequant = extract_quantized(int8_model, "int8")
    result["int8_raw"] = int8_raw
    result["int8_dequantized"] = int8_dequant

    # Extract UINT8 weights
    uint8_raw, uint8_dequant = extract_quantized(uint8_model, "uint8")
    result["uint8_raw"] = uint8_raw
    result["uint8_dequantized"] = uint8_dequant

    return result


def compute_all_layer_ranges(
    fp32_model: Optional[onnx.ModelProto], int8_model: Optional[onnx.ModelProto]
) -> List[Dict[str, Any]]:
    """Compute weight value ranges for all layers in both models.

    Args:
        fp32_model: FP32 ONNX model
        int8_model: INT8 quantized ONNX model

    Returns:
        List of dictionaries with keys:
        - name: Layer name
        - fp32_min: Minimum weight value in FP32 model
        - fp32_max: Maximum weight value in FP32 model
        - fp32_range: fp32_max - fp32_min
        - int8_min: Minimum weight value in INT8 model (dequantized)
        - int8_max: Maximum weight value in INT8 model (dequantized)
        - int8_range: int8_max - int8_min

        Returns empty list if models are None or have no common layers.
    """
    if fp32_model is None or int8_model is None:
        return []

    # Extract FP32 weight initializers
    fp32_weights = {}
    for init in fp32_model.graph.initializer:
        arr = nph.to_array(init)
        # Filter to weight tensors (2D+, float32)
        if arr.dtype == np.float32 and arr.ndim >= 2:
            fp32_weights[init.name] = arr

    # Extract INT8 weight initializers
    int8_initializers = {}
    for init in int8_model.graph.initializer:
        int8_initializers[init.name] = nph.to_array(init)

    # Find common layer prefixes
    layer_ranges = []

    for fp32_name, fp32_arr in fp32_weights.items():
        # Extract layer name prefix (e.g., "conv1" from "conv1.weight")
        layer_prefix = fp32_name.split(".")[0]

        # Find corresponding INT8 weight
        int8_arr = None

        for name, arr in int8_initializers.items():
            if layer_prefix in name and arr.ndim >= 2 and arr.dtype in [
                np.int8,
                np.uint8,
            ]:
                int8_arr = arr
                break

        if int8_arr is None:
            continue  # No corresponding INT8 weight found

        # Dequantize INT8 weights
        params = extract_layer_params(int8_model, layer_prefix)
        if params is not None:
            scale = params["scale"]
            zero_point = params["zero_point"]
            int8_dequantized = (int8_arr.astype(np.float32) - zero_point) * scale
        else:
            # If no params found, use raw values (fallback)
            int8_dequantized = int8_arr.astype(np.float32)

        # Compute ranges
        fp32_min = float(np.min(fp32_arr))
        fp32_max = float(np.max(fp32_arr))
        int8_min = float(np.min(int8_dequantized))
        int8_max = float(np.max(int8_dequantized))

        layer_ranges.append(
            {
                "name": layer_prefix,
                "fp32_min": fp32_min,
                "fp32_max": fp32_max,
                "fp32_range": fp32_max - fp32_min,
                "int8_min": int8_min,
                "int8_max": int8_max,
                "int8_range": int8_max - int8_min,
            }
        )

    # Sort by layer name for consistent ordering
    layer_ranges.sort(key=lambda x: x["name"])

    return layer_ranges


def get_layers_with_params(model: Optional[onnx.ModelProto]) -> Set[str]:
    """Return set of layer names that have QuantizeLinear/DequantizeLinear nodes.

    Args:
        model: ONNX ModelProto (can be None)

    Returns:
        Set of layer name strings. Returns empty set if model is None.
    """
    if model is None:
        return set()

    layers_with_params = set()

    for node in model.graph.node:
        if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
            # Add the node name itself
            if node.name:
                layers_with_params.add(node.name)

            # Extract layer prefixes from node name
            # E.g., "const_fold_opt__139_DequantizeLinear" -> "const_fold_opt__139"
            if "_QuantizeLinear" in node.name:
                prefix = node.name.replace("_QuantizeLinear", "")
                layers_with_params.add(prefix)
            elif "_DequantizeLinear" in node.name:
                prefix = node.name.replace("_DequantizeLinear", "")
                layers_with_params.add(prefix)

            # Also extract from inputs/outputs
            for inp in node.input:
                if inp:
                    layers_with_params.add(inp)
            for out in node.output:
                if out:
                    layers_with_params.add(out)

    return layers_with_params
