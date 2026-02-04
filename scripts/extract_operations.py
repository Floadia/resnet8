#!/usr/bin/env python3
"""Extract quantized operations from ONNX models to structured JSON.

Extracts QLinearConv, QLinearMatMul, QuantizeLinear, and DequantizeLinear
operations with their scales, zero-points, and attributes for documentation
purposes in subsequent phases.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

import onnx
import onnx.numpy_helper as nph
from onnx.helper import get_attribute_value


def extract_qlinear_operations(model_path: str) -> Dict[str, Any]:
    """Extract all quantized operations from ONNX model.

    Args:
        model_path: Path to ONNX model file

    Returns:
        Dictionary containing model metadata and quantized operations
    """
    # Load ONNX model
    model = onnx.load(model_path)
    graph = model.graph

    # Build initializer lookup dict mapping names to numpy arrays
    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = nph.to_array(init)

    # Extract quantized operations
    qlinear_ops = []
    target_ops = ["QLinearConv", "QLinearMatMul", "QuantizeLinear", "DequantizeLinear"]

    for node in graph.node:
        if node.op_type not in target_ops:
            continue

        # Extract node metadata
        node_data = {
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": {},
            "scales": {},
            "zero_points": {},
        }

        # Extract attributes using onnx.helper.get_attribute_value()
        for attr in node.attribute:
            value = get_attribute_value(attr)
            # Convert to JSON-serializable types if needed
            if hasattr(value, "tolist"):
                value = value.tolist()
            node_data["attributes"][attr.name] = value

        # Extract scales and zero-points from initializers
        for input_name in node.input:
            if input_name in initializers:
                value = initializers[input_name]
                # Convert numpy types to Python types for JSON serialization
                if value.ndim == 0:  # scalar
                    py_value = float(value) if value.dtype.kind == "f" else int(value)
                else:  # array
                    py_value = value.tolist()

                # Categorize by naming convention
                input_lower = input_name.lower()
                if "scale" in input_lower:
                    node_data["scales"][input_name] = py_value
                elif "zero" in input_lower or "zp" in input_lower:
                    node_data["zero_points"][input_name] = py_value

        qlinear_ops.append(node_data)

    # Build summary statistics
    op_type_counts = {}
    for op_type in target_ops:
        op_type_counts[op_type] = sum(
            1 for op in qlinear_ops if op["op_type"] == op_type
        )

    # Get opset version
    opset_version = model.opset_import[0].version if model.opset_import else None

    return {
        "model_path": model_path,
        "opset_version": opset_version,
        "operations": qlinear_ops,
        "summary": {
            "total_nodes": len(graph.node),
            "qlinear_nodes": len(qlinear_ops),
            "op_type_counts": op_type_counts,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract quantized operations from ONNX models to JSON"
    )
    parser.add_argument(
        "--model",
        default="models/resnet8_int8.onnx",
        help="Path to ONNX model file (default: models/resnet8_int8.onnx)",
    )
    parser.add_argument(
        "--output",
        default="models/resnet8_int8_operations.json",
        help="Output JSON file path (default: models/resnet8_int8_operations.json)",
    )
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        print("Please ensure the ONNX model has been created.", file=sys.stderr)
        sys.exit(1)

    # Extract operations
    print(f"Loading model: {args.model}")
    data = extract_qlinear_operations(args.model)
    print(f"Model opset version: {data['opset_version']}")
    print()

    # Print summary to stdout
    print("=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Total nodes in graph: {data['summary']['total_nodes']}")
    print(f"Quantized nodes found: {data['summary']['qlinear_nodes']}")
    print()
    print("Operations by type:")
    for op_type, count in data["summary"]["op_type_counts"].items():
        print(f"  {op_type:20s}: {count:3d}")
    print("=" * 50)
    print()

    # Write JSON output
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
