#!/usr/bin/env python3
"""Generate PNG/SVG visualizations of ONNX model graphs.

Creates Graphviz-based visualizations showing operator types and data flow
from input to output. Useful for understanding model architecture and
quantized operation placement.
"""

import argparse
import os
import subprocess
import sys

import onnx


def check_graphviz_installation() -> bool:
    """Check if Graphviz is installed and available.

    Returns:
        True if graphviz dot command is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["dot", "-V"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def visualize_onnx_graph(
    model_path: str, output_dir: str, rankdir: str = "TB"
) -> tuple[str, str, str]:
    """Generate .dot, .png, and .svg visualizations of ONNX model.

    Args:
        model_path: Path to ONNX model file
        output_dir: Output directory for visualization files
        rankdir: Graph layout direction ("TB" for top-to-bottom, "LR" for left-to-right)

    Returns:
        Tuple of (dot_path, png_path, svg_path)
    """
    # Check pydot import
    try:
        from onnx.tools.net_drawer import GetPydotGraph
    except ImportError:
        print("Error: pydot not installed.", file=sys.stderr)
        print("Install with: pip install pydot", file=sys.stderr)
        sys.exit(1)

    # Load ONNX model
    print(f"Loading model: {model_path}")
    model = onnx.load(model_path)
    graph = model.graph

    # Determine graph name from model or filename
    if graph.name:
        graph_name = graph.name
    else:
        base_name = os.path.basename(model_path)
        graph_name = os.path.splitext(base_name)[0]

    print(f"Graph name: {graph_name}")
    print(f"Total nodes: {len(graph.node)}")
    print()

    # Generate pydot graph
    print("Generating graph visualization...")
    pydot_graph = GetPydotGraph(
        graph, name=graph_name, rankdir=rankdir, embed_docstring=True
    )

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine output file paths
    base_name = os.path.basename(model_path)
    name_without_ext = os.path.splitext(base_name)[0]
    dot_path = os.path.join(output_dir, f"{name_without_ext}.dot")
    png_path = os.path.join(output_dir, f"{name_without_ext}.png")
    svg_path = os.path.join(output_dir, f"{name_without_ext}.svg")

    # Write .dot file
    print(f"Writing .dot file: {dot_path}")
    pydot_graph.write_dot(dot_path)

    # Convert to PNG using graphviz dot command
    print(f"Converting to PNG: {png_path}")
    subprocess.run(
        ["dot", "-Tpng", dot_path, "-o", png_path],
        check=True,
        capture_output=True,
    )

    # Convert to SVG using graphviz dot command
    print(f"Converting to SVG: {svg_path}")
    subprocess.run(
        ["dot", "-Tsvg", dot_path, "-o", svg_path],
        check=True,
        capture_output=True,
    )

    return dot_path, png_path, svg_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate PNG/SVG visualizations of ONNX model graphs"
    )
    parser.add_argument(
        "--model",
        default="models/resnet8_int8.onnx",
        help="Path to ONNX model file (default: models/resnet8_int8.onnx)",
    )
    parser.add_argument(
        "--output-dir",
        default="models/",
        help="Output directory for visualization files (default: models/)",
    )
    parser.add_argument(
        "--rankdir",
        default="TB",
        choices=["TB", "LR"],
        help="Graph layout direction: TB (top-to-bottom) or LR (left-to-right)",
    )
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        print("Please ensure the ONNX model has been created.", file=sys.stderr)
        sys.exit(1)

    # Check for graphviz installation
    if not check_graphviz_installation():
        print("Error: Graphviz not installed.", file=sys.stderr)
        print("Install with:", file=sys.stderr)
        print("  Ubuntu/Debian: sudo apt-get install graphviz", file=sys.stderr)
        print("  macOS: brew install graphviz", file=sys.stderr)
        print("  Windows: https://graphviz.org/download/", file=sys.stderr)
        sys.exit(1)

    # Generate visualizations
    dot_path, png_path, svg_path = visualize_onnx_graph(
        args.model, args.output_dir, args.rankdir
    )

    # Print summary
    print()
    print("=" * 50)
    print("VISUALIZATION COMPLETE")
    print("=" * 50)
    print(f"DOT file:  {dot_path}")
    print(f"PNG file:  {png_path}")
    print(f"SVG file:  {svg_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
