#!/usr/bin/env python3
"""
Keras to ONNX conversion script for ResNet8 CIFAR-10 model.

Converts pretrained Keras model to ONNX format with validation and logging.
"""

import logging

import onnx
import tensorflow as tf
import tf2onnx


def setup_logging(log_file):
    """Configure logging with file and console output"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def convert_keras_to_onnx(
    keras_path, onnx_path, input_shape, opset=15, log_file="logs/conversion.log"
):
    """
    Convert Keras .h5 model to ONNX format

    Args:
        keras_path: Path to .h5 model file
        onnx_path: Path to save .onnx model
        input_shape: Tuple of input dimensions (e.g., (None, 32, 32, 3))
        opset: ONNX opset version (default: 15)
        log_file: Path to log file

    Returns:
        True if conversion succeeded, False otherwise
    """
    logger = setup_logging(log_file)

    try:
        # Log conversion parameters
        logger.info(f"Converting {keras_path} to {onnx_path}")
        logger.info(f"Input shape: {input_shape}")
        logger.info(f"Opset version: {opset}")

        # Load Keras model
        logger.info("Loading Keras model")
        model = tf.keras.models.load_model(keras_path)
        logger.info(f"Model loaded: {model.name}")

        # Define input signature
        input_signature = [tf.TensorSpec(input_shape, tf.float32, name="input")]

        # Convert to ONNX
        logger.info("Starting ONNX conversion")
        onnx_model, _ = tf2onnx.convert.from_keras(
            model, input_signature=input_signature, opset=opset, output_path=onnx_path
        )
        logger.info("Conversion completed")

        # Verify ONNX model
        logger.info("Verifying ONNX model structure")
        onnx.checker.check_model(onnx_model)
        logger.info("Model verification passed")

        # Helper function for shape extraction
        def shape2tuple(shape):
            return tuple(getattr(d, "dim_value", 0) for d in shape.dim)

        # Log model structure
        for input_tensor in onnx_model.graph.input:
            shape = shape2tuple(input_tensor.type.tensor_type.shape)
            logger.info(f"Input: {input_tensor.name}, Shape: {shape}")

        for output_tensor in onnx_model.graph.output:
            shape = shape2tuple(output_tensor.type.tensor_type.shape)
            logger.info(f"Output: {output_tensor.name}, Shape: {shape}")

        node_count = len(onnx_model.graph.node)
        param_count = len(onnx_model.graph.initializer)
        logger.info(f"Nodes: {node_count}")
        logger.info(f"Parameters: {param_count}")

        # Verify shapes match expectations
        print("\n" + "=" * 60)
        print("CONVERSION VERIFICATION SUMMARY")
        print("=" * 60)

        for input_tensor in onnx_model.graph.input:
            shape = shape2tuple(input_tensor.type.tensor_type.shape)
            print(f"Input: {input_tensor.name}")
            print(f"  Shape: {shape}")
            # Verify spatial dimensions match CIFAR-10 (32x32x3)
            if shape[1:] == (32, 32, 3):
                print("  ✓ Shape matches CIFAR-10 format (batch, 32, 32, 3)")
            else:
                logger.warning(
                    f"Input shape {shape} does not match expected (batch, 32, 32, 3)"
                )

        for output_tensor in onnx_model.graph.output:
            shape = shape2tuple(output_tensor.type.tensor_type.shape)
            print(f"Output: {output_tensor.name}")
            print(f"  Shape: {shape}")
            # Verify class dimension matches CIFAR-10 (10 classes)
            if shape[-1] == 10 or (len(shape) == 2 and shape[1] == 10):
                print("  ✓ Shape matches CIFAR-10 classes (batch, 10)")
            else:
                logger.warning(
                    f"Output shape {shape} does not match expected (batch, 10)"
                )

        print("\nModel statistics:")
        print(f"  Total nodes: {node_count}")
        print(f"  Total parameters: {param_count}")
        print("=" * 60)
        print("Model verification passed ✓")
        print("=" * 60 + "\n")

        return True

    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    # Define paths
    keras_path = (
        "/mnt/ext1/references/tiny/benchmark/training/"
        "image_classification/trained_models/pretrainedResnet.h5"
    )
    onnx_path = "models/resnet8.onnx"
    log_file = "logs/conversion.log"

    # Convert with CIFAR-10 input shape (None for dynamic batch size)
    success = convert_keras_to_onnx(
        keras_path=keras_path,
        onnx_path=onnx_path,
        input_shape=(None, 32, 32, 3),  # CIFAR-10 format
        opset=15,
        log_file=log_file,
    )

    if success:
        print("Conversion completed successfully")
        print(f"ONNX model saved to: {onnx_path}")
        print(f"Conversion log saved to: {log_file}")
        exit(0)
    else:
        print(f"Conversion failed - check {log_file} for details")
        exit(1)
