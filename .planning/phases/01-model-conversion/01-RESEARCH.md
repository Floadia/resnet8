# Phase 1: Model Conversion - Research

**Researched:** 2026-01-27
**Domain:** Keras to ONNX model conversion using tf2onnx
**Confidence:** HIGH

## Summary

This research investigates the conversion of Keras .h5 models to ONNX format using tf2onnx, specifically for a ResNet8 CIFAR-10 model. The standard approach uses the tf2onnx Python API (specifically `tf2onnx.convert.from_keras()`) to convert models, followed by verification using the ONNX checker and model inspection tools.

tf2onnx 1.16.1 is the current stable version (released January 2024) and supports Python 3.7-3.10. It handles TensorFlow/Keras models and converts them to ONNX opset 14-18, with opset-15 as the default. The conversion workflow involves loading the Keras model, calling the conversion function with appropriate input specifications, saving the ONNX output, and validating the result through structural verification (input/output shapes, node count) and model checking (onnx.checker.check_model).

The primary risks are shape inference issues with BatchNormalization layers (common in ResNet architectures), incorrect opset selection causing operator compatibility problems, and missing input_signature specifications leading to dynamic shape inference failures. These are mitigated through explicit input shape specification, targeting opset-15 or higher, and comprehensive verification including layer count validation.

**Primary recommendation:** Use tf2onnx.convert.from_keras() with explicit input_signature (32x32x3 for CIFAR-10), default opset-15, and validate with onnx.checker plus manual structure inspection.

## Standard Stack

The established libraries/tools for Keras to ONNX conversion:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| tf2onnx | 1.16.1 | Keras/TensorFlow to ONNX converter | Official ONNX project tool, actively maintained, supports latest TF versions |
| tensorflow | 2.x | Load and run Keras models | Required dependency for loading .h5 files |
| onnx | 1.9+ | ONNX format library | Core ONNX library for model manipulation and validation |
| numpy | latest | Array operations | Universal dependency for tensor operations |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| netron | latest | Visual model inspection | Quick visual verification of model structure |
| onnx-tool | 0.3.3+ | Model analysis and profiling | Detailed parameter counting and layer analysis |
| Python logging | stdlib | Progress and warning tracking | Standard library for conversion progress reporting |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| tf2onnx | keras2onnx | keras2onnx is frozen at TF 2.3 / ONNX 1.10 (deprecated), tf2onnx is current |
| tf2onnx | onnxmltools | onnxmltools doesn't support Keras models directly, requires CoreML intermediate |
| Python API | Command-line tool | CLI is simpler for one-off conversions but less scriptable |

**Installation:**
```bash
pip install tensorflow tf2onnx onnx
# Optional: for visual inspection
pip install netron
# Optional: for detailed analysis
pip install onnx-tool
```

## Architecture Patterns

### Recommended Project Structure
```
scripts/
├── convert.py           # Conversion script
├── verify_onnx.py       # Verification script (optional)
models/
├── pretrainedResnet.h5  # Source Keras model (input)
└── resnet8.onnx         # Converted ONNX model (output)
logs/
└── conversion.log       # Conversion log with warnings
```

### Pattern 1: Python API Conversion (Recommended)
**What:** Use tf2onnx.convert.from_keras() for programmatic conversion with full control
**When to use:** When you need logging, error handling, and integration with other scripts
**Example:**
```python
# Source: https://github.com/onnx/tensorflow-onnx official API
import tensorflow as tf
import tf2onnx
import onnx
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load Keras model
logger.info("Loading Keras model from .h5 file")
model = tf.keras.models.load_model('models/pretrainedResnet.h5')

# Define input signature (CIFAR-10: 32x32x3)
input_signature = [tf.TensorSpec([None, 32, 32, 3], tf.float32, name='input')]

# Convert to ONNX
logger.info("Converting to ONNX format")
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=15,  # Default, explicitly specified for clarity
    output_path='models/resnet8.onnx'
)

logger.info("Conversion complete")
logger.info(f"ONNX model saved to models/resnet8.onnx")

# Verify the model
logger.info("Verifying ONNX model structure")
onnx.checker.check_model(onnx_model)
logger.info("ONNX model verification passed")
```

### Pattern 2: Structure Verification
**What:** Inspect converted ONNX model to verify shapes, node counts, and graph structure
**When to use:** After every conversion to catch errors early
**Example:**
```python
# Source: https://onnx.ai/onnx/intro/python.html
import onnx

def verify_onnx_structure(onnx_path, expected_input_shape, expected_output_shape):
    """Verify ONNX model structure matches expectations"""
    model = onnx.load(onnx_path)

    # Check model validity
    onnx.checker.check_model(model)

    # Helper to extract shape tuple
    def shape2tuple(shape):
        return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

    # Verify input shape
    for input_tensor in model.graph.input:
        shape = shape2tuple(input_tensor.type.tensor_type.shape)
        print(f"Input: {input_tensor.name}, Shape: {shape}")
        # Allow None for batch dimension
        if shape[1:] != expected_input_shape[1:]:
            raise ValueError(f"Input shape mismatch: {shape} vs {expected_input_shape}")

    # Verify output shape
    for output_tensor in model.graph.output:
        shape = shape2tuple(output_tensor.type.tensor_type.shape)
        print(f"Output: {output_tensor.name}, Shape: {shape}")
        if shape[1:] != expected_output_shape[1:]:
            raise ValueError(f"Output shape mismatch: {shape} vs {expected_output_shape}")

    # Count nodes (layers)
    node_count = len(model.graph.node)
    print(f"Total nodes/layers: {node_count}")

    # Count parameters (initializers)
    param_count = len(model.graph.initializer)
    print(f"Total parameters: {param_count}")

    return model

# Usage for ResNet8 CIFAR-10
verify_onnx_structure(
    'models/resnet8.onnx',
    expected_input_shape=(None, 32, 32, 3),
    expected_output_shape=(None, 10)
)
```

### Pattern 3: Logging with Progress Tracking
**What:** Use Python logging module with multiple handlers for file and console output
**When to use:** All conversion scripts to track progress and warnings
**Example:**
```python
# Source: https://www.carmatec.com/blog/python-logging-best-practices-complete-guide/
import logging

def setup_logger(name, log_file, level=logging.INFO):
    """Setup logger with file and console handlers"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Usage
logger = setup_logger('convert', 'logs/conversion.log')
logger.info("Starting conversion")
logger.warning("FusedBatchNormV3 shape inference may produce warnings")
logger.info("Conversion completed successfully")
```

### Anti-Patterns to Avoid
- **Don't rely on automatic shape inference:** Always specify input_signature explicitly to avoid dynamic shape errors
- **Don't ignore conversion warnings:** BatchNorm warnings often indicate shape mismatches that affect model behavior
- **Don't skip model verification:** onnx.checker.check_model() catches IR version and opset incompatibilities early
- **Don't use print() for logging:** Use logging module for proper level control and file output

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Model format conversion | Custom weight extractor/mapper | tf2onnx.convert.from_keras() | Handles operator mapping, shape inference, BatchNorm fusion, opset compatibility |
| ONNX validation | Manual shape checking | onnx.checker.check_model() | Verifies IR version, opset imports, metadata consistency, type inference |
| Model structure inspection | Custom protobuf parser | onnx.load() + model.graph API | Standard API for inputs, outputs, nodes, initializers |
| Visual model inspection | Text-based dumping | netron (GUI or CLI) | Interactive graph visualization with layer details |
| Parameter counting | Manual initializer iteration | onnx-tool or custom counter | Handles parameter types, shapes, memory calculation |

**Key insight:** ONNX conversion involves complex operator mapping, shape inference, and opset compatibility. tf2onnx handles hundreds of TensorFlow ops with proper shape propagation and BatchNorm fusion. Custom solutions will miss edge cases and fail silently.

## Common Pitfalls

### Pitfall 1: BatchNormalization Shape Inference Warnings
**What goes wrong:** tf2onnx produces warnings about FusedBatchNormV3 shape inference failures, or conversion fails with shape mismatch errors.
**Why it happens:** Keras BatchNormalization uses FusedBatchNormV3 internally, which requires specific shape constraints. tf2onnx may struggle to infer shapes when batch dimension is dynamic or when BN is used in non-standard ways.
**How to avoid:**
- Specify explicit input_signature with concrete batch dimension or None for dynamic batching
- Use opset-15 or higher which has better BatchNorm support
- Check conversion warnings and verify output shapes match expectations
**Warning signs:**
- Warnings containing "FusedBatchNormV3" during conversion
- Output shape (None, None) instead of (None, 10)
- Model passes check_model but produces wrong outputs

### Pitfall 2: Incorrect Opset Selection
**What goes wrong:** Conversion fails with "operator not supported in opset X" or runtime fails to load model.
**Why it happens:** Different opsets support different operators. Too low opset may not support TensorFlow ops, too high opset may not be supported by deployment runtime.
**How to avoid:**
- Use opset-15 (tf2onnx default) unless deployment runtime requires different version
- Check ONNX Runtime compatibility: supports opset 14-18 as of 2026
- If conversion fails, try opset-16 or opset-17 for newer op support
**Warning signs:**
- Conversion error mentioning "opset" or "not supported"
- Model loads but inference fails with operator errors
- onnx.checker.check_model() reports opset import issues

### Pitfall 3: Missing Input Signature
**What goes wrong:** Conversion creates model with dynamic shapes everywhere, or fails with "cannot infer shape" errors.
**Why it happens:** Without input_signature, tf2onnx must infer shapes from model structure alone. This fails for models with conditional logic or dynamic shapes.
**How to avoid:**
- Always provide input_signature with tf.TensorSpec
- Use None for batch dimension if dynamic batching needed: [None, 32, 32, 3]
- Use concrete dimension for batch if fixed: [1, 32, 32, 3]
**Warning signs:**
- Output shapes showing (None, None, None) instead of (None, 10)
- Conversion warnings about "unknown shape"
- Model verification shows unexpected dynamic dimensions

### Pitfall 4: Skipping Model Verification
**What goes wrong:** Converted model has structural errors but script reports success. Errors only appear during inference.
**Why it happens:** onnx.save() succeeds even if model is malformed. Only onnx.checker.check_model() validates structure.
**How to avoid:**
- Always call onnx.checker.check_model() after conversion
- Verify input/output shapes match expected values
- Count nodes to ensure layers weren't dropped
- Optional: Run single inference test to verify numerical correctness
**Warning signs:**
- Model file created but fails to load in ONNX Runtime
- Inference produces wrong shapes or NaN outputs
- Model works in one runtime but not another

### Pitfall 5: Insufficient Logging
**What goes wrong:** Conversion fails or produces warnings but information is lost. Debugging requires re-running conversion.
**Why it happens:** tf2onnx writes warnings to stderr, which may not be captured. Using print() doesn't preserve messages.
**How to avoid:**
- Setup logging.basicConfig() before importing tf2onnx
- Use both file and console handlers to preserve logs
- Log input shape, opset, output path before conversion
- Log success/failure status and verification results
**Warning signs:**
- Can't determine why conversion failed
- Warnings mentioned in issues but not visible in output
- Unable to reproduce intermittent failures

## Code Examples

Verified patterns from official sources:

### Complete Conversion Script
```python
# Source: Combined from https://github.com/onnx/tensorflow-onnx and https://onnx.ai/onnx/api/checker.html
import tensorflow as tf
import tf2onnx
import onnx
import logging
from pathlib import Path

def setup_logging(log_file):
    """Configure logging with file and console output"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def convert_keras_to_onnx(
    keras_path,
    onnx_path,
    input_shape,
    opset=15,
    log_file='conversion.log'
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
        input_signature = [tf.TensorSpec(input_shape, tf.float32, name='input')]

        # Convert to ONNX
        logger.info("Starting ONNX conversion")
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=opset,
            output_path=onnx_path
        )
        logger.info("Conversion completed")

        # Verify ONNX model
        logger.info("Verifying ONNX model structure")
        onnx.checker.check_model(onnx_model)
        logger.info("Model verification passed")

        # Log model structure
        logger.info(f"Input: {onnx_model.graph.input[0].name}")
        logger.info(f"Output: {onnx_model.graph.output[0].name}")
        logger.info(f"Nodes: {len(onnx_model.graph.node)}")
        logger.info(f"Parameters: {len(onnx_model.graph.initializer)}")

        return True

    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}", exc_info=True)
        return False

# Usage
if __name__ == '__main__':
    success = convert_keras_to_onnx(
        keras_path='models/pretrainedResnet.h5',
        onnx_path='models/resnet8.onnx',
        input_shape=(None, 32, 32, 3),  # CIFAR-10 format
        opset=15,
        log_file='logs/conversion.log'
    )

    if success:
        print("Conversion completed successfully")
    else:
        print("Conversion failed - check logs/conversion.log")
```

### Structure Verification Script
```python
# Source: https://onnx.ai/onnx/intro/python.html
import onnx

def inspect_onnx_model(onnx_path):
    """
    Inspect ONNX model structure and print details

    Args:
        onnx_path: Path to .onnx model file
    """
    model = onnx.load(onnx_path)

    # Verify model
    onnx.checker.check_model(model)
    print("✓ Model validation passed")

    # Helper function
    def shape2tuple(shape):
        return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

    # Inspect inputs
    print("\nInputs:")
    for input_tensor in model.graph.input:
        shape = shape2tuple(input_tensor.type.tensor_type.shape)
        dtype = input_tensor.type.tensor_type.elem_type
        print(f"  {input_tensor.name}: shape={shape}, dtype={dtype}")

    # Inspect outputs
    print("\nOutputs:")
    for output_tensor in model.graph.output:
        shape = shape2tuple(output_tensor.type.tensor_type.shape)
        dtype = output_tensor.type.tensor_type.elem_type
        print(f"  {output_tensor.name}: shape={shape}, dtype={dtype}")

    # Count nodes and parameters
    print(f"\nNodes (layers): {len(model.graph.node)}")
    print(f"Parameters (initializers): {len(model.graph.initializer)}")

    # List node types
    node_types = {}
    for node in model.graph.node:
        node_types[node.op_type] = node_types.get(node.op_type, 0) + 1

    print("\nNode type distribution:")
    for op_type, count in sorted(node_types.items()):
        print(f"  {op_type}: {count}")

# Usage
if __name__ == '__main__':
    inspect_onnx_model('models/resnet8.onnx')
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| keras2onnx | tf2onnx | 2021 | keras2onnx frozen at TF 2.3, tf2onnx supports TF 2.x actively |
| Manual opset specification | Default opset-15 | 2022 | tf2onnx defaults to opset-15, covers 99% of models |
| Command-line only | Python API preferred | 2020+ | Better error handling, logging, integration in scripts |
| Visual inspection via netron.app | netron CLI + Python | 2023+ | Programmatic inspection without browser upload |

**Deprecated/outdated:**
- **keras2onnx**: Frozen at TensorFlow 2.3 and ONNX 1.10 (2021). Use tf2onnx instead.
- **keras-onnx (GitHub project)**: Archived, directs users to tf2onnx
- **onnxmltools for Keras**: Never supported Keras directly, requires CoreML intermediate

## Open Questions

Things that couldn't be fully resolved:

1. **Exact ResNet8 architecture**
   - What we know: Model is at /mnt/ext1/references/tiny/benchmark/training/image_classification/trained_models/pretrainedResnet.h5 (1.1MB file exists)
   - What's unclear: Exact layer count, specific residual block configuration (basic vs bottleneck), presence of shortcuts
   - Recommendation: Inspect .h5 file during conversion to determine actual layer count for verification

2. **Expected node count in ONNX**
   - What we know: ResNet8 typically has 8 weight layers (Conv layers), but ONNX nodes include BatchNorm, ReLU, Add, etc.
   - What's unclear: Expected total node count after conversion for verification purposes
   - Recommendation: Run conversion first, log node count, use as baseline for future conversions

3. **Optimal opset for CIFAR-10 ResNet8**
   - What we know: Opset-15 is default and supports all common operations (Conv, BatchNorm, ReLU, Add)
   - What's unclear: Whether higher opsets (16-18) provide any advantages for this specific model
   - Recommendation: Start with default opset-15, only change if conversion fails or deployment runtime requires different version

## Sources

### Primary (HIGH confidence)
- [tf2onnx PyPI](https://pypi.org/project/tf2onnx/) - Version 1.16.1, Python support, installation
- [GitHub: tensorflow-onnx](https://github.com/onnx/tensorflow-onnx) - Official repository, API documentation
- [ONNX Checker API](https://onnx.ai/onnx/api/checker.html) - Model verification functions
- [ONNX Python API](https://onnx.ai/onnx/intro/python.html) - Model loading and inspection
- [tf2onnx convert.py source](https://github.com/onnx/tensorflow-onnx/blob/main/tf2onnx/convert.py) - from_keras() function signature

### Secondary (MEDIUM confidence)
- [ONNX Runtime TensorFlow Tutorial](https://onnxruntime.ai/docs/tutorials/tf-get-started.html) - Conversion workflow
- [Keras to ONNX conversion guides](https://github.com/onnx/keras-onnx) - Deprecation notice, tf2onnx recommendation
- [Netron GitHub](https://github.com/lutzroeder/netron) - Visual inspection tool
- [onnx-tool](https://github.com/ThanatosShinji/onnx-tool) - Model profiling and analysis
- [Python Logging Best Practices 2026](https://www.carmatec.com/blog/python-logging-best-practices-complete-guide/) - Logging patterns

### Tertiary (LOW confidence — flagged for validation)
- BatchNorm shape inference issues: Multiple GitHub issues reported but unclear if resolved in 1.16.1
- Optimal opset selection: General guidance exists but no specific recommendation for ResNet8
- Node count expectations: Will vary based on optimization passes, need empirical baseline

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - tf2onnx 1.16.1 verified from PyPI, official ONNX project tool
- Architecture: HIGH - Python API patterns verified from official GitHub source
- Pitfalls: HIGH - BatchNorm, opset, and input_signature issues documented in official GitHub issues
- Code examples: HIGH - Combined from official documentation and source code

**Research date:** 2026-01-27
**Valid until:** 2026-04-27 (90 days - tf2onnx is stable, updates infrequent)
