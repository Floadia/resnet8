# Phase 2: Accuracy Evaluation - Research

**Researched:** 2026-01-27
**Domain:** ONNX Runtime inference, CIFAR-10 evaluation
**Confidence:** HIGH

## Summary

This phase requires implementing ONNX Runtime inference on the CIFAR-10 test set to evaluate the converted ResNet8 model. The standard approach uses ONNX Runtime's Python API (`onnxruntime` package v1.23.2) with numpy for data loading and processing. CIFAR-10 test data is loaded from pickled batch files using Python's built-in pickle module, requiring proper reshaping (3072 → 32×32×3) and type conversion to float32.

The evaluation workflow follows a simple pattern: load ONNX model → create InferenceSession → load test data → batch inference → compute metrics. Per-class accuracy requires counting correct predictions per class using numpy operations. Critical attention must be paid to data type (float32 only), input shape matching, and proper preprocessing that matches the model's training normalization.

**Primary recommendation:** Use ONNX Runtime 1.23.2 with numpy-based data loading; implement batch inference with explicit float32 conversion; calculate per-class accuracy using numpy boolean indexing; avoid sklearn unless per-class metrics become complex.

## Standard Stack

The established libraries/tools for ONNX inference and CIFAR-10 evaluation:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| onnxruntime | 1.23.2+ | ONNX model inference | Official Microsoft runtime, optimized performance, cross-platform support |
| numpy | 1.26.4 | Array operations, data preprocessing | Universal scientific computing library, already constrained for project |
| pickle | stdlib | CIFAR-10 batch file loading | Built-in, CIFAR-10 official format |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| argparse | stdlib | CLI argument parsing | Configurable evaluation parameters (batch size, paths) |
| logging | stdlib | Structured output | Debugging and production monitoring |
| pathlib | stdlib | Path handling | Type-safe file path operations |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| numpy metrics | scikit-learn | sklearn adds dependency for simple accuracy calculation; use only if confusion matrix or complex metrics needed |
| Custom data loader | torchvision.datasets.CIFAR10 | Adds PyTorch dependency; unnecessary since raw pickled files are available |
| Batch-by-batch | Full dataset in memory | 10,000 images @ 32×32×3 = ~30MB; fits in memory, simpler code |

**Installation:**
```bash
# Core requirement
pip install onnxruntime>=1.23.2

# Already installed from Phase 1
# numpy==1.26.4
```

## Architecture Patterns

### Recommended Project Structure
```
scripts/
├── evaluate_onnx.py        # Main evaluation script
└── utils/
    ├── cifar10.py          # CIFAR-10 data loading utilities
    └── metrics.py          # Accuracy calculation utilities (optional)
```

### Pattern 1: ONNX Runtime InferenceSession Workflow
**What:** Standard three-step inference pattern for ONNX models
**When to use:** All ONNX Runtime inference tasks
**Example:**
```python
# Source: https://onnxruntime.ai/docs/get-started/with-python.html
import onnxruntime as ort
import numpy as np

# 1. Create inference session
session = ort.InferenceSession('model.onnx')

# 2. Get input name from model metadata
input_name = session.get_inputs()[0].name

# 3. Run inference with dict mapping input names to numpy arrays
outputs = session.run(None, {input_name: input_data.astype(np.float32)})
predictions = outputs[0]  # First output tensor
```

### Pattern 2: CIFAR-10 Pickle Data Loading
**What:** Official CIFAR-10 unpickle function for loading test batch
**When to use:** Loading CIFAR-10 from original pickled format
**Example:**
```python
# Source: https://www.cs.toronto.edu/~kriz/cifar.html
import pickle

def unpickle(file):
    """Load CIFAR-10 batch file (Python 3)"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Load test batch
test_dict = unpickle('cifar-10-batches-py/test_batch')
# test_dict[b'data']: 10000×3072 uint8 array
# test_dict[b'labels']: 10000-element list (0-9)
```

### Pattern 3: CIFAR-10 Image Reshaping
**What:** Convert flat 3072 array to 32×32×3 image format
**When to use:** Preprocessing CIFAR-10 data for inference
**Example:**
```python
# Source: https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html
# Data layout: first 1024 = red, next 1024 = green, final 1024 = blue
images = test_dict[b'data']  # Shape: (10000, 3072)

# Step 1: Reshape to (N, 3, 32, 32) - channels first
images = images.reshape(-1, 3, 32, 32)

# Step 2: Transpose to (N, 32, 32, 3) - channels last
images = images.transpose(0, 2, 3, 1)

# Now shape is (10000, 32, 32, 3)
```

### Pattern 4: Batch Inference with Dynamic Batch Size
**What:** Process all test data efficiently using ONNX Runtime's batch support
**When to use:** Model has dynamic batch dimension (None, 32, 32, 3)
**Example:**
```python
# Source: https://onnxruntime.ai/docs/api/python/api_summary.html
# Model input shape: (None, 32, 32, 3) - None allows any batch size

# Option A: Full dataset inference (10,000 images fits in memory)
all_predictions = session.run(None, {input_name: all_images.astype(np.float32)})[0]

# Option B: Batched inference if memory constrained
batch_size = 256
predictions = []
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size].astype(np.float32)
    batch_preds = session.run(None, {input_name: batch})[0]
    predictions.append(batch_preds)
predictions = np.concatenate(predictions, axis=0)
```

### Pattern 5: Per-Class Accuracy Calculation
**What:** Calculate accuracy for each of 10 CIFAR-10 classes using numpy
**When to use:** Meeting requirement EVAL-02 (per-class accuracy breakdown)
**Example:**
```python
# Source: https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/Problems/ComputeAccuracy.html
import numpy as np

# Get predicted classes from logits
predicted_classes = np.argmax(predictions, axis=1)  # Shape: (10000,)
true_classes = np.array(labels)  # Shape: (10000,)

# Overall accuracy
overall_accuracy = np.mean(predicted_classes == true_classes)

# Per-class accuracy
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

for class_id, class_name in enumerate(class_names):
    # Boolean mask for samples of this class
    class_mask = (true_classes == class_id)

    # Accuracy for this class only
    class_correct = np.sum((predicted_classes == true_classes) & class_mask)
    class_total = np.sum(class_mask)
    class_accuracy = class_correct / class_total if class_total > 0 else 0.0

    print(f"{class_name}: {class_correct}/{class_total} = {class_accuracy:.2%}")
```

### Pattern 6: Evaluation Script with Argparse
**What:** Configurable evaluation script with CLI arguments
**When to use:** Production evaluation scripts that need flexibility
**Example:**
```python
# Source: https://docs.python.org/3/howto/argparse.html
import argparse

def main():
    parser = argparse.ArgumentParser(description='Evaluate ONNX model on CIFAR-10')
    parser.add_argument('--model', type=str, default='models/resnet8.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--data-dir', type=str,
                        default='/mnt/ext1/references/tiny/benchmark/training/image_classification/cifar-10-batches-py',
                        help='Path to CIFAR-10 data directory')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for inference')
    args = parser.parse_args()

    # Use args.model, args.data_dir, args.batch_size
    evaluate(args.model, args.data_dir, args.batch_size)

if __name__ == '__main__':
    main()
```

### Anti-Patterns to Avoid
- **Loading data with torchvision**: Adds unnecessary PyTorch dependency when pickled files are directly available
- **Using float64**: ONNX Runtime only accepts float32; explicit conversion required
- **String input names**: Never hardcode "input" or "output"; always retrieve from model metadata using `get_inputs()[0].name`
- **Ignoring batch dimension**: Even single image needs shape (1, 32, 32, 3), not (32, 32, 3)
- **Per-sample loops**: Inefficient; leverage batch processing for 10-100× speedup

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CIFAR-10 class names | Hardcoded list | Load from batches.meta file | Official source, handles ordering correctly |
| Confusion matrix | Nested loops for TP/FP/TN/FN | `sklearn.metrics.confusion_matrix` | Edge cases (missing classes), normalization options |
| Classification report | Format string manipulation | `sklearn.metrics.classification_report` | Precision, recall, F1-score, support; standardized format |
| Data normalization | Manual mean/std calculation | Document training normalization | Must match training preprocessing exactly |

**Key insight:** CIFAR-10 evaluation is well-trodden territory. The complexity is in preprocessing consistency (matching training normalization), not in metric calculation. Focus on data pipeline correctness over custom metric implementations.

## Common Pitfalls

### Pitfall 1: Type Mismatch (float64 vs float32)
**What goes wrong:** ONNX Runtime raises `InvalidArgument` error: "Unexpected input data type. Actual: (tensor(double)), expected: (tensor(float))"
**Why it happens:** Numpy defaults to float64 for floating-point arrays; ONNX Runtime strictly requires float32
**How to avoid:** Always call `.astype(np.float32)` on input arrays before inference
**Warning signs:** Error message mentioning "tensor(double)" or dtype mismatch

### Pitfall 2: Input Shape Mismatch
**What goes wrong:** Runtime error about invalid dimensions or rank for input
**Why it happens:**
- Forgot batch dimension: passing (32, 32, 3) instead of (1, 32, 32, 3)
- Wrong channel order: (3, 32, 32) vs (32, 32, 3)
- Image not reshaped from flat 3072 array
**How to avoid:**
- Verify model input shape with `session.get_inputs()[0].shape`
- Test with single image first
- Print shapes during development: `print(f"Input shape: {input_array.shape}")`
**Warning signs:** Error message with "expected" vs "got" dimensions

### Pitfall 3: Incorrect Input Name
**What goes wrong:** Error "Invalid Feed Input Name" or no output returned
**Why it happens:** Hardcoding input name as "input" or "x" without checking model
**How to avoid:** Always retrieve name programmatically: `input_name = session.get_inputs()[0].name`
**Warning signs:** Runtime error mentioning feed input names

### Pitfall 4: Normalization Mismatch
**What goes wrong:** Accuracy significantly below expected (e.g., 10% instead of 85%+)
**Why it happens:** Model trained with normalized data (e.g., [0,1] or [-1,1]), but evaluation uses raw uint8 [0,255]
**How to avoid:**
- Document training normalization in Phase 1
- Apply identical preprocessing: divide by 255.0 for [0,1] normalization
- Test on small batch first; random guessing = ~10% accuracy for 10 classes
**Warning signs:** Accuracy near random chance (10% for CIFAR-10)

### Pitfall 5: Argmax on Wrong Axis
**What goes wrong:** Predictions array has wrong shape or nonsensical values
**Why it happens:** Model output shape is (batch_size, 10); argmax needs axis=1 to get class per sample, not axis=0
**How to avoid:**
- Verify output shape: `print(f"Output shape: {predictions.shape}")` should be (N, 10)
- Use `axis=1` or `axis=-1` to reduce across classes dimension
**Warning signs:** Predicted classes array has shape (10,) instead of (N,) or values outside [0,9]

### Pitfall 6: Label Key Encoding Issues (Python 3)
**What goes wrong:** KeyError when accessing dictionary keys from unpickled CIFAR-10 batch
**Why it happens:** Python 3's pickle.load with `encoding='bytes'` returns byte-string keys (b'data', b'labels') not string keys
**How to avoid:** Use byte-string literals: `test_dict[b'data']` not `test_dict['data']`
**Warning signs:** KeyError showing 'data' not found, but `test_dict.keys()` shows `b'data'`

### Pitfall 7: Off-by-One Errors in Class Indexing
**What goes wrong:** Class names don't match predictions (e.g., "automobile" labeled as "bird")
**Why it happens:** CIFAR-10 classes are 0-indexed (0-9); confusion with 1-indexed systems
**How to avoid:**
- Verify class order from batches.meta file
- Test with known samples (e.g., print a few predictions visually)
**Warning signs:** Per-class accuracies look shuffled or systematically wrong

### Pitfall 8: Dynamic Batch Size Warm-Up Delay
**What goes wrong:** First inference call takes minutes to complete
**Why it happens:** ONNX Runtime caches optimized graphs for specific batch sizes on first use
**How to avoid:**
- For production: warm up with representative batch sizes during initialization
- For evaluation script: acceptable since only run once; document in README
**Warning signs:** First batch processes slowly, subsequent batches are fast

## Code Examples

Verified patterns from official sources:

### Complete Evaluation Workflow
```python
# Combined from official ONNX Runtime and CIFAR-10 documentation
import onnxruntime as ort
import numpy as np
import pickle
from pathlib import Path

def unpickle(file):
    """Load CIFAR-10 batch (Python 3)"""
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def evaluate_onnx_model(model_path, data_dir):
    """Evaluate ONNX model on CIFAR-10 test set"""

    # 1. Load ONNX model
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    print(f"Model input: {input_name}, shape: {session.get_inputs()[0].shape}")

    # 2. Load CIFAR-10 test batch
    test_batch_path = Path(data_dir) / 'test_batch'
    test_dict = unpickle(str(test_batch_path))

    # 3. Preprocess data
    images = test_dict[b'data']  # (10000, 3072) uint8
    labels = np.array(test_dict[b'labels'])  # (10000,) int

    # Reshape to (10000, 32, 32, 3)
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Normalize to [0, 1] range (match training)
    images = images.astype(np.float32) / 255.0

    # 4. Run inference
    print(f"Running inference on {len(images)} images...")
    predictions = session.run(None, {input_name: images})[0]  # (10000, 10)
    predicted_classes = np.argmax(predictions, axis=1)  # (10000,)

    # 5. Calculate metrics
    correct = np.sum(predicted_classes == labels)
    total = len(labels)
    overall_accuracy = correct / total

    print(f"\nOverall Accuracy: {correct}/{total} = {overall_accuracy:.2%}")

    # 6. Per-class accuracy
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print("\nPer-Class Accuracy:")
    for class_id, class_name in enumerate(class_names):
        mask = (labels == class_id)
        class_correct = np.sum((predicted_classes == labels) & mask)
        class_total = np.sum(mask)
        class_acc = class_correct / class_total
        print(f"  {class_name:12s}: {class_correct}/{class_total:4d} = {class_acc:.2%}")

    return overall_accuracy, predicted_classes, labels

# Usage
if __name__ == '__main__':
    accuracy, preds, labels = evaluate_onnx_model(
        'models/resnet8.onnx',
        '/mnt/ext1/references/tiny/benchmark/training/image_classification/cifar-10-batches-py'
    )
```

### Loading CIFAR-10 Class Names from Metadata
```python
# Source: https://www.cs.toronto.edu/~kriz/cifar.html
def load_cifar10_metadata(data_dir):
    """Load official class names from batches.meta"""
    meta_path = Path(data_dir) / 'batches.meta'
    meta_dict = unpickle(str(meta_path))

    # Returns byte strings in Python 3
    label_names = [name.decode('utf-8') for name in meta_dict[b'label_names']]
    return label_names

# Usage
class_names = load_cifar10_metadata(data_dir)
# ['airplane', 'automobile', 'bird', 'cat', 'deer',
#  'dog', 'frog', 'horse', 'ship', 'truck']
```

### Batch Processing with Progress
```python
# Source: ONNX Runtime best practices
def evaluate_batched(session, images, labels, batch_size=256):
    """Evaluate with batched inference for memory efficiency"""
    input_name = session.get_inputs()[0].name
    n_samples = len(images)
    all_predictions = []

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = images[start_idx:end_idx].astype(np.float32)

        batch_preds = session.run(None, {input_name: batch})[0]
        all_predictions.append(batch_preds)

        if (start_idx // batch_size) % 10 == 0:
            print(f"  Processed {end_idx}/{n_samples} images...")

    predictions = np.concatenate(all_predictions, axis=0)
    predicted_classes = np.argmax(predictions, axis=1)

    accuracy = np.mean(predicted_classes == labels)
    return accuracy, predicted_classes
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| torchvision.datasets.CIFAR10 | Direct pickle loading | N/A | Removes PyTorch dependency for pure ONNX workflow |
| onnxruntime < 1.0 | onnxruntime 1.23.2+ | Ongoing | Better performance, Python 3.10-3.13 support |
| Manual metric loops | Vectorized numpy operations | Established | 10-100× faster for 10K samples |
| Single image inference | Batch inference | Established | Leverages ONNX Runtime optimizations |

**Deprecated/outdated:**
- **Python 2 cPickle**: Use Python 3's `pickle.load(fo, encoding='bytes')` with byte-string keys
- **onnxruntime-gpu without version spec**: Specify >=1.23.2 for CUDA 12.x support, Python 3.12+ compatibility
- **Manual data augmentation during eval**: Test set evaluation should be deterministic (no augmentation)

## Open Questions

Things that couldn't be fully resolved:

1. **Training Normalization Method**
   - What we know: Model was trained in Phase 1 (Keras); standard CIFAR-10 normalization is [0,1] via /255.0
   - What's unclear: Exact preprocessing used during training (might have mean/std normalization)
   - Recommendation: Check Phase 1 converter script; assume /255.0 normalization unless documented otherwise; if accuracy is ~10% (random), investigate normalization mismatch

2. **Expected Accuracy for ResNet8**
   - What we know: Requirement is >85%; modern ResNets achieve 90-95% on CIFAR-10
   - What's unclear: ResNet8 (shallow, 8 layers) may underperform ResNet18/34/50
   - Recommendation: 85% is reasonable baseline; if below 80%, investigate model architecture or training; if below 50%, check preprocessing

3. **Batch Size for 10K Images**
   - What we know: Full dataset is ~30MB (10K × 32×32×3 × 4 bytes), fits in memory
   - What's unclear: ONNX Runtime performance characteristics on target hardware
   - Recommendation: Start with full batch (10,000); if memory issues, use 256-1024 batch size; document in script args

4. **Output Format for Per-Class Accuracy**
   - What we know: Requirement EVAL-02 requires per-class breakdown
   - What's unclear: Desired format (stdout, CSV, JSON?)
   - Recommendation: Print to stdout with tabular format; add optional --output-json flag for machine-readable format if needed in future phases

## Sources

### Primary (HIGH confidence)
- [ONNX Runtime Python Documentation](https://onnxruntime.ai/docs/get-started/with-python.html) - Inference patterns, API usage
- [ONNX Runtime API Reference](https://onnxruntime.ai/docs/api/python/api_summary.html) - InferenceSession methods, input/output handling
- [ONNX Runtime Common Errors](https://onnxruntime.ai/docs/api/python/auto_examples/plot_common_errors.html) - Type mismatches, shape errors
- [CIFAR-10 Official Dataset Page](https://www.cs.toronto.edu/~kriz/cifar.html) - Unpickle function, data format, class labels
- [ONNX Runtime PyPI](https://pypi.org/project/onnxruntime/) - Version 1.23.2, Python 3.10-3.13 support

### Secondary (MEDIUM confidence)
- [NumPy Accuracy Calculation Tutorial](https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/Problems/ComputeAccuracy.html) - Vectorized accuracy with argmax
- [Binary Study CIFAR-10 Tutorial](https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html) - Reshape and transpose pattern
- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html) - Classification metrics, confusion matrix
- [Python Argparse Tutorial](https://docs.python.org/3/howto/argparse.html) - CLI argument parsing patterns
- [Python Logging Best Practices 2026](https://www.carmatec.com/blog/python-logging-best-practices-complete-guide/) - Structured logging patterns

### Tertiary (LOW confidence)
- [ONNX Runtime Batch Processing GitHub Issues](https://github.com/microsoft/onnxruntime/issues/9867) - Community discussions on batch inference
- [Medium: ML Inference Runtimes in 2026](https://medium.com/@digvijay17july/ml-inference-runtimes-in-2026-an-architects-guide-to-choosing-the-right-engine-d3989a87d052) - Ecosystem comparison
- [Stack Overflow discussions on CIFAR-10 preprocessing](https://groups.google.com/d/topic/keras-users/5em8x8YF4d8) - Community solutions

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - ONNX Runtime 1.23.2 verified on PyPI, official docs current
- Architecture: HIGH - Patterns verified against official ONNX Runtime and CIFAR-10 documentation
- Pitfalls: HIGH - Common errors documented in official ONNX Runtime troubleshooting guides
- Data preprocessing: HIGH - Official CIFAR-10 format documented by dataset authors
- Per-class metrics: MEDIUM - Multiple approaches exist (numpy vs sklearn); numpy sufficient for basic accuracy

**Research date:** 2026-01-27
**Valid until:** 2026-03-27 (60 days - stable domain, ONNX Runtime updates quarterly)
