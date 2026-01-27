# Architecture Patterns: Keras to PyTorch Model Conversion

**Domain:** Model Conversion (Keras .h5 to PyTorch .pt)
**Researched:** 2026-01-27
**Confidence:** MEDIUM

## Executive Summary

Keras to PyTorch conversion projects follow a **three-component architecture**: (1) Model Definition, (2) Weight Converter, and (3) Evaluation/Inference. For ResNet8 CIFAR-10 conversion, the recommended approach is **manual conversion** with layer-by-layer weight mapping rather than automated tools, given the specific architecture and need for verification.

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Conversion Project                        │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌──────────────┐
│ Model         │    │ Converter      │    │ Evaluation   │
│ Definition    │    │ Script         │    │ Script       │
│               │    │                │    │              │
│ resnet8.py    │◄───│ convert.py     │───►│ evaluate.py  │
│               │    │                │    │              │
│ - PyTorch     │    │ - Load .h5     │    │ - Load .pt   │
│   layers      │    │ - Extract      │    │ - Run        │
│ - Architecture│    │   weights      │    │   inference  │
│ - forward()   │    │ - Map layers   │    │ - Calculate  │
│               │    │ - Save .pt     │    │   metrics    │
└───────────────┘    └────────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Saved Weights    │
                    │                  │
                    │ Input:  .h5      │
                    │ Output: .pt      │
                    └──────────────────┘
```

### Data Flow

```
1. Keras Source (.h5)
   └─> Converter reads architecture metadata + trained weights
       └─> Extracts layer-by-layer weights as numpy arrays
           └─> PyTorch Model Definition instantiated
               └─> Weight mapping with shape transformations
                   └─> state_dict populated and saved (.pt)
                       └─> Evaluation script loads .pt
                           └─> Inference on test data
                               └─> Metrics comparison
```

## Component Boundaries

| Component | Responsibility | Inputs | Outputs | Communicates With |
|-----------|---------------|---------|---------|-------------------|
| **Model Definition** | Define PyTorch architecture matching Keras model | None (code only) | PyTorch model instance | Converter, Evaluator |
| **Converter Script** | Extract Keras weights, map to PyTorch, save checkpoint | `.h5` file, Model definition | `.pt` file (state_dict) | Model Definition |
| **Evaluation Script** | Load converted model, run inference, calculate metrics | `.pt` file, test data | Accuracy/metrics | Model Definition |

### Component Details

#### 1. Model Definition (`resnet8.py` or `models/resnet8.py`)

**Purpose:** Define PyTorch architecture that mirrors Keras model structure.

**Key elements:**
- Inherits from `torch.nn.Module`
- `__init__()`: Layer definitions (Conv2d, BatchNorm2d, ReLU, Linear, etc.)
- `forward()`: Forward pass logic (data flow through layers)
- No weights embedded - parameters registered automatically

**Critical considerations:**
- Layer ordering must match Keras model exactly
- Parameter names should be descriptive for weight mapping
- No hardcoded weights - structure only

**Example structure:**
```python
class ResNet8(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Define layers matching Keras architecture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # ... additional layers

    def forward(self, x):
        # Forward pass logic
        x = F.relu(self.bn1(self.conv1(x)))
        # ... forward flow
        return x
```

#### 2. Converter Script (`convert.py` or `convert_keras_to_pytorch.py`)

**Purpose:** Bridge Keras and PyTorch by extracting, transforming, and loading weights.

**Key operations:**
1. **Load Keras model** from .h5 file
2. **Instantiate PyTorch model** (from model definition)
3. **Extract Keras weights** layer by layer as numpy arrays
4. **Transform weight shapes** (Keras and PyTorch use different conventions)
5. **Map weights** to corresponding PyTorch parameters
6. **Validate dimensions** (catch shape mismatches early)
7. **Save state_dict** to .pt file

**Critical transformations:**
- **Conv2D weights**: Keras `(H, W, C_in, C_out)` → PyTorch `(C_out, C_in, H, W)`
- **Dense/Linear weights**: Keras `(C_out, C_in)` → PyTorch `(C_in, C_out)` (transpose)
- **BatchNorm**: Parameter names differ (gamma→weight, beta→bias)
- **Biases**: Usually no transformation needed (1D arrays)

**Weight mapping strategy:**
```python
# Pseudocode
keras_model = load_model('resnet8.h5')
pytorch_model = ResNet8()

# Layer-by-layer mapping
for keras_layer, pytorch_layer in zip(keras_model.layers, pytorch_model.modules()):
    weights = keras_layer.get_weights()
    # Apply shape transformations based on layer type
    transformed_weights = transform_weights(weights, layer_type)
    # Load into PyTorch
    pytorch_layer.load_state_dict(transformed_weights)

# Save complete model
torch.save(pytorch_model.state_dict(), 'resnet8.pt')
```

**Output format:**
- Save as state_dict (weights only) not full model
- Use `.pt` or `.pth` extension (convention)
- Include metadata (model architecture name, conversion date, etc.)

#### 3. Evaluation Script (`evaluate.py` or `eval.py`)

**Purpose:** Validate converted model produces correct predictions.

**Key operations:**
1. **Load PyTorch model** definition
2. **Load state_dict** from .pt file
3. **Set model to eval mode** (`model.eval()` - critical for BatchNorm/Dropout)
4. **Load test data** (CIFAR-10 test set)
5. **Run inference** (forward pass with `torch.no_grad()`)
6. **Calculate metrics** (accuracy, loss, etc.)
7. **Compare with Keras baseline** (optional but recommended)

**Critical settings:**
```python
model = ResNet8()
model.load_state_dict(torch.load('resnet8.pt', weights_only=True))
model.eval()  # MUST set to eval mode

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        # Calculate metrics
```

**Validation checklist:**
- Accuracy matches Keras model (within small tolerance)
- Loss values comparable
- Predictions on sample images match
- Inference speed reasonable

## Patterns to Follow

### Pattern 1: Manual Layer-by-Layer Conversion
**What:** Manually recreate architecture and map weights instead of using automated tools.

**When:**
- Custom architectures (ResNet8 is custom for CIFAR-10)
- Need full control and understanding
- Automated tools fail or produce incorrect results

**Why better than alternatives:**
- Full transparency and control
- Easier debugging (know exactly where weights came from)
- Learn architecture deeply
- No black-box tool dependencies

**Implementation:**
```python
# convert.py
import tensorflow as tf
import torch
from models.resnet8 import ResNet8

# Load Keras model
keras_model = tf.keras.models.load_model('resnet8_keras.h5')

# Instantiate PyTorch model
pytorch_model = ResNet8(num_classes=10)

# Manual mapping
pytorch_state_dict = {}

# Example: First conv layer
keras_conv_weights, keras_conv_bias = keras_model.layers[1].get_weights()
# Transform: (H, W, C_in, C_out) -> (C_out, C_in, H, W)
pytorch_conv_weights = torch.from_numpy(
    keras_conv_weights.transpose(3, 2, 0, 1)
)
pytorch_state_dict['conv1.weight'] = pytorch_conv_weights
pytorch_state_dict['conv1.bias'] = torch.from_numpy(keras_conv_bias)

# ... repeat for all layers

pytorch_model.load_state_dict(pytorch_state_dict)
torch.save(pytorch_model.state_dict(), 'resnet8.pt')
```

### Pattern 2: Separate Model Definition from Weights
**What:** Never hardcode weights in model definition. Keep architecture and parameters separate.

**Why:**
- Reusable model definition for different weight sets
- Standard PyTorch convention
- Easier testing (can instantiate with random weights)
- Cleaner code organization

**Implementation:**
```python
# models/resnet8.py - architecture only
class ResNet8(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Define structure, no weights

# Usage
model = ResNet8()  # Random init weights
model.load_state_dict(torch.load('converted.pt'))  # Load trained weights
```

### Pattern 3: Three-Script Architecture
**What:** Separate concerns into three independent scripts that can run standalone.

**Components:**
1. `models/resnet8.py` - Architecture definition (importable module)
2. `convert.py` - One-time conversion script
3. `evaluate.py` - Validation and inference script

**Why:**
- Clear separation of concerns
- Convert once, evaluate many times
- Each script testable independently
- Standard deep learning project pattern

**Directory structure:**
```
project/
├── models/
│   ├── __init__.py
│   └── resnet8.py          # Model definition
├── data/
│   └── cifar10.py          # Data loading utilities
├── convert.py              # Keras → PyTorch conversion
├── evaluate.py             # Model evaluation
├── checkpoints/
│   ├── resnet8_keras.h5    # Input (Keras weights)
│   └── resnet8.pt          # Output (PyTorch weights)
└── requirements.txt
```

### Pattern 4: Validation-Driven Conversion
**What:** Validate conversion correctness at multiple stages.

**Validation points:**
1. **Layer count match**: Same number of layers in both models
2. **Parameter count match**: Total parameters identical
3. **Shape validation**: Each layer's weight shapes correct after transformation
4. **Numerical validation**: Same input produces same output (forward pass comparison)
5. **Metric validation**: Accuracy on test set matches

**Implementation:**
```python
# In convert.py
def validate_conversion(keras_model, pytorch_model):
    # 1. Parameter count
    keras_params = keras_model.count_params()
    pytorch_params = sum(p.numel() for p in pytorch_model.parameters())
    assert keras_params == pytorch_params, f"Param mismatch: {keras_params} vs {pytorch_params}"

    # 2. Forward pass comparison
    test_input = np.random.randn(1, 32, 32, 3).astype(np.float32)
    keras_output = keras_model.predict(test_input)

    # Convert to PyTorch format: NHWC -> NCHW
    pytorch_input = torch.from_numpy(test_input.transpose(0, 3, 1, 2))
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(pytorch_input).numpy()

    # Compare outputs
    np.testing.assert_allclose(keras_output, pytorch_output, rtol=1e-5, atol=1e-5)
    print("✓ Conversion validated successfully")
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Using ONNX for Simple Conversions
**What:** Converting Keras → ONNX → PyTorch for simple architectures.

**Why bad:**
- ONNX adds complexity and potential failure points
- ONNX → PyTorch support is limited (mainly for inference, not native training)
- Loses PyTorch idioms and patterns
- Harder to debug when issues arise
- ONNX Runtime ≠ PyTorch model (can't fine-tune easily)

**Consequences:**
- Brittle conversion pipeline
- Models not true PyTorch (wrapped in ONNX runtime)
- Difficult to modify or extend model later

**Instead:**
Manual conversion for custom architectures like ResNet8. ONNX is better suited for deployment, not development conversion.

**When ONNX IS appropriate:**
- Deploying to edge devices or cross-platform inference
- Framework-agnostic model serving
- Large standard architectures with official ONNX support

### Anti-Pattern 2: Embedding Weights in Model Definition
**What:** Hardcoding weight values directly in the model class code.

**Why bad:**
- Violates separation of concerns
- Makes model definition non-reusable
- Huge file sizes (weights as source code)
- Can't train or fine-tune
- Standard PyTorch patterns won't work

**Consequences:**
```python
# DON'T DO THIS
class ResNet8(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1.weight = torch.tensor([...])  # Massive hardcoded array
```

**Instead:**
```python
# DO THIS
class ResNet8(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(...)  # Structure only

# Separately
model = ResNet8()
model.load_state_dict(torch.load('weights.pt'))
```

### Anti-Pattern 3: Ignoring Eval Mode
**What:** Running inference without calling `model.eval()`.

**Why bad:**
- BatchNorm uses training statistics (running mean/var) instead of frozen stats
- Dropout still active (random neuron dropping)
- Results non-deterministic and incorrect

**Consequences:**
- Accuracy significantly lower than expected
- Non-reproducible results
- Gradients still computed (slower, memory waste)

**Instead:**
```python
# ALWAYS do this for inference
model.eval()
with torch.no_grad():
    output = model(input)
```

### Anti-Pattern 4: Forgetting Data Format Differences
**What:** Not converting between Keras (NHWC) and PyTorch (NCHW) image formats.

**Why bad:**
- Keras uses (batch, height, width, channels)
- PyTorch uses (batch, channels, height, width)
- Silent failures or shape errors

**Consequences:**
```python
# Keras shape: (1, 32, 32, 3)
# PyTorch expects: (1, 3, 32, 32)

# WRONG - will fail or give incorrect results
pytorch_model(keras_image_tensor)

# CORRECT
pytorch_image = keras_image.transpose(0, 3, 1, 2)  # NHWC -> NCHW
pytorch_model(torch.from_numpy(pytorch_image))
```

**Critical for:**
- Test data preprocessing
- Weight conversion validation
- Inference pipeline

### Anti-Pattern 5: All-in-One Conversion Script
**What:** Single script that does conversion + evaluation + training.

**Why bad:**
- Hard to debug (which part failed?)
- Can't reuse components
- Violates single responsibility principle
- Must re-run everything to test one part

**Instead:**
Separate scripts per concern:
- `convert.py` - conversion only (run once)
- `evaluate.py` - evaluation only (run many times)
- `models/resnet8.py` - model definition (imported by both)

## Build Order and Dependencies

### Recommended Build Sequence

```
Phase 1: Model Definition
├─ Define PyTorch ResNet8 architecture
├─ Match Keras layer structure exactly
└─ Test: Instantiate model, check layer count

Phase 2: Converter Implementation
├─ Load Keras .h5 file
├─ Extract weights layer by layer
├─ Implement shape transformations
├─ Map to PyTorch state_dict
└─ Test: Parameter counts match, shapes correct

Phase 3: Validation
├─ Forward pass comparison (same input → same output)
├─ Save .pt file
└─ Test: Load saved .pt, verify integrity

Phase 4: Evaluation Script
├─ Load converted .pt model
├─ Load CIFAR-10 test data
├─ Run inference with proper settings
├─ Calculate accuracy
└─ Test: Accuracy matches Keras baseline
```

### Dependency Graph

```
resnet8.py (Model Definition)
    ↓
    ├─→ convert.py (needs model to instantiate and save)
    │       ↓
    │   resnet8.pt (converted weights)
    │       ↓
    └─→ evaluate.py (needs model to load and infer)
            ↓
        metrics.txt (accuracy results)
```

**Key insights:**
1. **Model Definition is foundational** - both converter and evaluator import it
2. **Converter runs once** - produces .pt artifact
3. **Evaluator runs many times** - consumes .pt artifact
4. **No circular dependencies** - clean unidirectional flow

### Component Independence

| Component | Can Run Standalone? | Dependencies | Produces |
|-----------|-------------------|--------------|----------|
| `models/resnet8.py` | Yes (import and instantiate) | None | Model instance |
| `convert.py` | Yes | models/resnet8.py, .h5 file | .pt file |
| `evaluate.py` | Yes | models/resnet8.py, .pt file | Metrics |

**This independence enables:**
- Testing each component separately
- Reusing model definition for other purposes (training, fine-tuning)
- Running evaluation without re-conversion
- Clean CI/CD pipelines (separate jobs per script)

## Framework-Specific Considerations

### Keras (.h5) Weight Format

**Storage structure:**
- `.h5` is HDF5 format (hierarchical data)
- Contains: model architecture (JSON) + layer weights (arrays)
- Weight access: `layer.get_weights()` returns list of numpy arrays
- Typical order: `[kernel, bias]` for conv/dense layers

**Key points:**
- Layer naming varies (conv2d_1, conv2d_2, etc.)
- Need to match Keras layer index to PyTorch module name
- BatchNorm stores: gamma, beta, moving_mean, moving_variance

### PyTorch (.pt) Weight Format

**Storage structure:**
- `.pt` is pickled Python dictionary (state_dict)
- Dictionary keys: parameter names (e.g., "conv1.weight", "conv1.bias")
- Values: torch.Tensor objects

**Best practices:**
- Use `torch.save(model.state_dict(), path)` not `torch.save(model, path)`
- Load with `weights_only=True` for security
- state_dict only contains parameters, not architecture

### Critical Format Differences

| Aspect | Keras | PyTorch | Conversion Action |
|--------|-------|---------|-------------------|
| Conv2D weight shape | (H, W, C_in, C_out) | (C_out, C_in, H, W) | Transpose: (3,2,0,1) |
| Dense/Linear weight shape | (out, in) | (in, out) | Transpose: (1,0) |
| Data format | NHWC | NCHW | Transpose: (0,3,1,2) |
| BatchNorm params | gamma, beta | weight, bias | Rename |
| Padding | "same", "valid" | int or tuple | Calculate manually |

## Scalability Considerations

For this specific project (ResNet8, single model conversion), scalability is not a primary concern. However, for context:

| Concern | ResNet8 Conversion | If Scaling to Many Models |
|---------|-------------------|---------------------------|
| **Model size** | Small (~2MB .pt file) | Use separate storage for checkpoints, streaming load |
| **Conversion time** | Seconds (one-time) | Batch conversion pipeline, parallel processing |
| **Evaluation throughput** | 10K test images, fast | GPU acceleration, DataLoader with workers |
| **Code reuse** | Single model | Generic converter base class, config-driven architecture |

**Current project scope:**
- One-time conversion (no need for production pipeline)
- Small model (fits in memory easily)
- Standard evaluation (CIFAR-10 test set)

**No over-engineering needed** - simple scripts sufficient for this use case.

## Confidence Assessment

| Aspect | Confidence | Rationale |
|--------|-----------|-----------|
| **Component structure** | HIGH | Standard PyTorch project patterns verified across multiple sources |
| **Manual conversion approach** | HIGH | Well-documented in PyTorch forums and tutorials |
| **Weight transformation details** | HIGH | Shape conversions verified in official docs and community sources |
| **ONNX limitations** | MEDIUM | Some sources recommend ONNX, others warn of limitations for development |
| **Build order** | HIGH | Natural dependencies clear from component purposes |

**Sources used:**
- PyTorch official tutorials (model structure, saving/loading)
- PyTorch forums (conversion discussions)
- Community articles (manual conversion workflows)
- Professional project templates (timm, Deep-Learning-Project-Template)
- ONNX documentation (tf2onnx for Keras)

## Gaps and Unknowns

**Low-risk gaps:**
1. **Exact ResNet8 architecture details** - Will need to inspect .h5 file to confirm layer structure
2. **Keras version differences** - Older Keras (standalone) vs tf.keras may have minor API differences
3. **Performance benchmarks** - No data on conversion time or evaluation speed for this specific model

**These gaps:**
- Are normal for pre-implementation research
- Will be resolved during implementation with .h5 file inspection
- Don't affect architecture recommendations (patterns are framework-agnostic)

**No critical blockers identified.**

## Sources

### High Confidence (Official Documentation)
- [PyTorch Model Definition Tutorial](https://docs.pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)
- [PyTorch Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [TensorFlow ONNX Converter (tf2onnx)](https://github.com/onnx/tensorflow-onnx)
- [PyTorch torchvision Models](https://docs.pytorch.org/vision/main/models.html)

### Medium Confidence (Verified Community Sources)
- [PyTorch Forums: Keras to PyTorch Conversion](https://discuss.pytorch.org/t/keras-to-pytorch-model-conversion/155153)
- [PyTorch Forums: Transferring Weights from Keras](https://discuss.pytorch.org/t/transferring-weights-from-keras-to-pytorch/9889)
- [Medium: Load Keras Weight to PyTorch](https://medium.com/analytics-vidhya/load-keras-weight-to-pytorch-and-transform-keras-architecture-to-pytorch-easily-8ff5dd18b86b)
- [Deep Learning Project Template](https://github.com/L1aoXingyu/Deep-Learning-Project-Template)
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models)

### Low Confidence (WebSearch, Needs Validation)
- [Pitfalls Porting Models Between Frameworks](https://shaoanlu.wordpress.com/2019/05/23/pitfalls-encountered-porting-models-to-keras-from-pytorch-and-tensorflow/) - Specific technical details should be verified
- [Deep Learning Model Convertor Collection](https://github.com/ysh329/deep-learning-model-convertor) - General ecosystem overview
- ONNX as primary conversion path - Some sources recommend, others suggest manual conversion for development

**Note on source recency:** Most conversion guidance is from 2019-2022. PyTorch and Keras APIs have remained stable in core conversion concepts, but specific tool versions should be checked during implementation.
