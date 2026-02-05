# Domain Pitfalls: Marimo Quantization Playground

**Domain:** Interactive Marimo notebook for quantization parameter inspection and modification
**Context:** ResNet8 ONNX and PyTorch quantized models, CIFAR-10 evaluation
**Purpose:** Prevent common mistakes when building interactive ML experimentation tools
**Researched:** 2026-02-05
**Confidence:** HIGH (Marimo official docs, PyTorch quantization APIs, ONNX Runtime documentation)

---

## Critical Pitfalls

Mistakes that cause broken interactivity, incorrect results, or unusable notebooks.

---

### Pitfall 1: Model Reloading on Every Cell Rerun Exhausts Memory

**What goes wrong:** Loading ONNX or PyTorch models inside reactive cells causes the model to reload every time a UI slider changes, eventually exhausting memory and crashing the kernel.

**Why it happens:**
- Marimo's reactive execution reruns cells when dependencies change
- Model loading cell depends on a file path or configuration variable
- Any UI element change that triggers the dependency chain causes full reload
- Each reload creates a new model instance in memory
- ONNX Runtime sessions and PyTorch models are not garbage collected immediately

**This project's exposure:**
```python
# WRONG: Model reloads every time any upstream variable changes
model_path = "models/resnet8_int8.onnx"  # Cell A
model = onnx.load(model_path)             # Cell B (reruns when A changes)
slider = mo.ui.slider(1, 100)             # Cell C
# If slider is accidentally referenced in A or B, model reloads on every slider move
```

**Consequences:**
- Memory usage grows with each interaction
- Eventually kernel crashes with OOM error
- Notebook becomes unusable after a few slider adjustments
- GPU memory not released (if using PyTorch with CUDA)
- User must restart kernel frequently

**Prevention:**

1. **Isolate model loading in dedicated cells with no UI dependencies:**
   ```python
   # Cell 1: Constants only (no UI elements)
   MODEL_PATH = "models/resnet8_int8.onnx"

   # Cell 2: Model loading (only depends on constants)
   @mo.cache  # Cache the expensive load
   def load_model(path):
       return onnx.load(path)

   model = load_model(MODEL_PATH)
   ```

2. **Use mo.cache for expensive operations:**
   ```python
   @mo.cache
   def get_onnx_session(model_path):
       """Cache ONNX Runtime session - only creates once per path."""
       return ort.InferenceSession(model_path)

   session = get_onnx_session("models/resnet8_int8.onnx")
   ```

3. **Separate UI cells from model cells:**
   ```python
   # Cell A: UI elements (can change freely)
   scale_slider = mo.ui.slider(0.001, 0.1, value=0.01, label="Scale")

   # Cell B: Model loading (does NOT reference scale_slider)
   model = load_cached_model()  # No reactive dependency on slider

   # Cell C: Computation (combines model + slider)
   result = modify_and_run(model, scale_slider.value)  # This cell reruns, not Cell B
   ```

**Detection:**

Warning signs during development:
- Memory usage steadily increases when moving sliders
- Kernel becomes unresponsive after extended use
- "Loading model..." message appears repeatedly in output
- Variables panel shows multiple model instances

**Phase to address:** Phase 1 (Notebook Setup) - Establish model loading pattern before adding interactivity

**Sources:**
- [Marimo Expensive Notebooks Guide](https://docs.marimo.io/guides/expensive_notebooks/)
- [Marimo Caching with mo.cache](https://docs.marimo.io/guides/best_practices/)

---

### Pitfall 2: Object Mutations Not Triggering Reactive Updates

**What goes wrong:** Modifying quantization parameters in-place (mutating objects) doesn't trigger downstream cell updates, so visualizations and evaluations don't refresh.

**Why it happens:**
- Marimo tracks variable assignments, not object mutations
- In-place modifications like `tensor[0] = new_value` or `dict["key"] = value` are invisible to the reactive engine
- ONNX graph modifications and PyTorch weight updates are typically in-place
- User changes a parameter but sees no update in downstream visualizations

**This project's exposure:**
```python
# WRONG: In-place mutation not tracked
scales = extract_scales(model)  # Cell A: extract scales dict
scales["conv1_scale"] = 0.05    # Cell B: modify in place - NOT TRACKED
display_scales(scales)          # Cell C: doesn't rerun after Cell B changes
```

**Consequences:**
- Visualizations show stale data
- User confusion: "I changed the parameter but nothing happened"
- Inconsistent notebook state
- Debugging nightmare: code looks correct but doesn't work

**Prevention:**

1. **Create new objects instead of mutating:**
   ```python
   # CORRECT: Create new dict with modified value
   original_scales = extract_scales(model)

   # In modification cell:
   modified_scales = {
       **original_scales,
       "conv1_scale": slider.value  # New value from UI
   }  # This is a new object, triggers reactivity
   ```

2. **Use mo.state for mutable state that must persist:**
   ```python
   # For state that needs to persist and trigger updates
   get_scales, set_scales = mo.state(initial_scales)

   # To update:
   set_scales({**get_scales(), "conv1_scale": new_value})
   ```

3. **Return modified copies from functions:**
   ```python
   def modify_scale(model_params: dict, layer_name: str, new_scale: float) -> dict:
       """Return a new dict with modified scale (don't mutate in place)."""
       return {**model_params, layer_name: new_scale}

   # Usage:
   updated_params = modify_scale(params, "conv1", slider.value)
   ```

4. **For ONNX models, create modified copies:**
   ```python
   import copy

   def modify_onnx_scale(model, node_name, new_scale):
       """Create modified model copy."""
       modified = copy.deepcopy(model)
       # Modify the copy
       for init in modified.graph.initializer:
           if init.name == node_name:
               # Update the initializer
               ...
       return modified  # Return new model, don't mutate original
   ```

**Detection:**

Test during development:
- Change a slider value - does the visualization update?
- Check Marimo's dependency graph - is the downstream cell connected?
- Add print statements in downstream cells - do they print on parameter change?

**Warning signs:**
- Cells have no connecting edges in the dependency graph
- UI changes don't trigger any cell execution
- "Stale" indicators don't appear on dependent cells

**Phase to address:** Phase 2 (Parameter Inspection) - Design data flow patterns before implementing interactions

**Sources:**
- [Marimo Best Practices - Mutations](https://docs.marimo.io/guides/best_practices/)
- [Marimo Troubleshooting - Cells Not Running](https://docs.marimo.io/guides/troubleshooting/)

---

### Pitfall 3: UI Element Values Reset When Definition Cell Reruns

**What goes wrong:** User adjusts a slider, then an unrelated change causes the slider's definition cell to rerun, resetting the slider to its initial value and losing user input.

**Why it happens:**
- UI elements are recreated when their definition cell runs
- New UI element has default `value=` parameter, not the user's selection
- Cell reruns can be triggered by any upstream dependency change
- Users don't expect their manual adjustments to disappear

**This project's exposure:**
```python
# PROBLEMATIC: Slider defined with formatting that depends on model
model = load_model()  # Cell A
layer_names = get_layer_names(model)  # Cell B

# Cell C: Slider definition depends on layer_names
layer_selector = mo.ui.dropdown(layer_names, label="Select Layer")
scale_slider = mo.ui.slider(0.001, 0.1, value=0.01, label="Scale")

# If model reloads or layer_names changes, both UI elements reset!
```

**Consequences:**
- User frustration: spending time adjusting parameters, then losing them
- Workflow interruption: must re-enter values repeatedly
- Makes experimentation tedious
- Users may stop using interactive features

**Prevention:**

1. **Isolate UI definitions in cells with minimal dependencies:**
   ```python
   # Cell A: UI elements with ONLY static configuration
   scale_slider = mo.ui.slider(0.001, 0.1, value=0.01, label="Scale")
   layer_dropdown = mo.ui.dropdown(
       ["conv1", "conv2", "conv3"],  # Static list, not computed
       label="Layer"
   )

   # Cell B: Model loading (completely separate)
   model = load_model()

   # Cell C: Combine UI values with model (this cell reruns, UI cells don't)
   selected_layer = layer_dropdown.value
   result = process(model, selected_layer, scale_slider.value)
   ```

2. **Use mo.state to persist values across cell reruns:**
   ```python
   # Persist slider value in state
   get_scale, set_scale = mo.state(0.01)

   # Slider that syncs with state
   scale_slider = mo.ui.slider(
       0.001, 0.1,
       value=get_scale(),
       on_change=lambda v: set_scale(v)
   )
   ```

3. **Compute dropdown options separately from dropdown definition:**
   ```python
   # Cell A: Compute options (may rerun)
   layer_names = get_layer_names(model)

   # Cell B: Dropdown definition (only reruns if layer_names changes structure)
   layer_dropdown = mo.ui.dropdown(layer_names, label="Layer")
   # Note: If layer_names content changes but structure is same, consider mo.state
   ```

4. **Use forms to batch UI elements:**
   ```python
   # Form batches updates - only triggers when submit is clicked
   param_form = mo.ui.batch(
       scale=mo.ui.slider(0.001, 0.1),
       layer=mo.ui.dropdown(["conv1", "conv2"]),
   ).form()

   # Access with param_form.value["scale"], param_form.value["layer"]
   ```

**Detection:**

During testing:
- Make a notebook change and check if sliders reset
- Observe which cells rerun when you make changes
- Use Marimo's dependency graph to trace UI element dependencies

**Warning signs:**
- Users complaining about lost settings
- Slider values jumping back to defaults
- UI elements flickering during computation

**Phase to address:** Phase 3 (Interactive Modification) - Design UI element isolation before building complex interactions

**Sources:**
- [Marimo FAQ - UI Elements](https://docs.marimo.io/faq/)
- [Marimo Troubleshooting - UI Values Resetting](https://docs.marimo.io/guides/troubleshooting/)

---

### Pitfall 4: Capturing Intermediate Values Requires Model Surgery

**What goes wrong:** Attempting to capture intermediate layer outputs during ONNX or PyTorch inference requires non-trivial model modifications, not simple "print at this layer" debugging.

**Why it happens:**
- ONNX Runtime doesn't expose intermediate tensors by default
- PyTorch quantized models don't have simple forward hooks on quantized layers
- Model graphs are compiled/optimized, hiding intermediate nodes
- Users expect Jupyter-like ability to inspect any variable

**This project's exposure:**

The existing codebase uses ONNX QDQ format which wraps operations:
- 98 QDQ nodes (32 QuantizeLinear + 66 DequantizeLinear)
- Intermediate values flow through the graph but aren't exposed
- PyTorch JIT traced model has similar opacity

```python
# User expectation (doesn't work):
model = load_model()
output = model.run(input)
print(model.intermediate["conv1_output"])  # No such API!

# What's actually required:
# 1. Modify ONNX graph to add intermediate outputs
# 2. Or use specialized debugging APIs
```

**Consequences:**
- Significant implementation effort to capture intermediates
- May need to maintain two model versions (normal + debug)
- Performance overhead from capturing all intermediate values
- Memory explosion if capturing all activations

**Prevention:**

1. **Use ONNX Runtime's quantization debug module:**
   ```python
   from onnxruntime.quantization.qdq_loss_debug import (
       modify_model_output_intermediate_tensors,
       collect_activations
   )

   # Augment model to expose all intermediate tensors
   augmented_model = modify_model_output_intermediate_tensors(model)

   # Collect activations during inference
   activations = collect_activations(augmented_model, calibration_reader)
   ```

2. **Add specific intermediate outputs to ONNX graph:**
   ```python
   import onnx

   def add_intermediate_output(model, tensor_name):
       """Add a tensor as a model output for inspection."""
       # Find the tensor in the graph
       for value_info in model.graph.value_info:
           if value_info.name == tensor_name:
               model.graph.output.append(value_info)
               return model

       # If not found, create value_info (may need shape inference first)
       new_output = onnx.helper.make_tensor_value_info(
           tensor_name, onnx.TensorProto.FLOAT, None
       )
       model.graph.output.append(new_output)
       return model
   ```

3. **For PyTorch, use register_forward_hook on the non-quantized model first:**
   ```python
   # Before quantization: attach hooks
   intermediates = {}

   def capture_hook(name):
       def hook(module, input, output):
           intermediates[name] = output.detach()
       return hook

   model.conv1.register_forward_hook(capture_hook("conv1"))
   # Then quantize
   ```

4. **Create a dedicated "debug mode" model variant:**
   ```python
   def create_debug_model(model_path, layers_to_capture):
       """Create model variant with intermediate outputs for playground use."""
       model = onnx.load(model_path)
       for layer in layers_to_capture:
           add_intermediate_output(model, layer)
       debug_path = model_path.replace(".onnx", "_debug.onnx")
       onnx.save(model, debug_path)
       return debug_path
   ```

**Detection:**

Early warning during planning:
- List the intermediate values you want to capture
- Check if those tensors are exposed in the model format
- Test capture mechanism on a small model first

**Warning signs:**
- Simple print statements don't show intermediate values
- Model.run() returns only final outputs
- Documentation mentions "model modification required"

**Phase to address:** Phase 2 (Parameter Inspection) - Implement intermediate capture infrastructure before building visualizations

**Sources:**
- [ONNX Runtime Quantization Debug](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [sklearn-onnx Intermediate Results](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_fbegin_investigate.html)
- [PyTorch Register Forward Hook](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)

---

### Pitfall 5: Modifying Quantized Weights Requires Understanding Internal Representation

**What goes wrong:** Attempting to modify quantized model parameters using familiar PyTorch patterns fails because quantized tensors have special internal representation.

**Why it happens:**
- Quantized weights are stored as `torch.qint8`, not regular tensors
- Direct assignment to `state_dict()` doesn't update actual weights
- ONNX initializers require specific modification patterns
- Scale and zero-point are tied to weight tensors

**This project's exposure:**

From existing `quantize_pytorch.py`:
```python
# PyTorch quantized model uses FX mode with JIT tracing
# Weights are qint8 with attached scale/zero_point
```

From existing `quantize_onnx.py`:
```python
# ONNX model has QDQ format with initializers
# Scale and zero-point stored as separate initializers
```

**Consequences:**
- Modifications appear to succeed but don't affect inference
- User thinks they changed a parameter but model behaves identically
- Debugging shows "correct" modified values but wrong outputs
- Confusion between display values and actual model state

**Prevention:**

1. **For PyTorch quantized weights, access directly through module:**
   ```python
   # WRONG: state_dict is read-only copy
   state = model.state_dict()
   state['layer.weight'] = new_weight  # Doesn't affect model!

   # CORRECT: Access through module directly
   model.features[0].weight = new_quantized_weight
   ```

2. **Reconstruct quantized tensors properly:**
   ```python
   import torch

   def modify_quantized_weight(weight_tensor, modification_fn):
       """Safely modify a quantized weight tensor."""
       # Get integer representation
       int_repr = weight_tensor.int_repr()
       scale = weight_tensor.q_per_channel_scales()
       zero_point = weight_tensor.q_per_channel_zero_points()
       axis = weight_tensor.q_per_channel_axis()

       # Apply modification to int representation
       modified_int = modification_fn(int_repr)

       # Reconstruct quantized tensor
       return torch._make_per_channel_quantized_tensor(
           modified_int, scale, zero_point, axis
       )
   ```

3. **For ONNX models, modify initializers directly:**
   ```python
   import numpy as np
   from onnx import numpy_helper

   def modify_onnx_initializer(model, initializer_name, new_value):
       """Modify an ONNX model initializer (scale, zero-point, or weight)."""
       for i, init in enumerate(model.graph.initializer):
           if init.name == initializer_name:
               # Create new initializer with modified value
               new_init = numpy_helper.from_array(
                   new_value.astype(numpy_helper.to_array(init).dtype),
                   name=initializer_name
               )
               model.graph.initializer[i].CopyFrom(new_init)
               return model
       raise ValueError(f"Initializer {initializer_name} not found")
   ```

4. **Verify modifications with inference test:**
   ```python
   def verify_modification(model, test_input, expected_change):
       """Verify that model modification actually changed outputs."""
       output_before = run_inference(model, test_input)
       # Apply modification
       modify_parameter(model, ...)
       output_after = run_inference(model, test_input)

       # Check that outputs actually changed
       assert not np.allclose(output_before, output_after), \
           "Modification had no effect - check modification pattern"
   ```

**Detection:**

During development:
- Run inference before and after modification - do outputs change?
- Print the modified parameter value - does it show expected value?
- Check model.state_dict() vs model.layer.weight - are they the same?

**Warning signs:**
- Modifications "succeed" silently but outputs are unchanged
- Documentation warns about "read-only copies"
- Quantized tensor operations behave differently than regular tensors

**Phase to address:** Phase 3 (Interactive Modification) - Implement and verify modification patterns before building UI

**Sources:**
- [PyTorch Forum: Changing Quantized Weights](https://discuss.pytorch.org/t/changing-quantized-weights/109060)
- [PyTorch Quantization API](https://pytorch.org/docs/stable/quantization-support.html)

---

## Moderate Pitfalls

Mistakes that cause confusion, performance issues, or maintenance burden.

---

### Pitfall 6: Slider Interactions Trigger Full Recomputation

**What goes wrong:** Moving a slider causes expensive recomputation even when only a display update is needed, making the UI sluggish and frustrating.

**Why it happens:**
- Every slider value change triggers reactive cell execution
- If inference is in the reactive chain, it runs on every slider move
- No debouncing by default in Marimo
- Users expect instant feedback from sliders

**This project's exposure:**
```python
# SLOW: Slider directly triggers inference
scale_slider = mo.ui.slider(0.001, 0.1)
modified_model = modify_scale(model, scale_slider.value)  # Runs on every slide
result = run_inference(modified_model, test_data)          # EXPENSIVE - runs every time!
display_result(result)
```

**Consequences:**
- UI feels unresponsive
- Sliders stutter and lag
- Users avoid using interactive features
- May trigger memory issues from rapid model operations

**Prevention:**

1. **Use mo.ui.run_button to require explicit execution:**
   ```python
   scale_slider = mo.ui.slider(0.001, 0.1, label="Scale")
   run_button = mo.ui.run_button(label="Run Inference")

   mo.stop(not run_button.value, "Click 'Run Inference' to execute")

   # Only runs when button is clicked
   result = run_inference(modified_model, test_data)
   ```

2. **Use .form() to batch slider changes:**
   ```python
   param_form = mo.ui.batch(
       scale=mo.ui.slider(0.001, 0.1),
       zero_point=mo.ui.slider(-128, 127),
   ).form(submit_button_label="Apply Changes")

   # Only runs when form is submitted
   modified_model = apply_params(model, param_form.value)
   ```

3. **Separate display from computation:**
   ```python
   # Cell A: Slider (changes freely)
   scale_slider = mo.ui.slider(0.001, 0.1)

   # Cell B: Display current value (fast, no inference)
   mo.md(f"Selected scale: {scale_slider.value}")

   # Cell C: Inference (only runs with explicit trigger)
   run_btn = mo.ui.run_button()
   mo.stop(not run_btn.value)
   result = run_inference(model, scale=scale_slider.value)  # Heavy work
   ```

4. **Cache inference results:**
   ```python
   @mo.cache
   def cached_inference(model_path, scale, zero_point):
       """Cache inference results for parameter combinations."""
       model = load_and_modify(model_path, scale, zero_point)
       return run_inference(model, test_data)

   # Repeated calls with same params use cache
   result = cached_inference("model.onnx", scale_slider.value, zp_slider.value)
   ```

**Detection:**

During development:
- Move a slider slowly - does UI respond smoothly?
- Check execution times in Marimo's status bar
- Use profiling to identify slow cells

**Warning signs:**
- Visible lag between slider movement and display update
- "Running..." indicator stays on for seconds during slider adjustment
- Kernel CPU usage spikes when touching sliders

**Phase to address:** Phase 3 (Interactive Modification) - Add execution controls before implementing modification logic

**Sources:**
- [Marimo Expensive Notebooks - mo.stop](https://docs.marimo.io/guides/expensive_notebooks/)
- [Marimo UI Forms](https://docs.marimo.io/api/inputs/form.html)

---

### Pitfall 7: PyTorch and ONNX Runtime Models Have Different Parameter Access Patterns

**What goes wrong:** Code written for ONNX model parameter access fails on PyTorch model, or vice versa, because the two frameworks have fundamentally different APIs.

**Why it happens:**
- ONNX stores parameters as graph initializers (static tensors)
- PyTorch stores parameters as module attributes (dynamic tensors)
- ONNX uses string names for lookup; PyTorch uses module hierarchy
- Quantization adds additional complexity in both frameworks

**This project's exposure:**

The project has both:
- `models/resnet8_int8.onnx` - ONNX QDQ format
- `models/resnet8_int8.pt` - PyTorch JIT TorchScript

```python
# ONNX parameter access
for init in onnx_model.graph.initializer:
    if "scale" in init.name:
        scale = numpy_helper.to_array(init)

# PyTorch parameter access (completely different!)
for name, param in pytorch_model.named_parameters():
    if "scale" in name:
        scale = param.data
```

**Consequences:**
- Duplicate code for each framework
- Subtle bugs from framework-specific behavior
- Maintenance burden when updating either framework
- Confusion about which model is being modified

**Prevention:**

1. **Create unified abstraction layer:**
   ```python
   class QuantizedModel:
       """Unified interface for ONNX and PyTorch quantized models."""

       def __init__(self, path):
           self.path = path
           self.framework = "onnx" if path.endswith(".onnx") else "pytorch"
           self._load()

       def get_scales(self) -> Dict[str, np.ndarray]:
           if self.framework == "onnx":
               return self._get_onnx_scales()
           else:
               return self._get_pytorch_scales()

       def set_scale(self, layer_name: str, value: np.ndarray):
           if self.framework == "onnx":
               self._set_onnx_scale(layer_name, value)
           else:
               self._set_pytorch_scale(layer_name, value)
   ```

2. **Document framework differences explicitly:**
   ```python
   def get_layer_names(model, framework: str) -> List[str]:
       """
       Get layer names from model.

       Note: ONNX uses node names like 'QuantizeLinear_0'
             PyTorch uses module paths like 'features.0.weight'
       """
       ...
   ```

3. **Use framework-specific modules:**
   ```python
   # onnx_utils.py
   def get_onnx_scales(model): ...
   def set_onnx_scale(model, name, value): ...

   # pytorch_utils.py
   def get_pytorch_scales(model): ...
   def set_pytorch_scale(model, name, value): ...

   # main.py
   if model_type == "onnx":
       from onnx_utils import get_onnx_scales as get_scales
   else:
       from pytorch_utils import get_pytorch_scales as get_scales
   ```

4. **Add model type indicator in UI:**
   ```python
   # Clear indication of which model is being inspected
   model_selector = mo.ui.dropdown(
       ["ONNX (int8)", "ONNX (uint8)", "PyTorch (int8)"],
       label="Model"
   )

   mo.md(f"**Currently inspecting:** {model_selector.value}")
   ```

**Detection:**

During development:
- Test every feature with both ONNX and PyTorch models
- Check for framework-specific imports in shared code
- Verify parameter names match between frameworks

**Warning signs:**
- Feature works for one model but fails for another
- KeyError or AttributeError with framework-specific messages
- Inconsistent parameter names in UI

**Phase to address:** Phase 2 (Parameter Inspection) - Design abstraction layer before implementing inspection features

**Sources:**
- [ONNX Python API](https://onnx.ai/onnx/repo-docs/PythonAPIOverview.html)
- [PyTorch Quantization API](https://pytorch.org/docs/stable/quantization-support.html)

---

### Pitfall 8: Accuracy Evaluation Takes Too Long for Interactive Use

**What goes wrong:** Running full CIFAR-10 evaluation (10,000 images) after each parameter change makes the interactive experience unusable.

**Why it happens:**
- Full evaluation takes minutes on CPU
- Users expect sub-second feedback from parameter changes
- Natural temptation to show "real" accuracy after each modification
- No built-in way to do incremental or approximate evaluation

**This project's exposure:**

From existing `evaluate.py`:
- Full CIFAR-10 test set: 10,000 images
- Current evaluation time: ~2-5 minutes on CPU
- Would need to run after each parameter change for "real" accuracy

**Consequences:**
- Interactive workflow becomes batch workflow
- Users lose context waiting for evaluation
- May lead to skipping evaluation entirely
- Frustration with tool usability

**Prevention:**

1. **Use stratified mini-batch for interactive feedback:**
   ```python
   def quick_evaluate(model, num_samples=100):
       """Fast evaluation on stratified sample (10 images per class)."""
       indices = stratified_sample(cifar10_test, samples_per_class=10)
       correct = 0
       for idx in indices:
           image, label = cifar10_test[idx]
           pred = run_inference(model, image)
           correct += (pred == label)
       return correct / num_samples  # Approximate accuracy

   # Use for interactive feedback (< 1 second)
   quick_acc = quick_evaluate(modified_model)
   mo.md(f"Quick accuracy (100 samples): {quick_acc:.1%}")
   ```

2. **Separate quick preview from full evaluation:**
   ```python
   # Cell A: Quick feedback (runs on slider change)
   quick_acc = quick_evaluate(model, num_samples=50)

   # Cell B: Full evaluation (requires explicit trigger)
   full_eval_btn = mo.ui.run_button(label="Run Full Evaluation (2-5 min)")
   mo.stop(not full_eval_btn.value)
   full_acc = full_evaluate(model)  # All 10,000 images
   ```

3. **Pre-compute evaluation on parameter grid:**
   ```python
   @mo.persistent_cache
   def precompute_accuracy_grid(model_path, scale_range, zp_range):
       """Pre-compute accuracy for parameter combinations."""
       results = {}
       for scale in scale_range:
           for zp in zp_range:
               model = load_and_modify(model_path, scale, zp)
               results[(scale, zp)] = full_evaluate(model)
       return results

   # Use cached grid for instant lookup
   grid = precompute_accuracy_grid(...)
   current_acc = grid[(scale_slider.value, zp_slider.value)]
   ```

4. **Show confidence interval for quick evaluation:**
   ```python
   def quick_evaluate_with_ci(model, num_samples=100, confidence=0.95):
       """Quick evaluation with confidence interval."""
       acc = quick_evaluate(model, num_samples)
       # Binomial confidence interval
       z = 1.96  # 95% CI
       ci = z * np.sqrt(acc * (1 - acc) / num_samples)
       return acc, ci

   acc, ci = quick_evaluate_with_ci(model)
   mo.md(f"Accuracy: {acc:.1%} +/- {ci:.1%} (n={num_samples})")
   ```

**Detection:**

During design:
- Estimate time for full evaluation
- Calculate required samples for acceptable confidence
- Test quick evaluation correlation with full evaluation

**Warning signs:**
- Users waiting more than 2 seconds for feedback
- Evaluation cell dominates execution time
- Users bypassing accuracy display

**Phase to address:** Phase 4 (Comparison) - Implement tiered evaluation before building comparison features

**Sources:**
- [Marimo Expensive Notebooks](https://docs.marimo.io/guides/expensive_notebooks/)
- Project context: 87.19% baseline accuracy on CIFAR-10

---

## Minor Pitfalls

Mistakes that cause inconvenience but are easily fixable.

---

### Pitfall 9: Closure Variable Capture in Lambda Functions

**What goes wrong:** Creating multiple UI elements in a loop causes all callbacks to reference the same (last) variable value.

**Why it happens:**
- Python lambdas capture variables by reference, not value
- By the time lambda executes, loop variable has final value
- Common when creating sliders for multiple layers dynamically

**This project's exposure:**
```python
# WRONG: All callbacks reference final layer_name
sliders = {}
for layer_name in layer_names:
    sliders[layer_name] = mo.ui.slider(
        0.001, 0.1,
        on_change=lambda v: update_layer(layer_name, v)  # BUG!
    )
# All sliders update the LAST layer_name
```

**Prevention:**

1. **Bind loop variable explicitly:**
   ```python
   for layer_name in layer_names:
       sliders[layer_name] = mo.ui.slider(
           0.001, 0.1,
           on_change=lambda v, name=layer_name: update_layer(name, v)
       )
   ```

2. **Use dict comprehension with factory function:**
   ```python
   def make_slider(name):
       return mo.ui.slider(
           0.001, 0.1,
           on_change=lambda v: update_layer(name, v)
       )

   sliders = {name: make_slider(name) for name in layer_names}
   ```

**Phase to address:** Phase 2 - Code review pattern for loop-created UI elements

**Sources:**
- [Marimo FAQ - Lambda Closure](https://docs.marimo.io/faq/)
- [Python Closures in Loops](https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result)

---

### Pitfall 10: Namespace Pollution with Intermediate Variables

**What goes wrong:** Many intermediate variables in global scope cause name collisions and clutter the variables panel.

**Why it happens:**
- Marimo cells share global namespace
- Data processing creates many intermediate variables
- Variables panel becomes unusable
- Easy to accidentally overwrite important variables

**This project's exposure:**
```python
# Cell 1
df = load_data()
filtered = df[df.value > 0]
normalized = filtered / filtered.max()
scaled = normalized * 255

# Cell 2
# Accidentally reuses 'filtered' for different purpose
filtered = model.graph.node[0]  # Overwrites previous 'filtered'!
```

**Prevention:**

1. **Use underscore prefix for local variables:**
   ```python
   # Cell 1
   _df = load_data()
   _filtered = _df[_df.value > 0]
   _normalized = _filtered / _filtered.max()
   processed_data = _normalized * 255  # Only export final result
   ```

2. **Encapsulate in functions:**
   ```python
   def process_data(raw_df):
       filtered = raw_df[raw_df.value > 0]
       normalized = filtered / filtered.max()
       return normalized * 255

   processed_data = process_data(load_data())
   ```

3. **Delete intermediates explicitly:**
   ```python
   df = load_data()
   filtered = df[df.value > 0]
   result = filtered / filtered.max()
   del df, filtered  # Clean up
   ```

**Phase to address:** Phase 1 - Establish naming conventions in notebook setup

**Sources:**
- [Marimo Best Practices - Variable Management](https://docs.marimo.io/guides/best_practices/)

---

### Pitfall 11: Missing Dependency Installation in Notebook Environment

**What goes wrong:** Notebook imports packages that aren't installed in the notebook's Python environment, causing ImportError.

**Why it happens:**
- Development machine has packages installed globally
- Notebook doesn't specify dependencies
- User runs notebook in fresh environment
- Optional visualization packages (plotly, altair) not in core requirements

**This project's exposure:**

Required packages for playground:
- `marimo` (notebook framework)
- `onnx` (model loading)
- `onnxruntime` (inference)
- `torch` (PyTorch model loading)
- `numpy` (data manipulation)
- `matplotlib` or `plotly` (visualization)

```python
# May fail if plotly not installed
import plotly.express as px  # ImportError in fresh environment
```

**Prevention:**

1. **Use Marimo's sandbox mode for dependency management:**
   ```python
   # At top of notebook, specify requirements
   # /// script
   # dependencies = [
   #   "onnx>=1.17.0",
   #   "onnxruntime>=1.23.2",
   #   "torch>=2.0.0",
   #   "numpy",
   #   "plotly",
   # ]
   # ///
   ```

2. **Add graceful fallbacks for optional packages:**
   ```python
   try:
       import plotly.express as px
       HAS_PLOTLY = True
   except ImportError:
       HAS_PLOTLY = False
       import matplotlib.pyplot as plt

   def plot_accuracy(data):
       if HAS_PLOTLY:
           return px.line(data)
       else:
           plt.plot(data)
           return plt.gcf()
   ```

3. **Update pyproject.toml with playground dependencies:**
   ```toml
   [project.optional-dependencies]
   playground = [
       "marimo>=0.10.0",
       "plotly>=5.0.0",
   ]
   ```

**Phase to address:** Phase 1 - Add dependencies before development starts

**Sources:**
- [Marimo Package Management](https://docs.marimo.io/guides/package_management/)

---

## Phase-Specific Warning Matrix

| Phase | Likely Pitfall | Priority | Mitigation |
|-------|---------------|----------|------------|
| **1. Notebook Setup** | Model reloading (#1) | CRITICAL | Use mo.cache, isolate loading cells |
| **1. Notebook Setup** | Namespace pollution (#10) | LOW | Establish naming conventions |
| **1. Notebook Setup** | Missing dependencies (#11) | LOW | Specify in pyproject.toml |
| **2. Parameter Inspection** | Intermediate capture (#4) | HIGH | Plan capture infrastructure first |
| **2. Parameter Inspection** | Framework differences (#7) | MEDIUM | Design abstraction layer |
| **2. Parameter Inspection** | Mutations not tracked (#2) | HIGH | Use immutable patterns |
| **3. Interactive Modification** | UI values reset (#3) | HIGH | Isolate UI definition cells |
| **3. Interactive Modification** | Weight modification (#5) | CRITICAL | Verify modification patterns |
| **3. Interactive Modification** | Slider performance (#6) | MEDIUM | Add run buttons, use forms |
| **3. Interactive Modification** | Lambda capture (#9) | LOW | Code review for loops |
| **4. Comparison** | Evaluation speed (#8) | HIGH | Implement tiered evaluation |

---

## Testing Checklist for Marimo Notebooks

### Reactivity Testing
- [ ] Model loading cell does NOT rerun when sliders change
- [ ] Parameter modifications trigger downstream visualization updates
- [ ] UI element values persist when unrelated cells rerun
- [ ] Circular dependencies are not created (check with `marimo check`)

### Performance Testing
- [ ] Slider interactions respond in < 500ms
- [ ] Quick evaluation completes in < 2 seconds
- [ ] Memory usage is stable over extended interaction sessions
- [ ] No "Loading model..." messages during parameter adjustment

### Correctness Testing
- [ ] Modified parameters actually affect inference output
- [ ] Both ONNX and PyTorch models work with all features
- [ ] Accuracy evaluation correlates with full test set
- [ ] Parameter values displayed match values in model

### Usability Testing
- [ ] Clear indication of which model is being inspected
- [ ] Progress indicators for long-running operations
- [ ] Undo/reset functionality for parameter changes
- [ ] Help text for complex interactions

---

## Research Sources

### Marimo Documentation (HIGH confidence)
- [Best Practices](https://docs.marimo.io/guides/best_practices/) - Official mutation and state management guidance
- [Expensive Notebooks](https://docs.marimo.io/guides/expensive_notebooks/) - Caching and execution control
- [Troubleshooting](https://docs.marimo.io/guides/troubleshooting/) - Common problems and solutions
- [FAQ](https://docs.marimo.io/faq/) - UI element patterns and gotchas

### PyTorch Quantization (HIGH confidence)
- [Changing Quantized Weights](https://discuss.pytorch.org/t/changing-quantized-weights/109060) - Weight modification patterns
- [Quantization API Reference](https://pytorch.org/docs/stable/quantization-support.html) - Official API documentation
- [Neural Network Quantization in PyTorch](https://arikpoz.github.io/posts/2025-04-16-neural-network-quantization-in-pytorch/) - Practical guide (2025)

### ONNX Runtime (HIGH confidence)
- [Quantization Documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) - Official quantization guide
- [Intermediate Results Investigation](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_fbegin_investigate.html) - Capturing intermediate values
- [ONNX Python API](https://onnx.ai/onnx/repo-docs/PythonAPIOverview.html) - Model manipulation

### Memory and Performance (MEDIUM confidence)
- [Memory-Efficient Model Loading](https://www.analyticsvidhya.com/blog/2024/10/memory-efficient-model-weight-loading-in-pytorch/) - PyTorch memory patterns
- [GPU Memory Issues](https://discuss.pytorch.org/t/running-out-of-gpu-memory-when-loading-a-huggingface-model/217053) - Common memory problems

---

**Confidence Assessment:** HIGH

- Marimo pitfalls verified with official documentation
- PyTorch quantization patterns from official forum discussions
- ONNX intermediate capture from official tutorials
- Integration pitfalls derived from project-specific codebase analysis

**Research Gaps:**

- No Marimo-specific examples for ML experimentation workflows (community is young)
- Limited documentation on Marimo + ONNX Runtime integration
- No benchmarks for reactive notebook performance with heavy ML workloads

These gaps are acceptable because the core patterns (reactivity, mutations, caching) are well-documented, and the ML-specific considerations can be derived from general best practices.
