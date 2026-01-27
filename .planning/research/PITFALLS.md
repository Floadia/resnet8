# Domain Pitfalls: Keras to PyTorch Model Conversion

**Domain:** Deep learning model conversion (Keras H5 to PyTorch)
**Researched:** 2026-01-27
**Confidence:** MEDIUM (based on community discussions and technical documentation)

## Critical Pitfalls

Mistakes that cause incorrect inference results or require complete rework.

### Pitfall 1: Channel Order Mismatch (NHWC vs NCHW)
**What goes wrong:** Keras uses channels_last format (NHWC: batch, height, width, channels) while PyTorch uses channels_first (NCHW: batch, channels, height, width). Weight tensors are loaded without transposition, causing silently incorrect computations.

**Why it happens:** Both frameworks accept 4D tensors, so no error is raised. The model runs but produces garbage outputs because spatial dimensions are interpreted as channels and vice versa.

**Consequences:**
- Model produces random predictions (near-chance accuracy)
- No error messages — silent failure
- Debugging is difficult because layer shapes appear correct
- For Conv2D layers, Keras weights are (height, width, input_channels, output_channels) while PyTorch expects (output_channels, input_channels, height, width)

**Prevention:**
1. Always transpose Conv2D weights from Keras (H, W, C_in, C_out) to PyTorch (C_out, C_in, H, W)
2. For Dense/Linear layers, transpose from Keras (output, input) to PyTorch (input, output)
3. Do NOT transpose bias vectors — they remain the same
4. Write unit tests comparing single-layer outputs between Keras and PyTorch
5. Verify the first Conv2D layer output matches exactly (within 1e-5) between frameworks

**Detection:**
- Accuracy near random chance (10% for CIFAR-10)
- Weight tensor shapes are permutations of expected shapes
- Intermediate layer activations have vastly different statistics
- First convolutional layer output differs by orders of magnitude

**Phase:** Architecture Translation phase — must verify weight shapes during initial model definition

---

### Pitfall 2: BatchNormalization Mode Confusion
**What goes wrong:** Model is not switched to evaluation mode before inference, causing BatchNorm to use batch statistics instead of learned running statistics. Accuracy drops significantly, especially on small batches or single images.

**Why it happens:**
- PyTorch requires explicit `model.eval()` call
- Keras automatically handles training/inference mode based on context
- Developers forget this is a stateful change, not a per-call parameter

**Consequences:**
- Validation accuracy 5-15% lower than expected
- Results are inconsistent across different batch sizes
- Single-image inference produces wildly incorrect results
- Running mean/variance from training data are ignored

**Prevention:**
1. Always call `model.eval()` before any inference or validation
2. Use context manager pattern for inference:
   ```python
   with torch.no_grad():
       model.eval()
       outputs = model(inputs)
   ```
3. Verify BatchNorm momentum parameter matches between frameworks (Keras default: 0.99, PyTorch default: 0.1)
4. Check that running_mean and running_var are loaded correctly from Keras
5. Test on single images — if results are unstable, BatchNorm mode is likely wrong

**Detection:**
- Validation accuracy drops when batch size changes
- Single-image predictions are random
- Intermediate BatchNorm layer statistics change during inference
- Model output varies across multiple forward passes with same input

**Phase:** Weight Loading phase — verify BatchNorm statistics transfer correctly; Evaluation phase — ensure eval() mode is set

---

### Pitfall 3: L2 Regularization vs Weight Decay Confusion
**What goes wrong:** Keras L2 regularization (λ=1e-4 in this project) is not equivalent to PyTorch weight_decay parameter. Direct value transfer causes different effective regularization strength, subtly altering inference behavior.

**Why it happens:**
- Keras adds L2 penalty to the loss function: `loss += λ * sum(w²)`
- PyTorch weight_decay in most optimizers implements L2 regularization, not true weight decay
- Mathematical difference: L2 regularization has a factor of 2 difference from weight decay
- For adaptive optimizers (Adam, RMSprop), behavior diverges significantly

**Consequences:**
- Converted model produces slightly different inference results (1-3% accuracy difference)
- Weight magnitudes drift from Keras values during any fine-tuning
- Model behavior is subtly incorrect but not obviously broken

**Prevention:**
1. **For inference only** (this project): L2 regularization doesn't affect inference, only training. Skip this entirely for pretrained model evaluation.
2. **For fine-tuning**: Convert Keras L2 parameter (λ) to PyTorch by dividing by 2: `weight_decay = λ / 2.0`
3. If using Adam/AdamW for fine-tuning, use `torch.optim.AdamW` with true weight decay, not L2 regularization
4. Document the conversion factor prominently in code comments

**Detection:**
- Small but consistent accuracy differences (85% vs 83%)
- Weight norms differ between frameworks
- Gradient magnitudes are slightly off during fine-tuning

**Phase:** Not applicable for inference-only conversion; becomes critical if fine-tuning is added later

---

### Pitfall 4: Weight Initialization Mismatch Ignored
**What goes wrong:** Developers assume initialization doesn't matter when loading pretrained weights. But missing or incorrectly initialized layers (e.g., shortcuts, projection layers) cause accuracy drops.

**Why it happens:**
- Keras uses Glorot/Xavier Uniform by default
- PyTorch defaults vary by layer type (Linear uses Kaiming, Conv2d uses custom)
- When weight loading is incomplete or partial, uninitialized layers use framework defaults
- Shortcut 1x1 convolutions are sometimes forgotten during weight transfer

**Consequences:**
- Model accuracy is 10-20% lower than expected
- Some layers have random weights despite "successful" weight loading
- Residual connections produce incorrect outputs

**Prevention:**
1. Explicitly verify ALL layers have loaded weights, not just convolutions
2. Print weight tensor statistics (min, max, mean, std) for each layer after loading
3. Compare PyTorch weight statistics to Keras weight statistics layer-by-layer
4. For shortcut/projection layers, ensure they're included in the weight mapping
5. Write assertions that fail if any layer's weight tensor contains initialization values

**Detection:**
- Weight tensors have suspicious statistics (e.g., all zeros, unit variance)
- Specific layers (often shortcuts) have very different magnitudes than Keras
- Layer-by-layer output comparison shows divergence at specific blocks
- Model accuracy is much worse than expected but better than random

**Phase:** Weight Loading phase — validate ALL weights are loaded, not just primary pathway

---

### Pitfall 5: Residual Connection Dimension Mismatch
**What goes wrong:** Shortcut connections in residual blocks fail to match dimensions when spatial size changes (stride=2) or channel count increases. Addition operation fails or produces incorrect results.

**Why it happens:**
- Keras automatically broadcasts in some cases; PyTorch is stricter
- 1x1 projection convolutions on shortcuts are implemented differently
- Stride handling differs: Keras may use stride in projection, PyTorch may use different patterns
- Channel dimension mismatch is not always caught by shape checks

**Consequences:**
- Runtime error: "RuntimeError: The size of tensor a (32) must match the size of tensor b (64)"
- Or worse: silent broadcasting produces incorrect results
- Residual blocks fail to learn proper shortcuts

**Prevention:**
1. For each residual block, verify input and output shapes match before addition
2. Implement shortcut projection layers with exact Keras architecture:
   - 1x1 Conv with stride matching the residual path stride
   - BatchNorm on projection path
   - No activation on shortcut (only on residual path output)
3. Test each residual block independently with known inputs
4. Log tensor shapes before addition operations during initial testing
5. Specifically verify stride=2 blocks where spatial dimensions halve

**Detection:**
- Shape mismatch errors during forward pass
- Residual blocks produce outputs with wrong spatial dimensions
- Manual shape inspection shows input/output dimension discrepancy
- Specific stacks (Stack 2, Stack 3 in ResNet8) fail while Stack 1 works

**Phase:** Architecture Translation phase — verify residual block structure matches Keras exactly

---

### Pitfall 6: Numerical Precision Drift
**What goes wrong:** Accumulation of small floating-point differences causes output to diverge from Keras results, even with correct architecture and weights.

**Why it happens:**
- Keras saves weights as float64, PyTorch typically uses float32
- Different operation ordering (addition is not associative in floating point)
- BLAS/CUDA library implementations differ between TensorFlow and PyTorch
- BatchNorm epsilon values differ (Keras default: 1e-3, PyTorch default: 1e-5)

**Consequences:**
- Outputs differ by small amounts (1e-3 to 1e-5 range)
- Accuracy drops 1-2% due to decision boundary shifts
- Exact reproducibility is impossible

**Prevention:**
1. Accept that exact bit-level reproducibility is impossible
2. Set BatchNorm epsilon to match Keras: `torch.nn.BatchNorm2d(..., eps=1e-3)`
3. Use float32 consistently (Keras will truncate float64 to float32 during loading)
4. Validate that output differences are small (< 1e-4) for known inputs
5. Focus on accuracy metrics, not exact output matching

**Detection:**
- Layer outputs differ by small amounts (1e-3 range) even with correct implementation
- Final predictions differ by tiny margins
- Accuracy is close but not identical (84.5% vs 85.1%)

**Phase:** Validation phase — establish acceptable tolerance thresholds

---

## Moderate Pitfalls

Mistakes that cause delays or require careful debugging but don't break the entire conversion.

### Pitfall 7: Softmax Location Ambiguity
**What goes wrong:** Unclear whether softmax is part of the model or should be applied externally during inference. Keras Dense layer can include softmax activation, but PyTorch Linear layer does not.

**Why it happens:**
- Keras combines final Dense layer with softmax activation in one layer definition
- PyTorch typically separates Linear layer from softmax
- Loss functions differ: CrossEntropyLoss includes softmax, NLLLoss requires manual softmax

**Prevention:**
1. Check Keras model definition: if final Dense has `activation='softmax'`, include it in PyTorch
2. For inference, apply `torch.softmax(outputs, dim=1)` to get probabilities
3. For accuracy calculation, softmax is unnecessary (argmax is the same)
4. Document whether model outputs are logits or probabilities

**Detection:**
- Output values are not in [0, 1] range when probabilities are expected
- Output values don't sum to 1.0
- Predictions are different from Keras despite same argmax

**Phase:** Architecture Translation and Evaluation phases

---

### Pitfall 8: Data Preprocessing Mismatch
**What goes wrong:** CIFAR-10 preprocessing differs between Keras and PyTorch defaults. Images are normalized differently, causing accuracy drops despite correct model.

**Why it happens:**
- Keras typically uses pixel values in [0, 1] range
- PyTorch torchvision datasets may use different normalization
- Mean/std normalization constants differ
- Image loading order (RGB vs BGR) can differ in some pipelines

**Prevention:**
1. Verify exact preprocessing from Keras training script
2. Match normalization exactly: divide by 255.0 if Keras used that
3. Check if Keras used dataset-specific mean/std subtraction
4. Test with a single known image and compare outputs

**Detection:**
- Model accuracy is significantly lower than expected
- First layer activations have very different magnitudes
- Single-image predictions are random but multi-image batches are better

**Phase:** Evaluation phase — verify data pipeline matches training preprocessing

---

### Pitfall 9: Missing .eval() and torch.no_grad()
**What goes wrong:** Forgetting `torch.no_grad()` context manager causes unnecessary gradient tracking, wasting memory and slowing inference. Combined with missing `.eval()`, causes incorrect results.

**Why it happens:**
- PyTorch defaults to training mode with gradient tracking
- Keras inference automatically disables training-specific behaviors
- Developers new to PyTorch forget these are separate concerns

**Prevention:**
1. Always use both together for inference:
   ```python
   model.eval()
   with torch.no_grad():
       outputs = model(inputs)
   ```
2. Use `@torch.inference_mode()` decorator for inference functions (PyTorch 1.9+)
3. Check memory usage — if it grows during inference, gradients are being tracked

**Detection:**
- High memory usage during inference
- Inference is slower than expected
- BatchNorm-related accuracy issues

**Phase:** Evaluation phase

---

## Minor Pitfalls

Mistakes that cause annoyance but are easily fixable.

### Pitfall 10: Bias Loading Oversight
**What goes wrong:** Bias vectors are not transposed like weight matrices, but developers apply transpose operation uniformly, causing shape mismatches.

**Why it happens:** Copy-paste error from weight transposition code

**Prevention:**
1. Bias vectors are 1D — load them directly without modification
2. Write separate loading logic for weights vs biases
3. Assert bias shapes are 1D before loading

**Detection:** Shape mismatch error when loading bias

**Phase:** Weight Loading phase

---

### Pitfall 11: Incorrect Weight File Path
**What goes wrong:** Hardcoded paths to Keras .h5 file break when running from different directories or on different machines.

**Why it happens:** Paths are not relative to script location

**Prevention:**
1. Use argparse to accept weight file path as command-line argument
2. Use pathlib for cross-platform path handling
3. Document expected file location in README

**Detection:** FileNotFoundError when loading weights

**Phase:** All phases — fix during initial setup

---

### Pitfall 12: Missing Dependency Versions
**What goes wrong:** Keras .h5 loading fails because h5py version is incompatible, or PyTorch version lacks needed features.

**Why it happens:** Environment setup without explicit version pinning

**Prevention:**
1. Pin PyTorch, TensorFlow/Keras, h5py versions in requirements.txt
2. Document Python version requirement
3. Test in clean virtual environment

**Detection:** Import errors or loading failures

**Phase:** Setup phase

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Architecture Definition | Channel order mismatch (NHWC vs NCHW) | Transpose Conv2D weights correctly; verify first layer output |
| Architecture Definition | Residual dimension mismatch | Test each block independently; verify stride=2 blocks |
| Weight Loading | BatchNorm statistics not loaded | Load running_mean, running_var, num_batches_tracked |
| Weight Loading | Initialization mismatch for missing layers | Verify ALL layers loaded; print weight statistics |
| Weight Loading | Bias transposition error | Load bias directly without transpose |
| Evaluation Setup | Missing model.eval() | Always call before inference; use context manager |
| Evaluation Setup | Data preprocessing mismatch | Match Keras preprocessing exactly (divide by 255.0) |
| Evaluation Setup | Softmax location ambiguity | Check if Keras used softmax activation; apply if needed |
| Validation | Numerical precision drift | Accept small differences; verify BatchNorm epsilon matches |
| Validation | Accuracy threshold unclear | Define acceptable range (>85% for this project) |

---

## Validation Checklist

Use this checklist to avoid pitfalls:

**Architecture Translation:**
- [ ] Conv2D weight shape transposed from (H,W,C_in,C_out) to (C_out,C_in,H,W)
- [ ] Dense/Linear weight shape transposed from (out,in) to (in,out)
- [ ] Bias vectors loaded without modification
- [ ] BatchNorm epsilon set to 1e-3 (Keras default)
- [ ] Residual shortcuts include projection layers where needed
- [ ] First layer output compared between Keras and PyTorch (diff < 1e-5)

**Weight Loading:**
- [ ] All layers show loaded weights (print statistics: min, max, mean, std)
- [ ] BatchNorm running_mean and running_var loaded
- [ ] Shortcut/projection layers included in weight mapping
- [ ] No layers show initialization patterns after loading

**Evaluation Setup:**
- [ ] model.eval() called before inference
- [ ] torch.no_grad() context manager used
- [ ] Data preprocessing matches Keras (check normalization)
- [ ] Softmax applied if needed for probability output

**Validation:**
- [ ] Test on single image — result should be stable
- [ ] Test on small batch (10 images) — accuracy reasonable
- [ ] Test on full test set — accuracy >85%
- [ ] Compare intermediate layer outputs to Keras on same inputs

---

## Sources

**Confidence: MEDIUM** — Based on community discussions and technical documentation. Specific issues verified across multiple sources.

### Critical Pitfalls (HIGH confidence)
- [TensorFlow/Keras to PyTorch translation - PyTorch Forums](https://discuss.pytorch.org/t/tensorflow-keras-to-pytorch-translation/174083)
- [Transferring weights from Keras to PyTorch - PyTorch Forums](https://discuss.pytorch.org/t/transferring-weights-from-keras-to-pytorch/9889)
- [Pitfalls encountered porting models to Keras from PyTorch/TensorFlow/MXNet](https://shaoanlu.wordpress.com/2019/05/23/pitfalls-encountered-porting-models-to-keras-from-pytorch-and-tensorflow/)
- [Load Keras Weight to PyTorch - Medium](https://medium.com/analytics-vidhya/load-keras-weight-to-pytorch-and-transform-keras-architecture-to-pytorch-easily-8ff5dd18b86b)
- [How to Transfer a Simple Keras Model to PyTorch - The Hard Way](https://gereshes.com/2019/06/24/how-to-transfer-a-simple-keras-model-to-pytorch-the-hard-way/)

### Channel Order Issues (HIGH confidence)
- [TensorRT UFF Tensorflow NHWC to NCHW conversion buggy - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/tensorrt-uff-tensorflow-nhwc-channels-last-to-nchw-channels-first-conversion-buggy/69282)
- [Channels Last Memory Format in PyTorch - PyTorch Tutorials](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)

### BatchNormalization (HIGH confidence)
- [Different results for batchnorm with pytorch and tensorflow/keras - PyTorch Forums](https://discuss.pytorch.org/t/different-results-for-batchnorm-with-pytorch-and-tensorflow-keras/151691)
- [What does model.eval() do for batchnorm layer? - PyTorch Forums](https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146)
- [Keras BatchNormalization layer documentation](https://keras.io/api/layers/normalization_layers/batch_normalization/)

### L2 Regularization vs Weight Decay (HIGH confidence)
- [Weight decay vs L2 regularization](https://bbabenko.github.io/weight-decay/)
- [Weight Decay is Not L2 Regularization - John Trimble](https://www.johntrimble.com/posts/weight-decay-is-not-l2-regularization/)
- [Understanding the difference between weight decay and L2 regularization](https://www.paepper.com/blog/posts/understanding-the-difference-between-weight-decay-and-l2-regularization/)
- [Weight decay vs L2 regularisation - PyTorch Forums](https://discuss.pytorch.org/t/weight-decay-vs-l2-regularisation/99186)

### Numerical Precision (MEDIUM confidence)
- [Numerical accuracy - PyTorch documentation](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [Loss of result precision from function converted from numpy/TFv1 to PyTorch - PyTorch Forums](https://discuss.pytorch.org/t/loss-of-result-precision-from-function-convereted-from-numpy-tfv1-to-pytorch/159275)
- [Same accuracy but different loss between PyTorch and Keras - PyTorch Forums](https://discuss.pytorch.org/t/same-accuracy-but-different-loss-between-pytorch-and-keras/83664)

### Weight Initialization (MEDIUM confidence)
- [Keras Layer weight initializers documentation](https://keras.io/api/layers/initializers/)
- [PyTorch Weight Initialization Techniques - TF Compare](https://apxml.com/courses/pytorch-for-tensorflow-developers/chapter-2-pytorch-nn-module-for-keras-users/weight-initialization-pytorch)

### Residual Connections (MEDIUM confidence)
- [Add Residual connection - PyTorch Forums](https://discuss.pytorch.org/t/add-residual-connection/20148)
- [ResNets fully explained with implementation from scratch using PyTorch - Medium](https://medium.com/@YasinShafiei/residual-networks-resnets-with-implementation-from-scratch-713b7c11f612)

### Model Validation (LOW confidence — general advice)
- [Loss of accuracy in recognizing when exporting a trained model - PaddleOCR Discussion](https://github.com/PaddlePaddle/PaddleOCR/discussions/14927)
- [Why do I get much worse results by using pytorch model than using keras - PyTorch Forums](https://discuss.pytorch.org/t/why-do-i-get-much-worse-results-by-using-pytorch-model-than-using-keras/16174)
