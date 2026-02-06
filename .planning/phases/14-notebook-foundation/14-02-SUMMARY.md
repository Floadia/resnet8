---
phase: 14-notebook-foundation
plan: 02
subsystem: notebook
tags: [marimo, layer-inspection, onnx, pytorch, interactive-ui, dropdown]

# Dependency graph
requires:
  - phase: 14-01
    provides: Marimo notebook infrastructure with cached model loading
provides:
  - Layer inspection utilities for ONNX and PyTorch models
  - Layer selection dropdown populated from model structure
  - Reactive layer info display
  - Complete notebook UI ready for visualization features
affects: [15-scale-parameter-visualization, 16-activation-distribution-analysis, 17-residual-analysis]

# Tech tracking
tech-stack:
  added: []
  patterns: [layer name extraction from ONNX (nodes + initializers), layer name extraction from PyTorch (named_modules), reactive UI updates via marimo dropdown]

key-files:
  created:
    - playground/utils/layer_inspector.py
    - playground/__init__.py
  modified:
    - playground/utils/__init__.py
    - playground/quantization.py
    - playground/utils/model_loader.py

key-decisions:
  - "Extract layer names from ONNX using both graph.node (operations) and graph.initializer (parameters)"
  - "Filter out PyTorch root module (empty name) to avoid confusion"
  - "Prioritize ONNX float model for layer list, fallback to PyTorch"
  - "Use file selection mode (not directory mode) for reliable model folder selection"
  - "Handle PyTorch dict format {'model': ..., 'epoch': ...} for quantized models"

patterns-established:
  - "Layer inspection pattern: separate utilities for ONNX vs PyTorch, unified get_all_layer_names interface"
  - "Dropdown population: extract from loaded models, show placeholder when no models"
  - "Reactive layer display: reference dropdown.value to trigger cell re-execution"

# Metrics
duration: 21h 21min (across sessions, with debugging)
completed: 2026-02-06
---

# Phase 14 Plan 02: Layer Inspection Utilities and Complete Notebook UI Summary

**Layer selection dropdown with ONNX/PyTorch layer name extraction, reactive updates, and file-based model folder selection**

## Performance

- **Duration:** 21h 21min (across 2 sessions with debugging and path fixes)
- **Started:** 2026-02-05T11:41:25Z
- **Completed:** 2026-02-06T09:02:17Z
- **Tasks:** 3 (2 auto + 1 checkpoint:human-verify)
- **Files modified:** 5

## Accomplishments
- Created layer_inspector.py with 4 functions for extracting layer names from ONNX and PyTorch models
- ONNX extraction includes both operations (nodes) and parameters (initializers)
- PyTorch extraction uses named_modules() with root module filtering
- Added layer selection dropdown to notebook populated from actual model structure
- Reactive layer info display showing name, type, and source framework
- Fixed module import issues by adding playground/__init__.py package marker
- Fixed file browser path resolution using file selection mode with absolute paths
- Fixed PyTorch model loading to handle dict format (quantized models saved with metadata)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create layer inspection utilities** - `01b8b63` (feat)
   - Implemented get_onnx_layer_names() extracting from nodes + initializers
   - Implemented get_pytorch_layer_names() using named_modules() with root filter
   - Implemented get_all_layer_names() with ONNX/PyTorch fallback priority
   - Implemented get_layer_type() returning operation/module type strings
   - Exported all functions from playground.utils

2. **Task 2: Add layer selection dropdown to notebook** - `c1c8f71` (feat)
   - Added imports for layer inspection utilities
   - Extracted layer names from loaded models
   - Created mo.ui.dropdown with layer names as options
   - Added reactive layer info display cell
   - Layer selection triggers immediate update (no button needed)

3. **Task 3: Human verification checkpoint** - Approved by user
   - Initial verification revealed import and path issues
   - Fixed via additional commits (see Deviations)

**Plan metadata:** (to be committed with this summary)

## Files Created/Modified

- `playground/utils/layer_inspector.py` - Layer name extraction for ONNX (nodes + initializers) and PyTorch (named_modules)
- `playground/utils/__init__.py` - Added exports for layer_inspector functions
- `playground/__init__.py` - Package marker to enable proper module imports
- `playground/quantization.py` - Added layer selection dropdown and reactive display cells
- `playground/utils/model_loader.py` - Fixed PyTorch loader to handle dict format, fixed path resolution

## Decisions Made

**1. Extract ONNX layers from both nodes and initializers**
- Nodes provide operation names (Conv, Relu, etc.)
- Initializers provide parameter names (weights, biases)
- Combined list gives complete view of model structure
- Deduplicated and sorted for clean dropdown

**2. Filter out PyTorch root module**
- named_modules() returns ('', model) for root
- Empty name confuses users in dropdown
- Per RESEARCH.md pitfall #6: always filter root module

**3. File selection mode for model folder picker**
- Directory selection mode navigates INTO folders (doesn't select them)
- File selection mode with .onnx/.pt filters lets users select a file
- Derive model folder from selected file's parent directory
- More reliable than directory mode for Marimo file browser

**4. Handle PyTorch dict format in model loader**
- Quantized models saved as {'model': ..., 'epoch': ..., 'optimizer_state': ...}
- Updated load_pytorch_model to check if result is dict and extract 'model' key
- Maintains backward compatibility with direct model saves

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added playground/__init__.py package marker**
- **Found during:** Task 3 verification
- **Issue:** Import error "No module named 'playground'" when running notebook
- **Root cause:** playground/ directory wasn't recognized as Python package
- **Fix:** Created playground/__init__.py (minimal package marker)
- **Files modified:** playground/__init__.py (new)
- **Commit:** c92ef87

**2. [Rule 1 - Bug] Fixed file browser path and PyTorch dict loading**
- **Found during:** Task 3 verification
- **Issue 1:** File browser initial_path="./models" resolved to playground/models (doesn't exist)
- **Issue 2:** PyTorch loader failed on quantized models (dict format not handled)
- **Fix 1:** Changed initial_path to "../models" for correct relative path
- **Fix 2:** Updated load_pytorch_model to extract model from dict when present
- **Files modified:** playground/quantization.py, playground/utils/model_loader.py
- **Commit:** b1c0f92

**3. [Rule 1 - Bug] Switched to file selection mode with absolute paths**
- **Found during:** Task 3 verification continued
- **Issue:** Directory selection mode in file browser navigates into folders rather than selecting them
- **Root cause:** Marimo file browser behavior - directory mode doesn't populate folder_picker.value
- **Fix:** Switched to file selection mode (.onnx/.pt filters), derive folder from parent path, use absolute paths computed from __file__
- **Files modified:** playground/quantization.py
- **Commit:** 2b3e8f5

## Issues Encountered

**Import and path resolution issues during verification:**
- Module import failures resolved by adding package __init__.py
- File browser path resolution required switching from directory to file selection mode
- PyTorch dict format required updating loader to handle quantized model structure
- All issues resolved via automatic fixes (Rule 1/3 deviations)

## User Setup Required

None - notebook is fully functional.

**To use:**
1. Launch: `marimo edit playground/quantization.py`
2. Click "Select a model file..." button
3. Navigate to models/ folder and select any .onnx or .pt file
4. Models load automatically from that folder (spinner shows progress)
5. Summary displays with available variants and layer count
6. Select a layer from dropdown to see layer info (name, type, source)

## Next Phase Readiness

**Ready for Phase 15 (Scale Parameter Visualization):**
- Layer inspection utilities complete and working
- Layer selection dropdown operational with reactive updates
- Can extract layer names from both ONNX and PyTorch models
- get_layer_type() available for displaying layer metadata
- Next: Add scale parameter extraction and visualization cells

**No blockers or concerns.**

---
*Phase: 14-notebook-foundation*
*Completed: 2026-02-06*
