---
phase: 01-model-conversion
plan: 01
subsystem: model-conversion
tags: [tensorflow, keras, onnx, tf2onnx, model-conversion, resnet, cifar10]

# Dependency graph
requires:
  - phase: none
    provides: "Initial phase - no dependencies"
provides:
  - Keras to ONNX conversion script (scripts/convert.py)
  - Working ONNX ResNet8 model for CIFAR-10 (315KB)
  - Python environment with TensorFlow, tf2onnx, ONNX
  - Conversion logging infrastructure
affects: [02-evaluation, model-evaluation, onnx-runtime]

# Tech tracking
tech-stack:
  added:
    - tensorflow>=2.20.0
    - tf2onnx (GitHub main branch for Python 3.12 compatibility)
    - onnx>=1.17.0
    - numpy>=1.26.4,<2.0
  patterns:
    - Python logging with dual handlers (file + console)
    - ONNX model validation with onnx.checker.check_model()
    - Shape verification for model correctness

key-files:
  created:
    - scripts/convert.py
    - requirements.txt
    - .gitignore
    - models/.gitkeep
    - logs/.gitkeep
  modified: []

key-decisions:
  - "Use tf2onnx from GitHub main branch instead of PyPI 1.16.1 for Python 3.12 compatibility"
  - "Use numpy 1.26.4 (last version <2.0 with Python 3.12 support) to avoid numpy.object/numpy.bool deprecation issues"
  - "Opset 15 (tf2onnx default) for conversion - broad ONNX Runtime compatibility"
  - "Dynamic batch dimension (None) for flexible inference batch sizes"

patterns-established:
  - "Virtual environment (venv/) for project isolation"
  - "Generated artifacts (.onnx, .log) excluded from git via .gitignore"
  - "Conversion script logs to both file and console for debugging"
  - "Explicit shape verification after conversion to catch errors early"

# Metrics
duration: 7min
completed: 2026-01-27
---

# Phase 01 Plan 01: Model Conversion Summary

**ResNet8 Keras model converted to ONNX format (315KB) with tf2onnx, verified input shape (batch, 32, 32, 3) and output shape (batch, 10) for CIFAR-10 classification**

## Performance

- **Duration:** 7 min
- **Started:** 2026-01-27T09:57:38Z
- **Completed:** 2026-01-27T10:04:26Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Keras ResNet8 model successfully converted to ONNX format
- ONNX model validated with correct input/output shapes for CIFAR-10
- Python environment established with TensorFlow 2.20.0, tf2onnx, and ONNX
- Conversion logging infrastructure for progress tracking and debugging

## Task Commits

Each task was committed atomically:

1. **Task 1: Create project structure and conversion script** - `c027898` (feat)

**Note:** Task 2 was verification-only (conversion already ran in Task 1) - no additional commit needed.

## Files Created/Modified
- `scripts/convert.py` - Keras to ONNX conversion with logging and shape verification
- `requirements.txt` - Python dependencies with version constraints for compatibility
- `.gitignore` - Exclude venv and generated artifacts from version control
- `models/.gitkeep` - Placeholder for ONNX output directory
- `logs/.gitkeep` - Placeholder for conversion logs directory

## Decisions Made

**1. tf2onnx version selection**
- Used tf2onnx from GitHub main branch instead of PyPI release 1.16.1
- Rationale: PyPI 1.16.1 uses deprecated numpy.object/numpy.bool attributes incompatible with numpy 1.20+, which is required for Python 3.12 compatibility

**2. numpy version constraint**
- Pinned to numpy>=1.26.4,<2.0
- Rationale: numpy 1.26.4 is the last <2.0 version with Python 3.12 binary wheel support, avoiding build-time compatibility issues with older numpy versions

**3. Virtual environment approach**
- Created project-local venv/ instead of system packages
- Rationale: System Python externally managed (PEP 668), required virtual environment for package installation

**4. Dynamic batch dimension**
- Used input shape (None, 32, 32, 3) in conversion
- Rationale: Allows flexible batch sizes during inference rather than fixed batch=1

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created virtual environment for dependency installation**
- **Found during:** Task 1 (running conversion script)
- **Issue:** System Python externally managed (PEP 668), pip install failed with error requiring --break-system-packages or venv
- **Fix:** Created `python3 -m venv venv` and installed packages in venv
- **Files modified:** venv/ directory created (gitignored)
- **Verification:** venv/bin/python successfully imports tensorflow, tf2onnx, onnx
- **Committed in:** c027898 (Task 1 commit includes .gitignore for venv/)

**2. [Rule 3 - Blocking] Fixed tf2onnx/numpy compatibility**
- **Found during:** Task 1 (importing tf2onnx)
- **Issue:** tf2onnx 1.16.1 from PyPI uses deprecated np.object/np.bool, fails with numpy 1.20+ and Python 3.12
- **Fix:** Installed tf2onnx from GitHub main branch (commit 4fed7d) which removes deprecated numpy attributes
- **Files modified:** requirements.txt updated to reference git+https://github.com/onnx/tensorflow-onnx.git
- **Verification:** `import tf2onnx` succeeds, conversion runs without AttributeError
- **Committed in:** c027898 (Task 1 commit)

**3. [Rule 2 - Missing Critical] Added requirements.txt and .gitignore**
- **Found during:** Task 1 (preparing to commit)
- **Issue:** Plan didn't specify requirements.txt or .gitignore, but necessary for reproducibility and clean git status
- **Fix:** Created requirements.txt with pinned dependency versions, .gitignore to exclude venv/ and generated files
- **Files created:** requirements.txt, .gitignore
- **Verification:** `git status` shows clean working tree (venv, .onnx, .log excluded)
- **Committed in:** c027898 (Task 1 commit)

---

**Total deviations:** 3 auto-fixed (2 blocking, 1 missing critical)
**Impact on plan:** All auto-fixes necessary for execution on Python 3.12 system and project hygiene. No scope creep - addressing environmental constraints and missing infrastructure files.

## Issues Encountered

**tf2onnx/numpy compatibility on Python 3.12:**
- tf2onnx 1.16.1 (latest PyPI release) incompatible with numpy 1.20+ due to deprecated numpy.object/numpy.bool
- numpy <1.24 cannot build on Python 3.12 (setuptools pkgutil.ImpImporter AttributeError)
- Solution: Use tf2onnx from GitHub main branch (post-1.16.1, includes numpy 2.0 compatibility fixes) with numpy 1.26.4

**Warnings in conversion log:**
- "Error in loading the saved optimizer state" - Expected, optimizer state not needed for inference
- "compile_metrics will be empty" - Expected, metrics not compiled until training/evaluation
- No impact on model correctness

## User Setup Required

None - no external service configuration required. Conversion runs locally with pre-trained model.

## Next Phase Readiness

**Ready for Phase 2 (Evaluation):**
- ONNX model available at `models/resnet8.onnx` (315KB)
- Model verified with onnx.checker.check_model()
- Input shape (batch, 32, 32, 3) matches CIFAR-10 format
- Output shape (batch, 10) matches CIFAR-10 classes
- Python environment ready for ONNX Runtime evaluation

**No blockers.**

**Context for next phase:**
- Input tensor name: `input`
- Output tensor name: `dense`
- Opset: 15
- 39 nodes, 35 parameters
- Conversion used dynamic batch dimension (0 in ONNX shape = None in TensorFlow)

---
*Phase: 01-model-conversion*
*Completed: 2026-01-27*
