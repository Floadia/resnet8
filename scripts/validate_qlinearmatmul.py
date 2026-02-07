#!/usr/bin/env python3
"""Validate QLinearMatMul manual implementation against ONNX specification.

This script demonstrates the two-stage QLinearMatMul computation pattern:
1. INT8×INT8→INT32 MAC operations with zero-point subtraction
2. Requantization to INT8 with scaling and saturation

Since the ResNet8 quantized model uses QDQ format (not QLinearMatMul), this
script creates synthetic examples to demonstrate the computation pattern.

Usage:
    python scripts/validate_qlinearmatmul.py
    python scripts/validate_qlinearmatmul.py --verbose
"""

import argparse
import sys

import numpy as np


def qlinear_matmul_manual(
    a: np.ndarray,
    a_scale: float,
    a_zero_point: int,
    b: np.ndarray,
    b_scale: float,
    b_zero_point: int,
    y_scale: float,
    y_zero_point: int,
    verbose: bool = False,
) -> np.ndarray:
    """Manual QLinearMatMul implementation with two-stage computation.

    Args:
        a: INT8 input [N, K]
        a_scale: Input scale factor
        a_zero_point: Input zero-point
        b: INT8 weights [K, M]
        b_scale: Weight scale factor (per-tensor)
        b_zero_point: Weight zero-point
        y_scale: Output scale factor
        y_zero_point: Output zero-point
        verbose: Print intermediate values

    Returns:
        INT8 output [N, M]
    """
    N, K = a.shape
    K_b, M = b.shape
    assert K == K_b, f"Shared dimension K mismatch: {K} != {K_b}"

    if verbose:
        print(f"Input a shape: {a.shape}")
        print(f"Weight b shape: {b.shape}")
        print(f"Output y shape: ({N}, {M})")
        print()

    # ========================================
    # Stage 1: INT32 MAC Accumulation
    # ========================================
    if verbose:
        print("=" * 60)
        print("STAGE 1: INT8×INT8→INT32 MAC Operations")
        print("=" * 60)

    acc = np.zeros((N, M), dtype=np.int32)

    for n in range(N):
        for m in range(M):
            for k in range(K):
                # Center values by subtracting zero-points
                a_val = np.int32(a[n, k]) - a_zero_point
                b_val = np.int32(b[k, m]) - b_zero_point

                # MAC operation
                acc[n, m] += a_val * b_val

            if verbose and n == 0 and m == 0:
                print(f"Output [0, 0] accumulator: {acc[n, m]}")
                print(f"  ({K} MACs: dot product of input row 0 with weight column 0)")

    if verbose:
        print(f"\nAccumulator range: [{np.min(acc)}, {np.max(acc)}]")
        print(f"Accumulator dtype: {acc.dtype} (INT32 required to prevent overflow)")
        print()

    # ========================================
    # Stage 2: Requantization to INT8
    # ========================================
    if verbose:
        print("=" * 60)
        print("STAGE 2: Requantization to INT8")
        print("=" * 60)

    # Combined scale factor
    scale_factor = (a_scale * b_scale) / y_scale
    if verbose:
        print(f"Scale factor: ({a_scale} * {b_scale}) / {y_scale} = {scale_factor}")
        print()

    # Apply scaling
    scaled = acc.astype(np.float32) * scale_factor

    # Round to nearest even
    rounded = np.round(scaled)

    # Add output zero-point
    with_zero_point = rounded + y_zero_point

    # Saturate to INT8 range
    y = np.clip(with_zero_point, -128, 127).astype(np.int8)

    if verbose:
        print(f"Output range: [{np.min(y)}, {np.max(y)}]")
        print(f"Output dtype: {y.dtype}")
        print()

    return y


def test_simple_matmul() -> bool:
    """Test 1: Simple 2×3 × 3×2 matrix multiplication."""
    print("=" * 70)
    print("Test 1: Simple matrix multiplication (2×3 × 3×2)")
    print("=" * 70)

    # Simple inputs (symmetric quantization)
    a = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int8)  # [2, 3]
    b = np.array([[5, -5], [10, -10], [15, -15]], dtype=np.int8)  # [3, 2]

    a_scale = 0.1
    a_zero_point = 0
    b_scale = 0.1
    b_zero_point = 0
    y_scale = 0.1
    y_zero_point = 0

    print(f"Input a:\n{a}")
    print(f"Weight b:\n{b}")
    print()

    # Manual computation
    y_manual = qlinear_matmul_manual(
        a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point
    )

    print(f"Output y (manual):\n{y_manual}")
    print()

    # Expected values (computed manually):
    # y[0,0] = 10*5 + 20*10 + 30*15 = 50 + 200 + 450 = 700 → 700 * (0.1*0.1/0.1) = 70
    # y[0,1] = 10*(-5) + 20*(-10) + 30*(-15) = -50 - 200 - 450 = -700 → -70
    # y[1,0] = 40*5 + 50*10 + 60*15 = 200 + 500 + 900 = 1600 → 160 (saturates to 127)
    # y[1,1] = 40*(-5) + 50*(-10) + 60*(-15) = -1600
    #   → -160 (saturates to -128)
    expected = np.array([[70, -70], [127, -128]], dtype=np.int8)

    print(f"Expected output:\n{expected}")
    print()

    match = np.array_equal(y_manual, expected)
    print(f"Result: {'PASS ✓' if match else 'FAIL ✗'}")
    print()

    return match


def test_resnet8_fc_layer() -> bool:
    """Test 2: ResNet8-like FC layer (64 features → 10 classes)."""
    print("=" * 70)
    print("Test 2: ResNet8 FC layer simulation (64 → 10)")
    print("=" * 70)

    # Simulate ResNet8 final layer
    N, K, M = 1, 64, 10
    np.random.seed(42)

    # Generate random INT8 values
    a = np.random.randint(-128, 128, size=(N, K), dtype=np.int8)
    b = np.random.randint(-128, 128, size=(K, M), dtype=np.int8)

    # Typical ResNet8 quantization parameters
    a_scale = 0.1903
    a_zero_point = 20
    b_scale = 0.0245
    b_zero_point = 0
    y_scale = 0.1585
    y_zero_point = 0

    print("Configuration:")
    print(f"  Batch size: {N}")
    print(f"  Input features: {K}")
    print(f"  Output features: {M}")
    print(f"  a_scale: {a_scale}")
    print(f"  a_zero_point: {a_zero_point}")
    print(f"  b_scale: {b_scale}")
    print(f"  b_zero_point: {b_zero_point}")
    print(f"  y_scale: {y_scale}")
    print(f"  y_zero_point: {y_zero_point}")
    print()

    # Manual computation (verbose for first output)
    y_manual = qlinear_matmul_manual(
        a,
        a_scale,
        a_zero_point,
        b,
        b_scale,
        b_zero_point,
        y_scale,
        y_zero_point,
        verbose=True,
    )

    print(f"Output y shape: {y_manual.shape}")
    print(f"Output y range: [{np.min(y_manual)}, {np.max(y_manual)}]")
    print(f"Output y (first 5 classes): {y_manual[0, :5]}")
    print()

    # Verify using numpy reference implementation
    # Dequantize → FP32 matmul → Quantize
    a_fp32 = (a.astype(np.float32) - a_zero_point) * a_scale
    b_fp32 = (b.astype(np.float32) - b_zero_point) * b_scale
    y_fp32 = np.matmul(a_fp32, b_fp32)
    y_quantized = np.round(y_fp32 / y_scale) + y_zero_point
    y_reference = np.clip(y_quantized, -128, 127).astype(np.int8)

    print(f"Reference output (first 5 classes): {y_reference[0, :5]}")
    print()

    match = np.array_equal(y_manual, y_reference)
    max_diff = np.max(np.abs(y_manual.astype(np.int32) - y_reference.astype(np.int32)))

    print(f"Match: {'Yes' if match else 'No'}")
    print(f"Max difference: {max_diff}")
    print(f"Result: {'PASS ✓' if match else 'FAIL ✗'}")
    print()

    return match


def test_asymmetric_quantization() -> bool:
    """Test 3: Asymmetric quantization (non-zero zero-points)."""
    print("=" * 70)
    print("Test 3: Asymmetric quantization (non-zero zero-points)")
    print("=" * 70)

    # Small test case
    a = np.array([[100, 120, 110]], dtype=np.int8)  # [1, 3]
    b = np.array([[50, 60], [70, 80], [90, 100]], dtype=np.int8)  # [3, 2]

    a_scale = 0.05
    a_zero_point = 100  # Non-zero
    b_scale = 0.08
    b_zero_point = 50  # Non-zero
    y_scale = 0.1
    y_zero_point = 10  # Non-zero

    print("Quantization parameters:")
    print(f"  a_zero_point: {a_zero_point} (non-zero)")
    print(f"  b_zero_point: {b_zero_point} (non-zero)")
    print(f"  y_zero_point: {y_zero_point} (non-zero)")
    print()

    y_manual = qlinear_matmul_manual(
        a,
        a_scale,
        a_zero_point,
        b,
        b_scale,
        b_zero_point,
        y_scale,
        y_zero_point,
        verbose=True,
    )

    # Reference implementation
    a_fp32 = (a.astype(np.float32) - a_zero_point) * a_scale
    b_fp32 = (b.astype(np.float32) - b_zero_point) * b_scale
    y_fp32 = np.matmul(a_fp32, b_fp32)
    y_quantized = np.round(y_fp32 / y_scale) + y_zero_point
    y_reference = np.clip(y_quantized, -128, 127).astype(np.int8)

    print(f"Manual output: {y_manual}")
    print(f"Reference output: {y_reference}")
    print()

    match = np.array_equal(y_manual, y_reference)
    print(f"Result: {'PASS ✓' if match else 'FAIL ✗'}")
    print()

    return match


def test_int32_accumulator_overflow() -> bool:
    """Test 4: Demonstrate INT32 accumulator requirement (INT16 would overflow)."""
    print("=" * 70)
    print("Test 4: INT32 accumulator overflow demonstration")
    print("=" * 70)

    # Worst-case: 64 MACs with maximum values
    K = 64
    a = np.full((1, K), 127, dtype=np.int8)
    b = np.full((K, 1), 127, dtype=np.int8)

    a_scale = 0.1
    a_zero_point = 0
    b_scale = 0.1
    b_zero_point = 0
    y_scale = 0.1
    y_zero_point = 0

    print("Configuration:")
    print(f"  K = {K} (number of MACs per output)")
    print("  Input values: all 127 (maximum INT8)")
    print("  Weight values: all 127 (maximum INT8)")
    print()

    # Compute accumulator value
    max_product = 127 * 127
    total_accumulation = max_product * K

    INT16_MAX = 32767
    INT32_MAX = 2147483647

    overflow_factor_int16 = total_accumulation / INT16_MAX
    margin_int32 = INT32_MAX / total_accumulation

    print("Analysis:")
    print(f"  Single product: 127 × 127 = {max_product}")
    print(f"  Total accumulation: {max_product} × {K} = {total_accumulation}")
    print()
    print(f"  INT16 max: {INT16_MAX}")
    print(f"  Overflow factor: {overflow_factor_int16:.1f}× (INT16 OVERFLOWS)")
    print()
    print(f"  INT32 max: {INT32_MAX}")
    print(f"  Safety margin: {margin_int32:.1f}× (INT32 safe)")
    print()

    # Verify with actual computation
    y_manual = qlinear_matmul_manual(
        a,
        a_scale,
        a_zero_point,
        b,
        b_scale,
        b_zero_point,
        y_scale,
        y_zero_point,
    )

    print(f"Manual computation result: {y_manual[0, 0]}")
    expected_val = total_accumulation * 0.1
    print(
        f"Expected: {total_accumulation} * (0.1*0.1/0.1)"
        f" = {expected_val} -> saturated to 127"
    )
    print()

    # Expected: 1032256 * 0.1 = 103225.6 → round to 103226 → saturate to 127
    expected = np.array([[127]], dtype=np.int8)
    match = np.array_equal(y_manual, expected)

    print(f"Result: {'PASS ✓' if match else 'FAIL ✗'}")
    print()

    return match


def main():
    parser = argparse.ArgumentParser(
        description="Validate QLinearMatMul implementation"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed computation steps"
    )
    parser.parse_args()

    # Run all tests
    tests = [
        ("Simple MatMul", test_simple_matmul),
        ("ResNet8 FC Layer", test_resnet8_fc_layer),
        ("Asymmetric Quantization", test_asymmetric_quantization),
        ("INT32 Accumulator Overflow", test_int32_accumulator_overflow),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results.append((name, False))

    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"{name:40s} {status}")

    all_passed = all(passed for _, passed in results)
    print()
    print(f"Overall: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
