#!/usr/bin/env python3
"""
Test overlap between basis functions with different exponents
"""

import numpy as np
from pymultiwfn.integrals.overlap import _calculate_primitive_overlap

# Test two S-type Gaussians on the same center with different exponents
alpha1 = 33.87
alpha2 = 5.095

overlap = _calculate_primitive_overlap(
    0,
    0,  # Both S-type
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),  # Same center
    alpha1,
    alpha2,  # Different exponents
    use_cache=False,
)

print(f"S-S overlap (same center, exponents {alpha1} and {alpha2}): {overlap:.10f}")
print(f"Should be < 1.0 and > 0")

# Test two S-type Gaussians on different centers
R = 1.4  # Distance between centers (in Bohr)

overlap_diff_center = _calculate_primitive_overlap(
    0,
    0,  # Both S-type
    (0.0, 0.0, 0.0),
    (R, 0.0, 0.0),  # Different centers
    alpha1,
    alpha1,  # Same exponent
    use_cache=False,
)

print(
    f"\nS-S overlap (different centers, distance {R} Bohr, exponent {alpha1}): {overlap_diff_center:.10f}"
)
print(f"Should be < 1.0 and > 0")

# Test Px-Px overlap
alpha = 1.407
overlap_PxPx = _calculate_primitive_overlap(
    1,
    1,  # Both Px-type
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),  # Same center
    alpha,
    alpha,  # Same exponent
    use_cache=False,
)

print(f"\nPx-Px overlap (same center, exponent {alpha}): {overlap_PxPx:.10f}")
print(f"Should be 1.0 for normalized functions")

# Test Px-Py overlap (should be 0)
overlap_PxPy = _calculate_primitive_overlap(
    1,
    2,  # Px and Py
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),  # Same center
    alpha,
    alpha,  # Same exponent
    use_cache=False,
)

print(f"\nPx-Py overlap (same center, exponent {alpha}): {overlap_PxPy:.10f}")
print(f"Should be 0.0 (orthogonal)")
