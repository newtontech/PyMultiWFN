#!/usr/bin/env python3
"""
Test normalization function
"""

import numpy as np
from pymultiwfn.integrals.overlap import _gaussian_normalization

# Test S-type Gaussian (l=m=n=0)
alpha = 1.0
N_S = _gaussian_normalization(alpha, 0, 0, 0)
print(f"S-type (l=m=n=0) normalization: N = {N_S:.10f}")
print(f"Expected: (2α/π)^0.75 = {(2*alpha/np.pi)**0.75:.10f}")
print(f"Match: {np.isclose(N_S, (2*alpha/np.pi)**0.75)}")

# Test Px-type Gaussian (l=1, m=n=0)
N_Px = _gaussian_normalization(alpha, 1, 0, 0)
print(f"\nPx-type (l=1,m=0,n=0) normalization: N = {N_Px:.10f}")

# Test Dxx-type Gaussian (l=2, m=n=0)
N_Dxx = _gaussian_normalization(alpha, 2, 0, 0)
print(f"\nDxx-type (l=2,m=0,n=0) normalization: N = {N_Dxx:.10f}")

# Check normalization by computing <φ|φ>
from pymultiwfn.integrals.overlap import _calculate_primitive_overlap

# For S-type at same center
overlap_SS = _calculate_primitive_overlap(
    0,
    0,  # Both S-type
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),  # Same center
    alpha,
    alpha,  # Same exponent
    use_cache=False,
)
print(f"\nS-S overlap (same center, same exponent): {overlap_SS:.10f}")
print(f"Should be 1.0 for normalized functions")
