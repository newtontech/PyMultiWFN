"""
Basis set evaluation module.
Vectorized implementation of Gaussian Type Orbitals (GTO) evaluation.
"""

import numpy as np

from pymultiwfn.core.data import Wavefunction


def evaluate_basis(wfn: Wavefunction, coords: np.ndarray) -> np.ndarray:
    """
    Evaluates all basis functions at the given coordinates.

    Args:
        wfn: Wavefunction object containing basis set info.
        coords: (N_points, 3) array of Cartesian coordinates.

    Returns:
        phi: (N_points, N_basis) array of basis function values.
    """
    n_points = coords.shape[0]
    n_basis = wfn.num_basis
    phi = np.zeros((n_points, n_basis))

    # Pre-allocate arrays for efficiency
    r_vec = np.empty((n_points, 3))
    r2 = np.empty(n_points)

    basis_idx = 0

    for shell in wfn.shells:
        # Get atom coordinates
        # Note: shell.center_idx is 0-based index of atom
        atom = wfn.atoms[shell.center_idx]
        atom_coord = np.array([atom.x, atom.y, atom.z])

        # Vector r = R_point - R_atom
        np.subtract(coords, atom_coord, out=r_vec)
        np.sum(np.square(r_vec), axis=1, out=r2)

        # Handle different shell types
        # 0=S, 1=P, -1=SP, 2=D (Cartesian 6 or Spherical 5?), etc.
        # FCHK usually implies Cartesian for P, D, F unless specified otherwise.
        # Multiwfn handles conversion. Here we assume Cartesian for simplicity of Phase 3 demo.

        if shell.type == 0:  # S shell
            radial = _eval_contraction(shell.exponents, shell.coefficients, r2)
            phi[:, basis_idx] = radial
            basis_idx += 1

        elif shell.type == 1:  # P shell
            radial = _eval_contraction(shell.exponents, shell.coefficients, r2)
            phi[:, basis_idx] = r_vec[:, 0] * radial  # X
            phi[:, basis_idx + 1] = r_vec[:, 1] * radial  # Y
            phi[:, basis_idx + 2] = r_vec[:, 2] * radial  # Z
            basis_idx += 3

        elif shell.type == -1:  # SP shell
            # SP shell uses the same contraction coefficients for both S and P
            # The coefficients array has shape (1, n_primitives)
            coeffs = shell.coefficients[0]  # Get the single row of coefficients

            # S component
            radial_s = _eval_contraction(shell.exponents, coeffs, r2)
            phi[:, basis_idx] = radial_s
            basis_idx += 1

            # P component
            radial_p = _eval_contraction(shell.exponents, coeffs, r2)
            phi[:, basis_idx] = r_vec[:, 0] * radial_p  # X
            phi[:, basis_idx + 1] = r_vec[:, 1] * radial_p  # Y
            phi[:, basis_idx + 2] = r_vec[:, 2] * radial_p  # Z
            basis_idx += 3

        elif shell.type == 2:  # D shell (Cartesian 6 functions)
            # Cartesian D: XX, YY, ZZ, XY, XZ, YZ
            radial = _eval_contraction(shell.exponents, shell.coefficients, r2)

            xx = r_vec[:, 0] * r_vec[:, 0]
            yy = r_vec[:, 1] * r_vec[:, 1]
            zz = r_vec[:, 2] * r_vec[:, 2]
            xy = r_vec[:, 0] * r_vec[:, 1]
            xz = r_vec[:, 0] * r_vec[:, 2]
            yz = r_vec[:, 1] * r_vec[:, 2]

            phi[:, basis_idx] = xx * radial
            phi[:, basis_idx + 1] = yy * radial
            phi[:, basis_idx + 2] = zz * radial
            phi[:, basis_idx + 3] = xy * radial
            phi[:, basis_idx + 4] = xz * radial
            phi[:, basis_idx + 5] = yz * radial
            basis_idx += 6

        elif shell.type == 3:  # F shell (Cartesian 10 functions)
            # Cartesian F: XXX, YYY, ZZZ, XXY, XXZ, YYZ, YZZ, XYY, XZZ, XYZ
            radial = _eval_contraction(shell.exponents, shell.coefficients, r2)

            x = r_vec[:, 0]
            y = r_vec[:, 1]
            z = r_vec[:, 2]

            xxx = x * x * x
            yyy = y * y * y
            zzz = z * z * z
            xxy = x * x * y
            xxz = x * x * z
            yyz = y * y * z
            yzz = y * z * z
            xyy = x * y * y
            xzz = x * z * z
            xyz = x * y * z

            phi[:, basis_idx] = xxx * radial
            phi[:, basis_idx + 1] = yyy * radial
            phi[:, basis_idx + 2] = zzz * radial
            phi[:, basis_idx + 3] = xxy * radial
            phi[:, basis_idx + 4] = xxz * radial
            phi[:, basis_idx + 5] = yyz * radial
            phi[:, basis_idx + 6] = yzz * radial
            phi[:, basis_idx + 7] = xyy * radial
            phi[:, basis_idx + 8] = xzz * radial
            phi[:, basis_idx + 9] = xyz * radial
            basis_idx += 10

        # TODO: Implement G, H and Spherical/Cartesian support

    return phi


def _eval_contraction(exps, coeffs, r2):
    """
    Evaluates the radial contraction: sum_k c_k * exp(-alpha_k * r^2)

    Args:
        exps: Array of exponents (n_primitives,)
        coeffs: Array of contraction coefficients (n_primitives,) or (1, n_primitives)
        r2: Array of squared distances (n_points,)

    Returns:
        res: Array of contracted values (n_points,)
    """
    # Ensure coeffs is 1D
    coeffs = np.asarray(coeffs).flatten()

    # Add numerical stability checks
    res = np.zeros_like(r2)
    for a, c in zip(exps, coeffs):
        # Avoid overflow in exp
        arg = -a * r2
        arg = np.where(arg < -700, -700, arg)  # Prevent exp(-inf) = 0
        res += c * np.exp(arg)
    return res
