"""
Electron density gradient calculation module.
Computes analytical gradients of electron density.
"""

import numpy as np

from pymultiwfn.core.data import Wavefunction
from pymultiwfn.math.basis import _eval_contraction


def calc_density_gradient(wfn: Wavefunction, coords: np.ndarray) -> np.ndarray:
    """
    Calculates the electron density gradient at given coordinates.

    The gradient ∇ρ(r) is computed analytically using the chain rule:
    ∇ρ(r) = Σ_ij P_ij [∇φ_i(r) φ_j(r) + φ_i(r) ∇φ_j(r)]

    Args:
        wfn: Wavefunction object containing basis set and coefficient info.
        coords: (N, 3) array of Cartesian coordinates.

    Returns:
        grad: (N, 3) array of gradient vectors (dρ/dx, dρ/dy, dρ/dz).
    """
    n_points = coords.shape[0]

    # Evaluate basis functions and their gradients
    phi = _evaluate_basis_all(wfn, coords)
    grad_phi = _evaluate_basis_gradient_all(wfn, coords)

    grad = np.zeros((n_points, 3))

    # Alpha / Total Density Gradient
    if wfn.coefficients is not None and wfn.occupations is not None:
        P = _make_density_matrix(wfn.coefficients, wfn.occupations)
        grad += _contract_gradient(phi, grad_phi, P)

    # Beta Density Gradient (if unrestricted)
    if wfn.is_unrestricted and wfn.coefficients_beta is not None:
        if wfn.occupations_beta is not None:
            P_beta = _make_density_matrix(wfn.coefficients_beta, wfn.occupations_beta)
            grad += _contract_gradient(phi, grad_phi, P_beta)

    return grad


def calc_density_laplacian(wfn: Wavefunction, coords: np.ndarray) -> np.ndarray:
    """
    Calculates the Laplacian of electron density (∇²ρ).

    Args:
        wfn: Wavefunction object.
        coords: (N, 3) array of Cartesian coordinates.

    Returns:
        laplacian: (N,) array of Laplacian values.
    """
    n_points = coords.shape[0]

    # Evaluate basis functions, gradients, and Laplacians
    phi = _evaluate_basis_all(wfn, coords)
    grad_phi = _evaluate_basis_gradient_all(wfn, coords)
    lap_phi = _evaluate_basis_laplacian_all(wfn, coords)

    laplacian = np.zeros(n_points)

    # Alpha / Total Density Laplacian
    if wfn.coefficients is not None and wfn.occupations is not None:
        P = _make_density_matrix(wfn.coefficients, wfn.occupations)
        laplacian += _contract_laplacian(phi, grad_phi, lap_phi, P)

    # Beta Density Laplacian (if unrestricted)
    if wfn.is_unrestricted and wfn.coefficients_beta is not None:
        if wfn.occupations_beta is not None:
            P_beta = _make_density_matrix(wfn.coefficients_beta, wfn.occupations_beta)
            laplacian += _contract_laplacian(phi, grad_phi, lap_phi, P_beta)

    return laplacian


def _evaluate_basis_all(wfn: Wavefunction, coords: np.ndarray) -> np.ndarray:
    """
    Evaluates all basis functions at given coordinates.
    Wrapper for basis.evaluate_basis for compatibility.
    """
    from pymultiwfn.math.basis import evaluate_basis

    return evaluate_basis(wfn, coords)


def _evaluate_basis_gradient_all(wfn: Wavefunction, coords: np.ndarray) -> np.ndarray:
    """
    Evaluates gradients of all basis functions at given coordinates.

    Returns:
        grad_phi: (N_points, 3, N_basis) array where
                  grad_phi[:, 0, :] = dφ/dx
                  grad_phi[:, 1, :] = dφ/dy
                  grad_phi[:, 2, :] = dφ/dz
    """

    n_points = coords.shape[0]
    n_basis = wfn.num_basis
    grad_phi = np.zeros((n_points, 3, n_basis))

    # Pre-allocate arrays
    r_vec = np.empty((n_points, 3))
    r2 = np.empty(n_points)

    basis_idx = 0

    for shell in wfn.shells:
        # Get atom coordinates
        atom = wfn.atoms[shell.center_idx]
        atom_coord = np.array([atom.x, atom.y, atom.z])

        # Vector r = R_point - R_atom
        np.subtract(coords, atom_coord, out=r_vec)
        np.sum(np.square(r_vec), axis=1, out=r2)

        # S shell (type=0)
        if shell.type == 0:
            radial = _eval_contraction(shell.exponents, shell.coefficients, r2)
            grad_radial = _eval_contraction_gradient(
                shell.exponents, shell.coefficients, r2, r_vec
            )
            # ∇φ = ∇radial (no angular component)
            grad_phi[:, :, basis_idx] = grad_radial
            basis_idx += 1

        # P shell (type=1)
        elif shell.type == 1:
            radial = _eval_contraction(shell.exponents, shell.coefficients, r2)
            grad_radial = _eval_contraction_gradient(
                shell.exponents, shell.coefficients, r2, r_vec
            )

            # φ_x = x * radial
            # ∇φ_x = [1*radial + x*d/dx(radial), x*d/dy(radial), x*d/dz(radial)]
            grad_phi[:, 0, basis_idx] = radial + r_vec[:, 0] * grad_radial[:, 0]
            grad_phi[:, 1, basis_idx] = r_vec[:, 0] * grad_radial[:, 1]
            grad_phi[:, 2, basis_idx] = r_vec[:, 0] * grad_radial[:, 2]

            # φ_y = y * radial
            grad_phi[:, 0, basis_idx + 1] = r_vec[:, 1] * grad_radial[:, 0]
            grad_phi[:, 1, basis_idx + 1] = radial + r_vec[:, 1] * grad_radial[:, 1]
            grad_phi[:, 2, basis_idx + 1] = r_vec[:, 1] * grad_radial[:, 2]

            # φ_z = z * radial
            grad_phi[:, 0, basis_idx + 2] = r_vec[:, 2] * grad_radial[:, 0]
            grad_phi[:, 1, basis_idx + 2] = r_vec[:, 2] * grad_radial[:, 1]
            grad_phi[:, 2, basis_idx + 2] = radial + r_vec[:, 2] * grad_radial[:, 2]

            basis_idx += 3

        # SP shell (type=-1)
        elif shell.type == -1:
            # S component
            grad_radial_s = _eval_contraction_gradient(
                shell.exponents, shell.coefficients[0], r2, r_vec
            )
            grad_phi[:, :, basis_idx] = grad_radial_s.T
            basis_idx += 1

            # P component
            radial_p = _eval_contraction(shell.exponents, shell.coefficients[1], r2)
            grad_radial_p = _eval_contraction_gradient(
                shell.exponents, shell.coefficients[1], r2, r_vec
            )

            grad_phi[:, 0, basis_idx] = radial_p + r_vec[:, 0] * grad_radial_p[:, 0]
            grad_phi[:, 1, basis_idx] = r_vec[:, 0] * grad_radial_p[:, 1]
            grad_phi[:, 2, basis_idx] = r_vec[:, 0] * grad_radial_p[:, 2]

            grad_phi[:, 0, basis_idx + 1] = r_vec[:, 1] * grad_radial_p[:, 0]
            grad_phi[:, 1, basis_idx + 1] = radial_p + r_vec[:, 1] * grad_radial_p[:, 1]
            grad_phi[:, 2, basis_idx + 1] = r_vec[:, 1] * grad_radial_p[:, 2]

            grad_phi[:, 0, basis_idx + 2] = r_vec[:, 2] * grad_radial_p[:, 0]
            grad_phi[:, 1, basis_idx + 2] = r_vec[:, 2] * grad_radial_p[:, 1]
            grad_phi[:, 2, basis_idx + 2] = radial_p + r_vec[:, 2] * grad_radial_p[:, 2]

            basis_idx += 3

        # D shell (type=2) - Cartesian
        elif shell.type == 2:
            radial = _eval_contraction(shell.exponents, shell.coefficients, r2)
            grad_radial = _eval_contraction_gradient(
                shell.exponents, shell.coefficients, r2, r_vec
            )

            x, y, z = r_vec[:, 0], r_vec[:, 1], r_vec[:, 2]
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z

            # φ_xx = xx * radial
            grad_phi[:, 0, basis_idx] = 2 * x * radial + xx * grad_radial[:, 0]
            grad_phi[:, 1, basis_idx] = xx * grad_radial[:, 1]
            grad_phi[:, 2, basis_idx] = xx * grad_radial[:, 2]

            # φ_yy = yy * radial
            grad_phi[:, 0, basis_idx + 1] = yy * grad_radial[:, 0]
            grad_phi[:, 1, basis_idx + 1] = 2 * y * radial + yy * grad_radial[:, 1]
            grad_phi[:, 2, basis_idx + 1] = yy * grad_radial[:, 2]

            # φ_zz = zz * radial
            grad_phi[:, 0, basis_idx + 2] = zz * grad_radial[:, 0]
            grad_phi[:, 1, basis_idx + 2] = zz * grad_radial[:, 1]
            grad_phi[:, 2, basis_idx + 2] = 2 * z * radial + zz * grad_radial[:, 2]

            # φ_xy = xy * radial
            grad_phi[:, 0, basis_idx + 3] = y * radial + xy * grad_radial[:, 0]
            grad_phi[:, 1, basis_idx + 3] = x * radial + xy * grad_radial[:, 1]
            grad_phi[:, 2, basis_idx + 3] = xy * grad_radial[:, 2]

            # φ_xz = xz * radial
            grad_phi[:, 0, basis_idx + 4] = z * radial + xz * grad_radial[:, 0]
            grad_phi[:, 1, basis_idx + 4] = xz * grad_radial[:, 1]
            grad_phi[:, 2, basis_idx + 4] = x * radial + xz * grad_radial[:, 2]

            # φ_yz = yz * radial
            grad_phi[:, 0, basis_idx + 5] = yz * grad_radial[:, 0]
            grad_phi[:, 1, basis_idx + 5] = z * radial + yz * grad_radial[:, 1]
            grad_phi[:, 2, basis_idx + 5] = y * radial + yz * grad_radial[:, 2]

            basis_idx += 6

        # TODO: Implement F, G, H and Spherical harmonics

    return grad_phi


def _evaluate_basis_laplacian_all(wfn: Wavefunction, coords: np.ndarray) -> np.ndarray:
    """
    Evaluates Laplacians of all basis functions at given coordinates.

    Returns:
        lap_phi: (N_points, N_basis) array of Laplacian values.
    """

    n_points = coords.shape[0]
    n_basis = wfn.num_basis
    lap_phi = np.zeros((n_points, n_basis))

    # Pre-allocate arrays
    r_vec = np.empty((n_points, 3))
    r2 = np.empty(n_points)

    basis_idx = 0

    for shell in wfn.shells:
        # Get atom coordinates
        atom = wfn.atoms[shell.center_idx]
        atom_coord = np.array([atom.x, atom.y, atom.z])

        # Vector r = R_point - R_atom
        np.subtract(coords, atom_coord, out=r_vec)
        np.sum(np.square(r_vec), axis=1, out=r2)

        # S shell (type=0)
        if shell.type == 0:
            lap_phi[:, basis_idx] = _eval_contraction_laplacian(
                shell.exponents, shell.coefficients, r2
            )
            basis_idx += 1

        # P shell (type=1)
        elif shell.type == 1:
            lap_radial = _eval_contraction_laplacian(
                shell.exponents, shell.coefficients, r2
            )
            grad_radial = _eval_contraction_gradient(
                shell.exponents, shell.coefficients, r2, r_vec
            )

            x, y, z = r_vec[:, 0], r_vec[:, 1], r_vec[:, 2]

            # φ_x = x * radial
            # ∇²φ_x = 2*d/dx(radial) + x*∇²radial
            lap_phi[:, basis_idx] = 2 * grad_radial[:, 0] + x * lap_radial

            # φ_y = y * radial
            lap_phi[:, basis_idx + 1] = 2 * grad_radial[:, 1] + y * lap_radial

            # φ_z = z * radial
            lap_phi[:, basis_idx + 2] = 2 * grad_radial[:, 2] + z * lap_radial

            basis_idx += 3

        # SP shell (type=-1)
        elif shell.type == -1:
            # S component
            lap_phi[:, basis_idx] = _eval_contraction_laplacian(
                shell.exponents, shell.coefficients[0], r2
            )
            basis_idx += 1

            # P component
            lap_radial = _eval_contraction_laplacian(
                shell.exponents, shell.coefficients[1], r2
            )
            grad_radial = _eval_contraction_gradient(
                shell.exponents, shell.coefficients[1], r2, r_vec
            )

            x, y, z = r_vec[:, 0], r_vec[:, 1], r_vec[:, 2]

            lap_phi[:, basis_idx] = 2 * grad_radial[:, 0] + x * lap_radial
            lap_phi[:, basis_idx + 1] = 2 * grad_radial[:, 1] + y * lap_radial
            lap_phi[:, basis_idx + 2] = 2 * grad_radial[:, 2] + z * lap_radial

            basis_idx += 3

        # D shell (type=2) - Cartesian
        elif shell.type == 2:
            radial = _eval_contraction(shell.exponents, shell.coefficients, r2)
            lap_radial = _eval_contraction_laplacian(
                shell.exponents, shell.coefficients, r2
            )
            grad_radial = _eval_contraction_gradient(
                shell.exponents, shell.coefficients, r2, r_vec
            )

            x, y, z = r_vec[:, 0], r_vec[:, 1], r_vec[:, 2]
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z

            # φ_xx = xx * radial
            # ∇²φ_xx = 2*radial + 4*x*d/dx(radial) + xx*∇²radial
            lap_phi[:, basis_idx] = (
                2 * radial + 4 * x * grad_radial[:, 0] + xx * lap_radial
            )

            # φ_yy = yy * radial
            lap_phi[:, basis_idx + 1] = (
                2 * radial + 4 * y * grad_radial[:, 1] + yy * lap_radial
            )

            # φ_zz = zz * radial
            lap_phi[:, basis_idx + 2] = (
                2 * radial + 4 * z * grad_radial[:, 2] + zz * lap_radial
            )

            # φ_xy = xy * radial
            # ∇²φ_xy = 2*(x*d/dy(radial) + y*d/dx(radial)) + xy*∇²radial
            lap_phi[:, basis_idx + 3] = (
                2 * (x * grad_radial[:, 1] + y * grad_radial[:, 0]) + xy * lap_radial
            )

            # φ_xz = xz * radial
            lap_phi[:, basis_idx + 4] = (
                2 * (x * grad_radial[:, 2] + z * grad_radial[:, 0]) + xz * lap_radial
            )

            # φ_yz = yz * radial
            lap_phi[:, basis_idx + 5] = (
                2 * (y * grad_radial[:, 2] + z * grad_radial[:, 1]) + yz * lap_radial
            )

            basis_idx += 6

    return lap_phi


def _eval_contraction_gradient(
    exps: np.ndarray, coeffs: np.ndarray, r2: np.ndarray, r_vec: np.ndarray
) -> np.ndarray:
    """
    Evaluates gradient of the radial contraction.

    For exp(-αr²): ∇[exp(-αr²)] = -2α * r * exp(-αr²)

    Returns:
        grad_radial: (N_points, 3) array of gradients.
    """
    grad = np.zeros((r2.shape[0], 3))

    for a, c in zip(exps, coeffs):
        # Avoid overflow
        arg = -a * r2
        arg = np.where(arg < -700, -700, arg)
        exp_val = np.exp(arg)

        # Gradient: -2α * r * exp(-αr²)
        factor = -2 * a * c * exp_val
        grad += factor[:, np.newaxis] * r_vec

    return grad


def _eval_contraction_laplacian(
    exps: np.ndarray, coeffs: np.ndarray, r2: np.ndarray
) -> np.ndarray:
    """
    Evaluates Laplacian of the radial contraction.

    For exp(-αr²): ∇²[exp(-αr²)] = (4α²r² - 6α) * exp(-αr²)

    Returns:
        laplacian: (N_points,) array of Laplacian values.
    """
    lap = np.zeros_like(r2)

    for a, c in zip(exps, coeffs):
        # Avoid overflow
        arg = -a * r2
        arg = np.where(arg < -700, -700, arg)
        exp_val = np.exp(arg)

        # Laplacian: (4α²r² - 6α) * exp(-αr²)
        factor = (4 * a * a * r2 - 6 * a) * c * exp_val
        lap += factor

    return lap


def _make_density_matrix(coeffs: np.ndarray, occs: np.ndarray) -> np.ndarray:
    """
    Constructs density matrix P from MO coefficients and occupations.

    P = C.T * diag(occ) * C

    Args:
        coeffs: (nmo, nbasis) array of MO coefficients.
        occs: (nmo,) array of orbital occupations.

    Returns:
        P: (nbasis, nbasis) density matrix.
    """
    # Optimization: Only use occupied orbitals
    occ_idx = occs > 1e-8

    if not np.any(occ_idx):
        return np.zeros((coeffs.shape[1], coeffs.shape[1]))

    C_occ = coeffs[occ_idx]
    n_occ = occs[occ_idx]

    # P_mu_nu = sum_i n_i C_i_mu C_i_nu
    P = np.einsum("i,ij,ik->jk", n_occ, C_occ, C_occ)

    return P


def _contract_gradient(
    phi: np.ndarray, grad_phi: np.ndarray, P: np.ndarray
) -> np.ndarray:
    """
    Contracts basis values and gradients with density matrix.

    ∇ρ(r) = 2 * Σ_ij P_ij φ_i(r) ∇φ_j(r)

    Args:
        phi: (N_points, N_basis) basis function values.
        grad_phi: (N_points, 3, N_basis) basis function gradients.
        P: (N_basis, N_basis) density matrix.

    Returns:
        grad: (N_points, 3) density gradient.
    """
    n_points = phi.shape[0]

    grad = np.zeros((n_points, 3))

    # For each coordinate direction
    for d in range(3):
        # grad_d = 2 * Σ_ij P_ij φ_i ∇φ_j_d
        # Using matrix multiplication: grad_d = 2 * phi @ P @ grad_phi_d.T
        # But grad_phi_d is (N_points, N_basis), so we need to transpose
        grad[:, d] = 2 * np.sum((phi @ P) * grad_phi[:, d, :], axis=1)

    return grad


def _contract_laplacian(
    phi: np.ndarray, grad_phi: np.ndarray, lap_phi: np.ndarray, P: np.ndarray
) -> np.ndarray:
    """
    Contracts basis values, gradients, and Laplacians with density matrix.

    ∇²ρ(r) = Σ_ij P_ij [∇²φ_i(r) φ_j(r) + 2 ∇φ_i(r) · ∇φ_j(r)]

    Args:
        phi: (N_points, N_basis) basis function values.
        grad_phi: (N_points, 3, N_basis) basis function gradients.
        lap_phi: (N_points, N_basis) basis function Laplacians.
        P: (N_basis, N_basis) density matrix.

    Returns:
        laplacian: (N_points,) density Laplacian.
    """
    n_points = phi.shape[0]

    # Compute P * lap_phi (N_basis, N_points)

    # First term: Σ_ij P_ij ∇²φ_i φ_j
    term1 = np.einsum("ij,ji->i", P, lap_phi.T * phi.T)

    # Second term: 2 * Σ_ij P_ij ∇φ_i · ∇φ_j
    # Compute dot product of gradients for each pair of basis functions
    # grad_dot[k, i, j] = ∇φ_i · ∇φ_j at point k
    term2 = np.zeros(n_points)
    for d in range(3):
        term2 += np.einsum("ij,ji->i", P, grad_phi[:, d, :].T * grad_phi[:, d, :].T)

    laplacian = term1 + 2 * term2

    return laplacian
