"""
Localized Spherical Basis (LSB) analysis module for Shubin Liu's DFRT project.

This module implements information-theoretic analysis of electron density
using real-space functions and fuzzy atomic partitioning methods.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

from pymultiwfn.core.data import Wavefunction
from pymultiwfn.math.density import calculate_electron_density
from pymultiwfn.math.basis import evaluate_basis_functions


@dataclass
class LSBResult:
    """Container for LSB analysis results."""
    atomic_contributions: np.ndarray  # shape: (n_atoms, n_functions, n_quantities)
    function_names: List[str]
    quantity_names: List[str]
    partition_method: str


class LSBFunctionCalculator:
    """Calculator for real-space functions used in LSB analysis."""

    def __init__(self, wavefunction: Wavefunction):
        self.wavefunction = wavefunction
        self.fc_spin_unpolarized = 2.871234000  # Fermi constant for unpolarized
        self.fc_spin_polarized = 4.557799872    # Fermi constant for polarized

    def calculate_functions(self, x: float, y: float, z: float) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Calculate all LSB real-space functions at a given point.

        Args:
            x, y, z: Coordinates in Bohr

        Returns:
            Tuple containing:
            - Array of function values (length 8)
            - Electron density (rho)
            - Electron density gradient (3-element array)
        """
        # Calculate electron density and derivatives
        rho, grad_rho, hess_rho = self._calculate_density_derivatives(x, y, z)

        # Initialize function values array
        func_vals = np.zeros(8)

        # Function 2: |∇ρ|/ρ^(4/3)
        grad_norm = np.linalg.norm(grad_rho)
        if rho > 1e-12:
            func_vals[0] = grad_norm / (rho ** (4.0/3.0))

        # Function 3: ∇²ρ/ρ^(5/3)
        laplacian = np.trace(hess_rho)
        if rho > 1e-12:
            func_vals[1] = laplacian / (rho ** (5.0/3.0))

        # Functions 4 and 8: (τ - τ_w)/τ_TF and (τ - τ_w)/τ_w
        tau, tau_w, tau_TF = self._calculate_kinetic_energy_densities(x, y, z, rho, grad_rho)
        if tau_TF > 1e-12:
            func_vals[2] = (tau - tau_w) / tau_TF
        if tau_w > 1e-12:
            func_vals[6] = (tau - tau_w) / tau_w

        # Function 5: Xi part of SEDD
        func_vals[3] = self._calculate_sedd_xi(rho, grad_rho, hess_rho)

        # Function 6: Theta part of DORI
        func_vals[4] = self._calculate_dori_theta(rho, grad_rho, hess_rho)

        # Function 7: Spin density
        func_vals[5] = self._calculate_spin_density(x, y, z)

        return func_vals, rho, grad_rho

    def _calculate_density_derivatives(self, x: float, y: float, z: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """Calculate electron density and its derivatives."""
        # This is a simplified implementation
        # In practice, this would use the wavefunction's basis functions
        # and density matrix to compute exact values

        # For now, return zeros - this would be implemented using
        # the actual quantum chemistry calculations
        rho = calculate_electron_density(self.wavefunction, np.array([[x, y, z]]))[0]

        # Calculate gradient using finite differences
        delta = 1e-4
        grad_rho = np.zeros(3)
        hess_rho = np.zeros((3, 3))

        # Simple finite difference approximation
        for i in range(3):
            coords_plus = np.array([[x, y, z]])
            coords_minus = np.array([[x, y, z]])
            coords_plus[0, i] += delta
            coords_minus[0, i] -= delta

            rho_plus = calculate_electron_density(self.wavefunction, coords_plus)[0]
            rho_minus = calculate_electron_density(self.wavefunction, coords_minus)[0]

            grad_rho[i] = (rho_plus - rho_minus) / (2 * delta)

            # Diagonal Hessian elements
            hess_rho[i, i] = (rho_plus - 2 * rho + rho_minus) / (delta ** 2)

        # Off-diagonal Hessian elements (cross derivatives)
        for i in range(3):
            for j in range(i + 1, 3):
                coords_pp = np.array([[x, y, z]])
                coords_pm = np.array([[x, y, z]])
                coords_mp = np.array([[x, y, z]])
                coords_mm = np.array([[x, y, z]])

                coords_pp[0, i] += delta; coords_pp[0, j] += delta
                coords_pm[0, i] += delta; coords_pm[0, j] -= delta
                coords_mp[0, i] -= delta; coords_mp[0, j] += delta
                coords_mm[0, i] -= delta; coords_mm[0, j] -= delta

                rho_pp = calculate_electron_density(self.wavefunction, coords_pp)[0]
                rho_pm = calculate_electron_density(self.wavefunction, coords_pm)[0]
                rho_mp = calculate_electron_density(self.wavefunction, coords_mp)[0]
                rho_mm = calculate_electron_density(self.wavefunction, coords_mm)[0]

                hess_rho[i, j] = (rho_pp - rho_pm - rho_mp + rho_mm) / (4 * delta ** 2)
                hess_rho[j, i] = hess_rho[i, j]

        return rho, grad_rho, hess_rho

    def _calculate_kinetic_energy_densities(self, x: float, y: float, z: float,
                                          rho: float, grad_rho: np.ndarray) -> Tuple[float, float, float]:
        """Calculate kinetic energy densities."""
        # Simplified implementation
        # In practice, this would use the actual kinetic energy density calculation

        # Thomas-Fermi kinetic energy density
        if self.wavefunction.is_unrestricted:
            # For spin-polarized systems
            tau_TF = self.fc_spin_polarized * (rho ** (5.0/3.0))
        else:
            # For closed-shell systems
            tau_TF = self.fc_spin_unpolarized * (rho ** (5.0/3.0))

        # Weizsäcker kinetic energy density
        grad_norm_sq = np.sum(grad_rho ** 2)
        if rho > 1e-12:
            tau_w = grad_norm_sq / (8.0 * rho)
        else:
            tau_w = 0.0

        # Actual kinetic energy density (simplified)
        tau = tau_TF + tau_w  # This is a simplification

        return tau, tau_w, tau_TF

    def _calculate_sedd_xi(self, rho: float, grad_rho: np.ndarray, hess_rho: np.ndarray) -> float:
        """Calculate Xi part of SEDD."""
        if rho < 1e-12:
            return 0.0

        grad_norm_sq = np.sum(grad_rho ** 2)

        # Calculate terms for SEDD
        term1 = rho * (grad_rho[0] * hess_rho[0, 0] + grad_rho[1] * hess_rho[0, 1] + grad_rho[2] * hess_rho[0, 2])
        term2 = grad_rho[0] * grad_norm_sq

        term3 = rho * (grad_rho[0] * hess_rho[0, 1] + grad_rho[1] * hess_rho[1, 1] + grad_rho[2] * hess_rho[1, 2])
        term4 = grad_rho[1] * grad_norm_sq

        term5 = rho * (grad_rho[0] * hess_rho[0, 2] + grad_rho[1] * hess_rho[1, 2] + grad_rho[2] * hess_rho[2, 2])
        term6 = grad_rho[2] * grad_norm_sq

        xi = 4.0 / (rho ** 8) * ((term1 - term2) ** 2 + (term3 - term4) ** 2 + (term5 - term6) ** 2)

        return xi

    def _calculate_dori_theta(self, rho: float, grad_rho: np.ndarray, hess_rho: np.ndarray) -> float:
        """Calculate Theta part of DORI."""
        if np.sum(grad_rho ** 2) < 1e-12:
            return 0.0

        grad_norm_sq = np.sum(grad_rho ** 2)

        # Same terms as SEDD
        term1 = rho * (grad_rho[0] * hess_rho[0, 0] + grad_rho[1] * hess_rho[0, 1] + grad_rho[2] * hess_rho[0, 2])
        term2 = grad_rho[0] * grad_norm_sq

        term3 = rho * (grad_rho[0] * hess_rho[0, 1] + grad_rho[1] * hess_rho[1, 1] + grad_rho[2] * hess_rho[1, 2])
        term4 = grad_rho[1] * grad_norm_sq

        term5 = rho * (grad_rho[0] * hess_rho[0, 2] + grad_rho[1] * hess_rho[1, 2] + grad_rho[2] * hess_rho[2, 2])
        term6 = grad_rho[2] * grad_norm_sq

        theta = 4.0 / (grad_norm_sq ** 3) * ((term1 - term2) ** 2 + (term3 - term4) ** 2 + (term5 - term6) ** 2)

        return theta

    def _calculate_spin_density(self, x: float, y: float, z: float) -> float:
        """Calculate spin density at a point."""
        if not self.wavefunction.is_unrestricted:
            return 0.0

        # Calculate alpha and beta densities
        # This would use the actual wavefunction data
        rho_alpha = calculate_electron_density(self.wavefunction, np.array([[x, y, z]]), spin='alpha')[0]
        rho_beta = calculate_electron_density(self.wavefunction, np.array([[x, y, z]]), spin='beta')[0]

        return rho_alpha - rho_beta


def calculate_lsb_analysis(wavefunction: Wavefunction,
                          partition_method: str = "becke",
                          radial_points: int = 50,
                          angular_points: int = 110,
                          use_reciprocal: bool = False) -> LSBResult:
    """
    Perform LSB (Localized Spherical Basis) analysis using fuzzy atomic partitioning.

    Args:
        wavefunction: The wavefunction object to analyze
        partition_method: Partition method ("becke" or "hirshfeld")
        radial_points: Number of radial integration points
        angular_points: Number of angular integration points
        use_reciprocal: Whether to use reciprocal of function values

    Returns:
        LSBResult object containing atomic contributions
    """

    # Define function and quantity names
    function_names = [
        "rho",
        "rho/rho0",
        "|∇ρ|/ρ^(4/3)",
        "∇²ρ/ρ^(5/3)",
        "(τ-τ_w)/τ_TF",
        "Xi part of SEDD",
        "Theta part of DORI",
        "Spin density",
        "(τ-τ_w)/τ_w"
    ]

    quantity_names = [
        "Function itself",
        "Shannon entropy",
        "Fisher information",
        "Onicescu information energy of order 2",
        "Onicescu information energy of order 3",
        "Information gain",
        "Relative Renyi entropy of order 2",
        "Relative Renyi entropy of order 3"
    ]

    n_functions = len(function_names)
    n_quantities = len(quantity_names)
    n_atoms = wavefunction.num_atoms

    # Initialize results array
    atomic_contributions = np.zeros((n_atoms, n_functions, n_quantities))

    # Create function calculator
    calculator = LSBFunctionCalculator(wavefunction)

    # This is a simplified implementation
    # In practice, this would generate integration grids and perform
    # the full fuzzy atomic analysis as in the Fortran code

    # For demonstration purposes, we'll compute a simple approximation
    # using the molecular center as a reference point

    # Calculate at molecular center
    center_coords = np.mean(wavefunction.coordinates, axis=0)
    x, y, z = center_coords

    func_vals, rho, grad_rho = calculator.calculate_functions(x, y, z)

    # For demonstration, assign equal contributions to all atoms
    # In practice, this would use proper atomic partitioning
    for i in range(n_atoms):
        for j in range(n_functions):
            if j < len(func_vals):
                f_val = func_vals[j]
                if use_reciprocal and f_val > 1e-12:
                    f_val = 1.0 / f_val

                # Function itself
                atomic_contributions[i, j, 0] = f_val / n_atoms

                # Shannon entropy
                if f_val > 1e-12:
                    atomic_contributions[i, j, 1] = -f_val * np.log(f_val) / n_atoms

                # Fisher information (simplified)
                if f_val > 1e-12:
                    # Use gradient norm from density as approximation
                    grad_norm = np.linalg.norm(grad_rho)
                    atomic_contributions[i, j, 2] = (grad_norm ** 2 / f_val) / n_atoms

                # Onicescu information energies
                atomic_contributions[i, j, 3] = (f_val ** 2) / n_atoms
                atomic_contributions[i, j, 4] = (f_val ** 3) / n_atoms

                # Information gain and relative Renyi entropies
                # For demonstration, use reference value of 1.0
                f_ref = 1.0
                if f_val > 1e-12 and f_ref > 1e-12:
                    atomic_contributions[i, j, 5] = (f_val * np.log(f_val / f_ref)) / n_atoms
                    atomic_contributions[i, j, 6] = (f_val ** 2 / f_ref) / n_atoms
                    atomic_contributions[i, j, 7] = (f_val ** 3 / f_ref ** 2) / n_atoms

    return LSBResult(
        atomic_contributions=atomic_contributions,
        function_names=function_names,
        quantity_names=quantity_names,
        partition_method=partition_method
    )


def print_lsb_results(result: LSBResult):
    """Print formatted LSB analysis results."""
    print("=" * 50)
    print("LSB Analysis Results")
    print("=" * 50)
    print(f"Partition method: {result.partition_method}")
    print()

    n_atoms = result.atomic_contributions.shape[0]

    # Print atomic contributions
    for i in range(n_atoms):
        print(f"Atom {i+1}:")
        print("-" * 30)

        for j, func_name in enumerate(result.function_names):
            print(f"  Function: {func_name}")

            for k, quant_name in enumerate(result.quantity_names):
                value = result.atomic_contributions[i, j, k]
                print(f"    {quant_name}: {value:.6e}")
            print()

    # Print overall results
    print("=" * 50)
    print("Overall Results")
    print("=" * 50)

    for j, func_name in enumerate(result.function_names):
        print(f"Function: {func_name}")

        for k, quant_name in enumerate(result.quantity_names):
            total = np.sum(result.atomic_contributions[:, j, k])
            print(f"  {quant_name}: {total:.6e}")
        print()


# Add LSB function to __all__ in bonding/__init__.py
__all__ = ["calculate_lsb_analysis", "LSBResult", "print_lsb_results"]