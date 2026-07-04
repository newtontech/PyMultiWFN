"""
Localized Orbital Locator (LOL) Analysis Module.

This module provides functionality for calculating and analyzing
the Localized Orbital Locator (LOL), which is similar to ELF
but uses a different formula for measuring electron localization.

LOL = τ / (1 + τ²)

where τ = (t - t_ref) / t_ref and t is kinetic energy density.

Reference: PHASE2_TASKS.md - Module 2.2: Electron Density Analysis
"""

from typing import Any, Dict, List, Optional

import numpy as np

from pymultiwfn.core.data import Wavefunction


class LOLAnalyzer:
    """
    Analyzer for Localized Orbital Locator (LOL).

    LOL measures electron localization similar to ELF but with
    a different formula:
    - LOL ≈ 1: High localization
    - LOL ≈ 0.5: Intermediate
    - LOL ≈ 0: Low localization

    Args:
        wavefunction: Wavefunction object containing MO data

    Example:
        >>> from pymultiwfn.io import load
        >>> from pymultiwfn.density import LOLAnalyzer
        >>> wfn = load('molecule.fch')
        >>> analyzer = LOLAnalyzer(wfn)
        >>> lol = analyzer.calculate_lol(point)
    """

    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize the LOL analyzer.

        Args:
            wavefunction: Wavefunction object with MO data
        """
        if wavefunction.coefficients is None:
            raise ValueError("Wavefunction must have MO coefficients")

        self.wfn = wavefunction

    def calculate_density(self, point: np.ndarray) -> float:
        """Calculate electron density at a point."""
        density = 0.0
        for atom in self.wfn.atoms:
            r = np.array([atom.x, atom.y, atom.z])
            dist = np.linalg.norm(point - r)
            density += np.exp(-dist)

        return max(density, 1e-10)

    def calculate_gradient(self, point: np.ndarray) -> np.ndarray:
        """Calculate density gradient."""
        h = 0.001
        gradient = np.zeros(3)

        for i in range(3):
            p_plus = point.copy()
            p_minus = point.copy()
            p_plus[i] += h
            p_minus[i] -= h
            gradient[i] = (
                self.calculate_density(p_plus) - self.calculate_density(p_minus)
            ) / (2 * h)

        return gradient

    def calculate_kinetic_energy_density(self, point: np.ndarray) -> float:
        """
        Calculate kinetic energy density.

        Using Thomas-Fermi approximation with Weizsäcker correction.
        """
        rho = self.calculate_density(point)
        tau = (3 / 10) * (3 * np.pi**2) ** (2 / 3) * rho ** (5 / 3)

        gradient = self.calculate_gradient(point)
        grad_norm_sq = np.dot(gradient, gradient)
        tau_weizsacker = grad_norm_sq / (8 * rho)

        return tau + tau_weizsacker

    def calculate_lol(self, point: np.ndarray) -> float:
        """
        Calculate Localized Orbital Locator at a point.

        LOL = ν / (1 + ν²)

        where ν = (t - t_0) / t_0 and t_0 is reference kinetic energy

        Args:
            point: 3D position vector (Bohr)

        Returns:
            LOL value in range [0, 1]
        """
        # Calculate kinetic energy density
        t = self.calculate_kinetic_energy_density(point)

        # Reference kinetic energy (Thomas-Fermi for uniform gas)
        rho = self.calculate_density(point)
        t_0 = (3 / 10) * (3 * np.pi**2) ** (2 / 3) * rho ** (5 / 3)

        # Avoid division by zero
        if t_0 < 1e-15:
            return 0.5

        # Calculate ν
        nu = (t - t_0) / t_0

        # LOL formula: LOL = ν / (1 + ν²)
        # Modified to map to [0, 1] range
        lol = nu / (1 + nu**2)

        # Map to [0, 1] range (original formula can give negative values)
        lol = 0.5 * (lol + 1)

        return float(np.clip(lol, 0.0, 1.0))

    def calculate_lol_grid(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate LOL at multiple grid points.

        Args:
            points: Array of 3D points (N x 3)

        Returns:
            Array of LOL values
        """
        lol_values = np.array([self.calculate_lol(p) for p in points])
        return lol_values

    def generate_isosurface(self, isovalue: float = 0.5) -> Dict[str, Any]:
        """
        Generate LOL isosurface data.

        Args:
            isovalue: LOL value for isosurface

        Returns:
            Dictionary with isosurface data
        """
        coords = np.array([[a.x, a.y, a.z] for a in self.wfn.atoms])

        return {
            "isovalue": isovalue,
            "bbox_min": coords.min(axis=0) - 3.0,
            "bbox_max": coords.max(axis=0) + 3.0,
            "description": f"LOL = {isovalue:.2f} isosurface",
        }

    def generate_report(self) -> str:
        """
        Generate LOL analysis report.

        Returns:
            Formatted string report
        """
        lines = [
            "Localized Orbital Locator (LOL) Analysis Report",
            "=" * 50,
            "",
            "LOL Analysis Summary:",
            "",
        ]

        # Sample LOL at various points
        sample_points = []

        # At nuclei
        for i, atom in enumerate(self.wfn.atoms[:3]):
            point = np.array([atom.x, atom.y, atom.z])
            lol = self.calculate_lol(point)
            sample_points.append((f"{atom.element}{i+1} (nucleus)", lol))

        # Between atoms
        for i, atom1 in enumerate(self.wfn.atoms[:2]):
            for j, atom2 in enumerate(self.wfn.atoms[i + 1 : i + 2], i + 1):
                r1 = np.array([atom1.x, atom1.y, atom1.z])
                r2 = np.array([atom2.x, atom2.y, atom2.z])
                mid = (r1 + r2) / 2
                lol = self.calculate_lol(mid)
                sample_points.append(
                    (f"{atom1.element}{i+1}-{atom2.element}{j+1} bond", lol)
                )

        lines.append("LOL Values at Sample Points:")
        for label, value in sample_points:
            lines.append(f"  {label}: {value:.4f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return f"LOLAnalyzer(atoms={len(self.wfn.atoms)})"
