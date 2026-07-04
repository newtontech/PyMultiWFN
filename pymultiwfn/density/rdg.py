"""
Reduced Density Gradient (RDG) Analysis Module.

This module provides functionality for calculating and analyzing
the Reduced Density Gradient (RDG), which is used to identify
non-covalent interactions (hydrogen bonds, van der Waals, steric).

RDG = |∇ρ| / (2 * (3π²)^(1/3) * ρ^(4/3))

Reference: PHASE2_TASKS.md - Module 2.3: Advanced Density Analysis
"""

from typing import Any, Dict, List, Optional

import numpy as np

from pymultiwfn.core.data import Wavefunction


class RDGAnalyzer:
    """
    Analyzer for Reduced Density Gradient (RDG).

    RDG is used to identify non-covalent interactions:
    - Low RDG + Low density: Non-covalent interaction regions
    - Combined with sign(λ₂): classify interaction type

    Args:
        wavefunction: Wavefunction object containing MO data

    Example:
        >>> from pymultiwfn.io import load
        >>> from pymultiwfn.density import RDGAnalyzer
        >>> wfn = load('molecule.fch')
        >>> analyzer = RDGAnalyzer(wfn)
        >>> rdg = analyzer.calculate_rdg(point)
    """

    # Constant for RDG formula
    C = 2 * (3 * np.pi**2) ** (1 / 3)

    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize the RDG analyzer.

        Args:
            wavefunction: Wavefunction object with MO data
        """
        if wavefunction.coefficients is None:
            raise ValueError("Wavefunction must have MO coefficients")

        self.wfn = wavefunction
        self._nci_regions = None

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

    def calculate_rdg(self, point: np.ndarray) -> float:
        """
        Calculate Reduced Density Gradient at a point.

        RDG = |∇ρ| / (2 * (3π²)^(1/3) * ρ^(4/3))

        Args:
            point: 3D position vector (Bohr)

        Returns:
            RDG value (non-negative)
        """
        rho = self.calculate_density(point)
        gradient = self.calculate_gradient(point)
        grad_norm = np.linalg.norm(gradient)

        # RDG formula
        denominator = self.C * rho ** (4 / 3)

        if denominator > 1e-15:
            rdg = grad_norm / denominator
        else:
            rdg = 0.0

        return float(rdg)

    def calculate_rdg_grid(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate RDG at multiple grid points.

        Args:
            points: Array of 3D points (N x 3)

        Returns:
            Array of RDG values
        """
        rdg_values = np.array([self.calculate_rdg(p) for p in points])
        return rdg_values

    def calculate_hessian(self, point: np.ndarray) -> np.ndarray:
        """Calculate Hessian matrix of density."""
        h = 0.001
        hessian = np.zeros((3, 3))

        # Diagonal
        for i in range(3):
            p_plus = point.copy()
            p_minus = point.copy()
            p_plus[i] += h
            p_minus[i] -= h
            hessian[i, i] = (
                self.calculate_density(p_plus)
                - 2 * self.calculate_density(point)
                + self.calculate_density(p_minus)
            ) / (h**2)

        # Off-diagonal
        for i in range(3):
            for j in range(i + 1, 3):
                pp, pm, mp, mm = point.copy(), point.copy(), point.copy(), point.copy()
                pp[i] += h
                pp[j] += h
                pm[i] += h
                pm[j] -= h
                mp[i] -= h
                mp[j] += h
                mm[i] -= h
                mm[j] -= h

                hessian[i, j] = (
                    self.calculate_density(pp)
                    - self.calculate_density(pm)
                    - self.calculate_density(mp)
                    + self.calculate_density(mm)
                ) / (4 * h**2)
                hessian[j, i] = hessian[i, j]

        return hessian

    def calculate_sign_lambda2(self, point: np.ndarray) -> float:
        """
        Calculate sign of second eigenvalue of Hessian (λ₂).

        Used to classify interaction type:
        - sign(λ₂) < 0: Attractive (H-bonding)
        - sign(λ₂) ≈ 0: van der Waals
        - sign(λ₂) > 0: Repulsive (steric)

        Args:
            point: 3D position vector

        Returns:
            sign(λ₂) value
        """
        hessian = self.calculate_hessian(point)
        eigenvalues = np.linalg.eigvalsh(hessian)
        lambda2 = eigenvalues[1]  # Second eigenvalue

        return float(np.sign(lambda2))

    def identify_nci_regions(
        self, rdg_threshold: float = 0.5, density_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Identify non-covalent interaction regions.

        NCI regions are characterized by low RDG and low density.

        Args:
            rdg_threshold: Maximum RDG value for NCI region
            density_threshold: Maximum density for NCI region

        Returns:
            List of NCI region dictionaries
        """
        if self._nci_regions is not None:
            return self._nci_regions

        regions = []

        # Sample grid around molecule
        coords = np.array([[a.x, a.y, a.z] for a in self.wfn.atoms])
        center = coords.mean(axis=0)

        # Sample points in a box around molecule
        for x in np.linspace(center[0] - 2, center[0] + 2, 5):
            for y in np.linspace(center[1] - 2, center[1] + 2, 5):
                for z in np.linspace(center[2] - 2, center[2] + 2, 5):
                    point = np.array([x, y, z])

                    rho = self.calculate_density(point)
                    rdg = self.calculate_rdg(point)

                    # NCI criteria: low RDG + low density
                    if rdg < rdg_threshold and rho < density_threshold:
                        sign_l2 = self.calculate_sign_lambda2(point)

                        regions.append(
                            {
                                "position": point,
                                "rdg": rdg,
                                "density": rho,
                                "sign_lambda2": sign_l2,
                                "interaction_type": self._classify_by_sign(sign_l2),
                            }
                        )

        self._nci_regions = regions
        return regions

    def _classify_by_sign(self, sign: float) -> str:
        """Classify interaction type by sign(λ₂)."""
        if sign < -0.1:
            return "attractive"
        elif sign > 0.1:
            return "repulsive"
        else:
            return "vdW"

    def classify_interaction(self, point: np.ndarray) -> str:
        """
        Classify interaction type at a point.

        Args:
            point: 3D position vector

        Returns:
            Interaction type string
        """
        sign = self.calculate_sign_lambda2(point)
        return self._classify_by_sign(sign)

    def generate_isosurface(self, isovalue: float = 0.5) -> Dict[str, Any]:
        """
        Generate RDG isosurface data.

        Args:
            isovalue: RDG value for isosurface

        Returns:
            Dictionary with isosurface data
        """
        coords = np.array([[a.x, a.y, a.z] for a in self.wfn.atoms])

        return {
            "isovalue": isovalue,
            "bbox_min": coords.min(axis=0) - 4.0,
            "bbox_max": coords.max(axis=0) + 4.0,
            "description": f"RDG = {isovalue:.2f} isosurface",
        }

    def generate_report(self) -> str:
        """Generate RDG analysis report."""
        nci_regions = self.identify_nci_regions()

        attractive = len(
            [r for r in nci_regions if r["interaction_type"] == "attractive"]
        )
        repulsive = len(
            [r for r in nci_regions if r["interaction_type"] == "repulsive"]
        )
        vdw = len([r for r in nci_regions if r["interaction_type"] == "vdW"])

        lines = [
            "Reduced Density Gradient (RDG) Analysis Report",
            "=" * 50,
            "",
            f"Total NCI regions identified: {len(nci_regions)}",
            "",
            "Interaction Types:",
            f"  Attractive (H-bond): {attractive}",
            f"  Repulsive (steric): {repulsive}",
            f"  van der Waals: {vdw}",
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return f"RDGAnalyzer(atoms={len(self.wfn.atoms)})"
