"""
Laplacian Analysis Module.

This module provides functionality for analyzing the Laplacian of
electron density (∇²ρ), which reveals electron concentration and
depletion regions important for chemical bonding analysis.

Reference: PHASE2_TASKS.md - Module 2.2: Electron Density Analysis
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pymultiwfn.core.data import Wavefunction


class LaplacianAnalyzer:
    """
    Analyzer for Laplacian of electron density.

    The Laplacian ∇²ρ = ∂²ρ/∂x² + ∂²ρ/∂y² + ∂²ρ/∂z² indicates:
    - ∇²ρ < 0: Electron concentration (covalent character)
    - ∇²ρ > 0: Electron depletion (ionic/closed-shell character)

    Args:
        wavefunction: Wavefunction object containing MO data

    Example:
        >>> from pymultiwfn.io import load
        >>> from pymultiwfn.density import LaplacianAnalyzer
        >>> wfn = load('molecule.fch')
        >>> analyzer = LaplacianAnalyzer(wfn)
        >>> laplacian = analyzer.calculate_laplacian(point)
    """

    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize the Laplacian analyzer.

        Args:
            wavefunction: Wavefunction object with MO data
        """
        if wavefunction.coefficients is None:
            raise ValueError("Wavefunction must have MO coefficients")

        self.wfn = wavefunction
        self._concentration_regions = None
        self._depletion_regions = None

    def calculate_density(self, point: np.ndarray) -> float:
        """
        Calculate electron density at a point.

        Args:
            point: 3D position vector (Bohr)

        Returns:
            Electron density value
        """
        # Simplified density calculation
        density = 0.0
        for atom in self.wfn.atoms:
            r = np.array([atom.x, atom.y, atom.z])
            dist = np.linalg.norm(point - r)
            density += np.exp(-dist)

        return density

    def calculate_gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of electron density.

        Args:
            point: 3D position vector (Bohr)

        Returns:
            Gradient vector (3D)
        """
        h = 0.001
        gradient = np.zeros(3)

        for i in range(3):
            point_plus = point.copy()
            point_minus = point.copy()
            point_plus[i] += h
            point_minus[i] -= h

            gradient[i] = (
                self.calculate_density(point_plus) - self.calculate_density(point_minus)
            ) / (2 * h)

        return gradient

    def calculate_hessian(self, point: np.ndarray) -> np.ndarray:
        """
        Calculate Hessian matrix of electron density.

        Args:
            point: 3D position vector (Bohr)

        Returns:
            Hessian matrix (3x3)
        """
        h = 0.001
        hessian = np.zeros((3, 3))

        # Diagonal elements
        for i in range(3):
            point_plus = point.copy()
            point_minus = point.copy()
            point_plus[i] += h
            point_minus[i] -= h

            hessian[i, i] = (
                self.calculate_density(point_plus)
                - 2 * self.calculate_density(point)
                + self.calculate_density(point_minus)
            ) / (h**2)

        # Off-diagonal elements
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

    def calculate_laplacian(self, point: np.ndarray) -> float:
        """
        Calculate Laplacian of electron density at a point.

        The Laplacian is the trace of the Hessian:
        ∇²ρ = ∂²ρ/∂x² + ∂²ρ/∂y² + ∂²ρ/∂z²

        Args:
            point: 3D position vector (Bohr)

        Returns:
            Laplacian value
        """
        hessian = self.calculate_hessian(point)
        return float(np.trace(hessian))

    def calculate_laplacian_grid(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate Laplacian at multiple grid points.

        Args:
            points: Array of 3D points (N x 3)

        Returns:
            Array of Laplacian values
        """
        laplacians = np.array([self.calculate_laplacian(p) for p in points])
        return laplacians

    def get_concentration_regions(self) -> List[Dict[str, Any]]:
        """
        Identify electron concentration regions (∇²ρ < 0).

        These regions indicate covalent bonding character where
        electron density is locally concentrated.

        Returns:
            List of concentration region dictionaries
        """
        if self._concentration_regions is not None:
            return self._concentration_regions

        regions = []

        # Sample points around each bond
        for i, atom1 in enumerate(self.wfn.atoms):
            for j, atom2 in enumerate(self.wfn.atoms[i + 1 :], i + 1):
                r1 = np.array([atom1.x, atom1.y, atom1.z])
                r2 = np.array([atom2.x, atom2.y, atom2.z])

                # Sample along bond
                for t in np.linspace(0.2, 0.8, 5):
                    point = r1 + t * (r2 - r1)
                    laplacian = self.calculate_laplacian(point)

                    if laplacian < 0:
                        regions.append(
                            {"position": point, "laplacian": laplacian, "bond": (i, j)}
                        )

        self._concentration_regions = regions
        return regions

    def get_depletion_regions(self) -> List[Dict[str, Any]]:
        """
        Identify electron depletion regions (∇²ρ > 0).

        These regions indicate closed-shell/ionic character where
        electron density is locally depleted.

        Returns:
            List of depletion region dictionaries
        """
        if self._depletion_regions is not None:
            return self._depletion_regions

        regions = []

        # Sample points around nuclei (outer regions)
        for i, atom in enumerate(self.wfn.atoms):
            center = np.array([atom.x, atom.y, atom.z])

            # Sample spherical shell around nucleus
            for r in [0.5, 1.0, 1.5]:
                for theta in np.linspace(0, np.pi, 3):
                    for phi in np.linspace(0, 2 * np.pi, 6):
                        point = center + r * np.array(
                            [
                                np.sin(theta) * np.cos(phi),
                                np.sin(theta) * np.sin(phi),
                                np.cos(theta),
                            ]
                        )
                        laplacian = self.calculate_laplacian(point)

                        if laplacian > 0:
                            regions.append(
                                {"position": point, "laplacian": laplacian, "atom": i}
                            )

        self._depletion_regions = regions
        return regions

    def classify_bond(self, point: np.ndarray) -> str:
        """
        Classify bond type based on Laplacian sign.

        Classification based on ∇²ρ at bond critical point:
        - ∇²ρ < 0: Shared (covalent) interaction
        - ∇²ρ > 0: Closed-shell (ionic) interaction

        Args:
            point: 3D position vector (typically BCP)

        Returns:
            Bond type string: 'shared', 'closed-shell', or 'transitional'
        """
        laplacian = self.calculate_laplacian(point)

        if laplacian < -0.1:
            return "shared"
        elif laplacian > 0.1:
            return "closed-shell"
        else:
            return "transitional"

    def get_laplacian_at_bcp(self, bond_index: int = 0) -> float:
        """
        Get Laplacian value at bond critical point.

        Args:
            bond_index: Index of bond

        Returns:
            Laplacian value at BCP
        """
        bonds = []
        for i, atom1 in enumerate(self.wfn.atoms):
            for j, atom2 in enumerate(self.wfn.atoms[i + 1 :], i + 1):
                bonds.append((i, j))

        if bond_index >= len(bonds):
            raise IndexError(f"Bond index {bond_index} out of range")

        i, j = bonds[bond_index]
        atom1 = self.wfn.atoms[i]
        atom2 = self.wfn.atoms[j]

        # BCP is approximately at bond midpoint
        r1 = np.array([atom1.x, atom1.y, atom1.z])
        r2 = np.array([atom2.x, atom2.y, atom2.z])
        bcp = (r1 + r2) / 2

        return self.calculate_laplacian(bcp)

    def generate_isosurface(self, isovalue: float = 0.0) -> Dict[str, Any]:
        """
        Generate Laplacian isosurface data.

        Args:
            isovalue: Laplacian value for isosurface (default: 0.0)

        Returns:
            Dictionary with isosurface data
        """
        # Simplified: return bounding box and isovalue
        coords = np.array([[a.x, a.y, a.z] for a in self.wfn.atoms])

        return {
            "isovalue": isovalue,
            "bbox_min": coords.min(axis=0) - 2.0,
            "bbox_max": coords.max(axis=0) + 2.0,
            "description": f"Laplacian = {isovalue:.3f} isosurface",
        }

    def generate_report(self) -> str:
        """
        Generate Laplacian analysis report.

        Returns:
            Formatted string report
        """
        concentration = self.get_concentration_regions()
        depletion = self.get_depletion_regions()

        lines = [
            "Laplacian Analysis Report",
            "=" * 50,
            "",
            f"Concentration regions (∇²ρ < 0): {len(concentration)}",
            f"Depletion regions (∇²ρ > 0): {len(depletion)}",
            "",
            "Bond Classification:",
        ]

        # Classify each bond
        for i, atom1 in enumerate(self.wfn.atoms):
            for j, atom2 in enumerate(self.wfn.atoms[i + 1 :], i + 1):
                r1 = np.array([atom1.x, atom1.y, atom1.z])
                r2 = np.array([atom2.x, atom2.y, atom2.z])
                bcp = (r1 + r2) / 2

                laplacian = self.calculate_laplacian(bcp)
                bond_type = self.classify_bond(bcp)

                lines.append(
                    f"  {atom1.element}{i+1}-{atom2.element}{j+1}: "
                    f"∇²ρ = {laplacian:.4f} ({bond_type})"
                )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return f"LaplacianAnalyzer(atoms={len(self.wfn.atoms)})"
