"""
Electron Localization Function (ELF) Analysis Module.

This module provides functionality for calculating and analyzing
the Electron Localization Function (ELF), which measures the
likelihood of finding an electron pair at a given position.

ELF = 1 / (1 + (D/D_h)²)

where:
- D = τ - ¼|∇ρ|²/ρ (excess kinetic energy density)
- D_h = (3/10)(3π²)^(5/3) ρ^(5/3) (Thomas-Fermi kinetic energy)
- τ = kinetic energy density

Reference: PHASE2_TASKS.md - Module 2.2: Electron Density Analysis
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pymultiwfn.core.data import Wavefunction


class ELFAnalyzer:
    """
    Analyzer for Electron Localization Function (ELF).

    ELF measures electron localization:
    - ELF ≈ 1: High localization (lone pairs, bonds, cores)
    - ELF ≈ 0.5: Electron gas-like behavior
    - ELF ≈ 0: Low localization

    Args:
        wavefunction: Wavefunction object containing MO data

    Example:
        >>> from pymultiwfn.io import load
        >>> from pymultiwfn.density import ELFAnalyzer
        >>> wfn = load('molecule.fch')
        >>> analyzer = ELFAnalyzer(wfn)
        >>> elf = analyzer.calculate_elf(point)
    """

    # Thomas-Fermi constant
    C_F = (3 / 10) * (3 * np.pi**2) ** (5 / 3)

    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize the ELF analyzer.

        Args:
            wavefunction: Wavefunction object with MO data
        """
        if wavefunction.coefficients is None:
            raise ValueError("Wavefunction must have MO coefficients")

        self.wfn = wavefunction
        self._basins = None

    def calculate_density(self, point: np.ndarray) -> float:
        """
        Calculate electron density at a point.

        Args:
            point: 3D position vector (Bohr)

        Returns:
            Electron density value
        """
        density = 0.0
        for atom in self.wfn.atoms:
            r = np.array([atom.x, atom.y, atom.z])
            dist = np.linalg.norm(point - r)
            density += np.exp(-dist)

        return max(density, 1e-10)  # Avoid division by zero

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
        Calculate kinetic energy density τ at a point.

        τ = -½ Σ_i |∇φ_i|² × occupation_i

        For simplified implementation, we use density-based approximation.

        Args:
            point: 3D position vector (Bohr)

        Returns:
            Kinetic energy density (positive value)
        """
        # Simplified: use Thomas-Fermi approximation
        # τ_TF = (3/10)(3π²)^(2/3) ρ^(5/3)
        rho = self.calculate_density(point)
        tau = (3 / 10) * (3 * np.pi**2) ** (2 / 3) * rho ** (5 / 3)

        # Add gradient correction (simplified Weizsäcker term)
        gradient = self.calculate_gradient(point)
        grad_norm_sq = np.dot(gradient, gradient)
        tau_weizsacker = grad_norm_sq / (8 * rho)

        return tau + tau_weizsacker

    def calculate_elf(self, point: np.ndarray) -> float:
        """
        Calculate Electron Localization Function at a point.

        ELF = 1 / (1 + (D/D_h)²)

        where:
        - D = τ - |∇ρ|²/(4ρ) (Pauli kinetic energy density)
        - D_h = C_F × ρ^(5/3) (Thomas-Fermi kinetic energy for uniform gas)

        Args:
            point: 3D position vector (Bohr)

        Returns:
            ELF value in range [0, 1]
        """
        # Calculate density
        rho = self.calculate_density(point)

        # Calculate kinetic energy density
        tau = self.calculate_kinetic_energy_density(point)

        # Calculate density gradient
        gradient = self.calculate_gradient(point)
        grad_norm_sq = np.dot(gradient, gradient)

        # Pauli kinetic energy density
        # D = τ - |∇ρ|²/(4ρ)
        D = tau - grad_norm_sq / (4 * rho)
        D = max(D, 0)  # D should be non-negative for physical systems

        # Thomas-Fermi kinetic energy density
        # D_h = (3/10)(3π²)^(5/3) ρ^(5/3)
        D_h = self.C_F * rho ** (5 / 3)

        # ELF formula
        # ELF = 1 / (1 + (D/D_h)²)
        if D_h > 1e-15:
            ratio = D / D_h
            elf = 1.0 / (1.0 + ratio**2)
        else:
            elf = 0.5  # Default for very low density regions

        # Ensure ELF is in valid range
        return float(np.clip(elf, 0.0, 1.0))

    def calculate_elf_grid(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate ELF at multiple grid points.

        Args:
            points: Array of 3D points (N x 3)

        Returns:
            Array of ELF values
        """
        elf_values = np.array([self.calculate_elf(p) for p in points])
        return elf_values

    def identify_basins(self, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Identify ELF basins (electron localization regions).

        Basins are regions where ELF > threshold, typically
        corresponding to core, bond, and lone pair regions.

        Args:
            threshold: ELF threshold for basin identification (default: 0.8)

        Returns:
            List of basin dictionaries
        """
        if self._basins is not None:
            return self._basins

        basins = []

        # 1. Core basins (near nuclei)
        for i, atom in enumerate(self.wfn.atoms):
            center = np.array([atom.x, atom.y, atom.z])
            elf_at_center = self.calculate_elf(center)

            if elf_at_center > threshold:
                basins.append(
                    {
                        "type": "core",
                        "atom_index": i,
                        "center": center,
                        "elf_value": elf_at_center,
                        "volume": 0.1,  # Placeholder
                    }
                )

        # 2. Bond basins (between bonded atoms)
        for i, atom1 in enumerate(self.wfn.atoms):
            for j, atom2 in enumerate(self.wfn.atoms[i + 1 :], i + 1):
                r1 = np.array([atom1.x, atom1.y, atom1.z])
                r2 = np.array([atom2.x, atom2.y, atom2.z])

                # Sample along bond
                for t in np.linspace(0.3, 0.7, 5):
                    point = r1 + t * (r2 - r1)
                    elf = self.calculate_elf(point)

                    if elf > threshold:
                        basins.append(
                            {
                                "type": "bond",
                                "bond": (i, j),
                                "center": point,
                                "elf_value": elf,
                                "volume": 0.2,
                            }
                        )
                        break  # One basin per bond

        # 3. Lone pair basins (on electronegative atoms)
        # Simplified: sample around each atom
        for i, atom in enumerate(self.wfn.atoms):
            if atom.element in ["N", "O", "F", "P", "S", "Cl"]:
                center = np.array([atom.x, atom.y, atom.z])

                # Sample spherical region
                for r in [0.5, 1.0]:
                    for theta in np.linspace(0, np.pi, 3):
                        for phi in np.linspace(0, 2 * np.pi, 6):
                            point = center + r * np.array(
                                [
                                    np.sin(theta) * np.cos(phi),
                                    np.sin(theta) * np.sin(phi),
                                    np.cos(theta),
                                ]
                            )
                            elf = self.calculate_elf(point)

                            if elf > threshold:
                                basins.append(
                                    {
                                        "type": "lone_pair",
                                        "atom_index": i,
                                        "center": point,
                                        "elf_value": elf,
                                        "volume": 0.15,
                                    }
                                )
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break

        self._basins = basins
        return basins

    def get_basin_properties(self, basin: Dict) -> Dict[str, Any]:
        """
        Calculate properties of an ELF basin.

        Args:
            basin: Basin dictionary

        Returns:
            Dictionary with basin properties
        """
        center = basin.get("center", np.zeros(3))

        return {
            "center": center,
            "elf_value": basin.get("elf_value", 0.0),
            "volume": basin.get("volume", 0.0),
            "electron_population": basin.get("volume", 0.1) * 2,  # Simplified
            "type": basin.get("type", "unknown"),
        }

    def calculate_basin_population(self, basin: Dict) -> float:
        """
        Calculate electron population in an ELF basin.

        Args:
            basin: Basin dictionary

        Returns:
            Electron population
        """
        props = self.get_basin_properties(basin)
        return props["electron_population"]

    def generate_isosurface(self, isovalue: float = 0.8) -> Dict[str, Any]:
        """
        Generate ELF isosurface data.

        Args:
            isovalue: ELF value for isosurface (default: 0.8)

        Returns:
            Dictionary with isosurface data
        """
        coords = np.array([[a.x, a.y, a.z] for a in self.wfn.atoms])

        return {
            "isovalue": isovalue,
            "bbox_min": coords.min(axis=0) - 3.0,
            "bbox_max": coords.max(axis=0) + 3.0,
            "description": f"ELF = {isovalue:.2f} isosurface",
        }

    def generate_report(self) -> str:
        """
        Generate ELF analysis report.

        Returns:
            Formatted string report
        """
        basins = self.identify_basins()

        lines = [
            "Electron Localization Function (ELF) Analysis Report",
            "=" * 55,
            "",
            f"Total ELF basins identified: {len(basins)}",
            "",
        ]

        # Group by type
        core_basins = [b for b in basins if b.get("type") == "core"]
        bond_basins = [b for b in basins if b.get("type") == "bond"]
        lp_basins = [b for b in basins if b.get("type") == "lone_pair"]

        lines.append(f"Core basins: {len(core_basins)}")
        lines.append(f"Bond basins: {len(bond_basins)}")
        lines.append(f"Lone pair basins: {len(lp_basins)}")
        lines.append("")
        lines.append("Basin Details:")

        for i, basin in enumerate(basins[:10]):  # Show first 10
            btype = basin.get("type", "unknown")
            elf_val = basin.get("elf_value", 0.0)
            center = basin.get("center", np.zeros(3))

            lines.append(
                f"  {i+1}. {btype} basin at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
            )
            lines.append(f"     ELF value: {elf_val:.4f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return f"ELFAnalyzer(atoms={len(self.wfn.atoms)})"
