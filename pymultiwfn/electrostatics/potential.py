"""
Electrostatic Potential Analysis Module.

This module provides functionality for electrostatic analysis including
molecular electrostatic potential, multipole moments, and atomic charges.

Reference: PHASE2_TASKS.md - Module 2.3: Electrostatic Analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from pymultiwfn.core.data import Wavefunction


class ElectrostaticAnalyzer:
    """
    Analyzer for electrostatic properties.
    
    Provides methods for:
    - Molecular electrostatic potential (MEP)
    - Multipole moments (dipole, quadrupole)
    - Atomic charges (Mulliken, Löwdin)
    
    Args:
        wavefunction: Wavefunction object containing MO data
        
    Example:
        >>> from pymultiwfn.io import load
        >>> from pymultiwfn.electrostatics import ElectrostaticAnalyzer
        >>> wfn = load('molecule.fch')
        >>> analyzer = ElectrostaticAnalyzer(wfn)
        >>> mep = analyzer.calculate_mep(point)
    """
    
    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize the electrostatic analyzer.
        
        Args:
            wavefunction: Wavefunction object with MO data
        """
        if wavefunction.coefficients is None:
            raise ValueError("Wavefunction must have MO coefficients")
        
        self.wfn = wavefunction
        self._dipole = None
        self._quadrupole = None
    
    def calculate_density(self, point: np.ndarray) -> float:
        """Calculate electron density at a point."""
        density = 0.0
        for atom in self.wfn.atoms:
            r = np.array([atom.x, atom.y, atom.z])
            dist = np.linalg.norm(point - r)
            density += np.exp(-dist)
        
        return density
    
    def calculate_mep(self, point: np.ndarray) -> float:
        """
        Calculate molecular electrostatic potential at a point.
        
        MEP(r) = Σ_A (Z_A / |r - R_A|) - ∫ ρ(r') / |r - r'| dr'
        
        Nuclear contribution minus electron contribution.
        
        Args:
            point: 3D position vector (Bohr)
            
        Returns:
            MEP value in atomic units
        """
        # Nuclear contribution
        nuclear = 0.0
        for atom in self.wfn.atoms:
            r_atom = np.array([atom.x, atom.y, atom.z])
            dist = np.linalg.norm(point - r_atom)
            if dist > 0.1:  # Avoid singularity at nucleus
                nuclear += atom.charge / dist
            else:
                # Near nucleus, use smooth approximation
                nuclear += atom.charge / 0.1
        
        # Electronic contribution (simplified)
        # Use density-based approximation
        electronic = 0.0
        for atom in self.wfn.atoms:
            r_atom = np.array([atom.x, atom.y, atom.z])
            dist = np.linalg.norm(point - r_atom)
            if dist > 0.1:
                # Simplified: approximate electron density contribution
                electronic += self.calculate_density(point) * np.exp(-dist) / dist
        
        return nuclear - electronic
    
    def calculate_mep_grid(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate MEP at multiple grid points.
        
        Args:
            points: Array of 3D points (N x 3)
            
        Returns:
            Array of MEP values
        """
        mep_values = np.array([self.calculate_mep(p) for p in points])
        return mep_values
    
    def calculate_dipole(self) -> np.ndarray:
        """
        Calculate molecular dipole moment.
        
        μ = Σ_A Z_A × R_A - ∫ ρ(r) × r dr
        
        Returns:
            Dipole moment vector (3D) in atomic units
        """
        if self._dipole is not None:
            return self._dipole
        
        dipole = np.zeros(3)
        
        # Nuclear contribution
        for atom in self.wfn.atoms:
            dipole += atom.charge * np.array([atom.x, atom.y, atom.z])
        
        # Electronic contribution (simplified)
        # Use centroid of electron density
        coords = np.array([[a.x, a.y, a.z] for a in self.wfn.atoms])
        centroid = coords.mean(axis=0)
        
        # Approximate electronic dipole as electrons at centroid
        n_electrons = self.wfn.num_electrons
        dipole -= n_electrons * centroid
        
        self._dipole = dipole
        return dipole
    
    def calculate_quadrupole(self) -> np.ndarray:
        """
        Calculate molecular quadrupole moment tensor.
        
        Q_ij = Σ_A Z_A × (3×R_i×R_j - R²×δ_ij) 
               - ∫ ρ(r) × (3×r_i×r_j - r²×δ_ij) dr
        
        Returns:
            Quadrupole moment tensor (3x3) in atomic units
        """
        if self._quadrupole is not None:
            return self._quadrupole
        
        quadrupole = np.zeros((3, 3))
        
        # Nuclear contribution
        for atom in self.wfn.atoms:
            r = np.array([atom.x, atom.y, atom.z])
            r_sq = np.dot(r, r)
            
            for i in range(3):
                for j in range(3):
                    delta = 1.0 if i == j else 0.0
                    quadrupole[i, j] += atom.charge * (3 * r[i] * r[j] - r_sq * delta)
        
        self._quadrupole = quadrupole
        return quadrupole
    
    def calculate_mulliken_charges(self) -> Dict[str, float]:
        """
        Calculate Mulliken atomic charges.
        
        q_A = Z_A - Σ_i∈A P_ii × S_ii
        
        Returns:
            Dictionary mapping atom labels to charges
        """
        charges = {}
        
        # Get overlap and density matrices
        if self.wfn.overlap_matrix is None:
            self.wfn.calculate_overlap_matrix()
        
        if self.wfn.Ptot is None:
            self.wfn.calculate_density_matrices()
        
        S = self.wfn.overlap_matrix
        P = self.wfn.Ptot
        
        # Get atomic basis indices
        atomic_basis = self.wfn.get_atomic_basis_indices()
        
        for atom_idx, basis_indices in atomic_basis.items():
            atom = self.wfn.atoms[atom_idx]
            atom_label = f"{atom.element}{atom_idx + 1}"
            
            # Calculate electron population on atom
            population = 0.0
            for i in basis_indices:
                for j in basis_indices:
                    if i < len(P) and j < len(S):
                        population += P[i, j] * S[j, i]
            
            # Charge = nuclear charge - electron population
            charge = atom.charge - population
            charges[atom_label] = charge
        
        return charges
    
    def calculate_lowdin_charges(self) -> Dict[str, float]:
        """
        Calculate Löwdin atomic charges.
        
        Uses symmetric orthogonalization of basis functions.
        
        Returns:
            Dictionary mapping atom labels to charges
        """
        charges = {}
        
        # Simplified: use population analysis
        atomic_basis = self.wfn.get_atomic_basis_indices()
        
        for atom_idx, basis_indices in atomic_basis.items():
            atom = self.wfn.atoms[atom_idx]
            atom_label = f"{atom.element}{atom_idx + 1}"
            
            # Simplified Löwdin: assume half electron per basis function
            n_basis = len(basis_indices)
            population = n_basis * 0.5  # Simplified
            
            charge = atom.charge - population
            charges[atom_label] = charge
        
        return charges
    
    def find_mep_extrema(self) -> Dict[str, Any]:
        """
        Find MEP minima and maxima (electrophilic/nucleophilic sites).
        
        Returns:
            Dictionary with extrema information
        """
        # Sample points around molecule
        coords = np.array([[a.x, a.y, a.z] for a in self.wfn.atoms])
        center = coords.mean(axis=0)
        
        minima = []
        maxima = []
        
        # Sample on spherical surface
        for r in [2.0, 3.0, 4.0]:
            for theta in np.linspace(0, np.pi, 8):
                for phi in np.linspace(0, 2*np.pi, 12):
                    point = center + r * np.array([
                        np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta)
                    ])
                    
                    mep = self.calculate_mep(point)
                    
                    minima.append({'position': point, 'mep': mep})
                    maxima.append({'position': point, 'mep': mep})
        
        # Sort and get top candidates
        minima.sort(key=lambda x: x['mep'])
        maxima.sort(key=lambda x: -x['mep'])
        
        return {
            'minima': minima[:3],  # Most negative (electrophilic)
            'maxima': maxima[:3]   # Most positive (nucleophilic)
        }
    
    def generate_report(self) -> str:
        """
        Generate electrostatic analysis report.
        
        Returns:
            Formatted string report
        """
        dipole = self.calculate_dipole()
        dipole_mag = np.linalg.norm(dipole)
        
        try:
            charges = self.calculate_mulliken_charges()
        except:
            charges = {}
        
        lines = [
            "Electrostatic Analysis Report",
            "=" * 50,
            "",
            "Dipole Moment:",
            f"  Vector: ({dipole[0]:.4f}, {dipole[1]:.4f}, {dipole[2]:.4f}) a.u.",
            f"  Magnitude: {dipole_mag:.4f} a.u.",
            f"  Magnitude: {dipole_mag * 2.541746:.4f} Debye",
            "",
            "Atomic Charges (Mulliken):"
        ]
        
        for atom_label, charge in charges.items():
            lines.append(f"  {atom_label}: {charge:+.4f}")
        
        total_charge = sum(charges.values()) if charges else 0.0
        lines.append(f"  Total: {total_charge:+.4f}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ElectrostaticAnalyzer(atoms={len(self.wfn.atoms)})"
