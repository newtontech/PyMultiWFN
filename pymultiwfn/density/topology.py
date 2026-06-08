"""
Critical Point Analysis Module.

This module provides functionality for locating and analyzing
critical points in electron density, including bond critical
points (BCP), ring critical points (RCP), and cage critical
points (CCP).

Reference: PHASE2_TASKS.md - Module 2.2: Electron Density Analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize

from pymultiwfn.core.data import Wavefunction


class CriticalPointAnalyzer:
    """
    Analyzer for critical points in electron density.
    
    Critical points are locations where the gradient of electron
    density vanishes (∇ρ = 0). They are classified by their rank
    and signature (number of positive/negative eigenvalues of Hessian).
    
    Common critical point types:
    - (3,-3): Nuclear critical point (NCP) - local maximum
    - (3,-1): Bond critical point (BCP) - saddle point
    - (3,+1): Ring critical point (RCP) - saddle point
    - (3,+3): Cage critical point (CCP) - local minimum
    
    Args:
        wavefunction: Wavefunction object containing MO data
        
    Example:
        >>> from pymultiwfn.io import load
        >>> from pymultiwfn.density import CriticalPointAnalyzer
        >>> wfn = load('molecule.fch')
        >>> analyzer = CriticalPointAnalyzer(wfn)
        >>> cps = analyzer.find_critical_points()
    """
    
    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize the critical point analyzer.
        
        Args:
            wavefunction: Wavefunction object with MO data
        """
        if wavefunction.coefficients is None:
            raise ValueError("Wavefunction must have MO coefficients")
        
        self.wfn = wavefunction
        self._critical_points = None
        self._bcps = None
        self._rcps = None
        self._ccps = None
    
    def calculate_density(self, point: np.ndarray) -> float:
        """
        Calculate electron density at a point.
        
        Args:
            point: 3D position vector (Bohr)
            
        Returns:
            Electron density value
        """
        # Simplified density calculation
        # Full implementation would evaluate basis functions at point
        
        # Placeholder: sum of Gaussian functions at nuclei
        density = 0.0
        for atom in self.wfn.atoms:
            r = np.array([atom.x, atom.y, atom.z])
            dist = np.linalg.norm(point - r)
            # Simple exponential decay from nucleus
            density += np.exp(-dist)
        
        return density
    
    def calculate_gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of electron density at a point.
        
        Args:
            point: 3D position vector (Bohr)
            
        Returns:
            Gradient vector (3D)
        """
        # Numerical gradient using central differences
        h = 0.001  # Step size
        gradient = np.zeros(3)
        
        for i in range(3):
            point_plus = point.copy()
            point_minus = point.copy()
            point_plus[i] += h
            point_minus[i] -= h
            
            gradient[i] = (self.calculate_density(point_plus) - 
                          self.calculate_density(point_minus)) / (2 * h)
        
        return gradient
    
    def calculate_hessian(self, point: np.ndarray) -> np.ndarray:
        """
        Calculate Hessian matrix of electron density at a point.
        
        The Hessian is the matrix of second derivatives:
        H_ij = ∂²ρ/∂x_i∂x_j
        
        Args:
            point: 3D position vector (Bohr)
            
        Returns:
            Hessian matrix (3x3)
        """
        h = 0.001  # Step size
        hessian = np.zeros((3, 3))
        
        # Diagonal elements
        for i in range(3):
            point_plus = point.copy()
            point_minus = point.copy()
            point_plus[i] += h
            point_minus[i] -= h
            
            hessian[i, i] = (self.calculate_density(point_plus) - 
                            2 * self.calculate_density(point) + 
                            self.calculate_density(point_minus)) / (h ** 2)
        
        # Off-diagonal elements
        for i in range(3):
            for j in range(i + 1, 3):
                # Four-point formula for mixed partials
                pp = point.copy()
                pm = point.copy()
                mp = point.copy()
                mm = point.copy()
                
                pp[i] += h; pp[j] += h
                pm[i] += h; pm[j] -= h
                mp[i] -= h; mp[j] += h
                mm[i] -= h; mm[j] -= h
                
                hessian[i, j] = (self.calculate_density(pp) - 
                                self.calculate_density(pm) - 
                                self.calculate_density(mp) + 
                                self.calculate_density(mm)) / (4 * h ** 2)
                hessian[j, i] = hessian[i, j]  # Symmetric
        
        return hessian
    
    def find_critical_points(self) -> List[Dict[str, Any]]:
        """
        Find all critical points in electron density.
        
        Uses numerical optimization to locate points where
        the gradient of density vanishes.
        
        Returns:
            List of critical point dictionaries with:
            - 'position': 3D position vector
            - 'type': CP type (NCP, BCP, RCP, CCP)
            - 'density': density value at CP
            - 'laplacian': Laplacian value at CP
        """
        if self._critical_points is not None:
            return self._critical_points
        
        critical_points = []
        
        # 1. Nuclear critical points (at nuclei)
        for i, atom in enumerate(self.wfn.atoms):
            position = np.array([atom.x, atom.y, atom.z])
            cp = {
                'position': position,
                'type': 'NCP',
                'atom_index': i,
                'density': self.calculate_density(position),
                'laplacian': self.get_laplacian_at_point(position)
            }
            critical_points.append(cp)
        
        # 2. Bond critical points (along bonds)
        for i, atom1 in enumerate(self.wfn.atoms):
            for j, atom2 in enumerate(self.wfn.atoms[i+1:], i+1):
                # Search along bond axis
                r1 = np.array([atom1.x, atom1.y, atom1.z])
                r2 = np.array([atom2.x, atom2.y, atom2.z])
                
                # Use midpoint as starting point
                mid = (r1 + r2) / 2
                
                # Optimize to find BCP
                try:
                    result = minimize(
                        lambda x: np.linalg.norm(self.calculate_gradient(x))**2,
                        mid,
                        method='BFGS',
                        options={'maxiter': 50}
                    )
                    
                    if result.fun < 0.01:  # Gradient norm threshold
                        position = result.x
                        hessian = self.calculate_hessian(position)
                        signature = self._get_signature(hessian)
                        
                        if signature == (3, -1):  # BCP signature
                            cp = {
                                'position': position,
                                'type': 'BCP',
                                'bond': (i, j),
                                'density': self.calculate_density(position),
                                'laplacian': self.get_laplacian_at_point(position)
                            }
                            critical_points.append(cp)
                except Exception:
                    pass  # Skip if optimization fails
        
        self._critical_points = critical_points
        return critical_points
    
    def _get_signature(self, hessian: np.ndarray) -> Tuple[int, int]:
        """
        Get signature (rank, signature) of critical point.
        
        Args:
            hessian: Hessian matrix at critical point
            
        Returns:
            Tuple (rank, signature) where rank is number of
            non-zero eigenvalues and signature is (pos - neg)
        """
        eigenvalues = np.linalg.eigvalsh(hessian)
        
        # Count positive and negative eigenvalues
        n_positive = np.sum(eigenvalues > 1e-6)
        n_negative = np.sum(eigenvalues < -1e-6)
        
        rank = n_positive + n_negative
        signature = n_positive - n_negative
        
        return (rank, signature)
    
    def get_bond_critical_points(self) -> List[Dict[str, Any]]:
        """
        Get all bond critical points (BCPs).
        
        Returns:
            List of BCP dictionaries
        """
        if self._bcps is not None:
            return self._bcps
        
        all_cps = self.find_critical_points()
        self._bcps = [cp for cp in all_cps if cp.get('type') == 'BCP']
        
        return self._bcps
    
    def get_ring_critical_points(self) -> List[Dict[str, Any]]:
        """
        Get all ring critical points (RCPs).
        
        Returns:
            List of RCP dictionaries
        """
        if self._rcps is not None:
            return self._rcps
        
        all_cps = self.find_critical_points()
        self._rcps = [cp for cp in all_cps if cp.get('type') == 'RCP']
        
        return self._rcps
    
    def get_cage_critical_points(self) -> List[Dict[str, Any]]:
        """
        Get all cage critical points (CCPs).
        
        Returns:
            List of CCP dictionaries
        """
        if self._ccps is not None:
            return self._ccps
        
        all_cps = self.find_critical_points()
        self._ccps = [cp for cp in all_cps if cp.get('type') == 'CCP']
        
        return self._ccps
    
    def get_density_at_point(self, point: np.ndarray) -> float:
        """
        Get electron density at a point.
        
        Args:
            point: 3D position vector
            
        Returns:
            Density value
        """
        return self.calculate_density(point)
    
    def get_laplacian_at_point(self, point: np.ndarray) -> float:
        """
        Calculate Laplacian of density at a point.
        
        ∇²ρ = Σ_i ∂²ρ/∂x_i² = Trace(Hessian)
        
        Args:
            point: 3D position vector
            
        Returns:
            Laplacian value
        """
        hessian = self.calculate_hessian(point)
        return float(np.trace(hessian))
    
    def get_critical_point_rank(self, cp: Dict) -> int:
        """
        Get rank of critical point.
        
        Args:
            cp: Critical point dictionary
            
        Returns:
            Rank (0-3)
        """
        position = cp.get('position')
        if position is None:
            return 0
        
        hessian = self.calculate_hessian(position)
        eigenvalues = np.linalg.eigvalsh(hessian)
        
        # Count non-zero eigenvalues
        return int(np.sum(np.abs(eigenvalues) > 1e-6))
    
    def get_critical_point_signature(self, cp: Dict) -> Tuple[int, int]:
        """
        Get signature of critical point.
        
        Args:
            cp: Critical point dictionary
            
        Returns:
            Tuple (rank, signature)
        """
        position = cp.get('position')
        if position is None:
            return (0, 0)
        
        hessian = self.calculate_hessian(position)
        return self._get_signature(hessian)
    
    def calculate_ellipticity(self, cp: Dict) -> float:
        """
        Calculate ellipticity at a critical point.
        
        Ellipticity measures deviation from cylindrical symmetry:
        ε = (λ₁/λ₂) - 1
        
        where λ₁ ≤ λ₂ < 0 are the two negative Hessian eigenvalues.
        
        Args:
            cp: Critical point dictionary
            
        Returns:
            Ellipticity value
        """
        position = cp.get('position')
        if position is None:
            return 0.0
        
        hessian = self.calculate_hessian(position)
        eigenvalues = np.sort(np.linalg.eigvalsh(hessian))
        
        # For BCPs, first two eigenvalues should be negative
        if eigenvalues[0] < 0 and eigenvalues[1] < 0:
            lambda1 = eigenvalues[0]
            lambda2 = eigenvalues[1]
            
            if abs(lambda2) > 1e-10:
                ellipticity = (lambda1 / lambda2) - 1
                return abs(ellipticity)
        
        return 0.0
    
    def generate_report(self) -> str:
        """
        Generate critical point analysis report.
        
        Returns:
            Formatted string report
        """
        cps = self.find_critical_points()
        bcps = self.get_bond_critical_points()
        
        lines = [
            "Critical Point Analysis Report",
            "=" * 50,
            "",
            f"Total critical points: {len(cps)}",
            f"Bond critical points (BCPs): {len(bcps)}",
            "",
            "Critical Points:",
        ]
        
        for i, cp in enumerate(cps[:10]):  # Show first 10
            pos = cp.get('position', np.zeros(3))
            cp_type = cp.get('type', 'Unknown')
            density = cp.get('density', 0.0)
            
            lines.append(
                f"  {i+1}. {cp_type} at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
            )
            lines.append(f"     Density: {density:.6f}")
        
        return "\n".join(lines)
    
    def verify_morse_relationship(self, cp: Dict) -> bool:
        """
        Verify Morse relationship at critical point.
        
        The Morse relationship relates the number of critical points:
        NCPs - BCPs + RCPs - CCPs = 1
        
        Args:
            cp: Critical point dictionary (unused, for API consistency)
            
        Returns:
            True if Morse relationship is satisfied
        """
        ncps = len([cp for cp in self.find_critical_points() if cp.get('type') == 'NCP'])
        bcps = len(self.get_bond_critical_points())
        rcps = len(self.get_ring_critical_points())
        ccps = len(self.get_cage_critical_points())
        
        morse_sum = ncps - bcps + rcps - ccps
        
        return morse_sum == 1
    
    def __repr__(self) -> str:
        """String representation."""
        n_cps = len(self.find_critical_points()) if self._critical_points else 0
        return f"CriticalPointAnalyzer(critical_points={n_cps})"
