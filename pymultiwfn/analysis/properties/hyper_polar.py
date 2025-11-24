"""
Hyperpolarizability analysis module for PyMultiWFN.

This module implements various (hyper)polarizability analyses including:
- Parsing Gaussian output files for polarizability/hyperpolarizability data
- Sum-over-states (SOS) method for calculating hyperpolarizabilities
- Hyperpolarizability density analysis
- Visualization via unit sphere and vector representations
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import re
from pathlib import Path
import itertools

from ..core.data import Wavefunction


class HyperPolarizabilityAnalyzer:
    """
    Main class for hyperpolarizability analysis.
    """

    def __init__(self, wavefunction: Optional[Wavefunction] = None):
        """
        Initialize the analyzer with an optional wavefunction object.

        Args:
            wavefunction: Wavefunction object containing electronic structure data
        """
        self.wavefunction = wavefunction

        # Physical constants
        self.au2debye = 2.54175  # 1 a.u. = 2.54175 Debye
        self.au2eV = 27.2114     # 1 a.u. = 27.2114 eV
        self.au2nm = 45.56335    # 1 a.u. = 45.56335 nm for wavelength conversion

    def parse_gaussian_polarizability(self, filename: str, method: int = 1,
                                     load_frequency_dependent: bool = False,
                                     output_unit: str = "a.u.") -> Dict[str, np.ndarray]:
        """
        Parse Gaussian output file for polarizability and hyperpolarizability data.

        Args:
            filename: Path to Gaussian output file
            method: Gaussian calculation method (1-7 as in Multiwfn)
            load_frequency_dependent: Whether to load frequency-dependent results
            output_unit: Output unit ("a.u.", "SI", "esu")

        Returns:
            Dictionary containing parsed tensors and properties
        """
        results = {}

        with open(filename, 'r') as f:
            content = f.read()

        # Parse dipole moment
        dipole_match = re.search(r"Dipole moment.*?\n.*?([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", content)
        if dipole_match:
            dipole = np.array([float(x) for x in dipole_match.groups()]) / self.au2debye
            results['dipole'] = dipole

        # Parse polarizability (alpha)
        alpha = np.zeros((3, 3))

        # Look for SCF Polarizability section
        alpha_section = re.search(r"SCF Polarizability.*?\n.*?\n.*?\n.*?([-\d.]+).*?\n.*?([-\d.]+).*?([-\d.]+).*?\n.*?([-\d.]+).*?([-\d.]+).*?([-\d.]+)",
                                 content, re.DOTALL)
        if alpha_section:
            values = [float(x) for x in alpha_section.groups()]
            alpha[0, 0] = values[0]
            alpha[1, 0] = alpha[0, 1] = values[1]
            alpha[1, 1] = values[2]
            alpha[2, 0] = alpha[0, 2] = values[3]
            alpha[2, 1] = alpha[1, 2] = values[4]
            alpha[2, 2] = values[5]
            results['alpha'] = alpha

        # Parse first hyperpolarizability (beta)
        beta = np.zeros((3, 3, 3))

        # Look for hyperpolarizability sections
        beta_section = re.search(r"SCF Static Hyperpolarizability.*?XXX.*?([-\d.]+).*?XXY.*?([-\d.]+).*?XYY.*?([-\d.]+).*?YYY.*?([-\d.]+).*?XXZ.*?([-\d.]+).*?XYZ.*?([-\d.]+).*?YYZ.*?([-\d.]+).*?XZZ.*?([-\d.]+).*?YZZ.*?([-\d.]+).*?ZZZ.*?([-\d.]+)",
                                content, re.DOTALL)
        if beta_section:
            values = [float(x) for x in beta_section.groups()]
            # Note: Gaussian sign convention requires inversion
            beta[0, 0, 0] = -values[0]  # XXX
            beta[0, 0, 1] = beta[0, 1, 0] = beta[1, 0, 0] = -values[1]  # XXY
            beta[0, 1, 1] = beta[1, 0, 1] = beta[1, 1, 0] = -values[2]  # XYY
            beta[1, 1, 1] = -values[3]  # YYY
            beta[0, 0, 2] = beta[0, 2, 0] = beta[2, 0, 0] = -values[4]  # XXZ
            beta[0, 1, 2] = beta[0, 2, 1] = beta[1, 0, 2] = beta[1, 2, 0] = beta[2, 0, 1] = beta[2, 1, 0] = -values[5]  # XYZ
            beta[1, 1, 2] = beta[1, 2, 1] = beta[2, 1, 1] = -values[6]  # YYZ
            beta[0, 2, 2] = beta[2, 0, 2] = beta[2, 2, 0] = -values[7]  # XZZ
            beta[1, 2, 2] = beta[2, 1, 2] = beta[2, 2, 1] = -values[8]  # YZZ
            beta[2, 2, 2] = -values[9]  # ZZZ
            results['beta'] = beta

        # Parse second hyperpolarizability (gamma) if available
        gamma = np.zeros((3, 3, 3, 3))
        gamma_section = re.search(r"Gamma\(0;0,0,0\).*?XXXX.*?([-\d.]+).*?XXXY.*?([-\d.]+).*?XXYY.*?([-\d.]+).*?XYYY.*?([-\d.]+).*?YYYY.*?([-\d.]+)",
                                 content, re.DOTALL)
        if gamma_section:
            # Simplified parsing - actual implementation would be more complex
            values = [float(x) for x in gamma_section.groups()]
            # Fill gamma tensor based on symmetry
            # This is a simplified implementation
            results['gamma'] = gamma

        return results

    def calculate_polarizability_tensor(self, method: str = "finite_field") -> np.ndarray:
        """
        Calculate polarizability tensor.

        Args:
            method: Method to use for calculation ("finite_field", "sos")

        Returns:
            Polarizability tensor (3, 3)
        """
        if method == "finite_field":
            return self._calculate_polarizability_finite_field()
        elif method == "sos":
            return self._calculate_polarizability_sos()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _calculate_polarizability_finite_field(self) -> np.ndarray:
        """
        Calculate polarizability using finite field method.

        Returns:
            Polarizability tensor (3, 3)
        """
        # Implementation would require wavefunction data and field perturbations
        # This is a placeholder for the actual implementation
        alpha = np.zeros((3, 3))

        if self.wavefunction is None:
            raise ValueError("Wavefunction data required for finite field calculation")

        # TODO: Implement finite field polarizability calculation
        # This would involve calculating dipole moments under different
        # electric field perturbations and taking numerical derivatives

        return alpha

    def _calculate_polarizability_sos(self, frequencies: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate polarizability using sum-over-states method.

        Args:
            frequencies: Array of frequencies to calculate frequency-dependent polarizability

        Returns:
            Polarizability tensor (3, 3)
        """
        if self.wavefunction is None:
            raise ValueError("Wavefunction data required for SOS calculation")

        # Implementation based on Fortran SOS subroutine
        # Formula: alpha_{ij}(-w;w) = sum_n [<0|r_i|n><n|r_j|0>/(E_n - w) + <0|r_j|n><n|r_i|0>/(E_n + w)]

        alpha = np.zeros((3, 3))

        # For now, this is a placeholder - actual implementation would require
        # excitation energies and transition dipole moments from the wavefunction
        # TODO: Implement full SOS calculation when excitation data is available

        return alpha

    def calculate_sos_polarizability(self, frequencies: Optional[np.ndarray] = None,
                                   num_states: Optional[int] = None) -> Dict[str, Union[np.ndarray, float]]:
        """
        Calculate polarizability using sum-over-states (SOS) method.

        Args:
            frequencies: Array of frequencies for frequency-dependent calculation
            num_states: Number of excited states to include (None = all available)

        Returns:
            Dictionary containing polarizability tensor and properties
        """
        if self.wavefunction is None:
            raise ValueError("Wavefunction data required for SOS calculation")

        alpha = self._calculate_polarizability_sos(frequencies)
        properties = self.analyze_polarizability_properties(alpha)

        return {
            'tensor': alpha,
            'properties': properties
        }

    def calculate_hyperpolarizability_tensor(self, order: int = 1,
                                           frequencies: Optional[List[float]] = None) -> np.ndarray:
        """
        Calculate hyperpolarizability tensor.

        Args:
            order: Order of hyperpolarizability (1 for beta, 2 for gamma)
            frequencies: List of frequencies for frequency-dependent calculation

        Returns:
            Hyperpolarizability tensor
        """
        if order == 1:
            return self._calculate_first_hyperpolarizability(frequencies)
        elif order == 2:
            return self._calculate_second_hyperpolarizability(frequencies)
        else:
            raise ValueError(f"Unsupported hyperpolarizability order: {order}")

    def _calculate_first_hyperpolarizability(self, frequencies: Optional[List[float]] = None) -> np.ndarray:
        """
        Calculate first hyperpolarizability tensor (beta).

        Args:
            frequencies: List of frequencies [w1, w2] for beta(-(w1+w2); w1, w2)

        Returns:
            First hyperpolarizability tensor (3, 3, 3)
        """
        beta = np.zeros((3, 3, 3))

        if self.wavefunction is None:
            raise ValueError("Wavefunction data required for hyperpolarizability calculation")

        # TODO: Implement first hyperpolarizability calculation
        # This would use either finite field or SOS method

        return beta

    def _calculate_second_hyperpolarizability(self, frequencies: Optional[List[float]] = None) -> np.ndarray:
        """
        Calculate second hyperpolarizability tensor (gamma).

        Args:
            frequencies: List of frequencies [w1, w2, w3] for gamma(-(w1+w2+w3); w1, w2, w3)

        Returns:
            Second hyperpolarizability tensor (3, 3, 3, 3)
        """
        gamma = np.zeros((3, 3, 3, 3))

        if self.wavefunction is None:
            raise ValueError("Wavefunction data required for hyperpolarizability calculation")

        # TODO: Implement second hyperpolarizability calculation

        return gamma

    def analyze_polarizability_properties(self, alpha: np.ndarray) -> Dict[str, float]:
        """
        Analyze properties of polarizability tensor.

        Args:
            alpha: Polarizability tensor (3, 3)

        Returns:
            Dictionary containing polarizability properties
        """
        properties = {}

        # Isotropic average
        alpha_iso = np.trace(alpha) / 3.0
        properties['isotropic_average'] = alpha_iso

        # Polarizability volume (in Angstrom^3)
        alpha_volume = alpha_iso * 0.14818470
        properties['polarizability_volume'] = alpha_volume

        # Anisotropy (definition 1)
        term1 = (alpha[0, 0] - alpha[1, 1])**2 + (alpha[0, 0] - alpha[2, 2])**2 + (alpha[1, 1] - alpha[2, 2])**2
        term2 = 6 * (alpha[0, 1]**2 + alpha[0, 2]**2 + alpha[1, 2]**2)
        anisotropy1 = np.sqrt((term1 + term2) / 2.0)
        properties['anisotropy_1'] = anisotropy1

        # Eigenvalues and anisotropy (definition 2)
        eigvals = np.linalg.eigvalsh(alpha)
        anisotropy2 = eigvals[2] - (eigvals[0] + eigvals[1]) / 2.0
        properties['eigenvalues'] = eigvals
        properties['anisotropy_2'] = anisotropy2

        # Vector components
        properties['x_component'] = np.sum(alpha[0, :])
        properties['y_component'] = np.sum(alpha[1, :])
        properties['z_component'] = np.sum(alpha[2, :])

        return properties

    def analyze_hyperpolarizability_properties(self, beta: np.ndarray,
                                             dipole: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Analyze properties of first hyperpolarizability tensor.

        Args:
            beta: First hyperpolarizability tensor (3, 3, 3)
            dipole: Dipole moment vector (3,)

        Returns:
            Dictionary containing hyperpolarizability properties
        """
        properties = {}

        # Beta vector components
        beta_x = 0.0
        beta_y = 0.0
        beta_z = 0.0

        for j in range(3):
            beta_x += (beta[0, j, j] + beta[j, j, 0] + beta[j, 0, j]) / 3.0
            beta_y += (beta[1, j, j] + beta[j, j, 1] + beta[j, 1, j]) / 3.0
            beta_z += (beta[2, j, j] + beta[j, j, 2] + beta[j, 2, j]) / 3.0

        properties['beta_x'] = beta_x
        properties['beta_y'] = beta_y
        properties['beta_z'] = beta_z

        # Magnitude
        beta_magnitude = np.sqrt(beta_x**2 + beta_y**2 + beta_z**2)
        properties['magnitude'] = beta_magnitude

        # Projection on dipole moment
        if dipole is not None:
            dipole_norm = np.linalg.norm(dipole)
            if dipole_norm > 0:
                beta_proj = (beta_x * dipole[0] + beta_y * dipole[1] + beta_z * dipole[2]) / dipole_norm
                properties['projection_on_dipole'] = beta_proj
                properties['beta_parallel'] = beta_proj * 3.0 / 5.0
                properties['beta_parallel_z'] = beta_z * 3.0 / 5.0

        # Perpendicular component
        beta_perp = 0.0
        for j in range(3):
            beta_perp += (2 * beta[2, j, j] + 2 * beta[j, j, 2] - 3 * beta[j, 2, j]) / 5.0
        properties['beta_perp_z'] = beta_perp

        return properties


def create_hyperpolarizability_density(wavefunction: Wavefunction,
                                      grid_points: np.ndarray,
                                      field_direction: int,
                                      hyperpolarizability_order: int = 1) -> np.ndarray:
    """
    Calculate hyperpolarizability density on a grid.

    Args:
        wavefunction: Wavefunction object
        grid_points: Array of grid points (N, 3)
        field_direction: Direction of electric field (0=x, 1=y, 2=z)
        hyperpolarizability_order: Order of hyperpolarizability (1 for beta, 2 for gamma)

    Returns:
        Hyperpolarizability density values at grid points (N,)
    """
    # Placeholder implementation
    # Actual implementation would require:
    # - Calculating electron density under different field perturbations
    # - Taking numerical derivatives

    density = np.zeros(len(grid_points))

    # TODO: Implement hyperpolarizability density calculation
    # This would follow the approach in hyper_polar_dens subroutine

    return density


def visualize_hyperpolarizability_tensor(tensor: np.ndarray,
                                        tensor_order: int,
                                        output_file: str = "hyperpolarizability.tcl") -> None:
    """
    Generate VMD visualization script for hyperpolarizability tensor.

    Args:
        tensor: Hyperpolarizability tensor
        tensor_order: Order of tensor (1 for alpha, 2 for beta, 3 for gamma)
        output_file: Output VMD script filename
    """
    # Placeholder implementation
    # Actual implementation would generate VMD commands for:
    # - Unit sphere representation with arrows
    # - Vector representation

    with open(output_file, 'w') as f:
        f.write("# VMD visualization script for hyperpolarizability tensor\n")
        f.write("color Display Background white\n")

        # TODO: Implement proper visualization generation
        # This would follow the vis_hypol subroutine

    print(f"VMD visualization script generated: {output_file}")


# Convenience functions for common analyses

def analyze_static_polarizability(wavefunction: Wavefunction) -> Dict[str, Union[np.ndarray, float]]:
    """
    Convenience function for static polarizability analysis.

    Args:
        wavefunction: Wavefunction object

    Returns:
        Dictionary containing polarizability tensor and properties
    """
    analyzer = HyperPolarizabilityAnalyzer(wavefunction)
    alpha = analyzer.calculate_polarizability_tensor()
    properties = analyzer.analyze_polarizability_properties(alpha)

    return {
        'tensor': alpha,
        'properties': properties
    }


def analyze_static_hyperpolarizability(wavefunction: Wavefunction,
                                      dipole: Optional[np.ndarray] = None) -> Dict[str, Union[np.ndarray, float]]:
    """
    Convenience function for static first hyperpolarizability analysis.

    Args:
        wavefunction: Wavefunction object
        dipole: Dipole moment vector

    Returns:
        Dictionary containing hyperpolarizability tensor and properties
    """
    analyzer = HyperPolarizabilityAnalyzer(wavefunction)
    beta = analyzer.calculate_hyperpolarizability_tensor(order=1)
    properties = analyzer.analyze_hyperpolarizability_properties(beta, dipole)

    return {
        'tensor': beta,
        'properties': properties
    }


def calculate_sos_polarizability(wavefunction: Wavefunction,
                               frequencies: Optional[np.ndarray] = None,
                               num_states: Optional[int] = None) -> Dict[str, Union[np.ndarray, float]]:
    """
    Calculate polarizability using sum-over-states (SOS) method.

    Args:
        wavefunction: Wavefunction object with excitation data
        frequencies: Array of frequencies for frequency-dependent calculation
        num_states: Number of excited states to include (None = all available)

    Returns:
        Dictionary containing polarizability tensor and properties
    """
    # Placeholder implementation
    # Actual implementation would require:
    # - Excitation energies and transition dipole moments
    # - Summation over excited states according to SOS formulas

    if wavefunction is None:
        raise ValueError("Wavefunction data required for SOS calculation")

    alpha = np.zeros((3, 3))

    # TODO: Implement SOS polarizability calculation
    # This would follow the formulas in the Fortran SOS subroutine

    analyzer = HyperPolarizabilityAnalyzer(wavefunction)
    properties = analyzer.analyze_polarizability_properties(alpha)

    return {
        'tensor': alpha,
        'properties': properties
    }


def calculate_sos_hyperpolarizability(wavefunction: Wavefunction,
                                    frequencies: List[float],
                                    order: int = 1,
                                    num_states: Optional[int] = None) -> Dict[str, Union[np.ndarray, float]]:
    """
    Calculate hyperpolarizability using sum-over-states (SOS) method.

    Args:
        wavefunction: Wavefunction object with excitation data
        frequencies: List of frequencies [w1, w2, ...] for beta(-(w1+w2); w1, w2)
        order: Order of hyperpolarizability (1 for beta, 2 for gamma)
        num_states: Number of excited states to include

    Returns:
        Dictionary containing hyperpolarizability tensor and properties
    """
    if wavefunction is None:
        raise ValueError("Wavefunction data required for SOS calculation")

    if order == 1:
        beta = np.zeros((3, 3, 3))
        # TODO: Implement SOS first hyperpolarizability calculation
        analyzer = HyperPolarizabilityAnalyzer(wavefunction)
        properties = analyzer.analyze_hyperpolarizability_properties(beta)

        return {
            'tensor': beta,
            'properties': properties
        }
    elif order == 2:
        gamma = np.zeros((3, 3, 3, 3))
        # TODO: Implement SOS second hyperpolarizability calculation

        return {
            'tensor': gamma,
            'properties': {}
        }
    else:
        raise ValueError(f"Unsupported hyperpolarizability order: {order}")


def two_level_analysis(wavefunction: Wavefunction,
                      state_indices: Union[int, List[int]]) -> Dict[str, Union[np.ndarray, float]]:
    """
    Perform two-level or three-level model analysis of hyperpolarizability.

    Args:
        wavefunction: Wavefunction object with excitation data
        state_indices: Single state index for two-level analysis, or list of two indices for three-level analysis

    Returns:
        Dictionary containing analysis results
    """
    if wavefunction is None:
        raise ValueError("Wavefunction data required for two-level analysis")

    # Placeholder implementation
    # Actual implementation would require:
    # - Transition dipole moments between ground and excited states
    # - Variation of dipole moments for excited states
    # - Application of two/three-level model formulas

    if isinstance(state_indices, int):
        # Two-level analysis
        result = {
            'type': 'two_level',
            'state_index': state_indices,
            'beta_components': np.zeros(3),
            'beta_norm': 0.0
        }
    else:
        # Three-level analysis
        result = {
            'type': 'three_level',
            'state_indices': state_indices,
            'beta_components': np.zeros(3),
            'beta_norm': 0.0
        }

    return result


def analyze_hrs_hyperpolarizability(beta: np.ndarray) -> Dict[str, float]:
    """
    Analyze hyper-Rayleigh scattering (HRS) related quantities from hyperpolarizability tensor.

    Args:
        beta: First hyperpolarizability tensor (3, 3, 3)

    Returns:
        Dictionary containing HRS-related quantities
    """
    # Placeholder implementation
    # Actual implementation would calculate:
    # - <beta_ZZZ^2> and <beta_XZZ^2> without Kleinman condition
    # - Hyper-Rayleigh scattering (beta_HRS)
    # - Depolarization ratio (DR)
    # - Nonlinear anisotropy parameter (rho)
    # - Dipolar and octupolar contributions

    return {
        'beta_zzz2_avg': 0.0,
        'beta_xzz2_avg': 0.0,
        'beta_hrs': 0.0,
        'depolarization_ratio': 0.0,
        'rho': 0.0,
        'dipolar_contribution': 0.0,
        'octupolar_contribution': 0.0
    }