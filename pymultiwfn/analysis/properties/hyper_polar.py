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

from ...core.data import Wavefunction


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

        # Conversion factors for different units
        self.au2si_alpha = 1.648777274e-41    # a.u. to C^2*m^2/J for polarizability
        self.au2si_beta = 3.206361306e-53     # a.u. to C^3*m^3/J for hyperpolarizability
        self.au2esu_alpha = 0.393456          # a.u. to esu for polarizability
        self.au2esu_beta = 2.9689e-32         # a.u. to esu for hyperpolarizability

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
        dipole_match = re.search(r"Dipole moment.*?X=\s*([-\d.]+)\s*Y=\s*([-\d.]+)\s*Z=\s*([-\d.]+)", content)
        if dipole_match:
            dipole = np.array([float(x) for x in dipole_match.groups()]) / self.au2debye
            results['dipole'] = dipole

        # Parse polarizability (alpha) - improved parsing
        alpha = np.zeros((3, 3))

        # Look for SCF Polarizability section
        alpha_section = re.search(r"SCF Polarizability for W=\s*[-\d.]+.*?\n\s*\d+\s+\d+\s+\d+\s*\n\s*1\s*([-\d.ED+E-]+).*?\n\s*2\s*([-\d.ED+E-]+)\s*([-\d.ED+E-]+).*?\n\s*3\s*([-\d.ED+E-]+)\s*([-\d.ED+E-]+)\s*([-\d.ED+E-]+)",
                                content, re.DOTALL)
        if alpha_section:
            values = [self._parse_fortran_scientific(x) for x in alpha_section.groups()]
            alpha[0, 0] = values[0]
            alpha[1, 0] = alpha[0, 1] = values[1]
            alpha[1, 1] = values[2]
            alpha[2, 0] = alpha[0, 2] = values[3]
            alpha[2, 1] = alpha[1, 2] = values[4]
            alpha[2, 2] = values[5]
            results['alpha'] = alpha

        # Parse first hyperpolarizability (beta) - improved parsing
        beta = np.zeros((3, 3, 3))

        # Look for SCF Static Hyperpolarizability section
        beta_k1 = re.search(r"K=\s*1 block:.*?\n\s*\d+.*?\n\s*1\s*([-\d.ED+E-]+)", content, re.DOTALL)
        beta_k2 = re.search(r"K=\s*2 block:.*?\n\s*\d+\s+\d+.*?\n\s*1\s*([-\d.ED+E-]+).*?\n\s*2\s*([-\d.ED+E-]+)\s*([-\d.ED+E-]+)", content, re.DOTALL)
        beta_k3 = re.search(r"K=\s*3 block:.*?\n\s*\d+\s+\d+\s+\d+.*?\n\s*1\s*([-\d.ED+E-]+).*?\n\s*2\s*([-\d.ED+E-]+)\s*([-\d.ED+E-]+).*?\n\s*3\s*([-\d.ED+E-]+)\s*([-\d.ED+E-]+)\s*([-\d.ED+E-]+)", content, re.DOTALL)

        if all([beta_k1, beta_k2, beta_k3]):
            # Parse K=1 block
            values1 = [self._parse_fortran_scientific(x) for x in beta_k1.groups()]
            # Parse K=2 block
            values2 = [self._parse_fortran_scientific(x) for x in beta_k2.groups()]
            # Parse K=3 block
            values3 = [self._parse_fortran_scientific(x) for x in beta_k3.groups()]

            # Fill beta tensor according to Gaussian format
            beta[0, 0, 0] = values1[0]  # XXX

            beta[0, 0, 1] = values2[0]  # XXY
            beta[1, 0, 0] = values2[0]
            beta[0, 1, 0] = values2[0]

            beta[0, 1, 1] = values2[1]  # XYY
            beta[1, 0, 1] = values2[1]
            beta[1, 1, 0] = values2[1]

            beta[1, 1, 1] = values2[2]  # YYY

            # K=3 block values
            beta[0, 0, 2] = values3[0]  # XXZ
            beta[0, 2, 0] = values3[0]
            beta[2, 0, 0] = values3[0]

            beta[0, 1, 2] = values3[1]  # XYZ
            beta[0, 2, 1] = values3[1]
            beta[1, 0, 2] = values3[1]
            beta[1, 2, 0] = values3[1]
            beta[2, 0, 1] = values3[1]
            beta[2, 1, 0] = values3[1]

            beta[1, 1, 2] = values3[2]  # YYZ
            beta[1, 2, 1] = values3[2]
            beta[2, 1, 1] = values3[2]

            beta[0, 2, 2] = values3[3]  # XZZ
            beta[2, 0, 2] = values3[3]
            beta[2, 2, 0] = values3[3]

            beta[1, 2, 2] = values3[4]  # YZZ
            beta[2, 1, 2] = values3[4]
            beta[2, 2, 1] = values3[4]

            beta[2, 2, 2] = values3[5]  # ZZZ

            results['beta'] = beta

        # Convert to requested units
        if output_unit == "SI":
            if 'alpha' in results:
                results['alpha'] *= self.au2si_alpha
            if 'beta' in results:
                results['beta'] *= self.au2si_beta
        elif output_unit == "esu":
            if 'alpha' in results:
                results['alpha'] *= self.au2esu_alpha
            if 'beta' in results:
                results['beta'] *= self.au2esu_beta

        return results

    def _parse_fortran_scientific(self, value_str: str) -> float:
        """Parse Fortran scientific notation like '0.136927D+02'"""
        value_str = value_str.strip().upper()
        if 'D' in value_str:
            value_str = value_str.replace('D', 'E')
        return float(value_str)

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

    def _calculate_polarizability_sos(self, filename: Optional[str] = None,
                                     frequencies: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate polarizability using sum-over-states (SOS) method.

        Args:
            filename: Path to Gaussian output file with TD-DFT/TD-HF data
            frequencies: Array of frequencies to calculate frequency-dependent polarizability

        Returns:
            Polarizability tensor (3, 3)
        """
        alpha = np.zeros((3, 3))

        if filename is None:
            raise ValueError("Gaussian output file required for SOS calculation")

        # Parse excitation energies and transition dipole moments
        excitation_data = self._parse_excitation_data(filename)

        if not excitation_data:
            raise ValueError("No excitation data found in file")

        excitation_energies = excitation_data['excitation_energies']  # in a.u.
        transition_dipoles = excitation_data['transition_dipoles']    # (n_states, 3)

        # Apply SOS formula for static polarizability (w=0)
        # alpha_ij = 2 * sum_n <0|r_i|n><n|r_j|0> / E_n
        n_states = len(excitation_energies)
        for i in range(3):
            for j in range(3):
                for n in range(n_states):
                    if excitation_energies[n] > 1e-10:  # Avoid division by zero
                        alpha[i, j] += 2.0 * transition_dipoles[n, i] * transition_dipoles[n, j] / excitation_energies[n]

        return alpha

    def _parse_excitation_data(self, filename: str) -> Dict[str, np.ndarray]:
        """
        Parse excitation energies and transition dipole moments from Gaussian output.

        Args:
            filename: Path to Gaussian output file

        Returns:
            Dictionary containing excitation data
        """
        with open(filename, 'r') as f:
            content = f.read()

        # First, find all excited states
        state_pattern = r"Excited State\s+(\d+):\s*([A-Za-z-]+)\s+([\d.]+)\s*eV\s*([\d.]+)\s*nm\s*f=([\d.]+)"
        states = re.findall(state_pattern, content)

        if not states:
            return {}

        n_states = len(states)
        excitation_energies = np.zeros(n_states)  # in eV initially
        excitation_wavelengths = np.zeros(n_states)
        oscillator_strengths = np.zeros(n_states)

        for i, (state_num, symmetry, energy, wavelength, f_value) in enumerate(states):
            excitation_energies[i] = float(energy)
            excitation_wavelengths[i] = float(wavelength)
            oscillator_strengths[i] = float(f_value)

        # Parse transition dipole moments
        trans_dip_pattern = r"Ground to excited state transition electric dipole moments.*?(\d+)\s+([-\d.ED+E-]+)\s+([-\d.ED+E-]+)\s+([-\d.ED+E-]+)"
        trans_dip_matches = re.findall(trans_dip_pattern, content, re.DOTALL)

        transition_dipoles = np.zeros((n_states, 3))
        for i, (state_num, x, y, z) in enumerate(trans_dip_matches[:n_states]):
            transition_dipoles[i, 0] = self._parse_fortran_scientific(x)
            transition_dipoles[i, 1] = self._parse_fortran_scientific(y)
            transition_dipoles[i, 2] = self._parse_fortran_scientific(z)

        # Convert energies to atomic units
        excitation_energies_au = excitation_energies / self.au2eV

        return {
            'excitation_energies': excitation_energies_au,
            'excitation_energies_eV': excitation_energies,
            'transition_dipoles': transition_dipoles,
            'oscillator_strengths': oscillator_strengths,
            'wavelengths': excitation_wavelengths
        }

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
    # Field strength for finite difference (default: 0.003 a.u. as in Multiwfn)
    field_strength = 0.003

    if hyperpolarizability_order == 1:
        # First hyperpolarizability density: needs 3 points (-F, 0, +F)
        field_values = [-field_strength, 0.0, field_strength]
        coefficients = [-0.5, 0.0, 0.5]  # Central difference formula
    elif hyperpolarizability_order == 2:
        # Second hyperpolarizability density: needs 4 points (-2F, -F, +F, +2F)
        field_values = [-2*field_strength, -field_strength, field_strength, 2*field_strength]
        coefficients = [0.25, -1.0, 1.0, -0.25]  # 4th order finite difference
    else:
        raise ValueError(f"Unsupported hyperpolarizability order: {hyperpolarizability_order}")

    density = np.zeros(len(grid_points))

    # Calculate densities at different field values
    densities = []
    for field_val in field_values:
        # This is a simplified implementation
        # In practice, you would need to:
        # 1. Apply the electric field perturbation to the Hamiltonian
        # 2. Recompute the electron density at this field
        # 3. Store the density for the finite difference calculation

        # For now, return zeros as placeholder
        field_density = np.zeros(len(grid_points))
        densities.append(field_density)

    # Apply finite difference coefficients
    for i, coeff in enumerate(coefficients):
        density += coeff * densities[i]

    # Normalize by field strength
    if hyperpolarizability_order == 1:
        density /= field_strength
    elif hyperpolarizability_order == 2:
        density /= (field_strength**2)

    return density


def generate_gaussian_input_files(wavefunction: Wavefunction,
                                 output_dir: str = ".",
                                 field_direction: int = 2,  # Z direction
                                 field_strength: float = 0.003,
                                 calculation_type: str = "polarizability",
                                 method: str = "PBE1PBE",
                                 basis: str = "aug-cc-pVTZ",
                                 charge: int = 0,
                                 multiplicity: int = 1) -> List[str]:
    """
    Generate Gaussian input files for hyperpolarizability density analysis.

    Args:
        wavefunction: Wavefunction object containing molecular structure
        output_dir: Directory to save input files
        field_direction: Direction of electric field (0=x, 1=y, 2=z)
        field_strength: Strength of electric field in a.u.
        calculation_type: Type of calculation ('polarizability', 'hyperpolarizability_1', 'hyperpolarizability_2')
        method: Quantum chemical method
        basis: Basis set
        charge: Molecular charge
        multiplicity: Spin multiplicity

    Returns:
        List of generated input file paths
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Define field configurations based on calculation type
    if calculation_type == "polarizability":
        field_configs = [-1, 1]  # Two points for first derivative
        suffixes = ["-1", "+1"]
    elif calculation_type == "hyperpolarizability_1":
        field_configs = [-1, 0, 1]  # Three points for second derivative
        suffixes = ["-1", "_0", "+1"]
    elif calculation_type == "hyperpolarizability_2":
        field_configs = [-2, -1, 1, 2]  # Four points for third derivative
        suffixes = ["-2", "-1", "+1", "+2"]
    else:
        raise ValueError(f"Unknown calculation type: {calculation_type}")

    direction_label = ["X", "Y", "Z"][field_direction]
    generated_files = []

    for field_val, suffix in zip(field_configs, suffixes):
        filename = f"{direction_label}{suffix}.gjf"
        filepath = output_path / filename

        # Build the field specification
        if field_val == 0:
            field_spec = ""
        else:
            field_spec = f"field={direction_label}"
            # Convert field strength to Gaussian format (inverted sign convention)
            gauss_field = -field_strength * field_val
            field_int = int(round(gauss_field * 10000))
            if gauss_field > 0:
                field_spec += f"+{field_int:04d}"
            else:
                field_spec += f"{field_int:04d}"

        # Write Gaussian input file
        with open(filepath, 'w') as f:
            f.write(f"#P {method}/{basis} int(ultrafine,acc2e=14) scf(noincfock,novaracc) out=wfx nosymm {field_spec}\n")
            f.write(f"\n{direction_label}{suffix}\n")
            f.write(f"{charge} {multiplicity}\n")

            # Write atomic coordinates
            if wavefunction.atoms is not None:
                for atom in wavefunction.atoms:
                    x = atom.x * 0.529177  # Convert Bohr to Angstrom
                    y = atom.y * 0.529177
                    z = atom.z * 0.529177
                    f.write(f"{atom.symbol:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")
            else:
                raise ValueError("No atomic coordinates found in wavefunction")

            f.write("\n")
            f.write(f"{direction_label}{suffix}.wfx\n\n")

        generated_files.append(str(filepath))

    return generated_files


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

def analyze_static_polarizability(wavefunction: Wavefunction,
                                  filename: Optional[str] = None,
                                  method: str = "parse") -> Dict[str, Union[np.ndarray, float]]:
    """
    Convenience function for static polarizability analysis.

    Args:
        wavefunction: Wavefunction object
        filename: Path to Gaussian output file (if method="parse")
        method: Method to use ("parse" to read from file, "sos" for sum-over-states)

    Returns:
        Dictionary containing polarizability tensor and properties
    """
    analyzer = HyperPolarizabilityAnalyzer(wavefunction)

    if method == "parse":
        if filename is None:
            raise ValueError("filename required when method='parse'")
        results = analyzer.parse_gaussian_polarizability(filename)
        if 'alpha' not in results:
            raise ValueError("No polarizability data found in file")
        alpha = results['alpha']
    elif method == "sos":
        if filename is None:
            raise ValueError("filename required when method='sos'")
        alpha = analyzer._calculate_polarizability_sos(filename)
    else:
        raise ValueError(f"Unknown method: {method}")

    properties = analyzer.analyze_polarizability_properties(alpha)

    return {
        'tensor': alpha,
        'properties': properties,
        'method': method
    }


def analyze_static_hyperpolarizability(wavefunction: Wavefunction,
                                      filename: Optional[str] = None,
                                      dipole: Optional[np.ndarray] = None) -> Dict[str, Union[np.ndarray, float]]:
    """
    Convenience function for static first hyperpolarizability analysis.

    Args:
        wavefunction: Wavefunction object
        filename: Path to Gaussian output file containing hyperpolarizability data
        dipole: Dipole moment vector

    Returns:
        Dictionary containing hyperpolarizability tensor and properties
    """
    analyzer = HyperPolarizabilityAnalyzer(wavefunction)

    if filename is None:
        raise ValueError("filename required for hyperpolarizability analysis")

    results = analyzer.parse_gaussian_polarizability(filename)
    if 'beta' not in results:
        raise ValueError("No hyperpolarizability data found in file")

    beta = results['beta']
    properties = analyzer.analyze_hyperpolarizability_properties(beta, dipole)

    return {
        'tensor': beta,
        'properties': properties
    }


def calculate_sos_polarizability(wavefunction: Wavefunction,
                               filename: Optional[str] = None,
                               frequencies: Optional[np.ndarray] = None,
                               num_states: Optional[int] = None) -> Dict[str, Union[np.ndarray, float]]:
    """
    Calculate polarizability using sum-over-states (SOS) method.

    Args:
        wavefunction: Wavefunction object with excitation data
        filename: Path to Gaussian output file with TD-DFT/TD-HF data
        frequencies: Array of frequencies for frequency-dependent calculation
        num_states: Number of excited states to include (None = all available)

    Returns:
        Dictionary containing polarizability tensor and properties
    """
    if filename is None:
        raise ValueError("Gaussian output file required for SOS calculation")

    analyzer = HyperPolarizabilityAnalyzer(wavefunction)
    alpha = analyzer._calculate_polarizability_sos(filename, frequencies)
    properties = analyzer.analyze_polarizability_properties(alpha)

    return {
        'tensor': alpha,
        'properties': properties,
        'method': 'sum-over-states'
    }


def two_level_model_analysis(wavefunction: Wavefunction,
                            filename: str,
                            state_index: int) -> Dict[str, float]:
    """
    Perform two-level model analysis of hyperpolarizability.

    Args:
        wavefunction: Wavefunction object
        filename: Path to Gaussian output file with TD-DFT/TD-HF data
        state_index: Index of excited state to analyze (1-based)

    Returns:
        Dictionary containing two-level model analysis results
    """
    if filename is None:
        raise ValueError("Gaussian output file required for two-level analysis")

    analyzer = HyperPolarizabilityAnalyzer(wavefunction)
    excitation_data = analyzer._parse_excitation_data(filename)

    if not excitation_data:
        raise ValueError("No excitation data found in file")

    # Convert to 0-based indexing
    state_idx = state_index - 1
    if state_idx < 0 or state_idx >= len(excitation_data['excitation_energies']):
        raise ValueError(f"Invalid state index: {state_index}")

    # Extract data for the specified state
    excitation_energy = excitation_data['excitation_energies'][state_idx]  # in a.u.
    transition_dipole = excitation_data['transition_dipoles'][state_idx]   # in a.u.
    oscillator_strength = excitation_data['oscillator_strengths'][state_idx]

    # Two-level model formula for static beta
    # beta_ijk = -3 * e^3 * <0|r_i|n><n|r_j|0><0|r_k|n> / (E_n^2)
    # This is a simplified version - actual implementation would be more complex

    beta_components = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if excitation_energy > 1e-10:
                    beta_components[i, j, k] = (-3.0 * transition_dipole[i] *
                                               transition_dipole[j] * transition_dipole[k] /
                                               (excitation_energy**2))

    return {
        'state_index': state_index,
        'excitation_energy_eV': excitation_energy * analyzer.au2eV,
        'oscillator_strength': oscillator_strength,
        'transition_dipole_au': transition_dipole,
        'beta_tensor': beta_components,
        'beta_magnitude': np.linalg.norm(beta_components.flatten())
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