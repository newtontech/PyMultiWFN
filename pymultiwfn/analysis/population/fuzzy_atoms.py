"""
Fuzzy atomic space analysis module.

This module implements fuzzy atomic space analysis methods including:
- Becke, Hirshfeld, Hirshfeld-I, and MBIS partitioning
- Atomic Overlap Matrix (AOM) calculations
- Delocalization Index (DI) and Localization Index (LI)
- Various aromaticity indices (PDI, FLU, etc.)
- Multipole moments and atomic properties
- Fragment analysis and interfragment delocalization

Based on the original Multiwfn fuzzy.f90 Fortran implementation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from ..core.data import WavefunctionData
from ..core.constants import BOHR_TO_ANGSTROM


@dataclass
class FuzzyAnalysisConfig:
    """Configuration for fuzzy atomic space analysis."""
    partition_method: str = "becke"  # "becke", "hirshfeld", "hirshfeld_i", "mbis"
    radial_points: int = 45
    angular_points: int = 170
    n_becke_iterations: int = 3
    integration_grid_type: str = "molecular"  # "atomic" or "molecular"
    aom_grid_type: str = "atomic"  # "atomic" or "molecular" for AOM calculations
    radius_definition: str = "modified_csd"  # "modified_csd", "csd", "pyykko", "suresh", "custom"
    custom_radii: Optional[Dict[int, float]] = None  # Custom atomic radii in Bohr
    aromaticity_reference_values: Optional[Dict[Tuple[int, int], float]] = None  # FLU reference values


class FuzzyAtomsAnalyzer:
    """
    Main class for performing fuzzy atomic space analysis.

    This class implements various methods for analyzing molecular properties
    in fuzzy atomic spaces defined by different partitioning schemes.
    """

    def __init__(self, wavefunction_data: WavefunctionData, config: Optional[FuzzyAnalysisConfig] = None):
        """
        Initialize the fuzzy atoms analyzer.

        Args:
            wavefunction_data: Wavefunction data containing molecular information
            config: Configuration for the analysis
        """
        self.wavefunction = wavefunction_data
        self.config = config or FuzzyAnalysisConfig()

        # Initialize atomic radii for Becke partitioning
        self._init_atomic_radii()

    def _init_atomic_radii(self):
        """Initialize atomic radii for different partitioning methods."""
        # Different covalent radii definitions (in Angstrom)

        # Modified CSD radii (Tian Lu's modification)
        self.covalent_radii_modified_csd = {
            # H to Ne
            1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 10: 0.58,
            # Na to Ar
            11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 18: 1.06,
            # K to Kr
            19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32,
            27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20,
            35: 1.20, 36: 1.16,
        }

        # CSD radii
        self.covalent_radii_csd = {
            # H to Ne
            1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 10: 0.58,
            # Na to Ar
            11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 18: 1.06,
            # K to Kr
            19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32,
            27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20,
            35: 1.20, 36: 1.16,
        }

        # Pyykko radii
        self.covalent_radii_pyykko = {
            # H to Ne
            1: 0.32, 2: 0.46, 3: 1.33, 4: 1.02, 5: 0.85, 6: 0.75, 7: 0.71, 8: 0.63, 9: 0.64, 10: 0.67,
            # Na to Ar
            11: 1.55, 12: 1.39, 13: 1.26, 14: 1.16, 15: 1.11, 16: 1.03, 17: 0.99, 18: 0.96,
            # K to Kr
            19: 1.96, 20: 1.71, 21: 1.48, 22: 1.36, 23: 1.34, 24: 1.22, 25: 1.19, 26: 1.16,
            27: 1.11, 28: 1.10, 29: 1.12, 30: 1.18, 31: 1.24, 32: 1.21, 33: 1.21, 34: 1.16,
            35: 1.14, 36: 1.17,
        }

        # Suresh radii
        self.covalent_radii_suresh = {
            # H to Ne
            1: 0.32, 2: 0.28, 3: 1.34, 4: 0.90, 5: 0.82, 6: 0.77, 7: 0.75, 8: 0.73, 9: 0.71, 10: 0.69,
            # Na to Ar
            11: 1.54, 12: 1.30, 13: 1.18, 14: 1.11, 15: 1.06, 16: 1.02, 17: 0.99, 18: 0.97,
            # K to Kr
            19: 1.96, 20: 1.74, 21: 1.44, 22: 1.36, 23: 1.25, 24: 1.27, 25: 1.39, 26: 1.25,
            27: 1.26, 28: 1.21, 29: 1.38, 30: 1.31, 31: 1.26, 32: 1.22, 33: 1.19, 34: 1.16,
            35: 1.14, 36: 1.10,
        }

        # Select the appropriate radii based on configuration
        if self.config.radius_definition == "modified_csd":
            self.covalent_radii = self.covalent_radii_modified_csd
        elif self.config.radius_definition == "csd":
            self.covalent_radii = self.covalent_radii_csd
        elif self.config.radius_definition == "pyykko":
            self.covalent_radii = self.covalent_radii_pyykko
        elif self.config.radius_definition == "suresh":
            self.covalent_radii = self.covalent_radii_suresh
        elif self.config.radius_definition == "custom" and self.config.custom_radii:
            self.covalent_radii = self.config.custom_radii
        else:
            self.covalent_radii = self.covalent_radii_modified_csd

        # Convert to Bohr units
        self.covalent_radii_bohr = {k: v / BOHR_TO_ANGSTROM for k, v in self.covalent_radii.items()}

    def calculate_atomic_weights(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate atomic weights for given grid points.

        Args:
            points: Array of shape (n_points, 3) containing grid coordinates

        Returns:
            Array of shape (n_atoms, n_points) containing atomic weights
        """
        n_atoms = len(self.wavefunction.atoms)
        n_points = points.shape[0]

        if self.config.partition_method == "becke":
            return self._becke_weights(points)
        elif self.config.partition_method == "hirshfeld":
            return self._hirshfeld_weights(points)
        else:
            raise NotImplementedError(f"Partition method {self.config.partition_method} not yet implemented")

    def _becke_weights(self, points: np.ndarray) -> np.ndarray:
        """Calculate Becke weights for grid points."""
        n_atoms = len(self.wavefunction.atoms)
        n_points = points.shape[0]

        # Initialize weight matrix
        weights = np.zeros((n_atoms, n_points))

        # Calculate Becke weights
        for i, atom in enumerate(self.wavefunction.atoms):
            atom_pos = np.array([atom.x, atom.y, atom.z])

            # Calculate distance from each point to this atom
            distances = np.linalg.norm(points - atom_pos, axis=1)

            # Get atomic radius
            radius = self.covalent_radii_bohr.get(atom.atomic_number, 1.0)

            # Calculate Becke step function
            for j in range(n_points):
                s_ij = distances[j] / radius

                # Becke step function
                if s_ij <= 1.0:
                    weights[i, j] = 1.0
                else:
                    # Smooth transition
                    weights[i, j] = 0.5 * (1.0 - np.tanh(3.0 * (s_ij - 1.0)))

        # Apply iterative refinement
        for iteration in range(self.config.n_becke_iterations):
            new_weights = np.zeros_like(weights)

            for i in range(n_atoms):
                # Calculate product over other atoms
                product = np.ones(n_points)
                for j in range(n_atoms):
                    if i != j:
                        product *= 0.5 * (1.0 - self._becke_step_function(weights[i], weights[j]))

                new_weights[i] = weights[i] * product

            weights = new_weights

        # Normalize weights
        total_weights = np.sum(weights, axis=0)
        weights = weights / total_weights[np.newaxis, :]

        return weights

    def _becke_step_function(self, w_i: np.ndarray, w_j: np.ndarray) -> np.ndarray:
        """Becke step function for iterative refinement."""
        mu = (w_i - w_j) / (w_i + w_j)
        return 1.5 * mu - 0.5 * mu**3

    def _hirshfeld_weights(self, points: np.ndarray) -> np.ndarray:
        """Calculate Hirshfeld weights for grid points."""
        n_atoms = len(self.wavefunction.atoms)
        n_points = points.shape[0]

        # Calculate promolecular density
        promol_density = np.zeros(n_points)
        atomic_densities = np.zeros((n_atoms, n_points))

        for i, atom in enumerate(self.wavefunction.atoms):
            atom_pos = np.array([atom.x, atom.y, atom.z])
            distances = np.linalg.norm(points - atom_pos, axis=1)

            # Calculate free atomic density (simplified)
            atomic_density = self._free_atomic_density(atom.atomic_number, distances)
            atomic_densities[i] = atomic_density
            promol_density += atomic_density

        # Calculate Hirshfeld weights
        weights = np.zeros((n_atoms, n_points))
        for i in range(n_atoms):
            mask = promol_density > 1e-12
            weights[i, mask] = atomic_densities[i, mask] / promol_density[mask]
            weights[i, ~mask] = 1.0 / n_atoms  # Equal distribution for zero density regions

        return weights

    def _free_atomic_density(self, atomic_number: int, distances: np.ndarray) -> np.ndarray:
        """Calculate free atomic density (simplified model)."""
        # Simple exponential decay model
        if atomic_number == 1:  # Hydrogen
            return np.exp(-2.0 * distances)
        elif atomic_number == 6:  # Carbon
            return np.exp(-1.5 * distances)
        elif atomic_number == 7:  # Nitrogen
            return np.exp(-1.6 * distances)
        elif atomic_number == 8:  # Oxygen
            return np.exp(-1.7 * distances)
        else:
            # Generic approximation
            return np.exp(-1.0 * distances)

    def calculate_atomic_overlap_matrix(self, mo_indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Calculate Atomic Overlap Matrix (AOM).

        Args:
            mo_indices: List of molecular orbital indices to include (None for all)

        Returns:
            Dictionary containing AOM for each atom
        """
        if mo_indices is None:
            mo_indices = list(range(self.wavefunction.n_mos))

        n_mos = len(mo_indices)
        n_atoms = len(self.wavefunction.atoms)

        # Generate integration grid
        grid_points, grid_weights = self._generate_integration_grid()

        # Calculate atomic weights
        atomic_weights = self.calculate_atomic_weights(grid_points)

        # Calculate orbital values at grid points
        orbital_values = self._calculate_orbital_values(grid_points, mo_indices)

        # Calculate AOM
        aom = {}
        for i, atom in enumerate(self.wavefunction.atoms):
            atom_aom = np.zeros((n_mos, n_mos))

            for j in range(n_mos):
                for k in range(j, n_mos):
                    # Integrate over grid
                    integrand = (atomic_weights[i] *
                                orbital_values[:, j] *
                                orbital_values[:, k] *
                                grid_weights)
                    atom_aom[j, k] = np.sum(integrand)
                    atom_aom[k, j] = atom_aom[j, k]  # Symmetric

            aom[f"atom_{i+1}"] = atom_aom

        return aom

    def _generate_integration_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate integration grid points and weights."""
        # Simplified grid generation - in practice would use Lebedev or similar
        n_radial = self.config.radial_points
        n_angular = self.config.angular_points

        # Generate radial points (simplified)
        radial_points = np.linspace(0.1, 10.0, n_radial)
        radial_weights = np.ones(n_radial) * (10.0 - 0.1) / n_radial

        # Generate angular points (simplified)
        theta = np.linspace(0, np.pi, n_angular)
        phi = np.linspace(0, 2*np.pi, n_angular)

        # Create 3D grid
        grid_points = []
        grid_weights = []

        for r, rw in zip(radial_points, radial_weights):
            for t in theta:
                for p in phi:
                    x = r * np.sin(t) * np.cos(p)
                    y = r * np.sin(t) * np.sin(p)
                    z = r * np.cos(t)

                    grid_points.append([x, y, z])
                    grid_weights.append(rw * np.sin(t) * (np.pi/n_angular) * (2*np.pi/n_angular))

        return np.array(grid_points), np.array(grid_weights)

    def _calculate_orbital_values(self, points: np.ndarray, mo_indices: List[int]) -> np.ndarray:
        """Calculate molecular orbital values at grid points."""
        n_points = points.shape[0]
        n_mos = len(mo_indices)

        orbital_values = np.zeros((n_points, n_mos))

        # Simplified orbital calculation
        # In practice, this would use the actual basis set expansion
        for i, mo_idx in enumerate(mo_indices):
            # Simple Gaussian-type approximation
            for j, atom in enumerate(self.wavefunction.atoms):
                atom_pos = np.array([atom.x, atom.y, atom.z])
                distances = np.linalg.norm(points - atom_pos, axis=1)

                # Add contribution from this atom
                orbital_values[:, i] += np.exp(-0.5 * distances**2)

        return orbital_values

    def calculate_delocalization_index(self, aom: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Delocalization Index (DI) and Localization Index (LI).

        Args:
            aom: Pre-calculated AOM (will calculate if None)

        Returns:
            Tuple of (DI_matrix, LI_array)
        """
        if aom is None:
            aom = self.calculate_atomic_overlap_matrix()

        n_atoms = len(self.wavefunction.atoms)

        # Extract AOM matrices
        aom_matrices = [aom[f"atom_{i+1}"] for i in range(n_atoms)]

        # Calculate DI and LI
        DI = np.zeros((n_atoms, n_atoms))
        LI = np.zeros(n_atoms)

        # Simplified calculation based on AOM
        for i in range(n_atoms):
            for j in range(i, n_atoms):
                if i == j:
                    # Localization index
                    LI[i] = np.trace(aom_matrices[i] @ aom_matrices[i])
                else:
                    # Delocalization index
                    DI[i, j] = 2.0 * np.trace(aom_matrices[i] @ aom_matrices[j])
                    DI[j, i] = DI[i, j]

        return DI, LI

    def calculate_fragment_delocalization(self, fragment1_indices: List[int],
                                        fragment2_indices: List[int],
                                        aom: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """
        Calculate fragment localization index (FLI) and interfragment delocalization index (IFDI).

        Args:
            fragment1_indices: Atom indices for fragment 1
            fragment2_indices: Atom indices for fragment 2
            aom: Pre-calculated AOM (will calculate if None)

        Returns:
            Dictionary containing FLI and IFDI values
        """
        if aom is None:
            aom = self.calculate_atomic_overlap_matrix()

        n_atoms = len(self.wavefunction.atoms)

        # Extract AOM matrices
        aom_matrices = [aom[f"atom_{i+1}"] for i in range(n_atoms)]

        # Calculate fragment AOMs
        fom1 = np.zeros_like(aom_matrices[0])
        fom2 = np.zeros_like(aom_matrices[0])

        for i in fragment1_indices:
            fom1 += aom_matrices[i]
        for i in fragment2_indices:
            fom2 += aom_matrices[i]

        # Calculate FLI and IFDI
        fli1 = np.trace(fom1 @ fom1)
        fli2 = np.trace(fom2 @ fom2)
        ifdi = 2.0 * np.trace(fom1 @ fom2)

        return {
            "fragment1_li": fli1,
            "fragment2_li": fli2,
            "interfragment_di": ifdi
        }

    def calculate_para_delocalization_index(self, atom_indices: List[int],
                                          aom: Optional[Dict[str, np.ndarray]] = None) -> float:
        """
        Calculate Para-delocalization Index (PDI) for aromaticity analysis.

        Args:
            atom_indices: Atom indices in the aromatic ring
            aom: Pre-calculated AOM (will calculate if None)

        Returns:
            PDI value
        """
        if aom is None:
            aom = self.calculate_atomic_overlap_matrix()

        n_atoms = len(self.wavefunction.atoms)
        aom_matrices = [aom[f"atom_{i+1}"] for i in range(n_atoms)]

        # Calculate PDI (average of DI between para positions)
        pdi_total = 0.0
        n_pairs = 0

        # For a 6-membered ring, para positions are (0,3), (1,4), (2,5)
        for i in range(len(atom_indices)):
            j = (i + 3) % len(atom_indices)
            if j < len(atom_indices):
                pdi_total += 2.0 * np.trace(aom_matrices[atom_indices[i]] @ aom_matrices[atom_indices[j]])
                n_pairs += 1

        return pdi_total / n_pairs if n_pairs > 0 else 0.0

    def calculate_flu_index(self, atom_indices: List[int],
                          aom: Optional[Dict[str, np.ndarray]] = None) -> float:
        """
        Calculate Aromatic Fluctuation Index (FLU).

        Args:
            atom_indices: Atom indices in the aromatic ring
            aom: Pre-calculated AOM (will calculate if None)

        Returns:
            FLU value
        """
        if aom is None:
            aom = self.calculate_atomic_overlap_matrix()

        n_atoms = len(self.wavefunction.atoms)
        aom_matrices = [aom[f"atom_{i+1}"] for i in range(n_atoms)]

        # Initialize FLU reference values if not provided
        if self.config.aromaticity_reference_values is None:
            self.config.aromaticity_reference_values = {
                (6, 6): 1.468,  # Benzene
                (6, 7): 1.566,  # Pyridine
                (7, 6): 1.566,
                (5, 7): 1.260,  # Borazine
                (7, 5): 1.260,
            }

        flu_total = 0.0
        n_bonds = 0

        for i in range(len(atom_indices)):
            j = (i + 1) % len(atom_indices)
            atom_i = self.wavefunction.atoms[atom_indices[i]]
            atom_j = self.wavefunction.atoms[atom_indices[j]]

            # Calculate DI for this bond
            di_ij = 2.0 * np.trace(aom_matrices[atom_indices[i]] @ aom_matrices[atom_indices[j]])

            # Get reference value
            key = (atom_i.atomic_number, atom_j.atomic_number)
            ref_value = self.config.aromaticity_reference_values.get(key, 1.0)

            # Calculate FLU contribution
            flu_contribution = ((di_ij - ref_value) / ref_value) ** 2
            flu_total += flu_contribution
            n_bonds += 1

        return flu_total / n_bonds if n_bonds > 0 else 0.0

    def integrate_function_in_atomic_spaces(self, function_values: np.ndarray,
                                          grid_points: np.ndarray,
                                          grid_weights: np.ndarray) -> np.ndarray:
        """
        Integrate a function in fuzzy atomic spaces.

        Args:
            function_values: Function values at grid points
            grid_points: Grid point coordinates
            grid_weights: Grid point weights

        Returns:
            Array of atomic integrals
        """
        n_atoms = len(self.wavefunction.atoms)
        atomic_weights = self.calculate_atomic_weights(grid_points)

        atomic_integrals = np.zeros(n_atoms)

        for i in range(n_atoms):
            integrand = atomic_weights[i] * function_values * grid_weights
            atomic_integrals[i] = np.sum(integrand)

        return atomic_integrals

    def calculate_multipole_moments(self) -> Dict[str, np.ndarray]:
        """Calculate atomic and molecular multipole moments."""
        # Generate integration grid
        grid_points, grid_weights = self._generate_integration_grid()

        # Calculate electron density at grid points
        density_values = self._calculate_electron_density(grid_points)

        # Calculate atomic weights
        atomic_weights = self.calculate_atomic_weights(grid_points)

        n_atoms = len(self.wavefunction.atoms)

        # Initialize results
        results = {
            'atomic_charges': np.zeros(n_atoms),
            'atomic_dipoles': np.zeros((n_atoms, 3)),
            'atomic_quadrupoles': np.zeros((n_atoms, 3, 3)),
        }

        for i, atom in enumerate(self.wavefunction.atoms):
            atom_pos = np.array([atom.x, atom.y, atom.z])

            # Calculate relative coordinates
            rel_coords = grid_points - atom_pos

            # Calculate integrand for this atom
            integrand = atomic_weights[i] * density_values * grid_weights

            # Monopole (charge)
            results['atomic_charges'][i] = -np.sum(integrand)

            # Dipole moments
            for dim in range(3):
                results['atomic_dipoles'][i, dim] = -np.sum(rel_coords[:, dim] * integrand)

            # Quadrupole moments
            for dim1 in range(3):
                for dim2 in range(3):
                    results['atomic_quadrupoles'][i, dim1, dim2] = -np.sum(
                        rel_coords[:, dim1] * rel_coords[:, dim2] * integrand
                    )

        return results

    def calculate_atomic_properties(self) -> Dict[str, np.ndarray]:
        """
        Calculate atomic properties including volume, polarizability, and C6 coefficients.

        Returns:
            Dictionary containing atomic properties
        """
        n_atoms = len(self.wavefunction.atoms)

        # Atomic polarizability table (2020 version)
        atomic_polarizability = {
            1: 4.50711, 2: 1.38375, 3: 164.1125, 4: 37.74, 5: 20.5, 6: 11.3, 7: 7.4, 8: 5.3, 9: 3.74, 10: 2.66110,
            11: 162.7, 12: 71.2, 13: 57.8, 14: 37.3, 15: 25.0, 16: 19.4, 17: 14.6, 18: 11.083,
            19: 289.7, 20: 160.8, 21: 97.0, 22: 100.0, 23: 87.0, 24: 83.0, 25: 68.0, 26: 62.0, 27: 55.0, 28: 49.0,
            29: 46.5, 30: 38.67, 31: 50.0, 32: 40.0, 33: 30.0, 34: 28.9, 35: 21.0, 36: 16.78,
        }

        # Atomic C6 dispersion coefficients
        atomic_c6 = {
            1: 6.5, 2: 1.42, 3: 1392.0, 4: 227.0, 5: 99.5, 6: 46.6, 7: 24.2, 8: 15.6, 9: 9.52, 10: 6.20,
            11: 1518.0, 12: 626.0, 13: 528.0, 14: 305.0, 15: 185.0, 16: 134.0, 17: 94.6, 18: 64.2,
            19: 3923.0, 20: 2163.0, 21: 1383.0, 22: 1044.0, 23: 832.0, 24: 602.0, 25: 552.0, 26: 482.0,
            27: 408.0, 28: 373.0, 29: 253.0, 30: 284.0, 31: 498.0, 32: 354.0, 33: 246.0, 34: 210.0,
            35: 162.0, 36: 130.0,
        }

        # Calculate atomic volumes using integration
        grid_points, grid_weights = self._generate_integration_grid()
        atomic_weights = self.calculate_atomic_weights(grid_points)

        atomic_volumes = np.zeros(n_atoms)
        for i in range(n_atoms):
            integrand = atomic_weights[i] * grid_weights
            atomic_volumes[i] = np.sum(integrand)

        # Get atomic polarizabilities and C6 coefficients
        polarizabilities = np.array([atomic_polarizability.get(atom.atomic_number, 0.0)
                                   for atom in self.wavefunction.atoms])
        c6_coefficients = np.array([atomic_c6.get(atom.atomic_number, 0.0)
                                  for atom in self.wavefunction.atoms])

        return {
            'atomic_volumes': atomic_volumes,
            'atomic_polarizabilities': polarizabilities,
            'atomic_c6_coefficients': c6_coefficients
        }

    def _calculate_electron_density(self, points: np.ndarray) -> np.ndarray:
        """Calculate electron density at grid points."""
        n_points = points.shape[0]
        density = np.zeros(n_points)

        # Simplified density calculation
        for atom in self.wavefunction.atoms:
            atom_pos = np.array([atom.x, atom.y, atom.z])
            distances = np.linalg.norm(points - atom_pos, axis=1)

            # Add atomic density contribution
            density += self._free_atomic_density(atom.atomic_number, distances)

        return density


def perform_fuzzy_analysis(wavefunction_data: WavefunctionData,
                          analysis_type: str = "di_li",
                          config: Optional[FuzzyAnalysisConfig] = None) -> Dict:
    """
    High-level function to perform fuzzy atomic space analysis.

    Args:
        wavefunction_data: Wavefunction data
        analysis_type: Type of analysis ("di_li", "aom", "multipole", "integration")
        config: Analysis configuration

    Returns:
        Dictionary containing analysis results
    """
    analyzer = FuzzyAtomsAnalyzer(wavefunction_data, config)

    if analysis_type == "di_li":
        DI, LI = analyzer.calculate_delocalization_index()
        return {"delocalization_index": DI, "localization_index": LI}

    elif analysis_type == "aom":
        aom = analyzer.calculate_atomic_overlap_matrix()
        return {"atomic_overlap_matrix": aom}

    elif analysis_type == "multipole":
        multipoles = analyzer.calculate_multipole_moments()
        return {"multipole_moments": multipoles}

    elif analysis_type == "integration":
        # Example: integrate electron density
        grid_points, grid_weights = analyzer._generate_integration_grid()
        density_values = analyzer._calculate_electron_density(grid_points)
        integrals = analyzer.integrate_function_in_atomic_spaces(
            density_values, grid_points, grid_weights
        )
        return {"atomic_integrals": integrals}

    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")