"""
Weak Interaction Analysis Module

This module implements visualization analysis of weak interactions in chemical systems,
including NCI (Non-covalent Interaction) analysis, IGM (Independent Gradient Model),
and related methods. Based on the visweak.f90 module from Multiwfn.

References:
- Tian Lu and Qinxue Chen, Visualization Analysis of Weak Interactions in Chemical Systems.
  In Comprehensive Computational Chemistry, vol. 2, pp. 240-264. Oxford: Elsevier (2024)
- Tian Lu, Visualization Analysis of Covalent and Noncovalent Interactions in Real Space,
  Angew. Chem. Int. Ed., 137, e202504895 (2025)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum

from ..core.data import Wavefunction
from ..math.density import calc_density
from ..math.basis import evaluate_basis
from ..config import Config


class AnalysisType(Enum):
    """Types of weak interaction analysis"""
    NCI = "nci"                    # Standard NCI analysis
    NCI_PROMOLECULAR = "nci_promolecular"  # NCI based on promolecular density
    ANCI = "anci"                  # Averaged NCI analysis
    IRI = "iri"                    # Interaction region indicator
    DORI = "dori"                  # Density overlap regions indicator
    VDW_POTENTIAL = "vdw_potential"  # van der Waals potential
    IGM = "igm"                    # Independent gradient model
    MIGM = "migm"                  # Modified IGM
    IGMH = "igmh"                  # IGM based on Hirshfeld partition
    AIGM = "aigm"                  # Averaged IGM
    AMIGM = "amigm"                # Averaged mIGM


@dataclass
class Fragment:
    """Definition of a fragment for interaction analysis"""
    atom_indices: List[int]
    name: str = ""


@dataclass
class WeakInteractionResult:
    """Results from weak interaction analysis"""
    analysis_type: AnalysisType
    grid_data: np.ndarray
    scatter_data: Tuple[np.ndarray, np.ndarray]  # (x, y) for scatter plot
    metadata: Dict

    def __post_init__(self):
        if isinstance(self.scatter_data, list):
            self.scatter_data = tuple(self.scatter_data)


class WeakInteractionAnalyzer:
    """
    Analyzer for weak interactions in chemical systems.

    This class provides various methods for visualizing and analyzing weak interactions,
    including NCI, IGM, and related approaches.
    """

    def __init__(self, wavefunction: Optional[Wavefunction] = None, config: Optional[Config] = None):
        """
        Initialize the weak interaction analyzer.

        Parameters
        ----------
        wavefunction : Wavefunction
            Wavefunction object containing molecular information
        config : Config, optional
            Configuration object
        """
        self.wfn = wavefunction
        self.config = config or Config()
        self.density_calc = calc_density if callable(calc_density) else None

        # Grid parameters
        self._grid_cache = {}

    def analyze_nci(self, use_promolecular: bool = False) -> WeakInteractionResult:
        """
        Perform NCI (Non-covalent Interaction) analysis.

        Parameters
        ----------
        use_promolecular : bool, optional
            Use promolecular density approximation, by default False

        Returns
        -------
        WeakInteractionResult
            Results containing RDG and sign(λ₂)ρ data
        """
        analysis_type = AnalysisType.NCI_PROMOLECULAR if use_promolecular else AnalysisType.NCI

        # Generate grid
        grid_points = self._generate_grid()

        # Calculate density and gradient
        density, gradient = self._calculate_density_gradient(grid_points, use_promolecular)

        # Calculate reduced density gradient (RDG)
        rdg = self._calculate_rdg(density, gradient)

        # Calculate sign(λ₂)ρ
        sign_lambda2_rho = self._calculate_sign_lambda2_rho(grid_points, use_promolecular)

        # Prepare scatter data
        valid_mask = (density > 0) & (np.linalg.norm(gradient, axis=0) > 0)
        scatter_x = sign_lambda2_rho[valid_mask]
        scatter_y = rdg[valid_mask]

        metadata = {
            'use_promolecular': use_promolecular,
            'n_points': len(grid_points[0]),
            'rdg_range': (np.min(rdg), np.max(rdg)),
            'sign_lambda2_rho_range': (np.min(sign_lambda2_rho), np.max(sign_lambda2_rho))
        }

        return WeakInteractionResult(
            analysis_type=analysis_type,
            grid_data={'rdg': rdg, 'sign_lambda2_rho': sign_lambda2_rho, 'density': density},
            scatter_data=(scatter_x, scatter_y),
            metadata=metadata
        )

    def analyze_igm(self, fragments: List[Fragment], igm_type: str = "igm") -> WeakInteractionResult:
        """
        Perform IGM (Independent Gradient Model) analysis.

        Parameters
        ----------
        fragments : List[Fragment]
            List of fragment definitions
        igm_type : str, optional
            Type of IGM analysis: 'igm', 'migm', 'igmh', by default 'igm'

        Returns
        -------
        WeakInteractionResult
            Results containing δg and sign(λ₂)ρ data
        """
        # Generate grid
        grid_points = self._generate_grid()

        # Calculate fragment gradients
        fragment_gradients = []
        for fragment in fragments:
            if igm_type == "igm":
                grad = self._calculate_fragment_gradient_promolecular(fragment.atom_indices, grid_points)
            elif igm_type == "migm":
                grad = self._calculate_fragment_gradient_hirshfeld_promolecular(fragment.atom_indices, grid_points)
            elif igm_type == "igmh":
                grad = self._calculate_fragment_gradient_hirshfeld(fragment.atom_indices, grid_points)
            else:
                raise ValueError(f"Unknown IGM type: {igm_type}")
            fragment_gradients.append(grad)

        # Calculate molecular gradient
        mol_gradient = self._calculate_molecular_gradient(grid_points, igm_type)

        # Calculate δg values
        delta_g_inter, delta_g_intra, delta_g_total = self._calculate_delta_g(
            fragment_gradients, mol_gradient
        )

        # Calculate sign(λ₂)ρ
        sign_lambda2_rho = self._calculate_sign_lambda2_rho(grid_points, False)

        # Prepare scatter data
        scatter_x = sign_lambda2_rho.flatten()
        scatter_y_delta_g_inter = delta_g_inter.flatten()
        scatter_y_delta_g_intra = delta_g_intra.flatten()
        scatter_y_delta_g_total = delta_g_total.flatten()

        metadata = {
            'igm_type': igm_type,
            'n_fragments': len(fragments),
            'fragment_sizes': [len(f.atom_indices) for f in fragments],
            'n_points': len(grid_points[0]),
            'delta_g_inter_range': (np.min(delta_g_inter), np.max(delta_g_inter)),
            'delta_g_intra_range': (np.min(delta_g_intra), np.max(delta_g_intra)),
            'delta_g_total_range': (np.min(delta_g_total), np.max(delta_g_total))
        }

        return WeakInteractionResult(
            analysis_type=AnalysisType.IGM,
            grid_data={
                'delta_g_inter': delta_g_inter,
                'delta_g_intra': delta_g_intra,
                'delta_g_total': delta_g_total,
                'sign_lambda2_rho': sign_lambda2_rho
            },
            scatter_data=(scatter_x, scatter_y_delta_g_inter, scatter_y_delta_g_intra, scatter_y_delta_g_total),
            metadata=metadata
        )

    def analyze_vdw_potential(self, probe_element: str = "Ar") -> WeakInteractionResult:
        """
        Calculate van der Waals potential for visualization.

        Parameters
        ----------
        probe_element : str, optional
            Element symbol for probe atom, by default "Ar"

        Returns
        -------
        WeakInteractionResult
            Results containing van der Waals potential data
        """
        # Generate grid with larger extension for vdW potential
        grid_points = self._generate_grid(extension=8.0)

        # Get UFF parameters for probe and system atoms
        probe_params = self._get_uff_parameters(probe_element)
        atom_params = [self._get_uff_parameters(atom.symbol) for atom in self.wfn.atoms]

        # Calculate vdW potential
        repulsion_grid, dispersion_grid, vdw_grid = self._calculate_vdw_potential(
            grid_points, atom_params, probe_params
        )

        metadata = {
            'probe_element': probe_element,
            'n_points': len(grid_points[0]),
            'repulsion_range': (np.min(repulsion_grid), np.max(repulsion_grid)),
            'dispersion_range': (np.min(dispersion_grid), np.max(dispersion_grid)),
            'vdw_range': (np.min(vdw_grid), np.max(vdw_grid))
        }

        return WeakInteractionResult(
            analysis_type=AnalysisType.VDW_POTENTIAL,
            grid_data={
                'repulsion': repulsion_grid,
                'dispersion': dispersion_grid,
                'vdw_total': vdw_grid
            },
            scatter_data=(),  # No scatter plot for vdW potential
            metadata=metadata
        )

    def _generate_grid(self, extension: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 3D grid points for calculation.

        Parameters
        ----------
        extension : float, optional
            Extension distance beyond molecular bounds in Bohr, by default 2.0

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            X, Y, Z coordinates of grid points
        """
        # Get molecular bounds
        coords = np.array([[atom.x, atom.y, atom.z] for atom in self.wfn.atoms])
        min_coords = np.min(coords, axis=0) - extension
        max_coords = np.max(coords, axis=0) + extension

        # Generate grid
        n_points = self.config.get('grid_points', 50)
        x = np.linspace(min_coords[0], max_coords[0], n_points)
        y = np.linspace(min_coords[1], max_coords[1], n_points)
        z = np.linspace(min_coords[2], max_coords[2], n_points)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        return X.flatten(), Y.flatten(), Z.flatten()

    def _calculate_density_gradient(self, grid_points: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                   use_promolecular: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate electron density and its gradient on grid points."""
        if use_promolecular:
            return self._calculate_promolecular_density_gradient(grid_points)
        else:
            return self.density_calc.calculate_density_and_gradient(grid_points)

    def _calculate_promolecular_density_gradient(self, grid_points: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate promolecular density and gradient."""
        density = np.zeros(len(grid_points[0]))
        gradient = np.zeros((3, len(grid_points[0])))

        for atom in self.wfn.atoms:
            atom_density, atom_gradient = self._calculate_atomic_density_gradient(
                grid_points, atom.index
            )
            density += atom_density
            gradient += atom_gradient

        return density, gradient

    def _calculate_atomic_density_gradient(self, grid_points: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                         atom_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate atomic density and gradient for isolated atom."""
        # This is a simplified implementation
        # In practice, this would use atomic density fitting parameters
        x, y, z = grid_points
        atom = self.wfn.atoms[atom_index]

        # Distance from atom center
        r = np.sqrt((x - atom.x)**2 + (y - atom.y)**2 + (z - atom.z)**2)

        # Simplified atomic density (exponential decay)
        # In practice, this would use fitted STO parameters
        alpha = 1.0  # Should be atom-dependent
        density = np.exp(-alpha * r)

        # Gradient
        gradient = np.zeros((3, len(r)))
        mask = r > 0
        gradient[0, mask] = -alpha * density[mask] * (x[mask] - atom.x) / r[mask]
        gradient[1, mask] = -alpha * density[mask] * (y[mask] - atom.y) / r[mask]
        gradient[2, mask] = -alpha * density[mask] * (z[mask] - atom.z) / r[mask]

        return density, gradient

    def _calculate_rdg(self, density: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Calculate reduced density gradient."""
        grad_norm = np.linalg.norm(gradient, axis=0)

        # Avoid division by zero
        mask = (density > 0) & (grad_norm > 0)
        rdg = np.zeros_like(density)

        # RDG = |∇ρ| / (2 * (3π²)^(1/3) * ρ^(4/3))
        constant = 0.161620459673995  # 1 / (2 * (3π²)^(1/3))
        rdg[mask] = constant * grad_norm[mask] / (density[mask] ** (4.0/3.0))

        # Handle special cases
        rdg[density >= self.config.get('RDG_max_density', 0.0)] = 100.0
        rdg[(grad_norm == 0) | (density == 0)] = 999.0

        return rdg

    def _calculate_sign_lambda2_rho(self, grid_points: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                   use_promolecular: bool = False) -> np.ndarray:
        """Calculate sign(λ₂)ρ where λ₂ is the second eigenvalue of Hessian."""
        # Calculate Hessian of density
        hessian = self._calculate_density_hessian(grid_points, use_promolecular)

        # Calculate eigenvalues of Hessian at each point
        sign_lambda2_rho = np.zeros(len(grid_points[0]))

        for i in range(len(grid_points[0])):
            hess_matrix = hessian[:, :, i]
            eigenvalues = np.linalg.eigvalsh(hess_matrix)
            eigenvalues.sort()  # Sort in ascending order

            lambda2 = eigenvalues[1]  # Second eigenvalue
            density = self._calculate_density_at_point(grid_points, i, use_promolecular)

            if lambda2 != 0:
                sign_lambda2_rho[i] = density * lambda2 / abs(lambda2)
            else:
                sign_lambda2_rho[i] = -density  # Around nuclei, always negative

        return sign_lambda2_rho

    def _calculate_density_hessian(self, grid_points: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                  use_promolecular: bool = False) -> np.ndarray:
        """Calculate Hessian matrix of electron density."""
        n_points = len(grid_points[0])
        hessian = np.zeros((3, 3, n_points))

        # Numerical differentiation for Hessian
        delta = 0.001  # Small displacement for numerical derivative

        for i in range(n_points):
            x, y, z = grid_points[0][i], grid_points[1][i], grid_points[2][i]

            # Calculate density at displaced points
            density_center = self._calculate_density_at_point(grid_points, i, use_promolecular)

            # Second derivatives
            for j, coord in enumerate([x, y, z]):
                coord_plus = coord + delta
                coord_minus = coord - delta

                # Create displaced grid points
                grid_plus = list(grid_points)
                grid_minus = list(grid_points)
                grid_plus[j][i] = coord_plus
                grid_minus[j][i] = coord_minus

                density_plus = self._calculate_density_at_point(grid_plus, i, use_promolecular)
                density_minus = self._calculate_density_at_point(grid_minus, i, use_promolecular)

                hessian[j, j, i] = (density_plus - 2 * density_center + density_minus) / (delta ** 2)

        return hessian

    def _calculate_density_at_point(self, grid_points: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                   index: int, use_promolecular: bool = False) -> float:
        """Calculate density at a specific grid point."""
        if use_promolecular:
            density = 0.0
            for atom in self.wfn.atoms:
                atom_density, _ = self._calculate_atomic_density_gradient(
                    (grid_points[0][index:index+1],
                     grid_points[1][index:index+1],
                     grid_points[2][index:index+1]),
                    atom.index
                )
                density += atom_density[0]
            return density
        else:
            return self.density_calc.calculate_density_at_point(
                grid_points[0][index], grid_points[1][index], grid_points[2][index]
            )

    def _calculate_fragment_gradient_promolecular(self, atom_indices: List[int],
                                                grid_points: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Calculate gradient for a fragment using promolecular approximation."""
        gradient = np.zeros((3, len(grid_points[0])))

        for atom_idx in atom_indices:
            _, atom_gradient = self._calculate_atomic_density_gradient(grid_points, atom_idx)
            gradient += atom_gradient

        return gradient

    def _calculate_fragment_gradient_hirshfeld_promolecular(self, atom_indices: List[int],
                                                           grid_points: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Calculate gradient for fragment using Hirshfeld partition of promolecular density."""
        # Calculate promolecular density
        promol_density, promol_gradient = self._calculate_promolecular_density_gradient(grid_points)

        # Calculate atomic densities for partitioning
        fragment_gradient = np.zeros((3, len(grid_points[0])))

        for atom_idx in atom_indices:
            atom_density, atom_gradient = self._calculate_atomic_density_gradient(grid_points, atom_idx)

            # Hirshfeld weight
            mask = promol_density > 0
            weight = np.zeros_like(promol_density)
            weight[mask] = atom_density[mask] / promol_density[mask]

            # Weighted gradient contribution
            fragment_gradient += weight * atom_gradient

        return fragment_gradient

    def _calculate_fragment_gradient_hirshfeld(self, atom_indices: List[int],
                                             grid_points: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Calculate gradient for fragment using Hirshfeld partition of actual density."""
        # Calculate actual molecular density and gradient
        mol_density, mol_gradient = self.density_calc.calculate_density_and_gradient(grid_points)

        # Calculate promolecular density for weighting
        promol_density, _ = self._calculate_promolecular_density_gradient(grid_points)

        # Calculate atomic densities
        fragment_gradient = np.zeros((3, len(grid_points[0])))

        for atom_idx in atom_indices:
            atom_density, atom_gradient = self._calculate_atomic_density_gradient(grid_points, atom_idx)

            # Hirshfeld weight
            mask = promol_density > 0
            weight = np.zeros_like(promol_density)
            weight[mask] = atom_density[mask] / promol_density[mask]

            # Special IGMH formula (see Multiwfn documentation)
            mask2 = promol_density > 0
            weighted_grad = np.zeros((3, len(grid_points[0])))

            weighted_grad[:, mask2] = (weight[:, mask2] * mol_gradient[:, mask2] -
                                      mol_density[mask2] * atom_gradient[:, mask2] / promol_density[mask2] +
                                      mol_density[mask2] * atom_density[mask2] *
                                      np.sum(atom_gradient[:, mask2], axis=0) / promol_density[mask2]**2)

            fragment_gradient += weighted_grad

        return fragment_gradient

    def _calculate_molecular_gradient(self, grid_points: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                   igm_type: str) -> np.ndarray:
        """Calculate molecular gradient based on IGM type."""
        if igm_type in ["igm", "migm"]:
            _, gradient = self._calculate_promolecular_density_gradient(grid_points)
        else:  # igmh
            _, gradient = self.density_calc.calculate_density_and_gradient(grid_points)
        return gradient

    def _calculate_delta_g(self, fragment_gradients: List[np.ndarray],
                          mol_gradient: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate δg values from fragment and molecular gradients."""
        # δg_inter: contribution from inter-fragment interactions
        # δg_intra: contribution from intra-fragment interactions
        # δg_total: total contribution

        grad_norm_mol = np.linalg.norm(mol_gradient, axis=0)

        # Calculate IGM gradient norm (sum of fragment gradient norms)
        igm_grad_norm = np.zeros_like(grad_norm_mol)
        for frag_grad in fragment_gradients:
            igm_grad_norm += np.linalg.norm(frag_grad, axis=0)

        # Total δg
        delta_g_total = igm_grad_norm - grad_norm_mol

        # For δg_inter and δg_intra, we need to consider how fragments combine
        # This is a simplified calculation - in practice, this is more complex
        combined_frag_gradient = np.zeros_like(mol_gradient)
        for frag_grad in fragment_gradients:
            combined_frag_gradient += frag_grad

        delta_g_inter = igm_grad_norm - np.linalg.norm(combined_frag_gradient, axis=0)
        delta_g_intra = delta_g_total - delta_g_inter

        return delta_g_inter, delta_g_intra, delta_g_total

    def _get_uff_parameters(self, element: str) -> Tuple[float, float]:
        """Get UFF force field parameters for an element."""
        # Simplified UFF parameters (A, B) for well depth and radius
        # In practice, this should contain all 103 elements
        uff_params = {
            'H': (0.044, 2.886),
            'C': (0.105, 3.851),
            'N': (0.069, 3.660),
            'O': (0.060, 3.500),
            'F': (0.050, 3.364),
            'P': (0.200, 4.044),
            'S': (0.198, 4.000),
            'Cl': (0.265, 3.947),
            'Br': (0.322, 4.220),
            'I': (0.350, 4.477),
            'Ar': (0.237, 3.823),  # Probe atom
        }

        return uff_params.get(element, (0.100, 4.000))  # Default values

    def _calculate_vdw_potential(self, grid_points: Tuple[np.ndarray, np.ndarray, np.ndarray],
                               atom_params: List[Tuple[float, float]],
                               probe_params: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate van der Waals potential on grid points."""
        x, y, z = grid_points
        n_points = len(x)

        repulsion_grid = np.zeros(n_points)
        dispersion_grid = np.zeros(n_points)

        probe_A, probe_B = probe_params

        for i, (atom, (atom_A, atom_B)) in enumerate(zip(self.wfn.atoms, atom_params)):
            # Distance from atom
            r = np.sqrt((x - atom.x)**2 + (y - atom.y)**2 + (z - atom.z)**2) * 0.529  # Convert to Angstrom

            # Avoid singularities
            r = np.maximum(r, 0.1)

            # Lennard-Jones parameters
            D_ij = np.sqrt(atom_A * probe_A)  # Well depth
            X_ij = np.sqrt(atom_B * probe_B)  # vdW distance

            # Calculate potential (only for atoms within 25 Angstrom)
            mask = r <= 25.0
            repulsion_grid[mask] += D_ij * (X_ij / r[mask])**12
            dispersion_grid[mask] -= 2 * D_ij * (X_ij / r[mask])**6

        vdw_grid = repulsion_grid + dispersion_grid

        return repulsion_grid, dispersion_grid, vdw_grid