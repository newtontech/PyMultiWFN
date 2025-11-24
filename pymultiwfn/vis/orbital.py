"""
Orbital Visualization Module for PyMultiWFN

This module provides functionality for visualizing molecular orbitals,
including isosurfaces, orbital plots, and related properties.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Optional, Union

from ..core.data import Wavefunction, Atom
from ..math.basis import evaluate_basis


class OrbitalVisualizer:
    """
    Class for visualizing molecular orbitals and orbital-related properties.

    This class provides methods for creating isosurfaces, contour plots,
    and other visualizations of molecular orbitals.
    """

    def __init__(self):
        """Initialize the orbital visualizer"""
        self.wavefunction = None
        self.current_orbital = None
        self.grid_data = None
        self.isovalue = 0.02

    def set_wavefunction(self, wavefunction: Wavefunction):
        """Set the wavefunction for orbital visualization"""
        self.wavefunction = wavefunction

    def generate_grid(self, n_points: int = 50, padding: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a 3D grid for visualization

        Args:
            n_points: Number of grid points in each dimension
            padding: Padding distance beyond molecular extent (in Angstroms)

        Returns:
            Tuple of (X, Y, Z) coordinate arrays
        """
        if not self.wavefunction:
            raise ValueError("No wavefunction set")

        atoms = self.wavefunction.atoms
        coords = np.array([[atom.x, atom.y, atom.z] for atom in atoms])

        # Determine molecular bounds
        min_coords = coords.min(axis=0) - padding
        max_coords = coords.max(axis=0) + padding

        # Create grid
        x = np.linspace(min_coords[0], max_coords[0], n_points)
        y = np.linspace(min_coords[1], max_coords[1], n_points)
        z = np.linspace(min_coords[2], max_coords[2], n_points)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        self.grid_data = {'X': X, 'Y': Y, 'Z': Z, 'coords': (x, y, z)}
        return X, Y, Z

    def evaluate_orbital(self, orbital_index: int, grid_points: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate orbital on a grid

        Args:
            orbital_index: Index of the orbital to evaluate
            grid_points: Optional grid points, uses stored grid if None

        Returns:
            Array of orbital values on the grid
        """
        if not self.wavefunction:
            raise ValueError("No wavefunction set")

        if orbital_index < 0 or orbital_index >= self.wavefunction.n_mo:
            raise ValueError(f"Orbital index {orbital_index} out of range")

        if grid_points is None:
            if self.grid_data is None:
                self.generate_grid()
            grid_points = np.stack([self.grid_data['X'].flatten(),
                                   self.grid_data['Y'].flatten(),
                                   self.grid_data['Z'].flatten()], axis=1)

        # Get orbital coefficients
        orbital_coeffs = self.wavefunction.mo_coeffs[:, orbital_index]

        # Evaluate orbital on grid points
        orbital_values = np.zeros(len(grid_points))

        # This would use the actual basis function evaluation
        # For now, provide a placeholder implementation
        n_basis = len(orbital_coeffs)
        for i in range(n_basis):
            if abs(orbital_coeffs[i]) > 1e-10:
                # Evaluate basis function i at all grid points
                basis_values = self._evaluate_basis_function(i, grid_points)
                orbital_values += orbital_coeffs[i] * basis_values

        return orbital_values.reshape(self.grid_data['X'].shape)

    def _evaluate_basis_function(self, basis_index: int, grid_points: np.ndarray) -> np.ndarray:
        """
        Evaluate a single basis function on grid points

        Args:
            basis_index: Index of the basis function
            grid_points: Grid points to evaluate

        Returns:
            Array of basis function values
        """
        # This is a placeholder - would need proper basis function evaluation
        # For demonstration, create a simple Gaussian-like function
        if not self.wavefunction or basis_index >= len(self.wavefunction.atoms):
            return np.zeros(len(grid_points))

        atom_idx = basis_index // 10  # Simplified mapping
        atom_idx = min(atom_idx, len(self.wavefunction.atoms) - 1)
        atom = self.wavefunction.atoms[atom_idx]
        atom_center = np.array([atom.x, atom.y, atom.z])

        # Simple Gaussian basis function
        alpha = 1.0
        distances = np.linalg.norm(grid_points - atom_center, axis=1)
        values = np.exp(-alpha * distances**2)

        return values

    def create_isosurface_plotly(self, orbital_index: int, isovalue: Optional[float] = None) -> go.Figure:
        """
        Create a 3D isosurface plot using plotly

        Args:
            orbital_index: Index of the orbital to visualize
            isovalue: Isovalue for surface generation

        Returns:
            plotly Figure object
        """
        if not self.wavefunction:
            raise ValueError("No wavefunction set")

        if isovalue is None:
            isovalue = self.isovalue

        # Generate grid and evaluate orbital
        self.generate_grid(n_points=30)  # Lower resolution for interactive plot
        orbital_data = self.evaluate_orbital(orbital_index)

        # Create isosurface using plotly
        fig = go.Figure(data=go.Isosurface(
            x=self.grid_data['X'].flatten(),
            y=self.grid_data['Y'].flatten(),
            z=self.grid_data['Z'].flatten(),
            value=orbital_data.flatten(),
            isomin=-isovalue,
            isomax=isovalue,
            surface_count=2,
            colorscale='RdBu',
            cmin=-isovalue,
            cmax=isovalue,
            opacity=0.7,
            colorbar=dict(title="Orbital Value")
        ))

        fig.update_layout(
            title=f'Molecular Orbital {orbital_index + 1} (Isovalue = ±{isovalue})',
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        return fig

    def create_contour_plot(self, orbital_index: int, plane: str = 'xy',
                          z_value: Optional[float] = None) -> plt.Figure:
        """
        Create a 2D contour plot of an orbital

        Args:
            orbital_index: Index of the orbital to visualize
            plane: Plane for contour plot ('xy', 'xz', 'yz')
            z_value: Fixed coordinate for the plane

        Returns:
            matplotlib Figure object
        """
        if not self.wavefunction:
            raise ValueError("No wavefunction set")

        # Generate grid
        self.generate_grid(n_points=50)
        orbital_data = self.evaluate_orbital(orbital_index)

        # Determine slice index
        atoms = self.wavefunction.atoms
        coords = np.array([[atom.x, atom.y, atom.z] for atom in atoms])

        if z_value is None:
            # Use center of mass as default
            z_value = coords[:, 2].mean() if plane == 'xy' else \
                     coords[:, 1].mean() if plane == 'xz' else \
                     coords[:, 0].mean()

        # Extract 2D slice
        if plane == 'xy':
            z_idx = np.argmin(np.abs(self.grid_data['coords'][2] - z_value))
            data_2d = orbital_data[:, :, z_idx]
            x_vals = self.grid_data['coords'][0]
            y_vals = self.grid_data['coords'][1]
            xlabel, ylabel = 'X (Å)', 'Y (Å)'
        elif plane == 'xz':
            y_idx = np.argmin(np.abs(self.grid_data['coords'][1] - z_value))
            data_2d = orbital_data[:, y_idx, :]
            x_vals = self.grid_data['coords'][0]
            y_vals = self.grid_data['coords'][2]
            xlabel, ylabel = 'X (Å)', 'Z (Å)'
        else:  # yz
            x_idx = np.argmin(np.abs(self.grid_data['coords'][0] - z_value))
            data_2d = orbital_data[x_idx, :, :]
            x_vals = self.grid_data['coords'][1]
            y_vals = self.grid_data['coords'][2]
            xlabel, ylabel = 'Y (Å)', 'Z (Å)'

        # Create contour plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Determine contour levels
        max_val = np.abs(data_2d).max()
        levels = np.linspace(-max_val, max_val, 20)

        contour = ax.contourf(x_vals, y_vals, data_2d, levels=levels, cmap='RdBu_r')
        ax.contour(x_vals, y_vals, data_2d, levels=levels, colors='black', linewidths=0.5, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Orbital Value')

        # Add atom positions if in plane
        tolerance = 0.5  # Angstroms
        if plane == 'xy':
            for atom in atoms:
                if abs(atom.z - z_value) < tolerance:
                    ax.plot(atom.x, atom.y, 'ko', markersize=8)
                    ax.text(atom.x, atom.y + 0.3, atom.element, ha='center', fontsize=10)
        elif plane == 'xz':
            for atom in atoms:
                if abs(atom.y - z_value) < tolerance:
                    ax.plot(atom.x, atom.z, 'ko', markersize=8)
                    ax.text(atom.x, atom.z + 0.3, atom.element, ha='center', fontsize=10)
        else:  # yz
            for atom in atoms:
                if abs(atom.x - z_value) < tolerance:
                    ax.plot(atom.y, atom.z, 'ko', markersize=8)
                    ax.text(atom.y, atom.z + 0.3, atom.element, ha='center', fontsize=10)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Molecular Orbital {orbital_index + 1} - {plane.upper()} Plane (z = {z_value:.2f} Å)')
        ax.set_aspect('equal')

        return fig

    def create_orbital_energy_diagram(self) -> plt.Figure:
        """
        Create an orbital energy diagram

        Returns:
            matplotlib Figure object
        """
        if not self.wavefunction:
            raise ValueError("No wavefunction set")

        fig, ax = plt.subplots(figsize=(8, 10))

        # Get orbital energies and occupations
        energies = self.wavefunction.mo_energies
        occupations = getattr(self.wavefunction, 'mo_occupations', None)

        n_orbitals = len(energies)
        n_electrons = 0
        if occupations is not None:
            n_electrons = int(occupations.sum())

        # Find HOMO and LUMO
        if occupations is not None:
            homo_idx = np.where(occupations > 0)[0].max()
            lumo_idx = homo_idx + 1
        else:
            # For closed-shell systems, assume half-filled
            n_electrons = len(energies) // 2
            homo_idx = n_electrons - 1
            lumo_idx = n_electrons if lumo_idx < n_orbitals else None

        # Plot orbital energies
        for i in range(n_orbitals):
            occupation = occupations[i] if occupations is not None else (2 if i < n_electrons else 0)

            if occupation > 0:
                color = 'blue'
                linewidth = 2
            else:
                color = 'red'
                linewidth = 1

            # Draw energy level
            ax.hlines(energies[i], i - 0.4, i + 0.4, colors=color, linewidth=linewidth)

            # Add occupation indicators
            if occupation > 0:
                n_electrons_orbital = int(occupation)
                for j in range(n_electrons_orbital):
                    x_offset = (j - n_electrons_orbital/2 + 0.25) * 0.15
                    ax.plot(i + x_offset, energies[i], 'ko', markersize=4)

            # Highlight HOMO and LUMO
            if i == homo_idx:
                ax.text(i + 0.5, energies[i], 'HOMO', fontsize=8, va='center')
            elif i == lumo_idx:
                ax.text(i + 0.5, energies[i], 'LUMO', fontsize=8, va='center')

        ax.set_xlabel('Orbital Index')
        ax.set_ylabel('Energy (Hartree)')
        ax.set_title('Molecular Orbital Energy Diagram')
        ax.grid(True, alpha=0.3)

        # Set reasonable limits
        energy_range = energies.max() - energies.min()
        ax.set_ylim(energies.min() - 0.1 * energy_range, energies.max() + 0.1 * energy_range)

        return fig

    def calculate_orbital_overlap(self, orb1_idx: int, orb2_idx: int) -> float:
        """
        Calculate overlap between two orbitals

        Args:
            orb1_idx: Index of first orbital
            orb2_idx: Index of second orbital

        Returns:
            Overlap integral value
        """
        if not self.wavefunction:
            raise ValueError("No wavefunction set")

        # Get orbital coefficients
        coeffs1 = self.wavefunction.mo_coeffs[:, orb1_idx]
        coeffs2 = self.wavefunction.mo_coeffs[:, orb2_idx]

        # This would need the actual overlap matrix
        # For now, provide a simplified calculation
        overlap = np.dot(coeffs1, coeffs2)  # Simplified: assumes orthonormal basis

        return overlap

    def get_orbital_character(self, orbital_index: int) -> Dict[str, float]:
        """
        Analyze atomic character contributions to an orbital

        Args:
            orbital_index: Index of the orbital to analyze

        Returns:
            Dictionary with atomic contributions
        """
        if not self.wavefunction:
            raise ValueError("No wavefunction set")

        orbital_coeffs = self.wavefunction.mo_coeffs[:, orbital_index]
        atoms = self.wavefunction.atoms

        # This is a simplified analysis
        # In reality, would need to map basis functions to atoms
        n_atoms = len(atoms)
        n_basis = len(orbital_coeffs)

        # Simple mapping: assume equal distribution of basis functions among atoms
        basis_per_atom = n_basis // n_atoms

        contributions = {}
        for i, atom in enumerate(atoms):
            start_idx = i * basis_per_atom
            end_idx = min((i + 1) * basis_per_atom, n_basis)
            atom_coeffs = orbital_coeffs[start_idx:end_idx]
            contribution = np.sum(atom_coeffs**2)
            contributions[atom.element] = contributions.get(atom.element, 0) + contribution

        # Normalize
        total = sum(contributions.values())
        if total > 0:
            contributions = {elem: val/total for elem, val in contributions.items()}

        return contributions