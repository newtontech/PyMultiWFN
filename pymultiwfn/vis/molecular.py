"""
Molecular Visualization Module for PyMultiWFN

This module provides comprehensive functionality for visualizing molecular structures,
including bonds, atoms, and molecular properties. Based on the molecular visualization
features from Multiwfn's GUI and display modules.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

from ..core.data import Wavefunction, Atom
from ..config import Config


class VisualizationStyle(Enum):
    """Styles for molecular visualization"""
    CPK = "cpk"           # Corey-Pauling-Koltun space-filling
    VDW = "vdw"           # van der Waals radii
    LINE = "line"         # Line/ball-and-stick model
    TUBE = "tube"         # Tubular bonds


class ColorScheme(Enum):
    """Color schemes for atoms"""
    DEFAULT = "default"
    CPK = "cpk"
    JMOL = "jmol"
    VMD = "vmd"


@dataclass
class VisualizationSettings:
    """Settings for molecular visualization"""
    style: VisualizationStyle = VisualizationStyle.LINE
    color_scheme: ColorScheme = ColorScheme.CPK
    scale_factor: float = 1.0
    show_hydrogens: bool = True
    show_labels: bool = False
    show_bonds: bool = True
    bond_threshold: float = 1.15  # Factor for bond detection
    atom_scale: Dict[str, float] = None  # Custom scaling per element
    colors: Dict[str, Tuple[float, float, float]] = None  # Custom colors per element


class MolecularVisualizer:
    """
    Class for visualizing molecular structures and properties.

    This class provides various methods for displaying molecular structures,
    bonds, and related properties using different visualization backends.
    Based on Multiwfn's molecular visualization capabilities.
    """

    def __init__(self, wavefunction: Wavefunction = None, config: Optional[Config] = None):
        """
        Initialize the molecular visualizer

        Parameters
        ----------
        wavefunction : Wavefunction, optional
            Wavefunction object containing molecular structure
        config : Config, optional
            Configuration object for visualization settings
        """
        self.wfn = wavefunction
        self.config = config or Config()

        # Default visualization settings
        self.settings = VisualizationSettings()

        # Standard atomic radii (in Angstroms)
        self.atomic_radii = {
            'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.85, 'C': 0.76,
            'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41,
            'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
            'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39,
            'Mn': 1.39, 'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22,
            'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.16, 'Br': 1.14, 'Kr': 1.17,
        }

        # Van der Waals radii (in Angstroms)
        self.vdw_radii = {
            'H': 1.20, 'He': 1.40, 'Li': 1.82, 'Be': 1.53, 'B': 1.92, 'C': 1.70,
            'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54, 'Na': 2.27, 'Mg': 1.73,
            'Al': 2.00, 'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
            'K': 2.75, 'Ca': 2.31, 'Sc': 2.30, 'Ti': 2.15, 'V': 2.05, 'Cr': 2.05,
            'Mn': 2.05, 'Fe': 2.05, 'Co': 2.00, 'Ni': 2.00, 'Cu': 2.00, 'Zn': 2.10,
            'Ga': 2.10, 'Ge': 2.10, 'As': 2.05, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02,
        }

        # CPK colors (RGB normalized to 0-1)
        self.cpk_colors = {
            'H': (1.0, 1.0, 1.0),      # White
            'He': (0.85, 0.85, 1.0),    # Light blue
            'Li': (0.8, 0.5, 1.0),      # Purple
            'Be': (0.76, 1.0, 0.0),     # Lime
            'B': (1.0, 0.71, 0.71),     # Pink
            'C': (0.25, 0.25, 0.25),    # Dark gray
            'N': (0.0, 0.0, 1.0),       # Blue
            'O': (1.0, 0.0, 0.0),       # Red
            'F': (0.56, 0.88, 0.31),    # Light green
            'Ne': (0.7, 0.89, 0.96),    # Light cyan
            'Na': (0.67, 0.36, 0.95),   # Indigo
            'Mg': (0.54, 1.0, 0.0),     # Bright green
            'Al': (0.75, 0.65, 0.65),   # Gray
            'Si': (0.94, 0.78, 0.63),   # Brown
            'P': (1.0, 0.5, 0.0),       # Orange
            'S': (1.0, 1.0, 0.0),       # Yellow
            'Cl': (0.12, 0.94, 0.12),   # Green
            'Ar': (0.5, 0.82, 0.89),    # Cyan
        }

        # Hex colors for matplotlib compatibility
        self.atom_colors = {
            'H': '#FFFFFF',  # White
            'C': '#404040',  # Dark gray
            'N': '#3050F8',  # Blue
            'O': '#FF0D0D',  # Red
            'F': '#90E050',  # Light green
            'P': '#FF8000',  # Orange
            'S': '#FFFF30',  # Yellow
            'Cl': '#1FF01F', # Green
            'Br': '#A62929', # Dark red
            'I': '#940094',  # Purple
        }

        # Legacy compatibility
        self.bond_threshold = 1.15

    def set_structure(self, wavefunction: Wavefunction):
        """Set the molecular structure to visualize (legacy compatibility)"""
        self.wfn = wavefunction
        self.current_structure = wavefunction

    def calculate_bonds(self) -> List[Tuple[int, int]]:
        """
        Calculate bonds based on interatomic distances

        Returns:
            List of tuples containing bonded atom indices
        """
        if not self.wfn and not self.current_structure:
            return []

        atoms = self.wfn.atoms if self.wfn else self.current_structure.atoms
        n_atoms = len(atoms)

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Calculate distance
                r1 = np.array([atoms[i].x, atoms[i].y, atoms[i].z])
                r2 = np.array([atoms[j].x, atoms[j].y, atoms[j].z])
                distance = np.linalg.norm(r1 - r2)

                # Get covalent radii sum for bond criterion
                elem1, elem2 = atoms[i].element, atoms[j].element
                r_cov1 = self.vdw_radii.get(elem1, 1.5)
                r_cov2 = self.vdw_radii.get(elem2, 1.5)

                if distance < self.bond_threshold * (r_cov1 + r_cov2):
                    bonds.append((i, j))

        return bonds

    def create_matplotlib_3d(self, show_bonds: bool = True,
                           show_labels: bool = True,
                           atom_style: str = 'CPK') -> plt.Figure:
        """
        Create a 3D matplotlib visualization of the molecule

        Args:
            show_bonds: Whether to show bonds between atoms
            show_labels: Whether to show atom labels
            atom_style: Style for atom representation ('CPK', 'VDW', 'Line')

        Returns:
            matplotlib Figure object
        """
        if not self.current_structure:
            raise ValueError("No molecular structure set")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        atoms = self.current_structure.atoms
        bonds = self.calculate_bonds() if show_bonds else []

        # Extract coordinates
        coords = np.array([[atom.x, atom.y, atom.z] for atom in atoms])
        elements = [atom.element for atom in atoms]

        # Plot bonds
        if show_bonds:
            for i, j in bonds:
                x_coords = [coords[i, 0], coords[j, 0]]
                y_coords = [coords[i, 1], coords[j, 1]]
                z_coords = [coords[i, 2], coords[j, 2]]
                ax.plot(x_coords, y_coords, z_coords, 'gray', alpha=0.6, linewidth=2)

        # Plot atoms
        for i, (elem, coord) in enumerate(zip(elements, coords)):
            color = self.atom_colors.get(elem, '#808080')

            if atom_style == 'CPK':
                size = 100
                ax.scatter(coord[0], coord[1], coord[2], c=color, s=size, alpha=0.8, edgecolors='black')
            elif atom_style == 'VDW':
                radius = self.vdw_radii.get(elem, 1.5) * 50
                ax.scatter(coord[0], coord[1], coord[2], c=color, s=radius, alpha=0.7)
            elif atom_style == 'Line':
                ax.scatter(coord[0], coord[1], coord[2], c=color, s=20, marker='o')

        # Add labels
        if show_labels:
            for i, (elem, coord) in enumerate(zip(elements, coords)):
                ax.text(coord[0], coord[1], coord[2], f'{elem}{i+1}', fontsize=8)

        # Set labels and title
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('Molecular Structure')

        # Make axes equal
        max_range = np.array([coords.max() - coords.min()]).max() / 2.0
        mid_x = (coords[:, 0].max() + coords[:, 0].min()) * 0.5
        mid_y = (coords[:, 1].max() + coords[:, 1].min()) * 0.5
        mid_z = (coords[:, 2].max() + coords[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        return fig

    def create_plotly_3d(self, show_bonds: bool = True,
                        show_labels: bool = True,
                        atom_style: str = 'CPK') -> go.Figure:
        """
        Create an interactive 3D plotly visualization of the molecule

        Args:
            show_bonds: Whether to show bonds between atoms
            show_labels: Whether to show atom labels
            atom_style: Style for atom representation ('CPK', 'VDW', 'Line')

        Returns:
            plotly Figure object
        """
        if not self.current_structure:
            raise ValueError("No molecular structure set")

        atoms = self.current_structure.atoms
        bonds = self.calculate_bonds() if show_bonds else []

        # Extract coordinates and properties
        coords = np.array([[atom.x, atom.y, atom.z] for atom in atoms])
        elements = [atom.element for atom in atoms]
        colors = [self.atom_colors.get(elem, '#808080') for elem in elements]

        # Determine atom sizes
        if atom_style == 'CPK':
            sizes = [15] * len(atoms)
        elif atom_style == 'VDW':
            sizes = [self.vdw_radii.get(elem, 1.5) * 10 for elem in elements]
        else:  # Line
            sizes = [5] * len(atoms)

        fig = go.Figure()

        # Add atoms
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=1, color='black')
            ),
            text=[f'{elem}{i+1}' for i, elem in enumerate(elements)],
            name='Atoms'
        ))

        # Add bonds
        if show_bonds:
            for i, j in bonds:
                bond_coords = coords[[i, j]]
                fig.add_trace(go.Scatter3d(
                    x=bond_coords[:, 0],
                    y=bond_coords[:, 1],
                    z=bond_coords[:, 2],
                    mode='lines',
                    line=dict(color='gray', width=3),
                    showlegend=False
                ))

        # Add labels if requested
        if show_labels:
            fig.add_trace(go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode='text',
                text=[f'{elem}{i+1}' for i, elem in enumerate(elements)],
                textposition='top center',
                textfont=dict(size=12),
                showlegend=False
            ))

        fig.update_layout(
            title='Molecular Structure',
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        return fig

    def create_distance_matrix_plot(self) -> plt.Figure:
        """
        Create a heatmap of interatomic distances

        Returns:
            matplotlib Figure object
        """
        if not self.current_structure:
            raise ValueError("No molecular structure set")

        atoms = self.current_structure.atoms
        n_atoms = len(atoms)
        coords = np.array([[atom.x, atom.y, atom.z] for atom in atoms])

        # Calculate distance matrix
        distance_matrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(n_atoms):
                distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

        # Create labels
        labels = [f'{atom.element}{i+1}' for i, atom in enumerate(atoms)]

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(distance_matrix, cmap='viridis')

        # Set ticks and labels
        ax.set_xticks(np.arange(n_atoms))
        ax.set_yticks(np.arange(n_atoms))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        for i in range(n_atoms):
            for j in range(n_atoms):
                text = ax.text(j, i, f'{distance_matrix[i, j]:.2f}',
                             ha="center", va="center", color="w")

        ax.set_title("Interatomic Distance Matrix (Å)")
        fig.tight_layout()

        return fig

    def save_structure(self, filename: str, format: str = 'xyz'):
        """
        Save molecular structure to file

        Args:
            filename: Output filename
            format: File format ('xyz', 'pdb', 'mol')
        """
        if not self.current_structure:
            raise ValueError("No molecular structure set")

        atoms = self.current_structure.atoms

        if format.lower() == 'xyz':
            with open(filename, 'w') as f:
                f.write(f"{len(atoms)}\n")
                f.write("PyMultiWFN generated structure\n")
                for atom in atoms:
                    f.write(f"{atom.element:2s} {atom.x:12.6f} {atom.y:12.6f} {atom.z:12.6f}\n")
        else:
            raise ValueError(f"Format {format} not yet implemented")

    def get_molecular_properties(self) -> Dict:
        """
        Calculate basic molecular properties

        Returns:
            Dictionary containing molecular properties
        """
        if not self.current_structure:
            raise ValueError("No molecular structure set")

        atoms = self.current_structure.atoms
        coords = np.array([[atom.x, atom.y, atom.z] for atom in atoms])

        # Calculate center of mass
        masses = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
                 'F': 18.998, 'P': 30.974, 'S': 32.06, 'Cl': 35.45}
        atom_masses = np.array([masses.get(atom.element, 12.011) for atom in atoms])
        total_mass = atom_masses.sum()
        com = np.sum(coords * atom_masses[:, np.newaxis], axis=0) / total_mass

        # Calculate moments of inertia
        centered_coords = coords - com
        inertia = np.zeros((3, 3))
        for i, coord in enumerate(centered_coords):
            mass = atom_masses[i]
            r2 = np.sum(coord**2)
            inertia += mass * (r2 * np.eye(3) - np.outer(coord, coord))

        # Principal moments of inertia
        principal_moments = np.linalg.eigvalsh(inertia)

        # Calculate radius of gyration
        rg = np.sqrt(np.sum(atom_masses[:, np.newaxis] * centered_coords**2) / total_mass)

        return {
            'n_atoms': len(atoms),
            'molecular_weight': total_mass,
            'center_of_mass': com,
            'moments_of_inertia': principal_moments,
            'radius_of_gyration': rg,
            'n_bonds': len(self.calculate_bonds())
        }