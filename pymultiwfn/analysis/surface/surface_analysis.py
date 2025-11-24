"""
Surface analysis module for PyMultiWFN.

This module provides functionality for quantitative molecular surface analysis,
including surface generation, property mapping, and statistical analysis.
Based on the improved Marching Tetrahedra algorithm as described in:
Tian Lu, Feiwu Chen, "Quantitative analysis of molecular surface based on
improved Marching Tetrahedra algorithm", J. Mol. Graph. Model., 38, 314-323 (2012).
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass

from pymultiwfn.core.data import Wavefunction
from pymultiwfn.math.density import calc_density


class SurfaceType(Enum):
    """Enumeration for different surface definition types."""
    ELECTRON_DENSITY = 1      # Isosurface of electron density
    REAL_SPACE_FUNCTION = 2   # Isosurface of a specific real space function
    HIRSHFELD = 5            # Hirshfeld surface (isosurface of Hirshfeld weight)
    BECKE = 6                # Becke surface (isosurface of Becke weight)
    EXTERNAL_GRID = 10       # Isosurface of external grid data
    MEMORY_GRID = 11         # Isosurface of grid data in memory


class MappedFunction(Enum):
    """Enumeration for functions that can be mapped on molecular surfaces."""
    USER_DEFINED = -1
    EXTERNAL_FILE = 0
    ESP = 1                  # Electrostatic potential
    ALIE = 2                 # Average local ionization energy
    ESP_ATOMIC = 3           # Electrostatic potential from atomic charges
    LEA = 4                  # Local electron affinity
    LEAE = -4                # Local electron attachment energy
    EDR = 5                  # Electron delocalization range function
    ORBITAL_OVERLAP = 6      # Orbital overlap distance function
    PAIR_DENSITY = 10        # Pair density
    ELECTRON_DENSITY = 11    # Electron density
    SIGN_LAMBDA2_RHO = 12    # Sign(lambda2)*rho
    DI = 20                  # Distance from nearest nucleus inside surface
    DE = 21                  # Distance from nearest nucleus outside surface
    DNORM = 22               # Normalized distance


@dataclass
class SurfaceData:
    """Data structure to store surface analysis results."""
    vertices: np.ndarray      # Surface vertex coordinates (N, 3)
    triangles: np.ndarray     # Surface triangle indices (M, 3)
    vertex_values: np.ndarray # Mapped function values at vertices (N,)
    surface_area: float       # Total surface area
    surface_volume: float     # Volume enclosed by surface
    fragment_areas: Optional[Dict[int, float]] = None  # Area per fragment
    fragment_stats: Optional[Dict[int, Dict[str, float]]] = None  # Statistics per fragment


class SurfaceAnalyzer:
    """
    Main class for performing quantitative molecular surface analysis.

    This class implements surface generation using various definitions (electron
    density isosurfaces, Hirshfeld/Becke surfaces, etc.) and mapping of various
    real space functions onto the generated surfaces.
    """

    def __init__(self, wavefunction: Wavefunction):
        """
        Initialize the SurfaceAnalyzer with wavefunction data.

        Args:
            wavefunction: Wavefunction object containing molecular information
        """
        self.wavefunction = wavefunction
        self.atoms = np.array([[atom.x, atom.y, atom.z] for atom in wavefunction.atoms])
        self.nuclear_charges = np.array([atom.charge for atom in wavefunction.atoms])
        self.atomic_numbers = np.array([atom.index for atom in wavefunction.atoms])

        # Default parameters
        self.grid_spacing = 0.25  # Bohr
        self.vdw_multiplier = 1.7
        self.merge_ratio = 0.5
        self.eliminate_redundant = True

    def generate_surface(self,
                        surface_type: SurfaceType,
                        isovalue: float = 0.001,
                        grid_spacing: Optional[float] = None,
                        fragment_atoms: Optional[List[int]] = None,
                        external_data: Optional[np.ndarray] = None,
                        grid_origin: Optional[np.ndarray] = None,
                        grid_spacing_3d: Optional[np.ndarray] = None) -> SurfaceData:
        """
        Generate molecular surface using specified definition.

        Args:
            surface_type: Type of surface to generate
            isovalue: Isovalue for surface definition
            grid_spacing: Grid spacing for surface generation (overwrites default)
            fragment_atoms: List of atomic indices for fragment surfaces (Hirshfeld/Becke)
            external_data: External grid data for surface definition
            grid_origin: Origin point for external data grid
            grid_spacing_3d: Spacing for external data grid

        Returns:
            SurfaceData object containing generated surface information
        """
        if grid_spacing is not None:
            self.grid_spacing = grid_spacing

        if surface_type == SurfaceType.ELECTRON_DENSITY:
            return self._generate_density_isosurface(isovalue)
        elif surface_type == SurfaceType.HIRSHFELD:
            if fragment_atoms is None:
                raise ValueError("fragment_atoms must be specified for Hirshfeld surface")
            return self._generate_hirshfeld_surface(isovalue, fragment_atoms)
        elif surface_type == SurfaceType.BECKE:
            if fragment_atoms is None:
                raise ValueError("fragment_atoms must be specified for Becke surface")
            return self._generate_becke_surface(isovalue, fragment_atoms)
        elif surface_type in [SurfaceType.EXTERNAL_GRID, SurfaceType.MEMORY_GRID]:
            if external_data is None:
                raise ValueError("external_data must be provided for external grid surfaces")
            return self._generate_external_isosurface(
                external_data, isovalue, grid_origin, grid_spacing_3d
            )
        else:
            raise ValueError(f"Surface type {surface_type} not yet implemented")

    def _generate_density_isosurface(self, isovalue: float) -> SurfaceData:
        """Generate electron density isosurface using marching tetrahedra."""
        # Create grid around molecule
        grid_origin, grid_dims, n_points = self._create_bounding_box()

        # Generate grid coordinates
        x = np.linspace(grid_origin[0], grid_origin[0] + grid_dims[0], n_points[0])
        y = np.linspace(grid_origin[1], grid_origin[1] + grid_dims[1], n_points[1])
        z = np.linspace(grid_origin[2], grid_origin[2] + grid_dims[2], n_points[2])

        # Calculate electron density on grid
        density_grid = self._calculate_density_grid(x, y, z)

        # Apply marching tetrahedra to extract surface
        vertices, triangles = self._marching_tetrahedra(density_grid, isovalue, grid_origin)

        # Calculate surface properties
        surface_area = self._calculate_surface_area(vertices, triangles)
        surface_volume = self._calculate_enclosed_volume(vertices, triangles)

        # Map electron density onto surface vertices
        vertex_values = self._interpolate_to_vertices(vertices, density_grid, grid_origin)

        return SurfaceData(
            vertices=vertices,
            triangles=triangles,
            vertex_values=vertex_values,
            surface_area=surface_area,
            surface_volume=surface_volume
        )

    def _generate_hirshfeld_surface(self, isovalue: float, fragment_atoms: List[int]) -> SurfaceData:
        """Generate Hirshfeld surface for specified fragment."""
        # Create grid around molecule
        grid_origin, grid_dims, n_points = self._create_bounding_box()

        # Generate grid coordinates
        x = np.linspace(grid_origin[0], grid_origin[0] + grid_dims[0], n_points[0])
        y = np.linspace(grid_origin[1], grid_origin[1] + grid_dims[1], n_points[1])
        z = np.linspace(grid_origin[2], grid_origin[2] + grid_dims[2], n_points[2])

        # Calculate Hirshfeld weights on grid
        hirshfeld_grid = self._calculate_hirshfeld_weights(x, y, z, fragment_atoms)

        # Apply marching tetrahedra to extract surface
        vertices, triangles = self._marching_tetrahedra(hirshfeld_grid, isovalue, grid_origin)

        # Calculate surface properties
        surface_area = self._calculate_surface_area(vertices, triangles)
        surface_volume = self._calculate_enclosed_volume(vertices, triangles)

        # Map Hirshfeld weights onto surface vertices
        vertex_values = self._interpolate_to_vertices(vertices, hirshfeld_grid, grid_origin)

        return SurfaceData(
            vertices=vertices,
            triangles=triangles,
            vertex_values=vertex_values,
            surface_area=surface_area,
            surface_volume=surface_volume
        )

    def _generate_becke_surface(self, isovalue: float, fragment_atoms: List[int]) -> SurfaceData:
        """Generate Becke surface for specified fragment."""
        # Create grid around molecule
        grid_origin, grid_dims, n_points = self._create_bounding_box()

        # Generate grid coordinates
        x = np.linspace(grid_origin[0], grid_origin[0] + grid_dims[0], n_points[0])
        y = np.linspace(grid_origin[1], grid_origin[1] + grid_dims[1], n_points[1])
        z = np.linspace(grid_origin[2], grid_origin[2] + grid_dims[2], n_points[2])

        # Calculate Becke weights on grid
        becke_grid = self._calculate_becke_weights(x, y, z, fragment_atoms)

        # Apply marching tetrahedra to extract surface
        vertices, triangles = self._marching_tetrahedra(becke_grid, isovalue, grid_origin)

        # Calculate surface properties
        surface_area = self._calculate_surface_area(vertices, triangles)
        surface_volume = self._calculate_enclosed_volume(vertices, triangles)

        # Map Becke weights onto surface vertices
        vertex_values = self._interpolate_to_vertices(vertices, becke_grid, grid_origin)

        return SurfaceData(
            vertices=vertices,
            triangles=triangles,
            vertex_values=vertex_values,
            surface_area=surface_area,
            surface_volume=surface_volume
        )

    def map_function_to_surface(self,
                               surface_data: SurfaceData,
                               mapped_function: MappedFunction,
                               **kwargs) -> SurfaceData:
        """
        Map a real space function onto the generated surface.

        Args:
            surface_data: Existing surface data to map function onto
            mapped_function: Function to map onto surface
            **kwargs: Additional arguments for specific functions

        Returns:
            Updated SurfaceData with mapped function values
        """
        if mapped_function == MappedFunction.ESP:
            vertex_values = self._calculate_esp(surface_data.vertices)
        elif mapped_function == MappedFunction.ALIE:
            vertex_values = self._calculate_alie(surface_data.vertices)
        elif mapped_function == MappedFunction.DI:
            vertex_values = self._calculate_di(surface_data.vertices)
        elif mapped_function == MappedFunction.DE:
            vertex_values = self._calculate_de(surface_data.vertices)
        elif mapped_function == MappedFunction.DNORM:
            vertex_values = self._calculate_dnorm(surface_data.vertices)
        else:
            raise ValueError(f"Mapped function {mapped_function} not yet implemented")

        # Update surface data with new mapped values
        surface_data.vertex_values = vertex_values
        return surface_data

    def analyze_surface_statistics(self, surface_data: SurfaceData,
                                  fragment_atoms: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Perform statistical analysis of surface properties.

        Args:
            surface_data: Surface data to analyze
            fragment_atoms: Optional fragment atoms for fragment-based analysis

        Returns:
            Dictionary containing statistical analysis results
        """
        stats = {
            'surface_area': surface_data.surface_area,
            'surface_volume': surface_data.surface_volume,
            'vertex_count': len(surface_data.vertices),
            'triangle_count': len(surface_data.triangles),
            'value_stats': {
                'mean': np.mean(surface_data.vertex_values),
                'std': np.std(surface_data.vertex_values),
                'min': np.min(surface_data.vertex_values),
                'max': np.max(surface_data.vertex_values),
                'positive_area': 0.0,
                'negative_area': 0.0
            }
        }

        # Calculate positive and negative surface areas
        if len(surface_data.triangles) > 0:
            positive_mask = surface_data.vertex_values > 0
            negative_mask = surface_data.vertex_values < 0

            pos_area = self._calculate_area_by_mask(
                surface_data.vertices, surface_data.triangles, positive_mask
            )
            neg_area = self._calculate_area_by_mask(
                surface_data.vertices, surface_data.triangles, negative_mask
            )

            stats['value_stats']['positive_area'] = pos_area
            stats['value_stats']['negative_area'] = neg_area

        # Fragment-based analysis if requested
        if fragment_atoms is not None:
            fragment_stats = self._analyze_fragments(surface_data, fragment_atoms)
            stats['fragment_stats'] = fragment_stats

        return stats

    # Private helper methods

    def _create_bounding_box(self) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
        """Create bounding box for grid generation."""
        min_coords = np.min(self.atoms, axis=0) - 5.0  # Add 5 Bohr margin
        max_coords = np.max(self.atoms, axis=0) + 5.0

        grid_origin = min_coords
        grid_dims = max_coords - min_coords
        n_points = tuple(np.ceil(grid_dims / self.grid_spacing).astype(int) + 1)

        return grid_origin, grid_dims, n_points

    def _calculate_density_grid(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Calculate electron density on a 3D grid."""
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        densities = calc_density(self.wavefunction, grid_coords)
        return densities.reshape(len(x), len(y), len(z))

    def _marching_tetrahedra(self, grid_data: np.ndarray, isovalue: float,
                           grid_origin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract isosurface using marching tetrahedra algorithm.

        This is a simplified implementation. The full algorithm would include
        tetrahedron decomposition, proper edge interpolation, and vertex elimination.
        """
        # Find boundary cells (where data crosses isovalue)
        boundary_mask = (grid_data[:-1, :-1, :-1] < isovalue) != (grid_data[1:, 1:, 1:] >= isovalue)

        # Generate vertices at boundary cell centers (simplified approach)
        boundary_indices = np.where(boundary_mask)
        vertices = np.column_stack([
            boundary_indices[0] * self.grid_spacing + grid_origin[0],
            boundary_indices[1] * self.grid_spacing + grid_origin[1],
            boundary_indices[2] * self.grid_spacing + grid_origin[2]
        ])

        # Generate simple triangulation (placeholder - proper marching tetrahedra needed)
        triangles = []
        n_cells = len(boundary_indices[0])

        # Create tetrahedra and triangulate (very simplified)
        for i in range(0, n_cells - 3, 4):
            if i + 3 < n_cells:
                triangles.extend([
                    [i, i+1, i+2],
                    [i, i+2, i+3]
                ])

        triangles = np.array(triangles) if triangles else np.array([]).reshape(0, 3)

        # Eliminate redundant vertices if requested
        if self.eliminate_redundant and len(vertices) > 0:
            vertices, triangles = self._eliminate_redundant_vertices(vertices, triangles)

        return vertices, triangles

    def _calculate_surface_area(self, vertices: np.ndarray, triangles: np.ndarray) -> float:
        """Calculate total surface area from vertices and triangles."""
        if len(triangles) == 0:
            return 0.0

        # Calculate triangle areas
        v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
        cross_products = np.cross(v1 - v0, v2 - v0)
        triangle_areas = 0.5 * np.linalg.norm(cross_products, axis=1)

        return np.sum(triangle_areas)

    def _calculate_enclosed_volume(self, vertices: np.ndarray, triangles: np.ndarray) -> float:
        """Calculate volume enclosed by surface using divergence theorem."""
        if len(triangles) == 0:
            return 0.0

        # Calculate signed volume contributions
        v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
        cross_products = np.cross(v1, v2)
        volumes = np.sum(v0 * cross_products, axis=1) / 6.0

        return abs(np.sum(volumes))

    def _interpolate_to_vertices(self, vertices: np.ndarray, grid_data: np.ndarray,
                                grid_origin: np.ndarray) -> np.ndarray:
        """Interpolate grid values to surface vertices."""
        # Simple trilinear interpolation (simplified approach)
        vertex_indices = ((vertices - grid_origin) / self.grid_spacing).astype(int)

        # Clamp indices to grid bounds
        max_indices = np.array(grid_data.shape) - 1
        vertex_indices = np.clip(vertex_indices, 0, max_indices)

        # Extract values at nearest grid points
        values = grid_data[
            vertex_indices[:, 0],
            vertex_indices[:, 1],
            vertex_indices[:, 2]
        ]

        return values

    def _eliminate_redundant_vertices(self, vertices: np.ndarray,
                                    triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Eliminate redundant surface vertices within threshold distance."""
        if len(vertices) == 0:
            return vertices, triangles

        threshold = self.grid_spacing * self.merge_ratio

        # Simple approach: remove vertices that are too close
        unique_vertices = []
        vertex_map = {}
        current_idx = 0

        for i, vertex in enumerate(vertices):
            is_duplicate = False
            for j, unique_vertex in enumerate(unique_vertices):
                if np.linalg.norm(vertex - unique_vertex) < threshold:
                    vertex_map[i] = j
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_vertices.append(vertex)
                vertex_map[i] = current_idx
                current_idx += 1

        # Update triangle indices
        new_triangles = []
        for triangle in triangles:
            new_triangle = [vertex_map[idx] for idx in triangle]
            if len(set(new_triangle)) == 3:  # Keep only valid triangles
                new_triangles.append(new_triangle)

        return np.array(unique_vertices), np.array(new_triangles)

    def _calculate_hirshfeld_weights(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                    fragment_atoms: List[int]) -> np.ndarray:
        """Calculate Hirshfeld weight function on grid."""
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        # Calculate promolecular densities for molecule and fragment
        total_density = self._calculate_promolecular_density(grid_coords)

        fragment_mask = np.zeros(len(self.atomic_numbers), dtype=bool)
        fragment_mask[fragment_atoms] = True
        fragment_density = self._calculate_promolecular_density(grid_coords, fragment_mask)

        # Hirshfeld weight = fragment_density / total_density
        weights = fragment_density / (total_density + 1e-10)
        return weights.reshape(len(x), len(y), len(z))

    def _calculate_becke_weights(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                fragment_atoms: List[int]) -> np.ndarray:
        """Calculate Becke weight function on grid."""
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        # Calculate Becke weights for each atom
        atomic_weights = self._calculate_atomic_becke_weights(grid_coords)

        # Sum weights for fragment atoms
        fragment_mask = np.zeros(len(self.atomic_numbers), dtype=bool)
        fragment_mask[fragment_atoms] = True
        fragment_weight = np.sum(atomic_weights[:, fragment_mask], axis=1)

        return fragment_weight.reshape(len(x), len(y), len(z))

    def _calculate_promolecular_density(self, grid_coords: np.ndarray,
                                      atom_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate promolecular electron density at grid points."""
        if atom_mask is None:
            relevant_atoms = self.atomic_numbers
            relevant_coords = self.atoms
        else:
            relevant_atoms = self.atomic_numbers[atom_mask]
            relevant_coords = self.atoms[atom_mask]

        density = np.zeros(len(grid_coords))

        # Simple Slater-type approximation for atomic densities
        for i, (Z, atom_coord) in enumerate(zip(relevant_atoms, relevant_coords)):
            distances = np.linalg.norm(grid_coords - atom_coord, axis=1)
            # Use a simple exponential decay model
            density += Z * np.exp(-2.0 * Z * distances)

        return density

    def _calculate_atomic_becke_weights(self, grid_coords: np.ndarray) -> np.ndarray:
        """Calculate Becke weights for all atoms at grid points."""
        n_atoms = len(self.atoms)
        n_points = len(grid_coords)
        weights = np.zeros((n_points, n_atoms))

        # Calculate distance matrix
        distances = np.linalg.norm(
            grid_coords[:, np.newaxis, :] - self.atoms[np.newaxis, :, :], axis=2
        )

        # Simple Becke weight calculation (simplified)
        for i in range(n_points):
            point_distances = distances[i]

            # Calculate cell functions
            cell_functions = np.zeros(n_atoms)
            for j in range(n_atoms):
                p_j = 1.0
                for k in range(n_atoms):
                    if j != k:
                        mu_ijk = (point_distances[j] - point_distances[k]) / np.linalg.norm(
                            self.atoms[j] - self.atoms[k]
                        )
                        f_ijk = 0.5 * (1 - mu_ijk + np.abs(mu_ijk))
                        p_j *= f_ijk
                cell_functions[j] = p_j

            # Normalize weights
            total_weight = np.sum(cell_functions)
            if total_weight > 0:
                weights[i] = cell_functions / total_weight

        return weights

    def _calculate_esp(self, vertices: np.ndarray) -> np.ndarray:
        """Calculate electrostatic potential at surface vertices."""
        esp = np.zeros(len(vertices))

        for i, vertex in enumerate(vertices):
            # Nuclear contribution
            for j, (atom_coord, Z) in enumerate(zip(self.atoms, self.nuclear_charges)):
                r = np.linalg.norm(vertex - atom_coord)
                if r > 1e-10:
                    esp[i] += Z / r

            # Electronic contribution (simplified - would need density integration)
            # This is a placeholder - actual implementation would require
            # integration of electron density over space
            esp[i] -= 0.0  # Placeholder electronic contribution

        return esp

    def _calculate_alie(self, vertices: np.ndarray) -> np.ndarray:
        """Calculate average local ionization energy at surface vertices."""
        # Placeholder implementation
        # ALIE requires orbital-specific densities and ionization potentials
        return np.ones(len(vertices)) * 0.5  # Placeholder value

    def _calculate_di(self, vertices: np.ndarray) -> np.ndarray:
        """Calculate distance from nearest nucleus inside surface."""
        distances = np.linalg.norm(
            vertices[:, np.newaxis, :] - self.atoms[np.newaxis, :, :], axis=2
        )
        return np.min(distances, axis=1) * 0.529177  # Convert Bohr to Angstrom

    def _calculate_de(self, vertices: np.ndarray) -> np.ndarray:
        """Calculate distance from nearest nucleus outside surface."""
        # Same as di for simplified implementation
        return self._calculate_di(vertices)

    def _calculate_dnorm(self, vertices: np.ndarray) -> np.ndarray:
        """Calculate normalized distance function."""
        di = self._calculate_di(vertices)
        de = self._calculate_de(vertices)

        # Simplified normalization
        vdw_radii = self._get_vdw_radii()
        min_distances = np.linalg.norm(
            vertices[:, np.newaxis, :] - self.atoms[np.newaxis, :, :], axis=2
        )
        nearest_indices = np.argmin(min_distances, axis=1)

        normalization = vdw_radii[nearest_indices] + 0.529177
        return (di + de) / normalization

    def _get_vdw_radii(self) -> np.ndarray:
        """Get van der Waals radii for atoms."""
        # Simplified van der Waals radii (in Angstrom)
        vdw_lookup = {
            1: 1.20,   # H
            6: 1.70,   # C
            7: 1.55,   # N
            8: 1.52,   # O
            9: 1.47,   # F
            15: 1.80,  # P
            16: 1.80,  # S
            17: 1.75,  # Cl
        }

        radii = []
        for Z in self.atomic_numbers:
            radii.append(vdw_lookup.get(Z, 1.70))  # Default to 1.70 Å

        return np.array(radii)

    def _calculate_area_by_mask(self, vertices: np.ndarray, triangles: np.ndarray,
                               mask: np.ndarray) -> float:
        """Calculate surface area for vertices satisfying a mask condition."""
        if len(triangles) == 0 or np.sum(mask) == 0:
            return 0.0

        # Filter triangles that have vertices satisfying the mask
        valid_triangles = []
        for triangle in triangles:
            if np.any(mask[triangle]):
                valid_triangles.append(triangle)

        if not valid_triangles:
            return 0.0

        valid_triangles = np.array(valid_triangles)
        v0, v1, v2 = vertices[valid_triangles[:, 0]], vertices[valid_triangles[:, 1]], vertices[valid_triangles[:, 2]]
        cross_products = np.cross(v1 - v0, v2 - v0)
        triangle_areas = 0.5 * np.linalg.norm(cross_products, axis=1)

        return np.sum(triangle_areas)

    def _analyze_fragments(self, surface_data: SurfaceData,
                          fragment_atoms: List[int]) -> Dict[int, Dict[str, float]]:
        """Analyze surface properties for different atomic fragments."""
        fragment_stats = {}

        # Assign each vertex to nearest atom
        distances = np.linalg.norm(
            surface_data.vertices[:, np.newaxis, :] - self.atoms[np.newaxis, :, :], axis=2
        )
        nearest_atoms = np.argmin(distances, axis=1)

        # Analyze each fragment
        for fragment_id in set(fragment_atoms):
            fragment_mask = np.isin(nearest_atoms, [fragment_id])

            if np.sum(fragment_mask) > 0:
                fragment_values = surface_data.vertex_values[fragment_mask]
                fragment_area = self._calculate_area_by_mask(
                    surface_data.vertices, surface_data.triangles, fragment_mask
                )

                fragment_stats[fragment_id] = {
                    'area': fragment_area,
                    'mean_value': np.mean(fragment_values),
                    'std_value': np.std(fragment_values),
                    'min_value': np.min(fragment_values),
                    'max_value': np.max(fragment_values),
                    'vertex_count': np.sum(fragment_mask)
                }

        return fragment_stats