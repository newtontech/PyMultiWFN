"""
Marching Tetrahedra algorithm implementation for surface extraction.

This module provides an optimized implementation of the Marching Tetrahedra algorithm
for extracting isosurfaces from 3D scalar fields. The implementation follows the
improved algorithm described in:
Tian Lu, Feiwu Chen, "Quantitative analysis of molecular surface based on
improved Marching Tetrahedra algorithm", J. Mol. Graph. Model., 38, 314-323 (2012).
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Tetrahedron vertex configurations for cubic cell decomposition
TETRAHEDRON_PATTERNS = [
    [0, 1, 3, 7],  # Tetrahedron 1
    [0, 1, 7, 5],  # Tetrahedron 2
    [0, 5, 7, 4],  # Tetrahedron 3
    [0, 3, 7, 6],  # Tetrahedron 4
    [0, 6, 7, 4],  # Tetrahedron 5
    [4, 5, 7, 6],  # Tetrahedron 6
]

# Edge table for tetrahedron triangulation (16 cases)
TETRAHEDRON_EDGE_TABLE = [
    [],  # Case 0: All vertices on one side
    [0, 1, 2],  # Case 1: Vertex 0 is inside
    [0, 1, 3],  # Case 2: Vertex 1 is inside
    [2, 1, 3],  # Case 3: Vertices 0,1 are inside
    [0, 2, 3],  # Case 4: Vertex 2 is inside
    [1, 0, 3],  # Case 5: Vertices 0,2 are inside
    [1, 2, 3],  # Case 6: Vertices 1,2 are inside
    [0, 2, 1],  # Case 7: Vertices 0,1,2 are inside
    [0, 3, 2],  # Case 8: Vertex 3 is inside
    [1, 0, 2],  # Case 9: Vertices 0,3 are inside
    [1, 3, 2],  # Case 10: Vertices 1,3 are inside
    [0, 3, 1],  # Case 11: Vertices 0,1,3 are inside
    [3, 0, 1],  # Case 12: Vertices 2,3 are inside
    [1, 0, 3],  # Case 13: Vertices 0,2,3 are inside
    [3, 2, 1],  # Case 14: Vertices 1,2,3 are inside
    [],  # Case 15: All vertices on other side
]

# Edge connectivity for tetrahedron
TETRAHEDRON_EDGES = [
    [0, 1],  # Edge 0
    [0, 2],  # Edge 1
    [0, 3],  # Edge 2
    [1, 2],  # Edge 3
    [1, 3],  # Edge 4
    [2, 3],  # Edge 5
]

@dataclass
class MarchingTetrahedraConfig:
    """Configuration for Marching Tetrahedra algorithm."""
    merge_threshold: float = 1e-6  # Threshold for merging duplicate vertices
    optimize_mesh: bool = True     # Enable mesh optimization
    calculate_normals: bool = False # Calculate surface normals


class MarchingTetrahedra:
    """
    Optimized implementation of the Marching Tetrahedra algorithm for surface extraction.

    This class provides efficient extraction of isosurfaces from 3D scalar fields using
    the tetrahedral decomposition approach. The implementation includes optimizations
    for memory usage and computational efficiency.
    """

    def __init__(self, config: Optional[MarchingTetrahedraConfig] = None):
        """
        Initialize the Marching Tetrahedra extractor.

        Args:
            config: Configuration object for algorithm parameters
        """
        self.config = config or MarchingTetrahedraConfig()
        self.vertices = []
        self.triangles = []
        self.normals = []

    def extract_surface(self,
                       scalar_field: np.ndarray,
                       isovalue: float,
                       grid_origin: np.ndarray,
                       grid_spacing: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Extract isosurface from scalar field using Marching Tetrahedra algorithm.

        Args:
            scalar_field: 3D array of scalar field values
            isovalue: Isovalue for surface extraction
            grid_origin: Origin coordinates of the 3D grid
            grid_spacing: Spacing between grid points

        Returns:
            Tuple containing:
            - vertices: Array of surface vertex coordinates (N, 3)
            - triangles: Array of triangle indices (M, 3)
            - normals: Array of vertex normals (N, 3) if calculate_normals=True
        """
        # Clear previous results
        self.vertices = []
        self.triangles = []
        self.normals = []

        # Convert to numpy arrays if needed
        scalar_field = np.asarray(scalar_field)
        grid_origin = np.asarray(grid_origin)

        # Get grid dimensions
        nx, ny, nz = scalar_field.shape

        # Process each cubic cell
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    # Process the six tetrahedra in this cubic cell
                    self._process_cubic_cell(
                        scalar_field, isovalue, grid_origin, grid_spacing, i, j, k
                    )

        # Convert to numpy arrays
        vertices = np.array(self.vertices) if self.vertices else np.array([]).reshape(0, 3)
        triangles = np.array(self.triangles) if self.triangles else np.array([]).reshape(0, 3)

        # Optimize mesh if requested
        if self.config.optimize_mesh and len(vertices) > 0:
            vertices, triangles = self._optimize_mesh(vertices, triangles)

        # Calculate normals if requested
        normals = None
        if self.config.calculate_normals and len(triangles) > 0:
            normals = self._calculate_normals(vertices, triangles)

        return vertices, triangles, normals

    def _process_cubic_cell(self, scalar_field: np.ndarray, isovalue: float,
                           grid_origin: np.ndarray, grid_spacing: float,
                           i: int, j: int, k: int):
        """Process a single cubic cell and its constituent tetrahedra."""
        # Get the 8 corner values of the cubic cell
        cell_values = np.array([
            scalar_field[i, j, k],       # Corner 0
            scalar_field[i+1, j, k],     # Corner 1
            scalar_field[i, j+1, k],     # Corner 2
            scalar_field[i+1, j+1, k],   # Corner 3
            scalar_field[i, j, k+1],     # Corner 4
            scalar_field[i+1, j, k+1],   # Corner 5
            scalar_field[i, j+1, k+1],   # Corner 6
            scalar_field[i+1, j+1, k+1], # Corner 7
        ])

        # Get the 8 corner coordinates of the cubic cell
        base_coord = grid_origin + grid_spacing * np.array([i, j, k])
        cell_coords = np.array([
            base_coord + grid_spacing * np.array([0, 0, 0]),  # Corner 0
            base_coord + grid_spacing * np.array([1, 0, 0]),  # Corner 1
            base_coord + grid_spacing * np.array([0, 1, 0]),  # Corner 2
            base_coord + grid_spacing * np.array([1, 1, 0]),  # Corner 3
            base_coord + grid_spacing * np.array([0, 0, 1]),  # Corner 4
            base_coord + grid_spacing * np.array([1, 0, 1]),  # Corner 5
            base_coord + grid_spacing * np.array([0, 1, 1]),  # Corner 6
            base_coord + grid_spacing * np.array([1, 1, 1]),  # Corner 7
        ])

        # Process each tetrahedron in the cubic cell
        for tetrahedron_pattern in TETRAHEDRON_PATTERNS:
            tetrahedron_values = cell_values[tetrahedron_pattern]
            tetrahedron_coords = cell_coords[tetrahedron_pattern]

            # Extract surface from this tetrahedron
            self._extract_from_tetrahedron(
                tetrahedron_values, tetrahedron_coords, isovalue
            )

    def _extract_from_tetrahedron(self, values: np.ndarray, coords: np.ndarray, isovalue: float):
        """Extract surface from a single tetrahedron."""
        # Determine which vertices are inside the isosurface
        inside_mask = values <= isovalue
        case_index = sum(bit << i for i, bit in enumerate(inside_mask))

        # Skip if no intersection
        if case_index == 0 or case_index == 15:
            return

        # Get triangulation for this case
        edge_indices = TETRAHEDRON_EDGE_TABLE[case_index]
        if len(edge_indices) < 3:
            return

        # Calculate intersection points on edges
        edge_vertices = []
        for edge_idx in edge_indices:
            edge = TETRAHEDRON_EDGES[edge_idx]
            v1, v2 = edge[0], edge[1]

            # Linear interpolation to find intersection point
            val1, val2 = values[v1], values[v2]
            if abs(val1 - val2) < 1e-10:
                # Edge is parallel to isosurface, use midpoint
                t = 0.5
            else:
                t = (isovalue - val1) / (val2 - val1)
                t = np.clip(t, 0.0, 1.0)

            # Calculate intersection vertex
            intersection = coords[v1] + t * (coords[v2] - coords[v1])
            edge_vertices.append(intersection)

        # Create triangles from edge vertices
        if len(edge_vertices) >= 3:
            for i in range(0, len(edge_vertices) - 2, 3):
                if i + 2 < len(edge_vertices):
                    triangle = [
                        self._add_vertex(edge_vertices[i]),
                        self._add_vertex(edge_vertices[i+1]),
                        self._add_vertex(edge_vertices[i+2])
                    ]
                    self.triangles.append(triangle)

    def _add_vertex(self, vertex: np.ndarray) -> int:
        """Add a vertex to the vertex list, checking for duplicates."""
        vertex = np.asarray(vertex)

        # Check for existing vertex within threshold
        for i, existing_vertex in enumerate(self.vertices):
            if np.linalg.norm(vertex - existing_vertex) < self.config.merge_threshold:
                return i

        # Add new vertex
        vertex_index = len(self.vertices)
        self.vertices.append(vertex.copy())
        return vertex_index

    def _optimize_mesh(self, vertices: np.ndarray, triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Optimize the mesh by removing unused vertices and degenerate triangles."""
        # Find used vertices
        used_vertices = np.zeros(len(vertices), dtype=bool)
        used_vertices[triangles.ravel()] = True

        # Create vertex mapping
        vertex_mapping = np.cumsum(used_vertices) - 1
        vertex_mapping[~used_vertices] = -1

        # Filter vertices and update triangle indices
        filtered_vertices = vertices[used_vertices]
        valid_triangles_mask = np.all(vertex_mapping[triangles] >= 0, axis=1)
        filtered_triangles = vertex_mapping[triangles[valid_triangles_mask]]

        return filtered_vertices, filtered_triangles

    def _calculate_normals(self, vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """Calculate vertex normals using triangle averaging."""
        normals = np.zeros_like(vertices)

        # Calculate triangle normals
        v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
        triangle_normals = np.cross(v1 - v0, v2 - v0)
        triangle_normals = triangle_normals / (np.linalg.norm(triangle_normals, axis=1, keepdims=True) + 1e-10)

        # Accumulate normals at vertices
        for i, triangle in enumerate(triangles):
            for vertex_idx in triangle:
                normals[vertex_idx] += triangle_normals[i]

        # Normalize vertex normals
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)

        return normals


def extract_isosurface(scalar_field: np.ndarray,
                      isovalue: float,
                      grid_origin: np.ndarray,
                      grid_spacing: float = 1.0,
                      config: Optional[MarchingTetrahedraConfig] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Convenience function to extract isosurface from scalar field.

    Args:
        scalar_field: 3D array of scalar field values
        isovalue: Isovalue for surface extraction
        grid_origin: Origin coordinates of the 3D grid
        grid_spacing: Spacing between grid points
        config: Configuration object for algorithm parameters

    Returns:
        Tuple containing vertices, triangles, and optional normals
    """
    extractor = MarchingTetrahedra(config)
    return extractor.extract_surface(scalar_field, isovalue, grid_origin, grid_spacing)