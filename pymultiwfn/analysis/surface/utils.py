"""
Utility functions for surface analysis.

This module provides helper functions and utilities for surface analysis,
including mathematical operations, data processing, and visualization helpers.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator

from .surface_analysis import SurfaceData


def calculate_surface_curvature(surface_data: SurfaceData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate principal curvatures and mean curvature at surface vertices.

    Args:
        surface_data: Surface data containing vertices and triangles

    Returns:
        Tuple containing:
        - mean_curvature: Mean curvature at each vertex
        - gaussian_curvature: Gaussian curvature at each vertex
        - principal_curvatures: Principal curvatures (k1, k2) at each vertex
    """
    if len(surface_data.triangles) == 0:
        return np.array([]), np.array([]), np.array([])

    vertices = surface_data.vertices
    triangles = surface_data.triangles

    # Build vertex-triangle adjacency
    vertex_triangles = _build_vertex_triangle_adjacency(vertices, triangles)

    # Initialize curvature arrays
    n_vertices = len(vertices)
    mean_curvature = np.zeros(n_vertices)
    gaussian_curvature = np.zeros(n_vertices)
    principal_curvatures = np.zeros((n_vertices, 2))

    for i in range(n_vertices):
        # Get neighboring triangles
        adjacent_triangles = vertex_triangles[i]
        if len(adjacent_triangles) < 3:
            continue

        # Calculate normal vectors for adjacent triangles
        triangle_normals = []
        for tri_idx in adjacent_triangles:
            v0, v1, v2 = vertices[triangles[tri_idx]]
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            triangle_normals.append(normal)

        triangle_normals = np.array(triangle_normals)

        # Calculate curvature using simplified approach
        # This is a simplified implementation - full curvature calculation would
        # require more sophisticated differential geometry
        if len(triangle_normals) > 2:
            # Estimate curvature from normal variation
            normal_variance = np.var(triangle_normals, axis=0)
            mean_curvature[i] = np.sqrt(np.sum(normal_variance))

            # Gaussian curvature (simplified approximation)
            area_sum = 0.0
            for tri_idx in adjacent_triangles:
                v0, v1, v2 = vertices[triangles[tri_idx]]
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                area_sum += area

            if area_sum > 0:
                gaussian_curvature[i] = (2 * np.pi) / area_sum

            # Principal curvatures (approximation)
            principal_curvatures[i, 0] = mean_curvature[i] + np.sqrt(max(0, mean_curvature[i]**2 - gaussian_curvature[i]))
            principal_curvatures[i, 1] = mean_curvature[i] - np.sqrt(max(0, mean_curvature[i]**2 - gaussian_curvature[i]))

    return mean_curvature, gaussian_curvature, principal_curvatures


def calculate_surface_descriptors(surface_data: SurfaceData) -> Dict[str, float]:
    """
    Calculate comprehensive surface descriptors.

    Args:
        surface_data: Surface data to analyze

    Returns:
        Dictionary containing surface descriptors
    """
    if len(surface_data.vertices) == 0:
        return {}

    vertices = surface_data.vertices
    triangles = surface_data.triangles
    values = surface_data.vertex_values

    descriptors = {
        'surface_area': surface_data.surface_area,
        'surface_volume': surface_data.surface_volume,
        'vertex_count': len(vertices),
        'triangle_count': len(triangles),
    }

    # Calculate basic statistical descriptors
    if len(values) > 0:
        descriptors.update({
            'value_mean': np.mean(values),
            'value_std': np.std(values),
            'value_min': np.min(values),
            'value_max': np.max(values),
            'value_range': np.max(values) - np.min(values),
        })

    # Calculate geometric descriptors
    if len(vertices) > 0:
        # Bounding box dimensions
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        bbox_dimensions = max_coords - min_coords
        bbox_volume = np.prod(bbox_dimensions)

        descriptors.update({
            'bbox_dimensions_x': bbox_dimensions[0],
            'bbox_dimensions_y': bbox_dimensions[1],
            'bbox_dimensions_z': bbox_dimensions[2],
            'bbox_volume': bbox_volume,
            'surface_to_volume_ratio': surface_data.surface_area / max(surface_data.surface_volume, 1e-10),
        })

        # Sphericity (measure of how sphere-like the surface is)
        if surface_data.surface_area > 0:
            equivalent_radius = np.sqrt(surface_data.surface_area / (4 * np.pi))
            sphericity = (np.pi**(1/3) * (6 * surface_data.surface_volume)**(2/3)) / surface_data.surface_area
            descriptors['sphericity'] = sphericity
            descriptors['equivalent_radius'] = equivalent_radius

    return descriptors


def smooth_surface(surface_data: SurfaceData, iterations: int = 1, alpha: float = 0.1) -> SurfaceData:
    """
    Apply Laplacian smoothing to surface vertices.

    Args:
        surface_data: Surface data to smooth
        iterations: Number of smoothing iterations
        alpha: Smoothing factor (0.0 = no smoothing, 1.0 = maximum smoothing)

    Returns:
        Smoothed surface data
    """
    if len(surface_data.triangles) == 0 or iterations <= 0:
        return surface_data

    vertices = surface_data.vertices.copy()
    triangles = surface_data.triangles

    # Build vertex adjacency
    vertex_neighbors = _build_vertex_adjacency(vertices, triangles)

    for _ in range(iterations):
        new_vertices = vertices.copy()

        for i in range(len(vertices)):
            if len(vertex_neighbors[i]) > 0:
                # Calculate Laplacian displacement
                neighbor_vertices = vertices[vertex_neighbors[i]]
                laplacian = np.mean(neighbor_vertices, axis=0) - vertices[i]
                new_vertices[i] = vertices[i] + alpha * laplacian

        vertices = new_vertices

    # Create new surface data with smoothed vertices
    return SurfaceData(
        vertices=vertices,
        triangles=triangles,
        vertex_values=surface_data.vertex_values.copy(),
        surface_area=_calculate_surface_area(vertices, triangles),
        surface_volume=_calculate_enclosed_volume(vertices, triangles),
        fragment_areas=surface_data.fragment_areas,
        fragment_stats=surface_data.fragment_stats
    )


def resample_surface(surface_data: SurfaceData, target_density: Optional[float] = None) -> SurfaceData:
    """
    Resample surface to achieve desired vertex density.

    Args:
        surface_data: Surface data to resample
        target_density: Target vertex density (vertices per unit area)

    Returns:
        Resampled surface data
    """
    if len(surface_data.triangles) == 0:
        return surface_data

    vertices = surface_data.vertices
    triangles = surface_data.triangles
    values = surface_data.vertex_values

    # Calculate current density
    current_density = len(vertices) / max(surface_data.surface_area, 1e-10)

    if target_density is None:
        # Default to moderate density increase
        target_density = current_density * 1.5

    # Subdivide triangles to increase density
    if target_density > current_density:
        # Determine subdivision level
        subdivision_ratio = target_density / current_density
        subdivision_level = max(1, int(np.log2(subdivision_ratio)))

        new_vertices = vertices.tolist()
        new_triangles = []
        new_values = values.tolist()

        for _ in range(subdivision_level):
            new_triangles = []
            for triangle in triangles:
                # Subdivide triangle into 4 smaller triangles
                v0, v1, v2 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]

                # Calculate edge midpoints
                mid01 = (v0 + v1) / 2
                mid12 = (v1 + v2) / 2
                mid20 = (v2 + v0) / 2

                # Add new vertices
                idx_mid01 = len(new_vertices)
                idx_mid12 = len(new_vertices) + 1
                idx_mid20 = len(new_vertices) + 2

                new_vertices.extend([mid01, mid12, mid20])

                # Interpolate values at midpoints
                val_mid01 = (values[triangle[0]] + values[triangle[1]]) / 2
                val_mid12 = (values[triangle[1]] + values[triangle[2]]) / 2
                val_mid20 = (values[triangle[2]] + values[triangle[0]]) / 2
                new_values.extend([val_mid01, val_mid12, val_mid20])

                # Create new triangles
                base_idx = triangle[0]
                new_triangles.extend([
                    [base_idx, idx_mid01, idx_mid20],
                    [idx_mid01, triangle[1], idx_mid12],
                    [idx_mid20, idx_mid12, triangle[2]],
                    [idx_mid01, idx_mid12, idx_mid20]
                ])

            # Update for next iteration
            vertices = np.array(new_vertices)
            triangles = np.array(new_triangles)
            values = np.array(new_values)

    return SurfaceData(
        vertices=vertices,
        triangles=triangles,
        vertex_values=values,
        surface_area=_calculate_surface_area(vertices, triangles),
        surface_volume=_calculate_enclosed_volume(vertices, triangles)
    )


def interpolate_to_grid(surface_data: SurfaceData,
                       grid_shape: Tuple[int, int, int],
                       grid_origin: np.ndarray,
                       grid_spacing: float) -> np.ndarray:
    """
    Interpolate surface values to a regular 3D grid.

    Args:
        surface_data: Surface data with vertex values
        grid_shape: Shape of the output grid (nx, ny, nz)
        grid_origin: Origin of the grid
        grid_spacing: Grid spacing

    Returns:
        3D array of interpolated values
    """
    if len(surface_data.vertices) == 0:
        return np.zeros(grid_shape)

    vertices = surface_data.vertices
    values = surface_data.vertex_values

    # Create grid coordinates
    nx, ny, nz = grid_shape
    x = grid_origin[0] + np.arange(nx) * grid_spacing
    y = grid_origin[1] + np.arange(ny) * grid_spacing
    z = grid_origin[2] + np.arange(nz) * grid_spacing

    # Use KDTree for efficient nearest neighbor interpolation
    tree = cKDTree(vertices)

    # Create grid mesh
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Find nearest vertices and interpolate values
    distances, indices = tree.query(grid_points, k=3)

    # Inverse distance weighting
    interpolated_values = np.zeros(len(grid_points))
    for i in range(len(grid_points)):
        # Avoid division by zero
        valid_distances = distances[i] > 1e-10
        if np.any(valid_distances):
            weights = 1.0 / distances[i][valid_distances]
            weights /= np.sum(weights)
            interpolated_values[i] = np.sum(weights * values[indices[i][valid_distances]])
        else:
            # If point coincides with vertex, use that value
            interpolated_values[i] = values[indices[i][0]]

    return interpolated_values.reshape(grid_shape)


def export_surface_to_obj(surface_data: SurfaceData, filename: str, include_values: bool = True):
    """
    Export surface data to OBJ file format.

    Args:
        surface_data: Surface data to export
        filename: Output filename
        include_values: Whether to include vertex values as vertex colors
    """
    with open(filename, 'w') as f:
        f.write("# Surface exported from PyMultiWFN\n")
        f.write(f"# Vertices: {len(surface_data.vertices)}\n")
        f.write(f"# Triangles: {len(surface_data.triangles)}\n")
        f.write(f"# Surface Area: {surface_data.surface_area:.6f}\n")
        f.write(f"# Surface Volume: {surface_data.surface_volume:.6f}\n\n")

        # Write vertices
        for i, vertex in enumerate(surface_data.vertices):
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}")

            # Include values as vertex colors if requested and available
            if include_values and i < len(surface_data.vertex_values):
                value = surface_data.vertex_values[i]
                # Normalize value to [0, 1] range for color
                if len(surface_data.vertex_values) > 1:
                    vmin, vmax = np.min(surface_data.vertex_values), np.max(surface_data.vertex_values)
                    if vmax > vmin:
                        normalized_value = (value - vmin) / (vmax - vmin)
                    else:
                        normalized_value = 0.5
                else:
                    normalized_value = 0.5
                f.write(f" {normalized_value:.6f} {normalized_value:.6f} {normalized_value:.6f}")

            f.write("\n")

        # Write triangles
        for triangle in surface_data.triangles:
            f.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n")


# Private helper functions

def _build_vertex_triangle_adjacency(vertices: np.ndarray, triangles: np.ndarray) -> List[List[int]]:
    """Build list of triangles adjacent to each vertex."""
    adjacency = [[] for _ in range(len(vertices))]
    for tri_idx, triangle in enumerate(triangles):
        for vertex_idx in triangle:
            adjacency[vertex_idx].append(tri_idx)
    return adjacency


def _build_vertex_adjacency(vertices: np.ndarray, triangles: np.ndarray) -> List[List[int]]:
    """Build list of neighboring vertices for each vertex."""
    adjacency = [[] for _ in range(len(vertices))]
    edge_set = set()

    for triangle in triangles:
        edges = [
            (min(triangle[0], triangle[1]), max(triangle[0], triangle[1])),
            (min(triangle[1], triangle[2]), max(triangle[1], triangle[2])),
            (min(triangle[2], triangle[0]), max(triangle[2], triangle[0]))
        ]

        for edge in edges:
            if edge not in edge_set:
                edge_set.add(edge)
                adjacency[edge[0]].append(edge[1])
                adjacency[edge[1]].append(edge[0])

    # Remove duplicates and sort
    for i in range(len(adjacency)):
        adjacency[i] = sorted(list(set(adjacency[i])))

    return adjacency


def _calculate_surface_area(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Calculate surface area from vertices and triangles."""
    if len(triangles) == 0:
        return 0.0

    v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    cross_products = np.cross(v1 - v0, v2 - v0)
    triangle_areas = 0.5 * np.linalg.norm(cross_products, axis=1)

    return np.sum(triangle_areas)


def _calculate_enclosed_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Calculate volume enclosed by surface using divergence theorem."""
    if len(triangles) == 0:
        return 0.0

    v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    cross_products = np.cross(v1, v2)
    volumes = np.sum(v0 * cross_products, axis=1) / 6.0

    return abs(np.sum(volumes))