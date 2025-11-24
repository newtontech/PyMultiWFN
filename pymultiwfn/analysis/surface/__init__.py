"""
Surface analysis module for PyMultiWFN.

This module provides comprehensive functionality for quantitative molecular surface analysis,
including:
- Surface generation using various definitions (electron density isosurfaces, Hirshfeld/Becke surfaces)
- Mapping of real space functions onto surfaces
- Statistical analysis of surface properties
- Fragment-based surface analysis
"""

from .surface_analysis import (
    SurfaceAnalyzer,
    SurfaceType,
    MappedFunction,
    SurfaceData,
)
from .marching_tetrahedra import (
    MarchingTetrahedra,
    MarchingTetrahedraConfig,
    extract_isosurface,
)
from .utils import (
    calculate_surface_curvature,
    calculate_surface_descriptors,
    smooth_surface,
    resample_surface,
    interpolate_to_grid,
    export_surface_to_obj,
)

__all__ = [
    'SurfaceAnalyzer',
    'SurfaceType',
    'MappedFunction',
    'SurfaceData',
    'MarchingTetrahedra',
    'MarchingTetrahedraConfig',
    'extract_isosurface',
    'calculate_surface_curvature',
    'calculate_surface_descriptors',
    'smooth_surface',
    'resample_surface',
    'interpolate_to_grid',
    'export_surface_to_obj',
]
