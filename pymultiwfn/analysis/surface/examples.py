"""
Examples demonstrating the use of the surface analysis module.

This module provides comprehensive examples of how to use the surface analysis
functionality for different types of molecular surface analysis.
"""

import numpy as np
from pymultiwfn.core.data import Wavefunction, Atom
from pymultiwfn.analysis.surface import (
    SurfaceAnalyzer,
    SurfaceType,
    MappedFunction,
    extract_isosurface,
    calculate_surface_descriptors,
    export_surface_to_obj
)


def example_water_molecule():
    """
    Example: Surface analysis of a water molecule.
    Demonstrates basic surface generation and analysis.
    """
    print("=== Water Molecule Surface Analysis ===")

    # Create a simple water molecule wavefunction (placeholder)
    atoms = [
        Atom(element='O', index=8, x=0.0, y=0.0, z=0.0, charge=8.0),
        Atom(element='H', index=1, x=0.957, y=0.0, z=0.0, charge=1.0),
        Atom(element='H', index=1, x=-0.239, y=0.927, z=0.0, charge=1.0)
    ]

    # Create wavefunction (this would normally be loaded from file)
    wfn = Wavefunction(atoms=atoms, num_electrons=10)

    # Initialize surface analyzer
    analyzer = SurfaceAnalyzer(wfn)

    # Generate electron density isosurface (vdW surface)
    surface_data = analyzer.generate_surface(
        SurfaceType.ELECTRON_DENSITY,
        isovalue=0.001,  # Common vdW surface definition
        grid_spacing=0.2
    )

    print(f"Surface generated successfully!")
    print(f"Number of vertices: {len(surface_data.vertices)}")
    print(f"Number of triangles: {len(surface_data.triangles)}")
    print(f"Surface area: {surface_data.surface_area:.3f} Bohr²")
    print(f"Surface volume: {surface_data.surface_volume:.3f} Bohr³")

    # Calculate surface descriptors
    descriptors = calculate_surface_descriptors(surface_data)
    print(f"\nSurface Descriptors:")
    for key, value in descriptors.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    # Map ESP onto surface
    surface_data = analyzer.map_function_to_surface(
        surface_data, MappedFunction.ESP
    )

    print(f"\nESP mapped onto surface:")
    print(f"  Mean ESP: {np.mean(surface_data.vertex_values):.6f}")
    print(f"  ESP range: [{np.min(surface_data.vertex_values):.6f}, {np.max(surface_data.vertex_values):.6f}]")

    # Export surface to OBJ file
    export_surface_to_obj(surface_data, 'water_surface.obj', include_values=True)
    print(f"Surface exported to 'water_surface.obj'")

    return surface_data


def example_hirshfeld_surface():
    """
    Example: Hirshfeld surface analysis of a molecule.
    Demonstrates fragment-based surface analysis.
    """
    print("\n=== Hirshfeld Surface Analysis ===")

    # Create a simple organic molecule (ethanol)
    atoms = [
        Atom(element='C', index=6, x=0.0, y=0.0, z=0.0, charge=6.0),
        Atom(element='C', index=6, x=1.54, y=0.0, z=0.0, charge=6.0),
        Atom(element='O', index=8, x=2.50, y=0.0, z=0.0, charge=8.0),
        Atom(element='H', index=1, x=-0.54, y=0.89, z=0.0, charge=1.0),
        Atom(element='H', index=1, x=-0.54, y=-0.89, z=0.0, charge=1.0),
        Atom(element='H', index=1, x=1.54, y=0.94, z=0.0, charge=1.0),
        Atom(element='H', index=1, x=1.54, y=-0.94, z=0.0, charge=1.0),
        Atom(element='H', index=1, x=3.07, y=0.89, z=0.0, charge=1.0),
    ]

    wfn = Wavefunction(atoms=atoms, num_electrons=26)
    analyzer = SurfaceAnalyzer(wfn)

    # Define fragment (e.g., the OH group)
    oh_fragment_atoms = [2, 7]  # O and H atom indices

    # Generate Hirshfeld surface for OH fragment
    hirshfeld_surface = analyzer.generate_surface(
        SurfaceType.HIRSHFELD,
        isovalue=0.5,  # Standard Hirshfeld surface isovalue
        fragment_atoms=oh_fragment_atoms,
        grid_spacing=0.15
    )

    print(f"Hirshfeld surface generated for OH fragment!")
    print(f"Surface area: {hirshfeld_surface.surface_area:.3f} Bohr²")
    print(f"Surface volume: {hirshfeld_surface.surface_volume:.3f} Bohr³")

    # Analyze fragment statistics
    stats = analyzer.analyze_surface_statistics(
        hirshfeld_surface, fragment_atoms=oh_fragment_atoms
    )

    print(f"\nFragment Statistics:")
    if 'fragment_stats' in stats:
        for atom_idx, atom_stats in stats['fragment_stats'].items():
            element = atoms[atom_idx].element
            print(f"  Atom {element}{atom_idx}:")
            print(f"    Area: {atom_stats['area']:.3f} Bohr²")
            print(f"    Mean value: {atom_stats['mean_value']:.6f}")
            print(f"    Vertex count: {atom_stats['vertex_count']}")

    return hirshfeld_surface


def example_surface_property_mapping():
    """
    Example: Mapping different properties onto molecular surfaces.
    Demonstrates the variety of properties that can be analyzed.
    """
    print("\n=== Surface Property Mapping ===")

    # Create a slightly larger molecule
    atoms = [
        Atom(element='C', index=6, x=0.0, y=0.0, z=0.0, charge=6.0),
        Atom(element='C', index=6, x=1.40, y=0.0, z=0.0, charge=6.0),
        Atom(element='C', index=6, x=2.10, y=1.21, z=0.0, charge=6.0),
        Atom(element='C', index=6, x=1.40, y=2.42, z=0.0, charge=6.0),
        Atom(element='C', index=6, x=0.0, y=2.42, z=0.0, charge=6.0),
        Atom(element='C', index=6, x=-0.70, y=1.21, z=0.0, charge=6.0),
    ]

    wfn = Wavefunction(atoms=atoms, num_electrons=30)
    analyzer = SurfaceAnalyzer(wfn)

    # Generate base surface
    base_surface = analyzer.generate_surface(
        SurfaceType.ELECTRON_DENSITY,
        isovalue=0.002,
        grid_spacing=0.2
    )

    print(f"Base surface generated: {len(base_surface.vertices)} vertices")

    # Map different properties
    properties_to_map = [
        (MappedFunction.ESP, "Electrostatic Potential"),
        (MappedFunction.DI, "Distance Inside"),
        (MappedFunction.DE, "Distance Outside"),
        (MappedFunction.DNORM, "Normalized Distance"),
    ]

    for mapped_func, name in properties_to_map:
        try:
            surface_with_property = analyzer.map_function_to_surface(
                base_surface.copy(), mapped_func
            )

            values = surface_with_property.vertex_values
            print(f"\n{name}:")
            print(f"  Mean: {np.mean(values):.6f}")
            print(f"  Std:  {np.std(values):.6f}")
            print(f"  Range: [{np.min(values):.6f}, {np.max(values):.6f}]")

        except ValueError as e:
            print(f"\n{name}: Not yet implemented ({e})")

    return base_surface


def example_surface_comparison():
    """
    Example: Comparing different surface definitions for the same molecule.
    """
    print("\n=== Surface Definition Comparison ===")

    # Use a simple molecule
    atoms = [
        Atom(element='C', index=6, x=0.0, y=0.0, z=0.0, charge=6.0),
        Atom(element='O', index=8, x=1.20, y=0.0, z=0.0, charge=8.0),
    ]

    wfn = Wavefunction(atoms=atoms, num_electrons=14)
    analyzer = SurfaceAnalyzer(wfn)

    # Define fragment for Becke/Hirshfeld analysis
    carbon_fragment = [0]

    # Compare different surface definitions
    surface_definitions = [
        (SurfaceType.ELECTRON_DENSITY, 0.001, "Density isosurface (vdW)"),
        (SurfaceType.ELECTRON_DENSITY, 0.002, "Density isosurface (higher)"),
        (SurfaceType.HIRSHFELD, 0.5, "Hirshfeld surface (C atom)"),
        (SurfaceType.BECKE, 0.5, "Becke surface (C atom)"),
    ]

    results = []

    for surf_type, isovalue, description in surface_definitions:
        try:
            if surf_type in [SurfaceType.HIRSHFELD, SurfaceType.BECKE]:
                surface = analyzer.generate_surface(
                    surf_type, isovalue, fragment_atoms=carbon_fragment
                )
            else:
                surface = analyzer.generate_surface(surf_type, isovalue)

            results.append({
                'type': description,
                'area': surface.surface_area,
                'volume': surface.surface_volume,
                'vertices': len(surface.vertices),
                'triangles': len(surface.triangles)
            })

        except Exception as e:
            print(f"Error generating {description}: {e}")

    # Display comparison
    print(f"{'Surface Type':<30} {'Area (Bohr²)':<15} {'Volume (Bohr³)':<15} {'Vertices':<10} {'Triangles':<10}")
    print("-" * 85)

    for result in results:
        print(f"{result['type']:<30} {result['area']:<15.3f} {result['volume']:<15.3f} "
              f"{result['vertices']:<10} {result['triangles']:<10}")

    return results


def example_direct_isosurface_extraction():
    """
    Example: Direct isosurface extraction from scalar field.
    Demonstrates the low-level marching tetrahedra functionality.
    """
    print("\n=== Direct Isosurface Extraction ===")

    # Create a simple 3D scalar field (e.g., a sphere)
    grid_shape = (20, 20, 20)
    grid_origin = np.array([-5.0, -5.0, -5.0])
    grid_spacing = 0.5

    # Generate spherical scalar field
    x = grid_origin[0] + np.arange(grid_shape[0]) * grid_spacing
    y = grid_origin[1] + np.arange(grid_shape[1]) * grid_spacing
    z = grid_origin[2] + np.arange(grid_shape[2]) * grid_spacing

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    scalar_field = np.sqrt(X**2 + Y**2 + Z**2)  # Distance from origin

    # Extract isosurface at radius = 2.0
    vertices, triangles, normals = extract_isosurface(
        scalar_field,
        isovalue=2.0,
        grid_origin=grid_origin,
        grid_spacing=grid_spacing
    )

    print(f"Isosurface extracted from scalar field:")
    print(f"  Grid shape: {grid_shape}")
    print(f"  Isovalue: 2.0 (sphere)")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Triangles: {len(triangles)}")

    if len(vertices) > 0:
        # Calculate expected surface area and volume for sphere with radius 2.0
        expected_area = 4 * np.pi * 2.0**2
        expected_volume = (4/3) * np.pi * 2.0**3

        # Calculate actual surface area
        from pymultiwfn.analysis.surface.utils import _calculate_surface_area
        actual_area = _calculate_surface_area(vertices, triangles)

        print(f"\nComparison with analytical sphere (r=2.0):")
        print(f"  Expected area: {expected_area:.6f}")
        print(f"  Actual area:   {actual_area:.6f}")
        print(f"  Error: {abs(actual_area - expected_area) / expected_area * 100:.2f}%")
        print(f"  Expected volume: {expected_volume:.6f}")

    return vertices, triangles


def run_all_examples():
    """Run all surface analysis examples."""
    print("PyMultiWFN Surface Analysis Examples")
    print("=" * 50)

    try:
        # Run examples
        water_surface = example_water_molecule()
        hirshfeld_surface = example_hirshfeld_surface()
        mapped_surface = example_surface_property_mapping()
        comparison_results = example_surface_comparison()
        vertices, triangles = example_direct_isosurface_extraction()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nGenerated files:")
        print("  - water_surface.obj (Water molecule surface)")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()