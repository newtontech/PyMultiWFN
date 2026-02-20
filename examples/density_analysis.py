#!/usr/bin/env python3
"""
Example: Density Analysis and Visualization

This script demonstrates electron density analysis:
1. Calculating density on a grid
2. Finding density maxima/minima
3. Integrating density (electron count)
4. Creating density profiles

Usage:
    python density_analysis.py molecule.wfn
"""

import sys
import numpy as np
from pymultiwfn.io.loader import load_wavefunction
from pymultiwfn.math.density import calc_density, clear_density_cache


def analyze_density(wfn):
    """Perform comprehensive density analysis."""
    
    print("\n" + "=" * 60)
    print("Electron Density Analysis")
    print("=" * 60)
    
    # 1. Density at atomic positions
    print("\n1. Density at Atomic Positions")
    print("-" * 60)
    atomic_coords = np.array([[atom.x, atom.y, atom.z] for atom in wfn.atoms])
    atomic_density = calc_density(wfn, atomic_coords)
    
    for i, atom in enumerate(wfn.atoms):
        print(f"  {atom.element:2s} ({i}): {atomic_density[i]:.6f}")
    
    # 2. Density on a grid
    print("\n2. Density on 3D Grid")
    print("-" * 60)
    
    # Create a coarse 3D grid around molecule
    # Find molecular extent
    coords = np.array([[atom.x, atom.y, atom.z] for atom in wfn.atoms])
    center = coords.mean(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0)
    box_size = extent.max() + 4.0  # Add 2 bohr padding on each side
    
    n_grid = 10  # Coarse grid for speed
    x = np.linspace(center[0] - box_size/2, center[0] + box_size/2, n_grid)
    y = np.linspace(center[1] - box_size/2, center[1] + box_size/2, n_grid)
    z = np.linspace(center[2] - box_size/2, center[2] + box_size/2, n_grid)
    
    # Generate grid points
    grid_points = []
    for xi in x:
        for yi in y:
            for zi in z:
                grid_points.append([xi, yi, zi])
    grid_points = np.array(grid_points)
    
    print(f"  Grid size: {n_grid} x {n_grid} x {n_grid} = {len(grid_points)} points")
    print(f"  Box size: {box_size:.2f} x {box_size:.2f} x {box_size:.2f} bohr")
    
    # Calculate density on grid
    clear_density_cache()  # Clear cache for fair timing
    grid_density = calc_density(wfn, grid_points)
    
    print(f"  ✓ Density calculated")
    print(f"  Min density: {grid_density.min():.6e}")
    print(f"  Max density: {grid_density.max():.6e}")
    print(f"  Mean density: {grid_density.mean():.6e}")
    
    # 3. Find density maxima
    print("\n3. Density Maxima (Top 5)")
    print("-" * 60)
    
    # Find indices of top 5 density values
    top_indices = np.argsort(grid_density)[-5:][::-1]
    
    for rank, idx in enumerate(top_indices, 1):
        point = grid_points[idx]
        density = grid_density[idx]
        print(f"  {rank}. Density: {density:.6f} at ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")
    
    # 4. Radial density profile
    print("\n4. Radial Density Profile (from molecular center)")
    print("-" * 60)
    
    # Sample points along radial direction
    n_radial = 20
    r_max = box_size / 2
    radii = np.linspace(0, r_max, n_radial)
    
    # Sample along x-axis from center
    radial_coords = np.array([[r, center[1], center[2]] for r in radii])
    radial_density = calc_density(wfn, radial_coords)
    
    print("  r (bohr)  |  Density")
    print("  " + "-" * 30)
    for r, rho in zip(radii, radial_density):
        print(f"  {r:8.3f}  |  {rho:.6e}")
    
    # 5. Approximate electron count (coarse integration)
    print("\n5. Approximate Electron Count")
    print("-" * 60)
    
    # Integrate density over grid volume
    # This is a very coarse approximation
    volume_element = (box_size / n_grid) ** 3
    total_electrons = np.sum(grid_density) * volume_element
    
    print(f"  Expected electrons: {wfn.num_electrons:.1f}")
    print(f"  Integrated electrons: {total_electrons:.2f}")
    print(f"  Relative error: {abs(total_electrons - wfn.num_electrons) / wfn.num_electrons * 100:.1f}%")
    print(f"  (Note: coarse grid gives rough approximation)")
    
    # 6. Performance statistics
    print("\n6. Performance Statistics")
    print("-" * 60)
    from pymultiwfn.math.density import get_cache_stats
    stats = get_cache_stats()
    print(f"  Cache size: {stats['cache_size']}")
    print(f"  Cache max size: {stats['max_size']}")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    
    if len(sys.argv) != 2:
        print("Usage: python density_analysis.py <wavefunction_file>")
        print("Example: python density_analysis.py molecule.wfn")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    try:
        wfn = load_wavefunction(filepath)
        print(f"\nLoaded: {filepath}")
        print(f"Molecule: {wfn.num_atoms} atoms, {wfn.num_electrons} electrons")
        analyze_density(wfn)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
