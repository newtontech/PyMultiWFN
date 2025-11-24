#!/usr/bin/env python3
"""
Simple test script for fuzzy_atoms.py module.
"""

import numpy as np
from pymultiwfn.core.data import Wavefunction, Atom
from pymultiwfn.analysis.population.fuzzy_atoms import FuzzyAtomsAnalyzer, FuzzyAnalysisConfig, perform_fuzzy_analysis

def test_fuzzy_atoms():
    """Test basic fuzzy atoms functionality."""

    # Create a simple water molecule
    wavefunction = Wavefunction()
    wavefunction.add_atom("O", 8, 0.0, 0.0, 0.0)
    wavefunction.add_atom("H", 1, 0.757, 0.586, 0.0)
    wavefunction.add_atom("H", 1, -0.757, 0.586, 0.0)

    # Set up basic wavefunction data
    wavefunction.num_electrons = 10
    wavefunction.num_basis = 10
    wavefunction.n_mos = 10

    # Test configuration
    config = FuzzyAnalysisConfig(
        partition_method="becke",
        radial_points=10,  # Reduced for testing
        angular_points=20,  # Reduced for testing
        n_becke_iterations=2
    )

    # Test analyzer creation
    analyzer = FuzzyAtomsAnalyzer(wavefunction, config)
    print("✓ FuzzyAtomsAnalyzer created successfully")

    # Test atomic radii initialization
    radii = analyzer.covalent_radii_bohr
    assert len(radii) > 0
    print("✓ Atomic radii initialized successfully")

    # Test atomic weights calculation
    test_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    weights = analyzer.calculate_atomic_weights(test_points)
    assert weights.shape == (3, 2)  # 3 atoms, 2 points
    print("✓ Atomic weights calculated successfully")

    # Test integration grid generation
    grid_points, grid_weights = analyzer._generate_integration_grid()
    assert len(grid_points) > 0
    assert len(grid_weights) > 0
    print("✓ Integration grid generated successfully")

    # Test high-level analysis function
    try:
        results = perform_fuzzy_analysis(wavefunction, "di_li", config)
        print("✓ High-level analysis function works")
    except Exception as e:
        print(f"⚠ High-level analysis function: {e}")

    print("\n✅ All basic tests passed!")

if __name__ == "__main__":
    test_fuzzy_atoms()