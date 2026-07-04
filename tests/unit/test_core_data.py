"""
Unit tests for pymultiwfn.core.data module.

Tests the Atom, Shell, and Wavefunction data structures.
"""

import numpy as np
import pytest

from pymultiwfn.core.data import Atom, Shell, Wavefunction


@pytest.mark.unit
class TestAtom:
    """Test cases for the Atom class."""

    def test_atom_creation(self, sample_atom):
        """Test that a sample atom can be created."""
        assert sample_atom.element == "H"
        assert sample_atom.index == 1
        assert sample_atom.charge == 1.0

    def test_atom_coord_property(self, sample_atom):
        """Test the coord property returns numpy array."""
        coord = sample_atom.coord
        assert isinstance(coord, np.ndarray)
        assert coord.shape == (3,)
        assert np.allclose(coord, [0.0, 0.0, 0.0])

    def test_atom_coordinates(self):
        """Test atom with non-zero coordinates."""
        atom = Atom(element="C", index=6, x=1.0, y=2.0, z=3.0, charge=6.0)
        assert np.allclose(atom.coord, [1.0, 2.0, 3.0])


@pytest.mark.unit
class TestShell:
    """Test cases for the Shell class."""

    def test_shell_creation(self, sample_shell):
        """Test that a sample shell can be created."""
        assert sample_shell.type == 0  # S shell
        assert sample_shell.center_idx == 0
        assert len(sample_shell.exponents) == 2
        assert len(sample_shell.coefficients) == 2

    def test_shell_exponents_are_numpy(self, sample_shell):
        """Test that exponents are stored as numpy array."""
        assert isinstance(sample_shell.exponents, np.ndarray)

    def test_shell_coefficients_are_numpy(self, sample_shell):
        """Test that coefficients are stored as numpy array."""
        assert isinstance(sample_shell.coefficients, np.ndarray)


@pytest.mark.unit
class TestWavefunction:
    """Test cases for the Wavefunction class."""

    def test_wavefunction_creation(self, sample_wavefunction):
        """Test that a sample wavefunction can be created."""
        assert len(sample_wavefunction.atoms) == 3  # H2O
        assert sample_wavefunction.num_electrons == 10.0
        assert sample_wavefunction.charge == 0
        assert sample_wavefunction.multiplicity == 1

    def test_wavefunction_atoms(self, sample_wavefunction):
        """Test that atoms are properly stored."""
        assert all(isinstance(atom, Atom) for atom in sample_wavefunction.atoms)

    def test_empty_wavefunction(self):
        """Test creating an empty wavefunction."""
        wf = Wavefunction()
        assert len(wf.atoms) == 0
        assert wf.num_electrons == 0.0
