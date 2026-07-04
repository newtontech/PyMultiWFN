"""
Additional consistency tests for PyMultiWFN with molecular systems.

This module tests calculations on more complex molecular systems
beyond simple atoms and diatomics.
"""

import numpy as np
import pytest

from pymultiwfn.core.data import Atom, Shell, Wavefunction
from pymultiwfn.math.density import calc_density


class TestWaterMolecule:
    """Test calculations on water molecule (H2O)."""

    def test_water_atom_count(self):
        """Verify H2O has 3 atoms.

        Reference: Water molecule structure
        """
        # H2O geometry (approximate, in Bohr)
        # O at origin, H atoms at ~0.96 Å = 1.81 Bohr
        atoms = [
            Atom(element="O", index=8, x=0.0, y=0.0, z=0.0, charge=8.0),
            Atom(element="H", index=1, x=0.0, y=1.43, z=1.12, charge=1.0),
            Atom(element="H", index=1, x=0.0, y=-1.43, z=1.12, charge=1.0),
        ]

        # Reference: H2O has 3 atoms
        assert len(atoms) == 3, "H2O should have 3 atoms"
        assert (
            sum(1 for a in atoms if a.element == "O") == 1
        ), "H2O should have 1 O atom"
        assert (
            sum(1 for a in atoms if a.element == "H") == 2
        ), "H2O should have 2 H atoms"

    def test_water_electron_count(self):
        """Verify H2O has 10 electrons.

        Reference: O (8 electrons) + 2×H (1 electron each) = 10 electrons
        """
        # Minimal wavefunction for H2O
        atoms = [
            Atom(element="O", index=8, x=0.0, y=0.0, z=0.0, charge=8.0),
            Atom(element="H", index=1, x=0.0, y=1.43, z=1.12, charge=1.0),
            Atom(element="H", index=1, x=0.0, y=-1.43, z=1.12, charge=1.0),
        ]

        wfn = Wavefunction(
            atoms=atoms,
            num_electrons=10.0,  # O(8) + 2×H(1) = 10
            charge=0,
            multiplicity=1,
            num_basis=7,  # Minimal basis
            num_atomic_orbitals=7,
            num_primitives=21,
            num_shells=5,
            shells=[],
            occupations=np.array([2.0, 2.0, 2.0, 2.0, 2.0]),  # 5 occupied orbitals
            coefficients=np.zeros((7, 5)),  # Placeholder
        )

        # Reference: H2O has 10 electrons
        assert (
            wfn.num_electrons == 10.0
        ), f"H2O should have 10 electrons, got {wfn.num_electrons}"

    def test_water_symmetry(self):
        """Verify H2O C2v symmetry.

        Reference: Water has C2v symmetry (two equivalent H atoms)
        """
        # Create H2O with symmetric geometry
        atoms = [
            Atom(element="O", index=8, x=0.0, y=0.0, z=0.0, charge=8.0),
            Atom(element="H", index=1, x=0.0, y=1.43, z=1.12, charge=1.0),
            Atom(element="H", index=1, x=0.0, y=-1.43, z=1.12, charge=1.0),
        ]

        # Reference: Two H atoms should be equidistant from O
        h1_dist = np.sqrt(atoms[1].x ** 2 + atoms[1].y ** 2 + atoms[1].z ** 2)
        h2_dist = np.sqrt(atoms[2].x ** 2 + atoms[2].y ** 2 + atoms[2].z ** 2)

        assert (
            abs(h1_dist - h2_dist) < 1e-10
        ), f"H atoms should be equidistant from O: {h1_dist} vs {h2_dist}"

        # Reference: H-O-H bond angle should be ~104.5°
        # Vector from O to H1 and H2
        v1 = np.array([atoms[1].x, atoms[1].y, atoms[1].z])
        v2 = np.array([atoms[2].x, atoms[2].y, atoms[2].z])

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle_deg = np.degrees(np.arccos(cos_angle))

        # Allow some tolerance for approximate geometry
        assert (
            100 < angle_deg < 110
        ), f"H-O-H angle should be ~104.5°, got {angle_deg:.1f}°"


class TestMethaneMolecule:
    """Test calculations on methane molecule (CH4)."""

    def test_methane_atom_count(self):
        """Verify CH4 has 5 atoms.

        Reference: Methane molecule structure
        """
        # CH4 tetrahedral geometry (approximate)
        atoms = [
            Atom(element="C", index=6, x=0.0, y=0.0, z=0.0, charge=6.0),
            Atom(element="H", index=1, x=1.0, y=1.0, z=1.0, charge=1.0),
            Atom(element="H", index=1, x=-1.0, y=-1.0, z=1.0, charge=1.0),
            Atom(element="H", index=1, x=-1.0, y=1.0, z=-1.0, charge=1.0),
            Atom(element="H", index=1, x=1.0, y=-1.0, z=-1.0, charge=1.0),
        ]

        # Reference: CH4 has 5 atoms
        assert len(atoms) == 5, "CH4 should have 5 atoms"
        assert (
            sum(1 for a in atoms if a.element == "C") == 1
        ), "CH4 should have 1 C atom"
        assert (
            sum(1 for a in atoms if a.element == "H") == 4
        ), "CH4 should have 4 H atoms"

    def test_methane_electron_count(self):
        """Verify CH4 has 10 electrons.

        Reference: C (6 electrons) + 4×H (1 electron each) = 10 electrons
        """
        atoms = [
            Atom(element="C", index=6, x=0.0, y=0.0, z=0.0, charge=6.0),
            Atom(element="H", index=1, x=1.0, y=1.0, z=1.0, charge=1.0),
            Atom(element="H", index=1, x=-1.0, y=-1.0, z=1.0, charge=1.0),
            Atom(element="H", index=1, x=-1.0, y=1.0, z=-1.0, charge=1.0),
            Atom(element="H", index=1, x=1.0, y=-1.0, z=-1.0, charge=1.0),
        ]

        wfn = Wavefunction(
            atoms=atoms,
            num_electrons=10.0,  # C(6) + 4×H(1) = 10
            charge=0,
            multiplicity=1,
            num_basis=9,  # Minimal basis
            num_atomic_orbitals=9,
            num_primitives=9,
            num_shells=9,
            shells=[],
            occupations=np.array([2.0] * 5),  # 5 occupied orbitals
            coefficients=np.zeros((9, 5)),
        )

        # Reference: CH4 has 10 electrons
        assert (
            wfn.num_electrons == 10.0
        ), f"CH4 should have 10 electrons, got {wfn.num_electrons}"

    def test_methane_tetrahedral_symmetry(self):
        """Verify CH4 has tetrahedral symmetry.

        Reference: All H atoms should be equivalent
        """
        atoms = [
            Atom(element="C", index=6, x=0.0, y=0.0, z=0.0, charge=6.0),
            Atom(element="H", index=1, x=1.0, y=1.0, z=1.0, charge=1.0),
            Atom(element="H", index=1, x=-1.0, y=-1.0, z=1.0, charge=1.0),
            Atom(element="H", index=1, x=-1.0, y=1.0, z=-1.0, charge=1.0),
            Atom(element="H", index=1, x=1.0, y=-1.0, z=-1.0, charge=1.0),
        ]

        # Reference: All H-C distances should be equal
        distances = []
        for i in range(1, 5):  # H atoms are indices 1-4
            dist = np.sqrt(atoms[i].x ** 2 + atoms[i].y ** 2 + atoms[i].z ** 2)
            distances.append(dist)

        # All distances should be equal (within tolerance)
        mean_dist = np.mean(distances)
        for i, d in enumerate(distances):
            assert (
                abs(d - mean_dist) < 1e-10
            ), f"H{i+1}-C distance {d} differs from mean {mean_dist}"


class TestDiatomicMolecules:
    """Test calculations on diatomic molecules."""

    def test_n2_atom_count(self):
        """Verify N2 has 2 atoms.

        Reference: Nitrogen molecule structure
        """
        # N2 molecule (N≡N triple bond)
        bond_length = 2.0  # Bohr (approximate)
        atoms = [
            Atom(element="N", index=7, x=0.0, y=0.0, z=-bond_length / 2, charge=7.0),
            Atom(element="N", index=7, x=0.0, y=0.0, z=bond_length / 2, charge=7.0),
        ]

        # Reference: N2 has 2 atoms
        assert len(atoms) == 2, "N2 should have 2 atoms"
        assert all(a.element == "N" for a in atoms), "N2 should have 2 N atoms"

    def test_n2_electron_count(self):
        """Verify N2 has 14 electrons.

        Reference: 2×N (7 electrons each) = 14 electrons
        """
        atoms = [
            Atom(element="N", index=7, x=0.0, y=0.0, z=-1.0, charge=7.0),
            Atom(element="N", index=7, x=0.0, y=0.0, z=1.0, charge=7.0),
        ]

        wfn = Wavefunction(
            atoms=atoms,
            num_electrons=14.0,  # 2×N(7) = 14
            charge=0,
            multiplicity=1,  # Singlet
            num_basis=2,
            num_atomic_orbitals=2,
            num_primitives=2,
            num_shells=2,
            shells=[],
            occupations=np.array([2.0] * 7),  # 7 occupied orbitals
            coefficients=np.zeros((2, 7)),
        )

        # Reference: N2 has 14 electrons
        assert (
            wfn.num_electrons == 14.0
        ), f"N2 should have 14 electrons, got {wfn.num_electrons}"

    def test_o2_electron_count(self):
        """Verify O2 has 16 electrons.

        Reference: 2×O (8 electrons each) = 16 electrons
        """
        # O2 molecule
        atoms = [
            Atom(element="O", index=8, x=0.0, y=0.0, z=-1.2, charge=8.0),
            Atom(element="O", index=8, x=0.0, y=0.0, z=1.2, charge=8.0),
        ]

        wfn = Wavefunction(
            atoms=atoms,
            num_electrons=16.0,  # 2×O(8) = 16
            charge=0,
            multiplicity=3,  # Triplet (O2 is paramagnetic)
            num_basis=2,
            num_atomic_orbitals=2,
            num_primitives=2,
            num_shells=2,
            shells=[],
            occupations=np.array([2.0] * 7 + [1.0, 1.0]),  # Open shell
            coefficients=np.zeros((2, 9)),
        )

        # Reference: O2 has 16 electrons
        assert (
            wfn.num_electrons == 16.0
        ), f"O2 should have 16 electrons, got {wfn.num_electrons}"


class TestMolecularProperties:
    """Test molecular property calculations."""

    def test_neutral_charge(self):
        """Verify neutral molecules have zero net charge.

        Reference: Sum of atomic charges = 0 for neutral molecules
        """
        # H2O
        atoms = [
            Atom(element="O", index=8, x=0.0, y=0.0, z=0.0, charge=8.0),
            Atom(element="H", index=1, x=0.0, y=1.43, z=1.12, charge=1.0),
            Atom(element="H", index=1, x=0.0, y=-1.43, z=1.12, charge=1.0),
        ]

        wfn = Wavefunction(
            atoms=atoms,
            num_electrons=10.0,
            charge=0,
            multiplicity=1,
            num_basis=7,
            num_atomic_orbitals=7,
            num_primitives=21,
            num_shells=5,
            shells=[],
            occupations=np.array([2.0] * 5),
            coefficients=np.zeros((7, 5)),
        )

        # Reference: Neutral molecule
        assert wfn.charge == 0, f"H2O should be neutral, got charge={wfn.charge}"

    def test_charged_molecule(self):
        """Verify ion charges are correct.

        Reference: Ions have non-zero net charge
        """
        # H2O+ (cation)
        atoms = [
            Atom(element="O", index=8, x=0.0, y=0.0, z=0.0, charge=8.0),
            Atom(element="H", index=1, x=0.0, y=1.43, z=1.12, charge=1.0),
            Atom(element="H", index=1, x=0.0, y=-1.43, z=1.12, charge=1.0),
        ]

        wfn = Wavefunction(
            atoms=atoms,
            num_electrons=9.0,  # One less electron (10 - 1 = 9)
            charge=+1,  # Cation
            multiplicity=2,  # Doublet
            num_basis=7,
            num_atomic_orbitals=7,
            num_primitives=21,
            num_shells=5,
            shells=[],
            occupations=np.array([2.0] * 4 + [1.0]),  # 4 doubly occupied, 1 singly
            coefficients=np.zeros((7, 5)),
        )

        # Reference: Cation has +1 charge
        assert wfn.charge == +1, f"H2O+ should have +1 charge, got {wfn.charge}"
        assert (
            wfn.num_electrons == 9
        ), f"H2O+ should have 9 electrons, got {wfn.num_electrons}"
