import math

import numpy as np
import pytest

from pymultiwfn.analysis.bonding.bondorder import calculate_mayer_bond_order
from pymultiwfn.core.data import Wavefunction

REFERENCE_DIATOMICS = [
    ("H2", "H", "H", 1.0),
    ("LiH", "Li", "H", 0.7),
    ("N2", "N", "N", 3.0),
    ("O2", "O", "O", 2.0),
    ("F2", "F", "F", 1.0),
]

ATOMIC_NUMBERS = {
    "H": 1,
    "Li": 3,
    "N": 7,
    "O": 8,
    "F": 9,
}


def _reference_diatomic(
    left_symbol: str, right_symbol: str, expected_bond_order: float
) -> Wavefunction:
    wfn = Wavefunction()
    wfn.add_atom(left_symbol, ATOMIC_NUMBERS[left_symbol], 0.0, 0.0, -0.7)
    wfn.add_atom(right_symbol, ATOMIC_NUMBERS[right_symbol], 0.0, 0.0, 0.7)
    wfn.num_basis = 2
    wfn.num_electrons = 2
    wfn.is_unrestricted = False
    wfn.overlap_matrix = np.eye(2)
    coupling = math.sqrt(expected_bond_order)
    wfn.Ptot = np.array([[1.0, coupling], [coupling, 1.0]])
    wfn.Palpha = wfn.Ptot / 2
    wfn.Pbeta = wfn.Ptot / 2
    wfn.get_atomic_basis_indices = lambda: {0: [0], 1: [1]}
    return wfn


@pytest.mark.parametrize(
    ("name", "left_symbol", "right_symbol", "expected_bond_order"),
    REFERENCE_DIATOMICS,
)
def test_reference_diatomic_mayer_bond_orders(
    name: str, left_symbol: str, right_symbol: str, expected_bond_order: float
):
    wfn = _reference_diatomic(left_symbol, right_symbol, expected_bond_order)

    result = calculate_mayer_bond_order(wfn)

    assert set(result) == {"total"}, name
    assert result["total"][0, 1] == pytest.approx(expected_bond_order)
    assert result["total"][1, 0] == pytest.approx(expected_bond_order)
