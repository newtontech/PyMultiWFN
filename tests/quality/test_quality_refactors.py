import logging

import numpy as np
import pytest

from pymultiwfn.analysis.base import BaseWavefunctionAnalysis
from pymultiwfn.analysis.bonding.bondorder import (
    _iter_atom_basis_pairs,
    calculate_mayer_bond_order,
    calculate_mulliken_bond_order,
)
from pymultiwfn.analysis.orbitals import OrbitalAnalyzer
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.core.definitions import get_atomic_number
from pymultiwfn.integrals.overlap import _type_to_lmn
from pymultiwfn.io.parsers.factory import ParserFactory
from pymultiwfn.io.parsers.gjf import GJFLoader


def _h2_wavefunction() -> Wavefunction:
    wfn = Wavefunction()
    wfn.add_atom("H", 1, 0.0, 0.0, 0.0)
    wfn.add_atom("H", 1, 1.4, 0.0, 0.0)
    wfn.num_electrons = 2
    wfn.num_basis = 2
    wfn.is_unrestricted = False
    wfn.overlap_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    wfn.Ptot = np.array([[1.0, 0.5], [0.5, 1.0]])
    wfn.Palpha = wfn.Ptot / 2
    wfn.Pbeta = wfn.Ptot / 2
    wfn.get_atomic_basis_indices = lambda: {0: [0], 1: [1]}
    return wfn


def test_canonical_atomic_number_lookup_is_case_insensitive():
    assert get_atomic_number("cl") == 17
    assert get_atomic_number(" CL ") == 17
    assert get_atomic_number("not-an-element") == 0
    assert get_atomic_number("", default=-1) == -1


def test_parser_factory_keeps_gaussian_input_aliases_unique():
    supported = ParserFactory.get_supported_formats()

    assert supported.count(".gjf") == 1
    assert ParserFactory.PARSERS[".gjf"] is GJFLoader
    assert ParserFactory.PARSERS[".com"] is GJFLoader
    assert "chgc" not in supported
    assert "locpot" not in supported


def test_parser_content_detection_logs_unreadable_candidates(tmp_path, caplog):
    candidate = tmp_path / "not-a-file"
    candidate.mkdir()

    with caplog.at_level(logging.DEBUG):
        assert ParserFactory._detect_from_content(str(candidate)) is None

    assert "Could not read parser content" in caplog.text


def test_shared_atom_pair_helper_drives_bond_order_calculations():
    wfn = _h2_wavefunction()

    assert _iter_atom_basis_pairs(wfn.get_atomic_basis_indices(), wfn.num_atoms) == [
        (0, [0], 1, [1])
    ]

    mayer = calculate_mayer_bond_order(wfn)["total"]
    mulliken = calculate_mulliken_bond_order(wfn)["total"]

    assert mayer.shape == (2, 2)
    assert mayer[0, 1] == pytest.approx(1.0)
    assert mulliken[0, 1] == pytest.approx(0.5)
    assert mayer[0, 0] == pytest.approx(mayer[0, 1])
    assert mulliken[1, 1] == pytest.approx(mulliken[1, 0])


def test_overlap_type_lookup_covers_cartesian_shells():
    assert _type_to_lmn(0) == (0, 0, 0)
    assert _type_to_lmn(7) == (1, 1, 0)
    assert _type_to_lmn(19) == (1, 1, 1)

    with pytest.raises(NotImplementedError):
        _type_to_lmn(99)


def test_analysis_base_contract_adopted_by_orbital_analyzer():
    wfn = Wavefunction()
    wfn.num_basis = 1
    wfn.coefficients = np.array([[1.0]])
    wfn.energies = np.array([-0.5])
    wfn.occupations = np.array([2.0])

    analyzer = OrbitalAnalyzer(wfn)

    assert isinstance(analyzer, BaseWavefunctionAnalysis)
    assert analyzer.wavefunction is wfn
    assert analyzer.wfn is wfn
    assert analyzer.get_orbital_properties(0)["energy"] == pytest.approx(-0.5)
