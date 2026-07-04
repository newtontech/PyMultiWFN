import subprocess
from pathlib import Path

import pytest

from consistency_verifier import (
    CaseSpec,
    ComparisonSpec,
    MultiwfnOracle,
    compare_value,
    extract_reference_values,
    load_case_specs,
)
from consistency_verifier.verifier import collect_pymultiwfn_values


def test_load_case_specs_uses_layered_suites(repo_root):
    cases = load_case_specs(
        repo_root / "consistency_verifier" / "cases",
        "pr",
        repo_root,
    )
    case_ids = {case.case_id for case in cases}

    assert "smoke_h2_metadata" in case_ids
    assert "pr_h2_density_gradient_orbitals" in case_ids
    assert "full_phenanthrene_metadata" not in case_ids


def test_extract_reference_values_from_multiwfn_output():
    output = """
    Total atoms: 2
    Total/Alpha/Beta electrons: 2.000000
    Net charge: 0.000000
    The number of orbitals: 28
    Total energy: -1.172345E+00
    """

    values = extract_reference_values(
        output,
        {
            "num_atoms": {
                "type": "int",
                "patterns": [r"Total atoms:\s*(\d+)"],
            },
            "num_electrons": {
                "type": "float",
                "patterns": [r"Total/Alpha/Beta electrons:\s*([-+0-9.]+)"],
            },
            "num_orbitals": {
                "type": "int",
                "patterns": [r"The number of orbitals:\s*(\d+)"],
            },
            "total_energy": {
                "type": "float",
                "patterns": [r"Total energy:\s*([-+0-9.Ee]+)"],
            },
        },
    )

    assert values == {
        "num_atoms": 2,
        "num_electrons": 2.0,
        "num_orbitals": 28,
        "total_energy": pytest.approx(-1.172345),
    }


def test_compare_value_respects_numeric_tolerance():
    spec = ComparisonSpec("density", kind="float", tolerance=1e-6)

    passed = compare_value(spec, 1.0, 1.0 + 5e-7)
    failed = compare_value(spec, 1.0, 1.0 + 5e-5)

    assert passed["status"] == "passed"
    assert failed["status"] == "failed"


def test_multiwfn_oracle_captures_subprocess(monkeypatch, tmp_path):
    input_file = tmp_path / "H2.wfn"
    input_file.write_text("placeholder", encoding="utf-8")

    def fake_run(command, **kwargs):
        assert command == ["/oracle/Multiwfn", str(input_file)]
        assert kwargs["input"] == "18\n1\nq\n"
        assert kwargs["cwd"] == str(tmp_path)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="Total atoms: 2\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    record = MultiwfnOracle(Path("/oracle/Multiwfn")).run(
        input_file,
        ["18", "1", "q"],
        timeout_seconds=10,
    )

    assert record.ok
    assert record.stdout == "Total atoms: 2\n"
    assert record.stdin == ["18", "1", "q"]


def test_collect_pymultiwfn_values_from_retained_h2(repo_root):
    case = CaseSpec(
        case_id="h2_probe",
        suite="smoke",
        input_path=repo_root
        / "Multiwfn_3.8_bin_Linux_noGUI"
        / "examples"
        / "H2_CCSD.wfn",
        commands=[],
        comparisons=[],
        pymultiwfn={"density_points": [[0.0, 0.0, 0.0]]},
    )

    values, warnings = collect_pymultiwfn_values(case)

    assert warnings == []
    assert values["num_atoms"] == 2
    assert values["num_electrons"] == pytest.approx(2.0)
    assert values["num_orbitals"] > 0
    assert "density.point_0" in values
