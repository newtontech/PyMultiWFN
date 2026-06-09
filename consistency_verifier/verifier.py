"""Manifest-driven consistency verification against the Multiwfn binary.

The verifier treats the retained Multiwfn noGUI executable as the oracle and
compares parsed oracle output against structured PyMultiWFN observations.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SUITE_ORDER = {"smoke": 0, "pr": 1, "full": 2}
DEFAULT_TIMEOUT_SECONDS = 60

DEFAULT_REFERENCE_EXTRACTORS: dict[str, dict[str, Any]] = {
    "num_atoms": {
        "type": "int",
        "patterns": [
            r"Total\s+atoms\s*:\s*(\d+)",
            r"Number\s+of\s+atoms\s*:\s*(\d+)",
            r"Atoms\s*:\s*(\d+)",
        ],
    },
    "num_electrons": {
        "type": "float",
        "patterns": [
            r"Total/Alpha/Beta\s+electrons\s*:\s*([-+]?\d+(?:\.\d+)?)",
            r"Number\s+of\s+electrons\s*:\s*([-+]?\d+(?:\.\d+)?)",
            r"Electrons\s*:\s*([-+]?\d+(?:\.\d+)?)",
        ],
    },
    "charge": {
        "type": "float",
        "patterns": [
            r"Net\s+charge\s*:\s*([-+]?\d+(?:\.\d+)?)",
            r"Charge\s*:\s*([-+]?\d+(?:\.\d+)?)",
        ],
    },
    "num_orbitals": {
        "type": "int",
        "patterns": [
            r"The\s+number\s+of\s+orbitals\s*:\s*(\d+)",
            r"Number\s+of\s+orbitals\s*:\s*(\d+)",
            r"Molecular\s+orbitals\s*:\s*(\d+)",
        ],
    },
    "num_basis": {
        "type": "int",
        "patterns": [
            r"Number\s+of\s+basis\s+functions\s*:\s*(\d+)",
            r"Basis\s+functions\s*:\s*(\d+)",
        ],
    },
    "total_energy": {
        "type": "float",
        "patterns": [
            r"Total\s+energy\s*:\s*([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)",
        ],
    },
}


@dataclass(frozen=True)
class ComparisonSpec:
    """A single field comparison from the case manifest."""

    field: str
    kind: str = "float"
    tolerance: float = 1e-6
    relative_tolerance: float = 0.0
    required: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ComparisonSpec":
        return cls(
            field=str(data["field"]),
            kind=str(data.get("kind", "float")),
            tolerance=float(data.get("tolerance", 1e-6)),
            relative_tolerance=float(data.get("relative_tolerance", 0.0)),
            required=bool(data.get("required", True)),
        )


@dataclass(frozen=True)
class CaseSpec:
    """A verifier case loaded from a JSON manifest."""

    case_id: str
    suite: str
    input_path: Path
    commands: list[str]
    comparisons: list[ComparisonSpec]
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    pymultiwfn: dict[str, Any] = field(default_factory=dict)
    reference_extractors: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], repo_root: Path) -> "CaseSpec":
        case_id = str(data["id"])
        suite = str(data.get("suite", "smoke"))
        if suite not in SUITE_ORDER:
            raise ValueError(f"Unknown suite '{suite}' in case {case_id}")

        input_path = _resolve_repo_path(repo_root, str(data["input"]))
        multiwfn_config = data.get("multiwfn", {})
        commands = [str(command) for command in multiwfn_config.get("commands", [])]
        comparisons = [
            ComparisonSpec.from_mapping(item)
            for item in data.get("comparisons", [])
        ]

        return cls(
            case_id=case_id,
            suite=suite,
            input_path=input_path,
            commands=commands,
            comparisons=comparisons,
            timeout_seconds=int(
                multiwfn_config.get(
                    "timeout_seconds",
                    data.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
                )
            ),
            pymultiwfn=dict(data.get("pymultiwfn", {})),
            reference_extractors=dict(data.get("reference_extractors", {})),
        )


@dataclass
class ExecutionRecord:
    """Captured execution details for a verifier subprocess."""

    command: list[str]
    cwd: str
    stdin: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out and self.returncode == 0


class MultiwfnOracle:
    """Run the retained Multiwfn executable with scripted menu commands."""

    def __init__(self, binary_path: Path):
        self.binary_path = binary_path

    def run(
        self,
        input_path: Path,
        commands: Iterable[str],
        timeout_seconds: int,
    ) -> ExecutionRecord:
        stdin_commands = [str(command) for command in commands]
        stdin_text = "\n".join(stdin_commands)
        if stdin_text:
            stdin_text += "\n"

        command = [str(self.binary_path), str(input_path)]
        start = time.perf_counter()

        try:
            completed = subprocess.run(
                command,
                input=stdin_text,
                cwd=str(input_path.parent),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - start
            return ExecutionRecord(
                command=command,
                cwd=str(input_path.parent),
                stdin=stdin_commands,
                returncode=None,
                stdout=_decode_process_text(exc.stdout),
                stderr=_decode_process_text(exc.stderr),
                elapsed_seconds=elapsed,
                timed_out=True,
                error=f"Multiwfn timed out after {timeout_seconds}s",
            )
        except OSError as exc:
            elapsed = time.perf_counter() - start
            return ExecutionRecord(
                command=command,
                cwd=str(input_path.parent),
                stdin=stdin_commands,
                returncode=None,
                stdout="",
                stderr="",
                elapsed_seconds=elapsed,
                error=f"Could not execute Multiwfn: {exc}",
            )

        elapsed = time.perf_counter() - start
        return ExecutionRecord(
            command=command,
            cwd=str(input_path.parent),
            stdin=stdin_commands,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed_seconds=elapsed,
            error=None if completed.returncode == 0 else "Multiwfn returned non-zero",
        )


class ConsistencyVerifier:
    """Run and compare PyMultiWFN with the original Multiwfn executable."""

    def __init__(
        self,
        multiwfn_path: str | Path | None = None,
        repo_root: str | Path | None = None,
    ):
        self.repo_root = Path(repo_root).resolve() if repo_root else _default_repo_root()
        self.multiwfn_path = Path(multiwfn_path) if multiwfn_path else (
            self.repo_root / "Multiwfn_3.8_bin_Linux_noGUI" / "Multiwfn"
        )
        if not self.multiwfn_path.is_absolute():
            self.multiwfn_path = (self.repo_root / self.multiwfn_path).resolve()
        self.oracle = MultiwfnOracle(self.multiwfn_path)

    def verify(self, test_file: str | Path, commands: list[str]) -> dict[str, Any]:
        """Backward-compatible single-case verification API.

        Older scripts call this method directly. It now performs a real oracle
        run and compares all common fields that can be parsed from both sides.
        """
        input_path = _resolve_repo_path(self.repo_root, str(test_file))
        case = CaseSpec(
            case_id="legacy_single_case",
            suite="smoke",
            input_path=input_path,
            commands=commands,
            comparisons=[
                ComparisonSpec("num_atoms", "int", required=False),
                ComparisonSpec("num_electrons", "float", tolerance=1e-6, required=False),
                ComparisonSpec("charge", "float", tolerance=1e-6, required=False),
                ComparisonSpec("num_orbitals", "int", required=False),
            ],
        )
        result = self.run_case(case)
        return {
            "match": result["status"] == "passed",
            "output_ref": result["multiwfn"]["stdout"],
            "output_py": json.dumps(result["pymultiwfn"]["values"], indent=2),
            "diff": _format_case_diff(result),
            "result": result,
        }

    def run_case(self, case: CaseSpec) -> dict[str, Any]:
        """Run one manifest case and return a JSON-serializable result."""
        if not case.input_path.exists():
            return {
                "id": case.case_id,
                "suite": case.suite,
                "input": str(case.input_path),
                "status": "error",
                "error": f"Input file not found: {case.input_path}",
                "comparisons": [],
            }

        py_started = time.perf_counter()
        try:
            py_values, py_warnings = collect_pymultiwfn_values(case)
            py_error = None
        except Exception as exc:  # pragma: no cover - exercised by CLI failures
            py_values = {}
            py_warnings = []
            py_error = f"PyMultiWFN failed: {exc}"
        py_elapsed = time.perf_counter() - py_started

        oracle_record = self.oracle.run(
            case.input_path,
            case.commands,
            timeout_seconds=case.timeout_seconds,
        )
        reference_values = extract_reference_values(
            oracle_record.stdout,
            _merged_extractors(case.reference_extractors),
        )

        comparison_results = compare_case_values(
            case.comparisons,
            reference_values,
            py_values,
        )

        status = _case_status(py_error, oracle_record, comparison_results)
        return {
            "id": case.case_id,
            "suite": case.suite,
            "input": str(case.input_path),
            "status": status,
            "error": py_error or oracle_record.error,
            "multiwfn": _execution_to_dict(oracle_record),
            "pymultiwfn": {
                "values": _json_safe(py_values),
                "warnings": py_warnings,
                "elapsed_seconds": py_elapsed,
            },
            "reference_values": _json_safe(reference_values),
            "comparisons": comparison_results,
        }

    def run_suite(
        self,
        suite: str,
        cases_dir: Path,
        results_dir: Path,
        case_ids: set[str] | None = None,
        write_report: bool = True,
        skip_oracle_if_unavailable: bool = False,
    ) -> dict[str, Any]:
        """Run a layered suite and optionally write report artifacts."""
        cases = load_case_specs(cases_dir, suite, self.repo_root, case_ids=case_ids)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = results_dir / run_id

        results = []
        for case in cases:
            if skip_oracle_if_unavailable and not _oracle_can_run_on_host(
                self.multiwfn_path
            ):
                result = _skipped_case_result(case, self.multiwfn_path)
            else:
                result = self.run_case(case)
            if write_report:
                _write_case_artifacts(run_dir, result)
            results.append(result)

        report = {
            "run_id": run_id,
            "suite": suite,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(self.repo_root),
            "multiwfn_bin": str(self.multiwfn_path),
            "environment": _environment_snapshot(),
            "summary": _summarize_results(results),
            "cases": results,
        }
        if write_report:
            run_dir.mkdir(parents=True, exist_ok=True)
            report_path = run_dir / "report.json"
            report_path.write_text(
                json.dumps(_json_safe(report), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            report["report_path"] = str(report_path)
        return report


def collect_pymultiwfn_values(case: CaseSpec) -> tuple[dict[str, Any], list[str]]:
    """Load a case input with PyMultiWFN and collect structured observations."""
    from pymultiwfn.io import load

    captured_stdout = io.StringIO()
    warnings: list[str] = []
    with contextlib.redirect_stdout(captured_stdout):
        wfn = load(case.input_path)

    loader_output = captured_stdout.getvalue().strip()
    if loader_output:
        warnings.append(f"PyMultiWFN loader wrote stdout: {loader_output[:200]}")

    values: dict[str, Any] = {
        "title": wfn.title,
        "method": wfn.method,
        "basis_set_name": wfn.basis_set_name,
        "num_atoms": int(wfn.num_atoms),
        "num_electrons": float(wfn.num_electrons),
        "charge": float(wfn.charge),
        "multiplicity": int(wfn.multiplicity),
        "num_basis": int(wfn.num_basis),
        "num_atomic_orbitals": int(wfn.num_atomic_orbitals),
        "num_primitives": int(wfn.num_primitives),
        "num_shells": int(wfn.num_shells),
        "num_orbitals": _infer_num_orbitals(wfn),
    }

    _collect_orbital_values(values, wfn, warnings)
    _collect_density_values(values, wfn, case.pymultiwfn, warnings)
    _collect_gradient_values(values, wfn, case.pymultiwfn, warnings)
    _collect_bond_values(values, wfn, case.pymultiwfn, warnings)

    return values, warnings


def extract_reference_values(
    output: str,
    extractors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Extract structured oracle values from Multiwfn text output."""
    values: dict[str, Any] = {}
    for field_name, extractor in extractors.items():
        for pattern in extractor.get("patterns", []):
            match = re.search(pattern, output, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                values[field_name] = _coerce_value(
                    match.group(1),
                    str(extractor.get("type", "string")),
                )
                break
    return values


def compare_case_values(
    comparisons: list[ComparisonSpec],
    reference_values: dict[str, Any],
    actual_values: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare all requested manifest fields."""
    return [
        compare_value(spec, reference_values.get(spec.field), actual_values.get(spec.field))
        for spec in comparisons
    ]


def compare_value(
    spec: ComparisonSpec,
    expected: Any,
    actual: Any,
) -> dict[str, Any]:
    """Compare one field according to its declared comparison kind."""
    if expected is None or actual is None:
        missing_side = "reference" if expected is None else "pymultiwfn"
        status = "failed" if spec.required else "skipped"
        return {
            "field": spec.field,
            "status": status,
            "kind": spec.kind,
            "expected": _json_safe(expected),
            "actual": _json_safe(actual),
            "required": spec.required,
            "message": f"Missing {missing_side} value",
        }

    if spec.kind == "int":
        passed = int(expected) == int(actual)
        difference: int | float = int(actual) - int(expected)
    elif spec.kind == "exact":
        passed = expected == actual
        difference = 0 if passed else math.nan
    elif spec.kind == "string":
        passed = str(expected) == str(actual)
        difference = 0 if passed else math.nan
    else:
        expected_float = float(expected)
        actual_float = float(actual)
        difference = actual_float - expected_float
        passed = math.isclose(
            actual_float,
            expected_float,
            abs_tol=spec.tolerance,
            rel_tol=spec.relative_tolerance,
        )

    return {
        "field": spec.field,
        "status": "passed" if passed else "failed",
        "kind": spec.kind,
        "expected": _json_safe(expected),
        "actual": _json_safe(actual),
        "difference": _json_safe(difference),
        "tolerance": spec.tolerance,
        "relative_tolerance": spec.relative_tolerance,
        "required": spec.required,
        "message": "ok" if passed else "values differ",
    }


def load_case_specs(
    cases_dir: Path,
    suite: str,
    repo_root: Path,
    case_ids: set[str] | None = None,
) -> list[CaseSpec]:
    """Load all JSON case manifests included in the requested suite layer."""
    if suite not in SUITE_ORDER:
        raise ValueError(f"Unknown suite '{suite}'")
    if not cases_dir.exists():
        raise FileNotFoundError(f"Cases directory not found: {cases_dir}")

    selected: list[CaseSpec] = []
    for manifest_path in sorted(cases_dir.glob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for raw_case in manifest.get("cases", []):
            case = CaseSpec.from_mapping(raw_case, repo_root)
            if case_ids is not None and case.case_id not in case_ids:
                continue
            if SUITE_ORDER[case.suite] <= SUITE_ORDER[suite]:
                selected.append(case)
    if not selected:
        raise ValueError(f"No verifier cases selected for suite '{suite}'")
    return sorted(selected, key=lambda case: (SUITE_ORDER[case.suite], case.case_id))


def normalize_output(text: str) -> str:
    """Normalize volatile text before creating diagnostic diffs."""
    normalized_lines = []
    skip_patterns = [
        r"^\s*CPU\s+time",
        r"^\s*Wall\s+time",
        r"^\s*Elapsed\s+time",
        r"^\s*Date\s*:",
    ]
    for line in text.replace("\r\n", "\n").splitlines():
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
            continue
        normalized_lines.append(re.sub(r"\s+", " ", line).strip())
    return "\n".join(line for line in normalized_lines if line)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        verifier = ConsistencyVerifier(args.multiwfn_bin, repo_root=args.repo_root)
        report = verifier.run_suite(
            suite=args.suite,
            cases_dir=Path(args.cases_dir),
            results_dir=Path(args.results_dir),
            case_ids=set(args.case) if args.case else None,
            write_report=not args.no_report,
            skip_oracle_if_unavailable=args.skip_oracle_if_unavailable,
        )
        _print_report_summary(report)
        summary = report["summary"]
        return 0 if summary["failed"] == 0 and summary["errors"] == 0 else 1

    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    repo_root = _default_repo_root()
    default_cases_dir = repo_root / "consistency_verifier" / "cases"
    default_results_dir = repo_root / "consistency_verifier" / "results"
    default_multiwfn_bin = repo_root / "Multiwfn_3.8_bin_Linux_noGUI" / "Multiwfn"

    parser = argparse.ArgumentParser(
        prog="python -m consistency_verifier",
        description="Run PyMultiWFN consistency checks against Multiwfn 3.8.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a verifier suite")
    run_parser.add_argument(
        "--suite",
        choices=sorted(SUITE_ORDER),
        default="smoke",
        help="Layered suite to run. pr includes smoke; full includes smoke and pr.",
    )
    run_parser.add_argument(
        "--multiwfn-bin",
        default=os.environ.get("MULTIWFN_BIN", str(default_multiwfn_bin)),
        help="Path to the Multiwfn executable oracle.",
    )
    run_parser.add_argument(
        "--cases-dir",
        default=str(default_cases_dir),
        help="Directory containing JSON verifier case manifests.",
    )
    run_parser.add_argument(
        "--results-dir",
        default=str(default_results_dir),
        help="Directory for generated verifier reports and transcripts.",
    )
    run_parser.add_argument(
        "--repo-root",
        default=str(repo_root),
        help="Repository root used to resolve relative manifest paths.",
    )
    run_parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only a specific case id. May be passed multiple times.",
    )
    run_parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write report or transcript files.",
    )
    run_parser.add_argument(
        "--skip-oracle-if-unavailable",
        action="store_true",
        help="Mark cases skipped instead of failing when the oracle is unavailable.",
    )
    return parser


def _collect_orbital_values(
    values: dict[str, Any],
    wfn: Any,
    warnings: list[str],
) -> None:
    if wfn.energies is None or wfn.occupations is None:
        return
    try:
        from pymultiwfn.orbitals.energies import OrbitalsAnalyzer

        analyzer = OrbitalsAnalyzer(wfn)
        values["homo_index"] = int(analyzer.homo_index)
        values["lumo_index"] = int(analyzer.lumo_index)
        values["homo_energy"] = float(analyzer.homo_energy)
        values["lumo_energy"] = float(analyzer.lumo_energy)
        values["homo_lumo_gap"] = float(analyzer.gap)
    except Exception as exc:
        warnings.append(f"Orbital analysis unavailable: {exc}")


def _collect_density_values(
    values: dict[str, Any],
    wfn: Any,
    config: dict[str, Any],
    warnings: list[str],
) -> None:
    points = config.get("density_points", [])
    if not points:
        return
    try:
        from pymultiwfn.math.density import calc_density

        coords = np.array(points, dtype=float)
        density = calc_density(wfn, coords)
        for index, value in enumerate(density):
            values[f"density.point_{index}"] = float(value)
    except Exception as exc:
        warnings.append(f"Density observations unavailable: {exc}")


def _collect_gradient_values(
    values: dict[str, Any],
    wfn: Any,
    config: dict[str, Any],
    warnings: list[str],
) -> None:
    points = config.get("gradient_points", [])
    if not points:
        return
    try:
        from pymultiwfn.math.gradient import calc_density_gradient

        coords = np.array(points, dtype=float)
        gradients = calc_density_gradient(wfn, coords)
        for index, vector in enumerate(gradients):
            values[f"gradient.point_{index}.x"] = float(vector[0])
            values[f"gradient.point_{index}.y"] = float(vector[1])
            values[f"gradient.point_{index}.z"] = float(vector[2])
            values[f"gradient.point_{index}.norm"] = float(np.linalg.norm(vector))
    except Exception as exc:
        warnings.append(f"Gradient observations unavailable: {exc}")


def _collect_bond_values(
    values: dict[str, Any],
    wfn: Any,
    config: dict[str, Any],
    warnings: list[str],
) -> None:
    pairs = config.get("bond_pairs", [])
    if not pairs:
        return

    try:
        from pymultiwfn.analysis.bonding.bondorder import calculate_mayer_bond_order
        from pymultiwfn.bonding.bonding import Bonding

        mayer_matrix = calculate_mayer_bond_order(wfn).get("total")
        bonding = Bonding(wfn)
        for left, right in pairs:
            left_int = int(left)
            right_int = int(right)
            key = f"{left_int}_{right_int}"
            if mayer_matrix is not None:
                values[f"mayer_bond_order.{key}"] = float(
                    mayer_matrix[left_int - 1, right_int - 1]
                )
            values[f"fuzzy_bond_order.{key}"] = float(
                bonding.get_fuzzy_bond_order(left_int, right_int)
            )
    except Exception as exc:
        warnings.append(f"Bond observations unavailable: {exc}")


def _case_status(
    py_error: str | None,
    oracle_record: ExecutionRecord,
    comparisons: list[dict[str, Any]],
) -> str:
    if py_error or not oracle_record.ok:
        return "error"
    if any(item["status"] == "failed" for item in comparisons):
        return "failed"
    return "passed"


def _coerce_value(value: str, value_type: str) -> Any:
    if value_type == "int":
        return int(float(value))
    if value_type == "float":
        return float(value.replace("D", "E").replace("d", "e"))
    return value.strip()


def _decode_process_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd": os.getcwd(),
    }


def _execution_to_dict(record: ExecutionRecord) -> dict[str, Any]:
    return {
        "command": record.command,
        "cwd": record.cwd,
        "stdin": record.stdin,
        "returncode": record.returncode,
        "stdout": record.stdout,
        "stderr": record.stderr,
        "elapsed_seconds": record.elapsed_seconds,
        "timed_out": record.timed_out,
        "error": record.error,
    }


def _format_case_diff(result: dict[str, Any]) -> str:
    lines = []
    for item in result.get("comparisons", []):
        if item["status"] != "passed":
            lines.append(
                f"{item['field']}: {item['message']} "
                f"(reference={item.get('expected')!r}, "
                f"pymultiwfn={item.get('actual')!r})"
            )
    if lines:
        return "\n".join(lines)
    return "no_diff"


def _infer_num_orbitals(wfn: Any) -> int:
    if wfn.energies is not None:
        return int(len(wfn.energies))
    if wfn.coefficients is not None:
        return int(wfn.coefficients.shape[0])
    return int(wfn.num_basis)


def _is_executable_file(path: Path) -> bool:
    return path.exists() and os.access(path, os.X_OK)


def _oracle_can_run_on_host(path: Path) -> bool:
    if not _is_executable_file(path):
        return False
    if platform.system() == "Linux":
        return True
    try:
        with path.open("rb") as handle:
            return handle.read(4) != b"\x7fELF"
    except OSError:
        return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _merged_extractors(
    case_extractors: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = dict(DEFAULT_REFERENCE_EXTRACTORS)
    merged.update(case_extractors)
    return merged


def _print_report_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(
        "Verifier summary: "
        f"{summary['passed']} passed, "
        f"{summary['failed']} failed, "
        f"{summary['errors']} errors, "
        f"{summary['skipped']} skipped "
        f"({summary['total']} total)"
    )
    if "report_path" in report:
        print(f"Report: {report['report_path']}")
    for case in report["cases"]:
        print(f"- {case['id']}: {case['status']}")


def _resolve_repo_path(repo_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _skipped_case_result(case: CaseSpec, binary_path: Path) -> dict[str, Any]:
    return {
        "id": case.case_id,
        "suite": case.suite,
        "input": str(case.input_path),
        "status": "skipped",
        "error": f"Oracle unavailable or not executable: {binary_path}",
        "multiwfn": {},
        "pymultiwfn": {"values": {}, "warnings": [], "elapsed_seconds": 0.0},
        "reference_values": {},
        "comparisons": [],
    }


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"total": len(results), "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for result in results:
        status = result["status"]
        if status == "passed":
            summary["passed"] += 1
        elif status == "failed":
            summary["failed"] += 1
        elif status == "skipped":
            summary["skipped"] += 1
        else:
            summary["errors"] += 1
    return summary


def _write_case_artifacts(run_dir: Path, result: dict[str, Any]) -> None:
    case_dir = run_dir / _safe_name(str(result["id"]))
    case_dir.mkdir(parents=True, exist_ok=True)
    multiwfn = result.get("multiwfn", {})
    (case_dir / "multiwfn.stdout.txt").write_text(
        str(multiwfn.get("stdout", "")),
        encoding="utf-8",
    )
    (case_dir / "multiwfn.stderr.txt").write_text(
        str(multiwfn.get("stderr", "")),
        encoding="utf-8",
    )
    (case_dir / "pymultiwfn.values.json").write_text(
        json.dumps(result.get("pymultiwfn", {}).get("values", {}), indent=2),
        encoding="utf-8",
    )
    (case_dir / "case_result.json").write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    raise SystemExit(main())
