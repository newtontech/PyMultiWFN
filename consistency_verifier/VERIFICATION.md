# Consistency Verification

Run parity checks from the repository root:

```bash
python -m consistency_verifier run --suite smoke
python -m consistency_verifier run --suite pr
python -m consistency_verifier run --suite full
```

The default oracle is:

```text
Multiwfn_3.8_bin_Linux_noGUI/Multiwfn
```

Override it with `MULTIWFN_BIN` or `--multiwfn-bin` when needed.

```bash
MULTIWFN_BIN=/path/to/Multiwfn python -m consistency_verifier run --suite smoke
python -m consistency_verifier run --suite smoke --multiwfn-bin /path/to/Multiwfn
```

The retained oracle binary is a Linux x86-64 executable. On non-Linux developer
machines, use the unit tests for the harness and run live oracle parity in a
Linux environment.

Reports are written under `consistency_verifier/results/<run-id>/` and include:

- `report.json`
- raw Multiwfn stdout/stderr transcripts
- structured PyMultiWFN values
- per-case comparison results
