"""
Integration tests for PyMultiWFN file loading.

These tests test the full workflow of loading various file formats.
"""

from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.requires_data
class TestFileLoading:
    """Integration tests for file loading workflows."""

    def test_discover_test_data_structure(self, test_data_dir):
        """Test that test data directories are properly structured."""
        assert test_data_dir.exists()
        assert (test_data_dir / "wfn").exists()
        assert (test_data_dir / "fchk").exists()
        assert (test_data_dir / "molden").exists()

    def test_load_wfn_file(self, test_data_dir):
        """Test loading a WFN file end-to-end."""
        from pymultiwfn.io.loader import load_wavefunction

        wfn_file = test_data_dir / "H2_CCSD.wfn"
        if not wfn_file.exists():
            pytest.skip("Example WFN file not found")

        wf = load_wavefunction(str(wfn_file))
        assert wf is not None
        assert len(wf.atoms) > 0
        assert wf.num_basis > 0
