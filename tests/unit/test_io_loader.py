"""
Unit tests for pymultiwfn.io.loader module.

Tests the file loading functionality.
"""

import pytest
from pathlib import Path
from pymultiwfn.io.loader import load_wavefunction


@pytest.mark.unit
class TestLoader:
    """Test cases for the loader module."""

    def test_load_wavefunction_file_not_found(self):
        """Test loading a non-existent file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            load_wavefunction("/nonexistent/path/to/file.wfn")

    @pytest.mark.requires_data
    def test_load_wavefunction_invalid_format(self, tmp_path):
        """Test loading an invalid file format."""
        # Create a file with invalid content
        invalid_file = tmp_path / "invalid.wfn"
        invalid_file.write_text("This is not a valid WFN file")

        with pytest.raises((ValueError, OSError)):
            load_wavefunction(str(invalid_file))


@pytest.mark.integration
class TestLoaderIntegration:
    """Integration tests for actual file loading."""

    @pytest.mark.requires_data
    def test_load_real_wfn_file(self, test_data_dir):
        """Test loading a real WFN file from test data."""
        # This test will only work when actual test data is added
        wfn_file = test_data_dir / "wfn" / "water_sto3g.wfn"

        if not wfn_file.exists():
            pytest.skip("Test data file not available")

        wavefunction = load_wavefunction(str(wfn_file))
        assert wavefunction is not None
        assert len(wavefunction.atoms) > 0
