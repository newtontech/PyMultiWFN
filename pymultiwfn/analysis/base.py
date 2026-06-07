"""Shared interfaces for wavefunction analysis classes."""

from __future__ import annotations

from abc import ABC

from pymultiwfn.core.data import Wavefunction


class BaseWavefunctionAnalysis(ABC):
    """Base class for analyses that operate on a Wavefunction."""

    def __init__(self, wavefunction: Wavefunction):
        self.wavefunction = wavefunction
        self.wfn = wavefunction
        self.validate_wavefunction()

    def validate_wavefunction(self) -> None:
        """Validate required wavefunction data before analysis starts."""
        if self.wavefunction is None:
            raise ValueError("A Wavefunction instance is required for analysis")
