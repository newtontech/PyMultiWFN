"""
Spectrum analysis module for PyMultiWFN.

Contains electronic excitation analysis, density of states,
and other spectral analysis tools.
"""

from .excitations import (
    ExcitedState,
    MOTransition,
    ExcitationAnalysis,
    ExcitationLoader,
    ExcitationAnalyzer,
    ExcitationFileType,
    load_excitation_data,
    analyze_excitation
)

__all__ = [
    'ExcitedState',
    'MOTransition',
    'ExcitationAnalysis',
    'ExcitationLoader',
    'ExcitationAnalyzer',
    'ExcitationFileType',
    'load_excitation_data',
    'analyze_excitation'
]