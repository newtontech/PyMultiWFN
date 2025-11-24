"""
PyMultiWFN Visualization Module

This module provides comprehensive visualization capabilities for quantum chemical analysis,
including weak interaction analysis, orbital visualization, molecular graphics, and interactive GUI.
Based on the visualization functionality from Multiwfn.
"""

from .display import Plotter
from .weak_interaction import WeakInteractionAnalyzer
from .orbital import OrbitalVisualizer
from .molecular import MolecularVisualizer

# GUI components are optional (require PyQt5)
try:
    from .gui.main_gui import MultiwfnGUI
    _GUI_AVAILABLE = True
except ImportError:
    MultiwfnGUI = None
    _GUI_AVAILABLE = False

__all__ = [
    'Plotter',
    'WeakInteractionAnalyzer',
    'OrbitalVisualizer',
    'MolecularVisualizer'
]

if _GUI_AVAILABLE:
    __all__.append('MultiwfnGUI')