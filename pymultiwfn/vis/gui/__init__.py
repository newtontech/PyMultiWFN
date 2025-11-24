"""
GUI Module for PyMultiWFN

This module contains the graphical user interface components for PyMultiWFN,
providing an interactive way to visualize molecular structures, orbitals,
and analysis results.
"""

from .main_gui import MultiwfnGUI
from .widgets import *

__all__ = ['MultiwfnGUI']