"""
Density analysis module for PyMultiWFN.

This module provides tools for electron density analysis including:
- Critical point analysis (BCP, RCP, CCP)
- Laplacian analysis
- Density topology

Reference: PHASE2_TASKS.md - Module 2.2: Electron Density Analysis
"""

from .topology import CriticalPointAnalyzer
from .laplacian import LaplacianAnalyzer

__all__ = ['CriticalPointAnalyzer', 'LaplacianAnalyzer']
