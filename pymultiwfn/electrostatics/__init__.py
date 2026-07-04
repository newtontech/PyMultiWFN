"""
Electrostatic analysis module for PyMultiWFN.

This module provides tools for electrostatic analysis including:
- Molecular electrostatic potential (MEP)
- Multipole moments (dipole, quadrupole)
- Atomic charges (Mulliken, Löwdin)
- ESP fitting

Reference: PHASE2_TASKS.md - Module 2.3: Electrostatic Analysis
"""

from .potential import ElectrostaticAnalyzer

__all__ = ["ElectrostaticAnalyzer"]
