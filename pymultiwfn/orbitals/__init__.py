"""
Orbital analysis module for PyMultiWFN.

This module provides tools for analyzing molecular orbital properties including:
- Orbital energies and HOMO-LUMO gaps
- Orbital compositions
- Orbital overlap analysis

Reference: PHASE2_TASKS.md - Module 2.1: Orbital Analysis
"""

from .energies import OrbitalsAnalyzer

__all__ = ['OrbitalsAnalyzer']
