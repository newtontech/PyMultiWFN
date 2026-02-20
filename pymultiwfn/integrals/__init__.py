"""
Integrals module for PyMultiWFN.

This module provides calculation of molecular integrals for Gaussian basis functions.
"""

from .overlap import calculate_overlap_matrix

__all__ = ["calculate_overlap_matrix"]
