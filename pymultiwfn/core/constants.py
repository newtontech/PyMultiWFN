"""
Physical constants and global parameters for PyMultiWFN.
Derived from define.f90 of Multiwfn.
"""

import numpy as np

# Mathematical constants
PI = np.pi

# Physical constants (2018 CODATA)
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
NM_TO_BOHR = 10.0 * ANGSTROM_TO_BOHR  # 1 nm = 10 Angstrom

# Energy conversion factors
AU_TO_KCAL_MOL = 627.51
AU_TO_KJ_MOL = 2625.5
AU_TO_EV = 27.2113838
AU_TO_WAVENUMBER = 219474.6363
CAL_TO_J = 4.184

# Other constants
ELECTRON_MASS = 9.10938215e-31  # kg
LIGHT_SPEED = 2.99792458e8      # m/s
AU_TO_DEBYE = 2.5417462
PLANCK_CONSTANT = 6.62606896e-34 # J*s
H_BAR = 1.054571628e-34          # J*s
AMU_TO_KG = 1.66053878e-27

BOLTZMANN_CONSTANT = 1.3806488e-23 # J/K
BOLTZMANN_CONSTANT_AU = 3.1668114e-6 # Hartree/K
BOLTZMANN_CONSTANT_EV = 8.6173324e-5 # eV/K

AVOGADRO_CONSTANT = 6.02214179e23
EV_TO_NM = 1239.842 # eV and nm conversion
AU_TO_NM = 45.563


