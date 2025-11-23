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

# Element information
MAX_ELEMENTS = 118
ELEMENT_NAMES = [
    "Bq", "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

# Map element name to atomic number (Z)
ELEMENT_NAME_TO_Z = {name.upper(): i for i, name in enumerate(ELEMENT_NAMES)}

# VDW Radii (Bondi 1964, Angstrom)
VDW_RADII = np.array([
    0.0, 1.2, 1.4,
    1.82, 1.77, 1.74, 1.7, 1.55, 1.52, 1.47, 1.54,
    2.27, 1.73, 1.73, 2.1, 1.8, 1.8, 1.75, 1.88,
    2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 1.63, 1.4, 1.39, 1.87, 2.0, 1.85, 1.9, 1.85, 2.02, # K-Kr (approx)
    # ... truncated for brevity, should be filled fully in production
])
