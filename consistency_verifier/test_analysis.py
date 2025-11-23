# consistency_verifier/test_bonding_analysis.py

import unittest
import os
import numpy as np

from pymultiwfn.io.parsers.fchk import FchkLoader
from pymultiwfn.analysis.bonding.mayer import calculate_mayer_bond_order
from pymultiwfn.analysis.bonding.mulliken import calculate_mulliken_bond_order # Added this line

class TestBondingAnalysis(unittest.TestCase): # Renamed class

    def setUp(self):
        # Placeholder for the FCHK file path
        # In a real scenario, this file would be provided or generated.
        self.fchk_file_path = "consistency_verifier/test_data/h2o.fchk"
        self.skip_tests_if_no_file = not os.path.exists(self.fchk_file_path)

        if self.skip_tests_if_no_file:
            self.skipTest(f"FCHK file not found: {self.fchk_file_path}. Skipping tests.")
            
        self.wfn = None
        self.overlap_matrix = None
        
        # Load wavefunction and overlap matrix once for both tests
        if not self.skip_tests_if_no_file:
            loader = FchkLoader(self.fchk_file_path)
            self.wfn = loader.load()
            self.overlap_matrix = self.wfn.overlap_matrix
            # Ensure density matrices are calculated after loading
            self.wfn.calculate_density_matrices()


    def test_h2o_mayer_bond_order(self):
        if self.skip_tests_if_no_file:
            return # Skip if file not present

        self.assertIsNotNone(self.wfn.overlap_matrix, "Overlap matrix was not loaded from FCHK.")
        self.assertIsNotNone(self.wfn.Ptot, "Total density matrix not calculated.")
        
        print(f"\n--- Testing Mayer Bond Order for H2O ({self.fchk_file_path}) ---")
        # Calculate Mayer bond order
        bnd_mattot, bnd_mata, bnd_matb = calculate_mayer_bond_order(self.wfn, self.overlap_matrix)

        self.assertIsNotNone(bnd_mattot, "Total Mayer bond order matrix is None.")
        self.assertEqual(bnd_mattot.shape, (self.wfn.num_atoms, self.wfn.num_atoms))

        print(f"Mayer Bond Order (Total) for H2O:\n{bnd_mattot}")

        # Example assertions for H2O (hypothetical values, need to be replaced with actual reference)
        # self.assertAlmostEqual(bnd_mattot[0, 1], 0.9, delta=0.2) 
        # self.assertAlmostEqual(bnd_mattot[0, 2], 0.9, delta=0.2)
        # self.assertAlmostEqual(bnd_mattot[1, 2], 0.0, delta=0.1) # H-H should be near zero

        # Further assertions for unrestricted case if applicable
        if self.wfn.is_unrestricted:
            self.assertIsNotNone(bnd_mata, "Alpha Mayer bond order matrix is None.")
            self.assertIsNotNone(bnd_matb, "Beta Mayer bond order matrix is None.")
            self.assertEqual(bnd_mata.shape, (self.wfn.num_atoms, self.wfn.num_atoms))
            self.assertEqual(bnd_matb.shape, (self.wfn.num_atoms, self.wfn.num_atoms))
            print(f"Mayer Bond Order (Alpha) for H2O:\n{bnd_mata}")
            print(f"Mayer Bond Order (Beta) for H2O:\n{bnd_matb}")

    def test_h2o_mulliken_bond_order(self): # Added this new test method
        if self.skip_tests_if_no_file:
            return # Skip if file not present

        self.assertIsNotNone(self.wfn.overlap_matrix, "Overlap matrix was not loaded from FCHK.")
        self.assertIsNotNone(self.wfn.Ptot, "Total density matrix not calculated.")
        
        print(f"\n--- Testing Mulliken Bond Order for H2O ({self.fchk_file_path}) ---")
        # Calculate Mulliken bond order
        bnd_mattot, bnd_mata, bnd_matb = calculate_mulliken_bond_order(self.wfn, self.overlap_matrix)

        self.assertIsNotNone(bnd_mattot, "Total Mulliken bond order matrix is None.")
        self.assertEqual(bnd_mattot.shape, (self.wfn.num_atoms, self.wfn.num_atoms))

        print(f"Mulliken Bond Order (Total) for H2O:\n{bnd_mattot}")

        # Example assertions for H2O (hypothetical values, need to be replaced with actual reference)
        # self.assertAlmostEqual(bnd_mattot[0, 1], 0.6, delta=0.2) 
        # self.assertAlmostEqual(bnd_mattot[0, 2], 0.6, delta=0.2)
        # self.assertAlmostEqual(bnd_mattot[1, 2], 0.0, delta=0.1) # H-H should be near zero

        # Further assertions for unrestricted case if applicable
        if self.wfn.is_unrestricted:
            self.assertIsNotNone(bnd_mata, "Alpha Mulliken bond order matrix is None.")
            self.assertIsNotNone(bnd_matb, "Beta Mulliken bond order matrix is None.")
            self.assertEqual(bnd_mata.shape, (self.wfn.num_atoms, self.wfn.num_atoms))
            self.assertEqual(bnd_matb.shape, (self.wfn.num_atoms, self.wfn.num_atoms))
            print(f"Mulliken Bond Order (Alpha) for H2O:\n{bnd_mata}")
            print(f"Mulliken Bond Order (Beta) for H2O:\n{bnd_matb}")
