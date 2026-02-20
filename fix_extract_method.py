#!/usr/bin/env python3
"""
Script to replace the _extract_wfn_basis_functions method in wfn.py
with the corrected version.
"""

import re

# Read the original file
with open("pymultiwfn/io/parsers/wfn.py", "r") as f:
    content = f.read()

# The new method implementation
new_method = '''    def _extract_wfn_basis_functions(self) -> List[dict]:
        """
        Extract basis functions from WFN-format data.

        This method extracts basis functions directly from WFN file's
        CENTRE ASSIGNMENTS, TYPE ASSIGNMENTS, and EXPONENTS, creating
        one basis function per entry.

        For WFN format:
        - Each entry in TYPE ASSIGNMENTS is a single basis function
        - Type values represent specific basis function components:
          - Type 1 = S
          - Type 2 = Px
          - Type 3 = Py
          - Type 4 = Pz
          - Type 5-10 = D components (xx, yy, zz, xy, xz, yz)
          - Type 11-20 = F components
        - Each basis function has its own exponent and centre

        Returns:
            List of basis function dictionaries compatible with overlap calculation
        """
        if not hasattr(self.wfn, '_centre_assignments'):
            raise ValueError("WFN file missing centre assignments")
        if not hasattr(self.wfn, '_type_assignments'):
            raise ValueError("WFN file missing type assignments")
        if not hasattr(self.wfn, '_exponents'):
            raise ValueError("WFN file missing exponents")

        centre_assignments = self.wfn._centre_assignments
        type_assignments = self.wfn._type_assignments
        exponents = self.wfn._exponents

        basis_functions = []

        # Map WFN types directly to overlap calculation 'type' field
        # In overlap.py, type field uses:
        # - 0 = S
        # - 1 = Px, 2 = Py, 3 = Pz
        # - 4 = D_xx, 5 = D_yy, 6 = D_zz, 7 = D_xy, 8 = D_xz, 9 = D_yz
        # - 10-19 = F components
        wfn_type_to_overlap_type = {
            1: 0,   # S
            2: 1,   # Px
            3: 2,   # Py
            4: 3,   # Pz
            5: 4,   # D_xx
            6: 5,   # D_yy
            7: 6,   # D_zz
            8: 7,   # D_xy
            9: 8,   # D_xz
            10: 9,  # D_yz
            11: 10, # F_xxx
            12: 11, # F_yyy
            13: 12, # F_zzz
            14: 13, # F_xxy
            15: 14, # F_xxz
            16: 15, # F_xyy
            17: 16, # F_yyz
            18: 17, # F_xzz
            19: 18, # F_yzz
            20: 19, # F_xyz
        }

        # Map WFN types to shell types (for grouping)
        # - S = 0, P = 1, D = 2, F = 3
        wfn_type_to_shell_type = {
            1: 0,   # S
            2: 1,   # P
            3: 1,   # P
            4: 1,   # P
            5: 2,   # D
            6: 2,   # D
            7: 2,   # D
            8: 2,   # D
            9: 2,   # D
            10: 2,  # D
        }
        # Add F types (11-20) to shell type mapping
        for wfn_type in range(11, 21):
            wfn_type_to_shell_type[wfn_type] = 3  # F

        # Counter for generating unique basis function indices
        bf_idx = 0

        for i in range(len(centre_assignments)):
            centre_idx = centre_assignments[i]
            wfn_type = type_assignments[i]
            exp = exponents[i]

            atom = self.wfn.atoms[centre_idx]
            coords = tuple(atom.coord)

            # Map WFN type to overlap calculation type
            overlap_type = wfn_type_to_overlap_type.get(wfn_type, wfn_type - 1)
            shell_type = wfn_type_to_shell_type.get(wfn_type, 0)

            # Create basis function dictionary
            basis_functions.append({
                'type': overlap_type,
                'center': centre_idx,
                'coords': coords,
                'exponents': np.array([exp]),
                'coefficients': np.array([1.0]),
                'shell_type': shell_type,
                'shell_idx': i,
                'bf_idx': bf_idx,
            })
            bf_idx += 1

        return basis_functions'''

# Pattern to match the entire method
# We'll match from the method definition to the return statement
pattern = r"    def _extract_wfn_basis_functions\(self\) -> List\[dict\]:.*?return basis_functions"

# Replace the method
new_content = re.sub(pattern, new_method.strip(), content, flags=re.DOTALL)

# Write the modified content back
with open("pymultiwfn/io/parsers/wfn.py", "w") as f:
    f.write(new_content)

print("Successfully replaced _extract_wfn_basis_functions method")
