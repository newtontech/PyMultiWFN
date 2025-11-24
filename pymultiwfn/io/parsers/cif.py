"""
Parser for Crystallographic Information Files (.cif).
CIF format is used for crystallographic data storage and exchange.
"""

import re
import numpy as np
from pymultiwfn.core.data import Wavefunction
from pymultiwfn.core.constants import ANGSTROM_TO_BOHR

class CIFLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse CIF file and return Wavefunction object."""
        with open(self.filename, 'r') as f:
            content = f.read()

        self._parse_cif(content)

        self.wfn._infer_occupations()
        return self.wfn

    def _parse_cif(self, content: str):
        """Parse CIF format."""
        # Remove comments
        lines = []
        for line in content.split('\n'):
            if line.strip() and not line.strip().startswith('#'):
                lines.append(line.strip())

        # Parse CIF structure
        in_loop = False
        current_loop = {}
        atom_data = []

        for line in lines:
            line = line.strip()

            # Handle data blocks
            if line.startswith('data_'):
                block_name = line[5:]
                if block_name:
                    self.wfn.title = block_name
                continue

            # Handle loops
            if line.startswith('loop_'):
                in_loop = True
                current_loop = {'headers': [], 'data': []}
                continue

            if in_loop:
                # Check if this is a header line
                if line.startswith('_'):
                    current_loop['headers'].append(line)
                else:
                    # This is data line
                    if line:
                        current_loop['data'].append(line.split())
                    continue

            # Handle single-value data items
            if line.startswith('_'):
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0]
                    value = ' '.join(parts[1:])
                    self._process_single_value(key, value)

            # Check if loop ended
            if in_loop and line and not line.startswith('_') and not line.startswith('loop_'):
                # Process the completed loop
                if current_loop['headers'] and current_loop['data']:
                    loop_type = self._determine_loop_type(current_loop['headers'])
                    if loop_type == 'atom':
                        atom_data = self._process_atom_loop(current_loop)
                    elif loop_type == 'cell':
                        self._process_cell_loop(current_loop)

                in_loop = False
                current_loop = {}

        # Process any remaining loop
        if in_loop and current_loop['headers'] and current_loop['data']:
            loop_type = self._determine_loop_type(current_loop['headers'])
            if loop_type == 'atom':
                atom_data = self._process_atom_loop(current_loop)
            elif loop_type == 'cell':
                self._process_cell_loop(current_loop)

        # Add atoms to wavefunction
        for atom in atom_data:
            element = atom['element']
            atomic_num = self._element_to_atomic_number(element)
            atom_info = {
                'label': atom.get('label', ''),
                'type_symbol': atom.get('type_symbol', ''),
                'occupancy': atom.get('occupancy', 1.0),
                'U_iso_or_equiv': atom.get('U_iso_or_equiv', 0.0)
            }
            self.wfn.add_atom(
                element,
                atomic_num,
                atom['x'],
                atom['y'],
                atom['z'],
                float(atomic_num),
                atom_info
            )

    def _determine_loop_type(self, headers):
        """Determine the type of loop based on headers."""
        header_str = ' '.join(headers).lower()

        if any(keyword in header_str for keyword in ['_atom_site_', '_atom']):
            return 'atom'
        elif any(keyword in header_str for keyword in ['_cell_', '_cell_']):
            return 'cell'
        return 'unknown'

    def _process_atom_loop(self, loop_data):
        """Process atom site loop."""
        headers = loop_data['headers']
        data_rows = loop_data['data']

        # Create header mapping
        header_map = {}
        for i, header in enumerate(headers):
            header_map[header] = i

        atoms = []
        for row in data_rows:
            atom = {}

            # Extract common CIF atom fields
            if '_atom_site_label' in header_map:
                atom['label'] = row[header_map['_atom_site_label']]

            if '_atom_site_type_symbol' in header_map:
                atom['type_symbol'] = row[header_map['_atom_site_type_symbol']]

            if '_atom_site_fract_x' in header_map:
                atom['x'] = float(row[header_map['_atom_site_fract_x']])  # Fractional
            elif '_atom_site_Cartesian_x' in header_map:
                atom['x'] = float(row[header_map['_atom_site_Cartesian_x']]) * ANGSTROM_TO_BOHR

            if '_atom_site_fract_y' in header_map:
                atom['y'] = float(row[header_map['_atom_site_fract_y']])
            elif '_atom_site_Cartesian_y' in header_map:
                atom['y'] = float(row[header_map['_atom_site_Cartesian_y']]) * ANGSTROM_TO_BOHR

            if '_atom_site_fract_z' in header_map:
                atom['z'] = float(row[header_map['_atom_site_fract_z']])
            elif '_atom_site_Cartesian_z' in header_map:
                atom['z'] = float(row[header_map['_atom_site_Cartesian_z']]) * ANGSTROM_TO_BOHR

            if '_atom_site_occupancy' in header_map:
                atom['occupancy'] = float(row[header_map['_atom_site_occupancy']])

            if '_atom_site_U_iso_or_equiv' in header_map:
                atom['U_iso_or_equiv'] = float(row[header_map['_atom_site_U_iso_or_equiv']])

            # Extract element symbol
            element = ''
            if 'type_symbol' in atom:
                element = atom['type_symbol']
            elif 'label' in atom:
                element = self._extract_element_from_label(atom['label'])

            if not element:
                element = 'C'  # Default to carbon

            atom['element'] = element

            # Convert fractional coordinates to Cartesian if needed
            if all(key in atom for key in ['x', 'y', 'z']) and self._are_fractional_coords(headers):
                atom['x'], atom['y'], atom['z'] = self._fractional_to_cartesian(
                    atom['x'], atom['y'], atom['z']
                )

            atoms.append(atom)

        return atoms

    def _process_cell_loop(self, loop_data):
        """Process unit cell parameters."""
        headers = loop_data['headers']
        data_rows = loop_data['data']

        if not data_rows:
            return

        row = data_rows[0]  # Usually only one row for cell data

        # Create header mapping
        header_map = {}
        for i, header in enumerate(headers):
            header_map[header] = i

        cell_info = {}

        if '_cell_length_a' in header_map:
            cell_info['a'] = float(row[header_map['_cell_length_a']]) * ANGSTROM_TO_BOHR
        if '_cell_length_b' in header_map:
            cell_info['b'] = float(row[header_map['_cell_length_b']]) * ANGSTROM_TO_BOHR
        if '_cell_length_c' in header_map:
            cell_info['c'] = float(row[header_map['_cell_length_c']]) * ANGSTROM_TO_BOHR
        if '_cell_angle_alpha' in header_map:
            cell_info['alpha'] = float(row[header_map['_cell_angle_alpha']])
        if '_cell_angle_beta' in header_map:
            cell_info['beta'] = float(row[header_map['_cell_angle_beta']])
        if '_cell_angle_gamma' in header_map:
            cell_info['gamma'] = float(row[header_map['_cell_angle_gamma']])

        if cell_info:
            self.wfn.crystal_info = cell_info

    def _are_fractional_coords(self, headers):
        """Check if coordinates are fractional."""
        header_str = ' '.join(headers).lower()
        return 'fract' in header_str

    def _fractional_to_cartesian(self, x_frac, y_frac, z_frac):
        """Convert fractional coordinates to Cartesian."""
        # This is simplified - proper conversion requires cell parameters
        if not hasattr(self.wfn, 'crystal_info') or not self.wfn.crystal_info:
            # Default to orthogonal unit cell
            return (x_frac, y_frac, z_frac)

        # Get cell parameters
        a = self.wfn.crystal_info.get('a', 1.0)
        b = self.wfn.crystal_info.get('b', 1.0)
        c = self.wfn.crystal_info.get('c', 1.0)
        alpha = self.wfn.crystal_info.get('alpha', 90.0)
        beta = self.wfn.crystal_info.get('beta', 90.0)
        gamma = self.wfn.crystal_info.get('gamma', 90.0)

        # Convert angles to radians
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)

        # Build transformation matrix (simplified for orthogonal cells)
        # For proper crystallographic conversion, this would need more complex matrix math
        x_cart = x_frac * a
        y_cart = y_frac * b
        z_cart = z_frac * c

        return (x_cart, y_cart, z_cart)

    def _process_single_value(self, key: str, value: str):
        """Process single CIF data item."""
        if key.startswith('_cell_'):
            if not hasattr(self.wfn, 'crystal_info'):
                self.wfn.crystal_info = {}

            if key == '_cell_length_a':
                self.wfn.crystal_info['a'] = float(value) * ANGSTROM_TO_BOHR
            elif key == '_cell_length_b':
                self.wfn.crystal_info['b'] = float(value) * ANGSTROM_TO_BOHR
            elif key == '_cell_length_c':
                self.wfn.crystal_info['c'] = float(value) * ANGSTROM_TO_BOHR
            elif key == '_cell_angle_alpha':
                self.wfn.crystal_info['alpha'] = float(value)
            elif key == '_cell_angle_beta':
                self.wfn.crystal_info['beta'] = float(value)
            elif key == '_cell_angle_gamma':
                self.wfn.crystal_info['gamma'] = float(value)

        elif key.startswith('_chemical_'):
            if key == '_chemical_name_systematic':
                self.wfn.title = value
            elif key == '_chemical_formula_sum':
                self.wfn.chemical_formula = value

    def _extract_element_from_label(self, label: str) -> str:
        """Extract element symbol from atom label."""
        if not label:
            return ''

        # Remove any numeric suffixes and extract element symbol
        # Common CIF labeling patterns
        label = label.strip()

        # Pattern 1: Element followed by numbers (e.g., C1, O2, Fe3)
        match = re.match(r'^([A-Z][a-z]?)\d*', label)
        if match:
            return match.group(1)

        # Pattern 2: Element at the beginning
        for i in range(1, min(3, len(label))):  # Check first 1-2 characters
            candidate = label[:i].title()
            if self._element_to_atomic_number(candidate) > 0:
                return candidate

        # Default to carbon
        return 'C'

    def _element_to_atomic_number(self, element: str) -> int:
        """Convert element symbol to atomic number."""
        element_mapping = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
            'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
            'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
            'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
            'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
            'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
            'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
            'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
            'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
            'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
            'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
            'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
        }
        return element_mapping.get(element.title(), 0)