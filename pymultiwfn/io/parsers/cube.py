"""
Parser for Gaussian Cube files (.cub, .cube).
Cube format is used for volumetric data such as electron density, electrostatic potential, etc.
"""

import re
import numpy as np
from pymultiwfn.core.data import Wavefunction

class CubeLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()
        self.grid_data = None

    def load(self) -> Wavefunction:
        """Parse Cube file and return Wavefunction object."""
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        self._parse_cube(lines)

        return self.wfn

    def _parse_cube(self, lines):
        """Parse Cube format."""
        if len(lines) < 6:
            raise ValueError("Cube file is too short")

        # First two lines: comments/title
        self.wfn.title = lines[0].strip() + " " + lines[1].strip()

        # Third line: number of atoms and origin
        try:
            natom_line = lines[2].split()
            num_atoms = int(natom_line[0])
            origin_x = float(natom_line[1])
            origin_y = float(natom_line[2])
            origin_z = float(natom_line[3])
        except (ValueError, IndexError):
            raise ValueError("Invalid atom/origin line in cube file")

        # Next three lines: grid information
        grid_info = []
        for i in range(3, 6):
            try:
                parts = lines[i].split()
                n_points = int(parts[0])
                vector = [float(parts[1]), float(parts[2]), float(parts[3])]
                grid_info.append((n_points, vector))
            except (ValueError, IndexError):
                raise ValueError(f"Invalid grid line {i} in cube file")

        # Parse atomic coordinates
        for i in range(6, 6 + num_atoms):
            if i >= len(lines):
                break

            line = lines[i].strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 5:
                try:
                    atomic_num = int(parts[0])
                    charge = float(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])

                    # Get element symbol (simplified)
                    element = f"X{atomic_num}"
                    if 1 <= atomic_num <= 118:
                        from pymultiwfn.core.definitions import ELEMENT_NAMES
                        element = ELEMENT_NAMES[atomic_num] if atomic_num < len(ELEMENT_NAMES) else f"X{atomic_num}"

                    self.wfn.add_atom(element, atomic_num, x, y, z, charge)
                except (ValueError, IndexError):
                    continue

        # Parse volumetric data
        data_start = 6 + num_atoms
        data_values = []

        for line in lines[data_start:]:
            line = line.strip()
            if line:
                parts = line.split()
                for part in parts:
                    try:
                        value = float(part)
                        data_values.append(value)
                    except ValueError:
                        continue

        # Store grid information
        self.grid_data = {
            'origin': np.array([origin_x, origin_y, origin_z]),
            'vectors': np.array([info[1] for info in grid_info]),
            'dimensions': [info[0] for info in grid_info],
            'data': np.array(data_values)
        }

        # Store as attribute of wavefunction
        self.wfn.cube_data = self.grid_data

    def get_grid_coordinates(self):
        """Generate 3D grid coordinates."""
        if self.grid_data is None:
            return None

        origin = self.grid_data['origin']
        vectors = self.grid_data['vectors']
        dimensions = self.grid_data['dimensions']

        # Create coordinate arrays
        x = np.arange(dimensions[0])
        y = np.arange(dimensions[1])
        z = np.arange(dimensions[2])

        # Create mesh grid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Transform to real space
        coords = np.stack([X, Y, Z], axis=-1)
        real_coords = coords.dot(vectors.T) + origin

        return real_coords

    def interpolate_value(self, point):
        """Interpolate grid value at arbitrary point."""
        if self.grid_data is None:
            return 0.0

        # This is a simplified nearest-neighbor interpolation
        # For better results, implement trilinear interpolation
        origin = self.grid_data['origin']
        vectors = self.grid_data['vectors']
        dimensions = self.grid_data['dimensions']
        data = self.grid_data['data'].reshape(dimensions)

        # Convert point to grid coordinates
        rel_pos = point - origin
        grid_coords = rel_pos @ np.linalg.inv(vectors.T)

        # Find nearest grid point
        ix = int(np.round(grid_coords[0]))
        iy = int(np.round(grid_coords[1]))
        iz = int(np.round(grid_coords[2]))

        # Check bounds
        ix = max(0, min(ix, dimensions[0] - 1))
        iy = max(0, min(iy, dimensions[1] - 1))
        iz = max(0, min(iz, dimensions[2] - 1))

        return data[ix, iy, iz]