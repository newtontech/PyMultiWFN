"""
Parser for OpenDX format files (.dx).
OpenDX format is used for volumetric data visualization.
"""

import re
import numpy as np
from pymultiwfn.core.data import Wavefunction

class DXLoader:
    def __init__(self, filename: str):
        self.filename = filename
        self.wfn = Wavefunction()

    def load(self) -> Wavefunction:
        """Parse DX file and return Wavefunction object."""
        with open(self.filename, 'r') as f:
            content = f.read()

        self._parse_dx(content)
        return self.wfn

    def _parse_dx(self, content: str):
        """Parse OpenDX format for volumetric data."""
        lines = content.strip().split('\n')

        # Parse header information
        self._parse_header(lines)

        # Find data section
        data_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith('object 1 class gridpositions counts'):
                # Extract grid dimensions
                parts = line.strip().split()
                if len(parts) >= 8:
                    nx, ny, nz = int(parts[7]), int(parts[8]), int(parts[9])
                    self.wfn.grid_shape = (nx, ny, nz)
                data_start = i + 1
                break

        if data_start is None:
            raise ValueError("No grid data found in DX file")

        # Parse grid data
        self._parse_grid_data(lines, data_start)

    def _parse_header(self, lines):
        """Parse DX header for metadata."""
        for line in lines:
            line = line.strip()

            # Extract title if available
            if 'object' in line and 'class' in line and 'field' in line:
                parts = line.split()
                if len(parts) >= 3 and parts[0] == 'object':
                    self.wfn.title = ' '.join(parts[3:]).strip('"')
                    break

    def _parse_grid_data(self, lines, start_idx):
        """Parse volumetric grid data from DX file."""
        grid_data = []
        points = []

        # Skip grid position lines and find actual data
        line_idx = start_idx

        # Find origin and spacing
        origin = None
        spacing = None

        # First 3 lines after gridpositions contain the grid vectors/spacing
        for i in range(3):
            if line_idx + i < len(lines):
                parts = lines[line_idx + i].strip().split()
                if len(parts) >= 3:
                    if origin is None:
                        origin = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
                    else:
                        # These are delta vectors, calculate spacing
                        vector = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
                        if spacing is None:
                            spacing = np.linalg.norm(vector)

        line_idx += 4  # Skip position lines

        # Look for the actual data section
        data_section_found = False
        for i in range(line_idx, len(lines)):
            line = lines[i].strip()

            # Check for data section marker
            if 'object 2 class array type float rank 1 shape' in line:
                data_section_found = True
                # Find the actual data values (usually after "data follows")
                for j in range(i + 1, len(lines)):
                    if 'data follows' in lines[j].lower():
                        line_idx = j + 1
                        break
                break

        if not data_section_found:
            raise ValueError("No data section found in DX file")

        # Parse actual data values
        for i in range(line_idx, len(lines)):
            line = lines[i].strip()

            # Check for end of data
            if line.startswith('attribute') or line.startswith('object'):
                break

            if line:
                try:
                    # Split line and convert to float
                    values = [float(x) for x in line.split()]
                    grid_data.extend(values)
                except ValueError:
                    continue

        # Convert to numpy array
        if grid_data and self.wfn.grid_shape:
            grid_data = np.array(grid_data)

            # Reshape according to grid dimensions
            expected_size = self.wfn.grid_shape[0] * self.wfn.grid_shape[1] * self.wfn.grid_shape[2]
            if len(grid_data) >= expected_size:
                grid_data = grid_data[:expected_size].reshape(self.wfn.grid_shape)
                self.wfn.grid_data = grid_data

                # Store grid information
                if origin is not None:
                    self.wfn.grid_origin = origin
                if spacing is not None:
                    self.wfn.grid_spacing = spacing

    def get_volumetric_data(self):
        """Get the parsed volumetric data."""
        if hasattr(self.wfn, 'grid_data'):
            return self.wfn.grid_data
        else:
            raise ValueError("No volumetric data loaded")

    def get_grid_info(self):
        """Get grid information (shape, origin, spacing)."""
        info = {}
        if hasattr(self.wfn, 'grid_shape'):
            info['shape'] = self.wfn.grid_shape
        if hasattr(self.wfn, 'grid_origin'):
            info['origin'] = self.wfn.grid_origin
        if hasattr(self.wfn, 'grid_spacing'):
            info['spacing'] = self.wfn.grid_spacing
        return info