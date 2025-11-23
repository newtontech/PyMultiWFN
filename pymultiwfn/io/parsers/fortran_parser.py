# pymultiwfn/io/parsers/fortran_parser.py

import re
from typing import Dict, List, Any

class FortranParser:
    """
    A parser for Fortran source files, primarily aimed at extracting information
    relevant for migrating Fortran code to Python, such as COMMON blocks and PARAMETER statements.
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.content = ""

    def load(self) -> Dict[str, Any]:
        """
        Parses the Fortran source file and extracts relevant information.
        Returns a dictionary containing extracted data.
        """
        print(f"Parsing Fortran source file: {self.filename}")
        with open(self.filename, 'r') as f:
            self.content = f.read()

        extracted_data = {
            "common_blocks": self._parse_common_blocks(),
            "parameters": self._parse_parameters(),
            # Add other parsing methods here as needed
        }
        return extracted_data

    def _parse_common_blocks(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Parses Fortran COMMON blocks.
        Example: COMMON /MyBlock/ Var1, Var2(10), Var3
        """
        common_blocks = {}
        # Regex to find COMMON blocks. Fortran is case-insensitive, but parsing usually handles specific cases.
        # We'll look for lines starting with "COMMON /BlockName/"
        # This regex is simplified and might need refinement for all Fortran COMMON block variations.
        common_pattern = re.compile(r"COMMON\s*/(\w+)/\s*(.*)", re.IGNORECASE)

        for line in self.content.splitlines():
            match = common_pattern.match(line.strip())
            if match:
                block_name = match.group(1)
                variables_str = match.group(2)
                variables = []
                # Split variables by comma, handle array dimensions
                for var_str in variables_str.split(','):
                    var_str = var_str.strip()
                    if var_str:
                        # Basic attempt to separate name and potential dimension
                        var_match = re.match(r"(\w+)(?:\((\d+)\))?", var_str)
                        if var_match:
                            var_name = var_match.group(1)
                            var_dim = var_match.group(2) if var_match.group(2) else "1"
                            variables.append({"name": var_name, "dimension": var_dim})
                        else:
                            variables.append({"name": var_str, "dimension": "1"})

                common_blocks[block_name] = variables
        return common_blocks

    def _parse_parameters(self) -> List[Dict[str, str]]:
        """
        Parses Fortran PARAMETER statements.
        Example: REAL*8, PARAMETER :: ONE = 1.0D0, ZERO = 0.0D0
        Example: INTEGER, PARAMETER :: MAX_ATOMS = 100
        """
        parameters = []
        # Regex for PARAMETER statements. This can be complex due to type declarations.
        # This is a simplified regex to catch basic PARAMETER definitions.
        parameter_pattern = re.compile(r"PARAMETER\s*::\s*(.*)", re.IGNORECASE)

        for line in self.content.splitlines():
            match = parameter_pattern.search(line.strip())
            if match:
                definitions_str = match.group(1)
                # Split by comma, handling multiple assignments on one line
                for definition in definitions_str.split(','):
                    definition = definition.strip()
                    if '=' in definition:
                        name, value = definition.split('=', 1)
                        parameters.append({"name": name.strip(), "value": value.strip()})
        return parameters

    # Additional methods could be added for:
    # - _parse_subroutines_functions()
    # - _parse_module_variables()
    # - _parse_types()