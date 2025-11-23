"""
Configuration management for PyMultiWFN.
"""

import os

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        # Defaults
        self.nthreads = 4
        self.output_format = "png"
        self.silent = False
        
        # Load from environment or file (placeholder)
        if "OMP_NUM_THREADS" in os.environ:
            try:
                self.nthreads = int(os.environ["OMP_NUM_THREADS"])
            except ValueError:
                pass

config = Config()
