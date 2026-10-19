"""JWST molecule screener: calibrated CH4 and CO2 calls for exoplanet transmission spectra.

    from jwst_molecule_screener import screen, read_table
    print(screen(read_table("spectrum.txt")))
"""

from .core import bin_spectrum, read_table, screen

__all__ = ["screen", "read_table", "bin_spectrum"]
__version__ = "1.0.0"
