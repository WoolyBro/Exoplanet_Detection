"""Command line: python -m jwst_molecule_screener spectrum.txt [more spectra of the same planet ...]"""

import argparse
import json

from .core import read_table, screen


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrated CH4 and CO2 calls for JWST transmission spectra of one planet.")
    ap.add_argument("files", nargs="+", help="tables of wavelength (micron), depth, uncertainty; several = several spectra of one planet")
    ap.add_argument("--depth-unit", choices=["ppm", "percent", "fraction"], default="ppm")
    ap.add_argument("--cols", default="0,1,2", help="column indices of wavelength, depth, uncertainty")
    a = ap.parse_args()
    w, d, e = (int(c) for c in a.cols.split(","))
    print(json.dumps(screen([read_table(f, w, d, e, a.depth_unit) for f in a.files]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
