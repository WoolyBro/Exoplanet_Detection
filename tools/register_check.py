"""
register_check.py
=================

Check the prospective molecule register (eda/MOLECULE_REGISTER.md): recompute its SHA-256 against the
registered hash, and list every entry with its call. Resolve an entry by adding a row to
outputs/stage_b_register/resolutions.csv (planet, gas, outcome = detected | not_detected, source); the
script then scores it as a hit, a miss, or an open ambiguous call.

    python tools/register_check.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1]
REG = R / "outputs" / "stage_b_register" / "molecule_register.csv"
RES = R / "outputs" / "stage_b_register" / "resolutions.csv"
REGISTERED = "66c65066e4397b045224011847337eb91dbd23e0703024403df8fdca2ac653fb"


def main() -> int:
    h = hashlib.sha256(REG.read_bytes()).hexdigest()
    print(f"register hash {'OK' if h == REGISTERED else 'CHANGED - the register is void'} ({h[:16]}...)")
    reg = pd.read_csv(REG)
    if RES.is_file():
        res = pd.read_csv(RES)
        reg = reg.merge(res, on=["planet", "gas"], how="left")
    else:
        reg["outcome"] = None
    def score(r):
        if pd.isna(r.get("outcome")):
            return "open"
        if r.call == "ambiguous":
            return "ambiguous (not scored)"
        return "hit" if (r.call == "present") == (r.outcome == "detected") else "miss"
    reg["status"] = reg.apply(score, axis=1)
    pd.set_option("display.width", 200)
    print(reg[["planet", "gas", "call", "p_gbm", "p_cnn", "status"]].to_string(index=False))
    print(reg.status.value_counts().to_dict())
    return 0 if h == REGISTERED else 1


if __name__ == "__main__":
    sys.exit(main())
