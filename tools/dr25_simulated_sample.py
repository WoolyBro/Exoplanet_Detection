"""
dr25_simulated_sample.py
========================

Draws the fixed, seeded samples of the Kepler DR25 simulated-data benchmark (eda/DR25_SIMULATED_PROTOCOL.md):

  INJ1  1,000 recovered injected transits (Christiansen 2017, KSCI-19110; Thompson et al. 2018): real planets
        by construction. Their light curves are the archive's injected PDC flux.
  INV   1,000 TCEs found by the Kepler pipeline in INVERTED light curves (Coughlin 2017, KSCI-19114): false
        alarms by construction, made of the systematics and variability that look transit-like.

Only stars that host no KOI in any Kepler split are eligible, so neither model has seen any light curve of
these stars. Robovetter dispositions and scores (kplr_dr25_<set>_tces.txt) are carried for the
head-to-head comparison.

    python tools/dr25_simulated_sample.py
Output: external_data/kepler_dr25_simulated/sample_{inj1,inv}.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
D = R / "external_data" / "kepler_dr25_simulated"
N, SEED = 1000, 20261018


def table(s: str) -> pd.DataFrame:
    from astropy.io import ascii
    return ascii.read(D / f"kplr_dr25_{s}_tces.txt", format="ipac").to_pandas()


def main() -> int:
    seen = set(pd.read_csv(R / "splits" / "koi_cumulative_split_v2.csv", low_memory=False, usecols=["kepid"]).kepid)
    for s in ["inj1", "inv"]:
        t = table(s)
        t = t[~t.KIC.isin(seen) & t.period.notna() & t.epoch.notna() & (t.duration > 0)]
        pick = t.sample(n=N, random_state=SEED).sort_values(["KIC", "TCE_ID"])
        pick.insert(0, "set", s.upper())
        pick.to_csv(D / f"sample_{s}.csv", index=False)
        print(f"{s}: {len(pick)} TCEs on {pick.KIC.nunique()} stars; Robovetter PC {(pick.Disp == 'PC').mean():.3f}; "
              f"MES median {pick.MES.median():.1f}; period median {pick.period.median():.1f} d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
