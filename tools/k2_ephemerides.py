"""
k2_ephemerides.py
=================

Ephemerides for the K2 blind test (eda/K2_BLIND_TEST_PROTOCOL.md). One row per K2 planet or candidate of
splits/k2_planets_candidates_split.csv (the mentor's split, read-only; its labels are used unchanged).

The split keeps only each planet's DEFAULT parameter set, in which 492 of 1,806 rows lack a period, epoch
or (mostly) duration. The NASA Exoplanet Archive's k2pandc table holds every published parameter set, so:

  1. catalogue  - the default set's value;
  2. other set  - else the value from another published set for the same planet (with that set's error);
  3. estimated  - durations only: T14 = (P / pi) asin[(R*/a) <sqrt((1+k)^2 - b^2)>], with a/R* from the
                  catalogue (pl_ratdor) or from Kepler's law and the stellar mass (or log g and radius),
                  k = Rp/R*, and b from the catalogue or averaged over b ~ U(0, 1). The estimator is checked
                  against every row that has a catalogue duration, and its accuracy is written out.

Rows are tagged with duration_source, so the protocol can use catalogue durations for the primary
analysis and add the estimated ones only in a secondary analysis.

    python tools/k2_ephemerides.py            # downloads the k2pandc table (TAP) once, then builds
Output: external_data/k2/k2pandc_all_<date>.csv (archive snapshot), external_data/k2/k2_ephemerides.csv
"""

from __future__ import annotations

import io
import json
import ssl
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
OUT = R / "external_data" / "k2"
SPLIT = R / "splits" / "k2_planets_candidates_split.csv"
G, MSUN, RSUN, DAY = 6.674e-8, 1.989e33, 6.957e10, 86400.0
REARTH_PER_RSUN = 109.1
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
COLS = ["pl_name", "default_flag", "pl_refname", "rowupdate", "pl_orbper", "pl_orbpererr1", "pl_tranmid",
        "pl_tranmiderr1", "pl_trandur", "pl_trandurerr1", "pl_ratdor", "pl_imppar", "pl_ratror"]


def snapshot() -> pd.DataFrame:
    have = sorted(OUT.glob("k2pandc_all_*.csv"))
    if have:
        return pd.read_csv(have[-1])
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    url = TAP + "?" + urllib.parse.urlencode({"query": f"select {', '.join(COLS)} from k2pandc", "format": "csv"})
    req = urllib.request.Request(url, headers={"User-Agent": "research (exoplanet K2 blind test)"})
    data = urllib.request.urlopen(req, timeout=300, context=ctx).read()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"k2pandc_all_{date.today().isoformat()}.csv"
    path.write_bytes(data)
    return pd.read_csv(io.BytesIO(data))


def pick(sets: pd.DataFrame, value: str, err: str | None) -> pd.DataFrame:
    """Per planet: the default set's value if present, else the most recently updated other set's."""
    s = sets.dropna(subset=[value]).sort_values(["pl_name", "default_flag", "rowupdate"], ascending=[True, False, False])
    first = s.groupby("pl_name").head(1).set_index("pl_name")
    out = pd.DataFrame({value: first[value], f"{value}_source": np.where(first.default_flag == 1, "catalogue", "other set")})
    if err:
        out[err] = first[err]
    return out


def mean_chord(k: np.ndarray, b: np.ndarray) -> np.ndarray:
    """sqrt((1+k)^2 - b^2), or its average over b ~ U(0, 1) where b is unknown."""
    grid = np.linspace(0, 1, 201)
    avg = np.array([np.mean(np.sqrt(np.clip((1 + kk) ** 2 - grid ** 2, 0, None))) for kk in k])
    known = np.sqrt(np.clip((1 + k) ** 2 - np.nan_to_num(b) ** 2, 0, None))
    return np.where(np.isfinite(b), known, avg)


def estimate_duration_h(df: pd.DataFrame) -> pd.Series:
    P = df.pl_orbper.to_numpy(float)
    rs = df.st_rad.to_numpy(float)
    mass = df.st_mass.to_numpy(float)
    from_logg = 10 ** (df.st_logg.to_numpy(float) - 4.438) * rs ** 2            # M/Msun from log g and R
    mass = np.where(np.isfinite(mass), mass, from_logg)
    a_over_r = (G * mass * MSUN * (P * DAY) ** 2 / (4 * np.pi ** 2)) ** (1 / 3) / (rs * RSUN)
    a_over_r = np.where(np.isfinite(df.pl_ratdor_set.to_numpy(float)), df.pl_ratdor_set.to_numpy(float), a_over_r)
    k = df.pl_ratror_set.to_numpy(float)
    k = np.where(np.isfinite(k), k, df.pl_rade.to_numpy(float) / (rs * REARTH_PER_RSUN))
    k = np.where(np.isfinite(k), k, 0.02)                                         # a typical small planet
    chord = mean_chord(k, df.pl_imppar_set.to_numpy(float))
    x = np.clip(chord / a_over_r, 0, 1)
    return pd.Series(P / np.pi * np.arcsin(x) * 24, index=df.index)


def main() -> int:
    sets = snapshot()
    split = pd.read_csv(SPLIT, low_memory=False)
    split = split[split.epic_hostname.notna()].copy()                              # one microlensing planet
    split["epic"] = split.epic_hostname.str.split().str[-1].astype(np.int64)
    base = split[["pl_name", "epic_hostname", "epic", "tic_id", "k2_campaigns", "disposition", "label_harmonized",
                  "split", "pl_rade", "st_rad", "st_mass", "st_logg", "st_teff", "sy_kepmag", "pl_trandep"]].set_index("pl_name")
    for value, err in [("pl_orbper", "pl_orbpererr1"), ("pl_tranmid", "pl_tranmiderr1"), ("pl_trandur", "pl_trandurerr1")]:
        base = base.join(pick(sets, value, err), how="left")
    for extra in ["pl_ratdor", "pl_imppar", "pl_ratror"]:
        base = base.join(pick(sets, extra, None)[[extra]].rename(columns={extra: f"{extra}_set"}), how="left")
    base = base.reset_index()
    base.loc[base.pl_trandur <= 0, ["pl_trandur", "pl_trandur_source"]] = [np.nan, None]

    # duration estimator, validated where a catalogue duration exists
    est = estimate_duration_h(base)
    have = base.pl_trandur.notna() & est.notna() & (est > 0)
    ratio = (est[have] / base.pl_trandur[have]).to_numpy()
    check = {"n": int(have.sum()), "median_est_over_catalogue": round(float(np.median(ratio)), 3),
             "p16_p84": [round(float(np.percentile(ratio, 16)), 3), round(float(np.percentile(ratio, 84)), 3)],
             "fraction_within_30pct": round(float(np.mean(np.abs(ratio - 1) < 0.3)), 3)}
    fill = base.pl_trandur.isna() & est.notna() & (est > 0)
    base["duration_source"] = np.where(base.pl_trandur.notna(), base.pl_trandur_source, np.where(fill, "estimated", "missing"))
    base.loc[fill, "pl_trandur"] = est[fill]
    base["ephemeris_complete"] = base[["pl_orbper", "pl_tranmid", "pl_trandur"]].notna().all(axis=1)
    keep = ["pl_name", "epic_hostname", "epic", "tic_id", "k2_campaigns", "disposition", "label_harmonized", "split",
            "pl_orbper", "pl_orbpererr1", "pl_orbper_source", "pl_tranmid", "pl_tranmiderr1", "pl_tranmid_source",
            "pl_trandur", "pl_trandurerr1", "duration_source", "ephemeris_complete", "pl_rade", "pl_trandep",
            "st_rad", "st_teff", "sy_kepmag"]
    base[keep].to_csv(OUT / "k2_ephemerides.csv", index=False)
    summary = {"rows": int(len(base)), "complete": int(base.ephemeris_complete.sum()),
               "duration_source": base.duration_source.value_counts().to_dict(),
               "period_source": base.pl_orbper_source.value_counts(dropna=False).astype(int).to_dict(),
               "epoch_source": base.pl_tranmid_source.value_counts(dropna=False).astype(int).to_dict(),
               "complete_by_label": base[base.ephemeris_complete].label_harmonized.value_counts().to_dict(),
               "duration_estimator_check": check}
    (OUT / "k2_ephemerides_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
