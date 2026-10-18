"""
candidate_factor.py
===================

Part (c) of HABITABILITY_FACTOR_PROTOCOL.md: the habitability factor for UNCONFIRMED candidates, where the
CNN-LSTM vetter enters directly. factor = P(real) x Stage C score, with P(real) the vetter's calibrated
planet-like probability and the Stage C score from exactly the same physics and weights as the confirmed
targets (target_physics.monte_carlo, build_priority_table.WEIGHTS).

Candidates with an unbiased vetter score only:
  * the 564 TESS v2-test candidates: TESS ensemble, Platt map fitted on TESS validation (test read once);
  * the K2 candidates of the blind read: TESS ensemble score with the same TESS Platt map (it transferred to
    K2, ECE 0.087), excluding the stars the TESS model had seen.
Parameters come from the local split files (the archive's TOI and K2 tables); TESS J magnitudes from the TIC
(MAST), and orbital distances from the period and log g.

    python stage_c_priority_fusion/candidate_factor.py
Output: outputs/stage_c/candidate_factor.csv, candidate_shortlist.csv, candidate_factor.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE))
import build_priority_table as bpt  # noqa: E402
import target_physics as tp  # noqa: E402

OUT = RESEARCH / "outputs" / "stage_c"
TESS_PLATT = (1.0121, 1.0341)
G, MSUN, RSUN, AU, DAY = 6.674e-8, 1.989e33, 6.957e10, 1.496e13, 86400.0


def platt(score):
    s = np.clip(np.asarray(score, float), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(TESS_PLATT[0] * np.log(s / (1 - s)) + TESS_PLATT[1])))


def tic_table(tics) -> pd.DataFrame:
    """TIC v8 stellar parameters (MAST) for the given TIC ids: Teff, rad, mass, Jmag."""
    from astroquery.mast import Catalogs
    rows = []
    ids = sorted({str(int(t)) for t in tics if pd.notna(t)})
    for i in range(0, len(ids), 200):
        tab = Catalogs.query_criteria(catalog="Tic", ID=ids[i:i + 200])
        rows.append(pd.DataFrame({c: np.asarray(tab[c]) for c in ["ID", "Teff", "rad", "mass", "Jmag"]}))
    t = pd.concat(rows, ignore_index=True)
    t["ID"] = t.ID.astype(str)
    return t.set_index("ID").apply(pd.to_numeric, errors="coerce")


def complete(d: pd.DataFrame, tic_col: str) -> pd.DataFrame:
    """Fill missing stellar parameters from the TIC; make insolation and orbit consistent; flag what was done."""
    tic = tic_table(d[tic_col])
    key = d[tic_col].map(lambda x: str(int(x)) if pd.notna(x) else None)
    notes = [[] for _ in range(len(d))]
    for col, tcol in [("st_teff", "Teff"), ("st_rad", "rad"), ("st_mass", "mass"), ("sy_jmag", "Jmag")]:
        if col not in d:
            d[col] = np.nan
        fill = d[col].isna() & key.map(tic[tcol]).notna()
        d.loc[fill, col] = key[fill].map(tic[tcol])
        for i in np.flatnonzero(fill.to_numpy()):
            notes[i].append(f"{col} from TIC")
    if "st_logg" in d:                                                   # mass from log g where TIC has none
        m_g = 10 ** (d.st_logg - 4.438) * d.st_rad ** 2
        fill = d.st_mass.isna() & m_g.notna()
        d.loc[fill, "st_mass"] = m_g[fill]
    lum = d.st_rad ** 2 * (d.st_teff / 5772.0) ** 4
    a_p = (G * d.st_mass * MSUN * (d.pl_orbper * DAY) ** 2 / (4 * np.pi ** 2)) ** (1 / 3) / AU
    s_p = lum / a_p ** 2
    if "pl_insol" not in d:
        d["pl_insol"] = np.nan
    ratio = d.pl_insol / s_p
    conflict = d.pl_insol.notna() & s_p.notna() & ((ratio > 3) | (ratio < 1 / 3))
    derived = d.pl_insol.isna() & s_p.notna()
    d.loc[derived, "pl_insol"] = s_p[derived]
    d["pl_orbsmax"] = np.where(conflict | d.pl_orbper.isna(), np.sqrt(lum / d.pl_insol), a_p)
    for i in np.flatnonzero(conflict.to_numpy()):
        notes[i].append(f"catalogue insolation is {ratio.iloc[i]:.2g}x the period's: period likely an alias; insolation used")
    for i in np.flatnonzero(derived.to_numpy()):
        notes[i].append("insolation derived from period and stellar parameters (30 % error)")
    d.loc[derived, "pl_insolerr1"] = 0.3 * d.loc[derived, "pl_insol"]
    d.loc[derived, "pl_insolerr2"] = -0.3 * d.loc[derived, "pl_insol"]
    d["pl_orbsmaxerr1"], d["pl_orbsmaxerr2"] = 0.05 * d.pl_orbsmax, -0.05 * d.pl_orbsmax
    d["period_insolation_conflict"] = conflict.to_numpy()
    d["param_notes"] = ["; ".join(x) for x in notes]
    return d


def tess_candidates() -> pd.DataFrame:
    p = pd.read_csv(RESEARCH / "stage_a_transit_model" / "runs" / "v2_final" / "tess_test_predictions.csv")
    p = p[p.true_label == "CANDIDATE"]
    t = pd.read_csv(RESEARCH / "splits" / "toi_tess_candidates_split_v2.csv", low_memory=False)
    t["object_id"] = "TOI " + t.toi.map(lambda x: f"{x:.2f}")
    d = p.merge(t, on="object_id")
    d["p_real"] = d.calibrated_score
    d = complete(d, "tid")
    d["pl_name"], d["mission"], d["host"] = d.object_id, "TESS", "TIC " + d.tid.astype(int).astype(str)
    return d


def k2_candidates() -> pd.DataFrame:
    p = pd.read_csv(RESEARCH / "stage_a_transit_model" / "runs" / "k2_blind" / "k2_predictions.csv")
    p = p[(p.true_label == "CANDIDATE") & ~p.tess_overlap.fillna(False).astype(bool)]
    k = pd.read_csv(RESEARCH / "splits" / "k2_planets_candidates_split.csv", low_memory=False)
    d = p.merge(k, left_on="object_id", right_on="pl_name")
    d["p_real"] = platt(d.score_tess)
    d["tic"] = pd.to_numeric(d.tic_id.astype(str).str.replace("TIC", "").str.strip(), errors="coerce")
    d = complete(d, "tic")
    d["mission"], d["host"] = "K2", d.epic_hostname
    return d


def main() -> int:
    frames = []
    for d in (tess_candidates(), k2_candidates()):
        cols = {c: d[c] if c in d else np.nan for c in
                ["pl_name", "mission", "host", "p_real", "pl_rade", "pl_radeerr1", "pl_radeerr2", "pl_insol", "pl_insolerr1",
                 "pl_insolerr2", "st_teff", "st_tefferr1", "st_tefferr2", "st_rad", "st_raderr1", "st_raderr2", "st_lum",
                 "st_lumerr1", "st_lumerr2", "pl_orbsmax", "pl_orbsmaxerr1", "pl_orbsmaxerr2", "sy_jmag", "pl_orbper",
                 "period_insolation_conflict", "param_notes"]}
        frames.append(pd.DataFrame(cols))
    c = pd.concat(frames, ignore_index=True)
    need = ["pl_rade", "pl_insol", "st_teff", "st_rad", "pl_orbsmax", "sy_jmag"]
    usable = c.dropna(subset=need).copy()
    usable["tran_flag"], usable["discoverymethod"] = 1, "Transit"
    usable["pl_bmassprov"], usable["pl_bmasse"], usable["pl_bmasselim"] = "Estimated", np.nan, 0
    usable["pl_bmasseerr1"] = usable["pl_bmasseerr2"] = np.nan
    usable = usable.reset_index(drop=True)
    usable["st_lum"] = np.nan                                            # luminosity from R and T_eff, as derived above
    phys, draws = tp.monte_carlo(usable, n=2000, return_draws=True)
    score = np.median(bpt.score_draws(draws, bpt.WEIGHTS), axis=1)
    out = usable[["pl_name", "mission", "host", "p_real", "pl_rade", "pl_insol", "st_teff", "sy_jmag", "pl_orbper",
                  "period_insolation_conflict", "param_notes"]].assign(
        p_hz_optimistic=phys.p_hz_optimistic, p_hz_conservative=phys.p_hz_conservative, p_radius_lt_1p5=phys.p_radius_lt_1p5,
        p_terrestrial_atm=draws["terrestrial_atm"].mean(axis=1), tsm_median=phys.tsm_median, stage_c_score=score,
        factor=usable.p_real * score)
    out = out.sort_values("factor", ascending=False)
    out.to_csv(OUT / "candidate_factor.csv", index=False)
    short = out[(out.p_real >= 0.5) & (out.p_hz_optimistic >= 0.5)]
    short.to_csv(OUT / "candidate_shortlist.csv", index=False)
    confirmed = pd.read_csv(OUT / "final_priority_table.csv")
    top10_score = confirmed[confirmed.eligible].priority_score.nlargest(10).min()
    res = {"n_candidates": int(len(c)), "n_with_parameters": int(len(usable)),
           "by_mission": usable.mission.value_counts().to_dict(),
           "n_shortlist": int(len(short)), "n_shortlist_period_alias_flag": int(short.period_insolation_conflict.sum()),
           "shortlist": short.head(25).round(3).to_dict(orient="records"),
           "n_shortlist_above_confirmed_top10_score": int((short.stage_c_score >= top10_score).sum()),
           "confirmed_top10_min_score": round(float(top10_score), 3),
           "expectations": {"C1 shortlist not empty": bool(len(short) > 0)}}
    (OUT / "candidate_factor.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "shortlist"}, indent=2, default=float))
    pd.set_option("display.width", 220)
    print(short.head(25).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
