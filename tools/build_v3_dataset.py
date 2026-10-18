"""
build_v3_dataset.py
===================

The data of vetter v3 (eda/VETTER_V3_PROTOCOL.md): what the v2 vetter lacked, measured on 2026-10-18.

  1. REAL FALSE ALARMS (NTP, non-transiting phenomena). DR25 TCEs that never became KOIs. v2 trained on
     KOIs only - signals that had already passed the pipeline's transit-like triage - and so let 24 % of
     inverted-light-curve false alarms through (eda/DR25_SIMULATED_RESULTS.md).
  2. DIAGNOSTICS. Raw Robovetter measurements per TCE (centroid score, ghost/halo ratio, odd/even and
     secondary-eclipse significances, Modelshift, LPP and CHASES shape metrics, SES/MES ratios), from the
     public Robovetter input tables (nasa/kepler-robovetter). The Robovetter's DECISION flags
     (ephem_disp, cent_disp, cflag*, iflag*) are excluded so the model cannot copy its verdicts.
  3. FRESH SIMULATED TESTS on held-out stars: INJ1 (planets), INV (false alarms), INJ2 (transits on a
     neighbouring star: the centroid test) and INJ3 (eclipsing-binary mimics).

Split rule, no leakage by construction:
  * NTP TCEs are taken only from stars that host no KOI.
  * Signal families (split_signal_families.signal_twins: same period within 1e-4 and coincident epoch)
    are built over KOIs + NTP together. An NTP family that contains a KOI inherits that KOI's v2 split;
    every other family is assigned train/val/test 70/15/15 by a seeded draw.
  * Simulated test signals come only from stars that are neither KOI hosts nor in any NTP family of
    train or val, and none of them is in the 2026-10-18 DR25 benchmark sample.

    python tools/build_v3_dataset.py [--fetch]
Output: external_data/kepler_dr25_simulated/v3/{ntp_sample.csv, eval_{inj1,inv,inj2,inj3}.csv, diagnostics.csv, summary.json}
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
SIM = R / "external_data" / "kepler_dr25_simulated"
RVIN = SIM / "robovetter_inputs"
OUT = SIM / "v3"
SETS = ["obs", "inj1", "inj2", "inj3", "inv"]
SEED = 20261019
N_NTP, N_EVAL = 6000, 1000
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
# raw measurements only; decision flags are deliberately absent
FEATURES = ["cent_score", "halo_ghost", "oesig_dv", "mod_oe_dv", "mod_sig_pri_dv", "mod_sig_sec_dv", "mod_sig_ter_dv",
            "mod_sig_pos_dv", "mod_fred_dv", "mod_fa1_dv", "mod_fa2_dv", "mod_depth_sec_dv", "mod_ph_sec_dv", "alb_dv",
            "sdepthsig_dv", "lpp_dv", "all_tran_chases", "sweet_snr", "shape_metric", "ses_to_mes", "mes", "new_mes",
            "nrealt", "ngoodt", "period", "duration", "depth"]


def fetch() -> None:
    RVIN.mkdir(parents=True, exist_ok=True)
    for s in SETS:
        dest = RVIN / f"kplr_dr25_{s}_robovetter_input.txt"
        if dest.is_file():
            continue
        tgz = RVIN / f"{s}.tar.gz"
        urllib.request.urlretrieve(f"https://github.com/nasa/kepler-robovetter/raw/master/kplr_dr25_{s}_robovetter_input.tar.gz", tgz)
        with tarfile.open(tgz) as t:
            t.extractall(RVIN, filter="data")
        tgz.unlink()


def robovetter(s: str) -> pd.DataFrame:
    from astropy.io import ascii
    return ascii.read(RVIN / f"kplr_dr25_{s}_robovetter_input.txt", format="ipac").to_pandas()


def main() -> int:
    import split_signal_families as ssf

    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    koi = pd.read_csv(R / "splits" / "koi_cumulative_split_v2.csv", low_memory=False,
                      usecols=["kepoi_name", "kepid", "koi_tce_plnt_num", "koi_period", "koi_time0bk", "koi_duration", "split"])
    koi_stars = set(koi.kepid)
    obs = robovetter("obs")
    ntp = obs[~obs.KIC.isin(koi_stars)].copy()            # TCEs on stars that host no KOI
    ntp = ntp[ntp.period.notna() & ntp.epoch.notna() & (ntp.duration > 0)]

    # signal families over KOIs + NTP together
    k = koi.rename(columns={"kepid": "star", "koi_period": "P", "koi_time0bk": "T", "koi_duration": "D"})
    k = k.assign(id=k.kepoi_name, kind="KOI")[["id", "star", "P", "T", "D", "split", "kind"]]
    n = ntp.assign(id=ntp.TCE_ID + "_OBS", star=ntp.KIC, P=ntp.period, T=ntp.epoch, D=ntp.duration, split=None, kind="NTP")
    allc = pd.concat([k, n[["id", "star", "P", "T", "D", "split", "kind"]]], ignore_index=True)
    cfg = {"star": "star", "period": "P", "epoch": "T", "duration": "D", "duration_unit_h": True}
    fam, n_pairs = ssf.signal_families(allc, cfg)
    allc["family"] = fam
    # NTP links can chain KOI families of DIFFERENT v2 splits into one family (e.g. one 0.57 d contaminating
    # binary seen on hundreds of stars). Their NTP rows would bridge splits, so they are not used at all.
    koi_splits = allc[allc.kind == "KOI"].groupby("family").split.nunique()
    bridging = set(koi_splits[koi_splits > 1].index)
    n_bridging = int(((allc.kind == "NTP") & allc.family.isin(bridging)).sum())
    allc = allc[~((allc.kind == "NTP") & allc.family.isin(bridging))].reset_index(drop=True)
    fam_split = allc[allc.kind == "KOI"].groupby("family").split.agg(lambda s: s.iloc[0])
    families = sorted(set(allc[allc.kind == "NTP"].family) - set(fam_split.index))
    draw = rng.choice(list(SPLIT_FRACTIONS), size=len(families), p=list(SPLIT_FRACTIONS.values()))
    fam_split = pd.concat([fam_split, pd.Series(draw, index=families)])
    allc["split"] = allc.family.map(fam_split)
    n = allc[allc.kind == "NTP"].copy()
    inherited = int(allc[allc.kind == "NTP"].family.isin(set(allc[allc.kind == "KOI"].family)).sum())

    # sample NTP, proportionally to the split sizes
    pick = pd.concat([g.sample(n=min(len(g), int(round(N_NTP * SPLIT_FRACTIONS[sp]))), random_state=SEED)
                      for sp, g in n.groupby("split")])
    ntp_s = ntp.assign(id=ntp.TCE_ID + "_OBS").merge(pick[["id", "split", "family"]], on="id")
    ntp_s.insert(0, "set", "OBS")
    ntp_s.to_csv(OUT / "ntp_sample.csv", index=False)

    # held-out stars for the simulated tests
    used_fams = set(n[n.split.isin(["train", "val"])].family)
    busy = set(n[n.family.isin(used_fams)].star) | koi_stars
    old = set()
    for s in ["inj1", "inv"]:
        f = SIM / f"sample_{s}.csv"
        if f.is_file():
            old |= set(pd.read_csv(f).TCE_ID)
    summary = {"seed": SEED, "obs_tces": int(len(obs)), "ntp_pool": int(len(ntp)), "twin_pairs_koi_ntp": int(n_pairs),
               "ntp_in_koi_families": inherited, "ntp_dropped_bridging_koi_splits": n_bridging,
               "ntp_sample_by_split": ntp_s.split.value_counts().to_dict(), "eval": {}}
    for s in ["inj1", "inv", "inj2", "inj3"]:
        t = robovetter(s)
        t = t[~t.KIC.isin(busy) & ~t.TCE_ID.isin(old) & t.period.notna() & t.epoch.notna() & (t.duration > 0)]
        e = t.sample(n=min(N_EVAL, len(t)), random_state=SEED).sort_values(["KIC", "TCE_ID"])
        e.insert(0, "set", s.upper())
        e.to_csv(OUT / f"eval_{s}.csv", index=False)
        summary["eval"][s] = {"n": int(len(e)), "stars": int(e.KIC.nunique()), "pool": int(len(t))}

    # diagnostics for every row the model will see: KOIs (via KIC + planet number), NTP and the tests
    keyed = []
    km = koi.merge(obs, left_on=["kepid", "koi_tce_plnt_num"], right_on=["KIC", "pn"], how="left")
    keyed.append(km.assign(object_id=km.kepoi_name)[["object_id"] + FEATURES])
    keyed.append(ntp_s.assign(object_id=ntp_s.id)[["object_id"] + FEATURES])
    for s in ["inj1", "inv", "inj2", "inj3"]:
        e = pd.read_csv(OUT / f"eval_{s}.csv")
        keyed.append(e.assign(object_id=e.TCE_ID + "_" + s.upper())[["object_id"] + FEATURES])
    diag = pd.concat(keyed, ignore_index=True)
    diag.to_csv(OUT / "diagnostics.csv", index=False)
    summary["diagnostics_rows"] = int(len(diag))
    summary["koi_without_dr25_tce"] = int(km.KIC.isna().sum())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=int))
    print(json.dumps(summary, indent=2, default=int))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
