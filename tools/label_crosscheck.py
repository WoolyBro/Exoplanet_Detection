"""
label_crosscheck.py
===================

An independent check of the CH4 labels in place of a second human labeller: our 43 labelled planets
(outputs/labels/: verified_gas_labels, ch4_labels_extended, _round2, _round3) against the molecules
listed by The Extrasolar Planets Encyclopaedia (exoplanet.eu), compiled by a different team.

exoplanet.eu's `species` field is served by the Paris Observatory EPN-TAP service (table
exoplanet.epn_core). It lists every species reported for a planet: detections, claims and tentative
mentions, from any instrument or epoch, without a significance threshold. Ours require a JWST
transmission paper stating a detection (or a non-detection / upper limit). Disagreements are therefore
expected where the two definitions differ, and each one is classified by reading our cited source.

    python tools/label_crosscheck.py          # uses external_data/exoplanet_eu/exoplanet_eu_species_<date>.csv
    python tools/label_crosscheck.py --fetch  # re-download the species table first
Output: outputs/labels/ch4_label_crosscheck_exoplanet_eu.csv, ..._summary.json
"""

from __future__ import annotations

import argparse
import io
import json
import re
import socket
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1]
LABELS = R / "outputs" / "labels"
EU_DIR = R / "external_data" / "exoplanet_eu"
FILES = ["verified_gas_labels.csv", "ch4_labels_extended.csv", "ch4_labels_round2.csv", "ch4_labels_round3.csv"]
TAP = "http://voparis-tap-planeto.obspm.fr/tap/sync"     # the service refuses https
# Why each disagreement arises, from reading the source cited in our label files (2026-10-18).
REASONS = {
    "HD 189733 b": "exoplanet.eu keeps pre-JWST CH4 claims; JWST gives a 5-sigma upper limit of 0.1 ppm (Fu et al. 2024, Nature 632, 752)",
    "HD 209458 b": "exoplanet.eu keeps a pre-JWST (ground high-resolution) CH4 claim; JWST finds no evidence of CH4 (Xue et al. 2024, ApJL 963, L5)",
    "TOI-270 d": "JWST CH4 at 3.8-4.9 sigma (Holmberg & Madhusudhan 2024, A&A 683, L2) not yet in exoplanet.eu (lists H2O only)",
    "TOI-5205 b": "JWST CH4 ~10 sigma (Canas et al. 2026, AJ 171, 260): planet has no species entry in exoplanet.eu yet",
    "LP 791-18 c": "JWST CH4 ~4.5 sigma (Roy et al. 2025, Nature Astronomy): no species entry in exoplanet.eu yet",
    "LTT 3780 c": "JWST CH4 3.0-4.6 sigma (Rigby et al. 2025, arXiv:2512.15844): no species entry in exoplanet.eu yet",
    "HATS-6 b": "JWST CH4 strong evidence (Guzman Caloca et al. 2026, AJ 172, 193): no species entry in exoplanet.eu yet",
}


def fetch() -> Path:
    socket.setdefaulttimeout(120)
    adql = ("select target_name, alt_target_name, star_name, species, detection_type, modification_date, updated "
            "from exoplanet.epn_core where species is not null and species <> ''")
    url = TAP + "?" + urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql})
    data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research"}), timeout=120).read()
    EU_DIR.mkdir(parents=True, exist_ok=True)
    out = EU_DIR / f"exoplanet_eu_species_{date.today().isoformat()}.csv"
    pd.read_csv(io.BytesIO(data)).to_csv(out, index=False)
    return out


def key(name: str) -> str:
    n = str(name).strip()
    n = re.sub(r"\s+A\s*([b-h])$", r" \1", n)          # "HAT-P-18 Ab" -> "HAT-P-18 b"
    n = re.sub(r"A([b-h])$", r" \1", n)
    return re.sub(r"[^a-z0-9]", "", n.lower())


def main() -> int:
    from sklearn.metrics import cohen_kappa_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    src = fetch() if a.fetch else sorted(EU_DIR.glob("exoplanet_eu_species_*.csv"))[-1]
    lab = pd.concat([pd.read_csv(LABELS / f, dtype=str, keep_default_na=False)[["planet", "CH4"]].assign(label_file=f) for f in FILES])
    lab = lab[lab.CH4.isin(["0", "1"])].drop_duplicates("planet").reset_index(drop=True)
    eu = pd.read_csv(src)
    alt = eu.dropna(subset=["alt_target_name"]).assign(target_name=lambda d: d.alt_target_name)
    idx = pd.concat([eu, alt]).assign(k=lambda d: d.target_name.map(key)).drop_duplicates("k").set_index("k").species
    lab["exoplanet_eu_species"] = lab.planet.map(key).map(idx)
    lab["ours_ch4"] = lab.CH4.astype(int)
    lab["exoplanet_eu_lists_ch4"] = lab.exoplanet_eu_species.fillna("").str.split("#").map(lambda s: "CH4" in s).astype(int)
    lab["agree"] = lab.ours_ch4 == lab.exoplanet_eu_lists_ch4
    lab["reason_if_disagree"] = [("" if ag else REASONS.get(p, "UNEXPLAINED")) for p, ag in zip(lab.planet, lab.agree)]
    listed = lab.exoplanet_eu_species.notna()
    summary = {"source": src.name, "n_labelled": int(len(lab)), "n_ours_ch4": int(lab.ours_ch4.sum()),
               "n_with_exoplanet_eu_species": int(listed.sum()),
               "agreement_all": round(float(lab.agree.mean()), 3),
               "kappa_all": round(float(cohen_kappa_score(lab.ours_ch4, lab.exoplanet_eu_lists_ch4)), 3),
               "agreement_listed_only": round(float(lab[listed].agree.mean()), 3),
               "kappa_listed_only": round(float(cohen_kappa_score(lab[listed].ours_ch4, lab[listed].exoplanet_eu_lists_ch4)), 3),
               "confusion_rows_ours_cols_exoplanet_eu": pd.crosstab(lab.ours_ch4, lab.exoplanet_eu_lists_ch4).to_dict(orient="index"),
               "n_disagree": int((~lab.agree).sum()),
               "n_unexplained": int((lab.reason_if_disagree == "UNEXPLAINED").sum())}
    lab.drop(columns=["CH4"]).to_csv(LABELS / "ch4_label_crosscheck_exoplanet_eu.csv", index=False)
    (LABELS / "ch4_label_crosscheck_exoplanet_eu_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(lab[~lab.agree][["planet", "ours_ch4", "exoplanet_eu_lists_ch4", "reason_if_disagree"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
