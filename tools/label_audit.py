"""
label_audit.py
==============

A blind human check of the molecule labels, which were extracted by an AI assistant.

    python tools/label_audit.py --make       # writes paper/Label_Audit_Sheet.csv, with the AI labels hidden
    python tools/label_audit.py --score      # after the sheet is filled in: agreement and Cohen's kappa

Fill in column your_label for each row from the linked paper: 1 for a reported detection at 3 sigma or higher, 0 for a reported
non-detection, upper limit, or featureless spectrum across the band, and unknown for tentative or conflicting reports.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

R = Path(__file__).resolve().parents[1]
L = R / "outputs" / "labels"
SHEET = R / "paper" / "Label_Audit_Sheet.csv"


def link(s) -> str:
    m = re.search(r"(?:arXiv:|\()(\d{4}\.\d{4,5})", str(s))
    return f"https://arxiv.org/abs/{m.group(1)}" if m else ""


def labels() -> pd.DataFrame:
    rows = []
    v = pd.read_csv(L / "verified_gas_labels.csv", dtype=str)
    for r in v.itertuples():
        for gas in ("CH4", "CO2"):
            arx = getattr(r, f"{gas}_arxiv", "")
            rows.append((r.planet, gas, str(getattr(r, gas)), str(getattr(r, f"{gas}_bibcode", "")) + " " + str(arx), f"https://arxiv.org/abs/{arx}" if str(arx) not in ("nan", "") else ""))
    for f in ["ch4_labels_extended.csv", "ch4_labels_round2.csv", "ch4_labels_round3.csv"]:
        d = pd.read_csv(L / f, dtype=str)
        rows += [(r.planet, "CH4", r.CH4, r.source, link(r.source)) for r in d.itertuples()]
    d = pd.read_csv(L / "co2_labels.csv", dtype=str)
    rows += [(r.planet, "CO2", r.CO2, r.source, link(r.source)) for r in d.itertuples()]
    t = pd.DataFrame(rows, columns=["planet", "gas", "ai_label", "source", "paper"]).drop_duplicates(["planet", "gas"], keep="last")
    return t[t.ai_label.isin(["0", "1"])].reset_index(drop=True)


def main() -> int:
    t = labels()
    if "--make" in sys.argv:
        s = t.sample(frac=1, random_state=20261031).reset_index(drop=True)
        s = s.drop(columns="ai_label").assign(your_label="", notes="")
        s.to_csv(SHEET, index=False)
        print(f"{len(s)} rows written to {SHEET}")
    if "--score" in sys.argv:
        s = pd.read_csv(SHEET, dtype=str).merge(t[["planet", "gas", "ai_label"]], on=["planet", "gas"])
        s = s[s.your_label.isin(["0", "1", "unknown"])]
        for g, d in list(s.groupby("gas")) + [("all", s)]:
            k = d[d.your_label != "unknown"]
            print(g, "rows", len(d), "agreement", round(float((k.your_label == k.ai_label).mean()), 3), "kappa", round(float(cohen_kappa_score(k.your_label, k.ai_label)), 2), "unknown", int((d.your_label == "unknown").sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
