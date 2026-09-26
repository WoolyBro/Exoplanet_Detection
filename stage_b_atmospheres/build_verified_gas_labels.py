#!/usr/bin/env python3
"""
build_verified_gas_labels.py
============================

Writes labels/verified_gas_labels.csv: a small, bibcode-traceable replacement for the
quarantined demo file gas_labels_literature.csv (which failed verification).

Scope
  Planets: only those with JWST transmission spectra in the mentor pack
  (04_atmospheric_spectra/metadata/spectra_metadata.csv) whose papers could be checked.
  Molecules: H2O, CO2, CH4, CO, SO2 only (Na/K lie outside a 1-5 um grid, He escape is
  unresolvable at 100 bins, clouds/haze are not a discrete molecule).

Label definitions (every value below was read from the cited paper's arXiv abstract or
text on 2026-09-17; nothing is copied from the demo file)
  1        the paper reports a detection, with every significance it reports >= 3 sigma
           (or several independent papers in the pack confirm it)
  0        the paper explicitly reports a non-detection / upper limit for that molecule,
           within wavelengths the data cover
  unknown  anything else: not addressed, tentative (a reported range dipping below 3 sigma),
           or contradicted between papers analysing the same data

Planets whose papers give no usable per-molecule result are excluded (listed in
EXCLUDED with the reason), rather than guessed.
"""

from pathlib import Path

import pandas as pd

LABELS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "labels"  # stage_b_atmospheres/ -> root
OUT = LABELS_DIR / "verified_gas_labels.csv"
EXCLUDED_OUT = LABELS_DIR / "verified_gas_labels_excluded.csv"
MOLECULES = ["H2O", "CO2", "CH4", "CO", "SO2"]
U = "unknown"

# planet -> molecule -> (value, bibcode, arXiv id, significance/evidence as stated, note)
LABELS = {
    "WASP-39 b": {
        "H2O": (1, "2023Natur.614..659R", "2211.10487", "33 sigma (NIRSpec PRISM)", "also Alderson+2023 21.5 sigma; Ahrer+2023 NIRCam detection"),
        "CO2": (1, "2023Natur.614..659R", "2211.10487", "28 sigma (NIRSpec PRISM)", "also Alderson+2023 28.5 sigma; ERS Team 2022 (2208.11692) 26 sigma"),
        "CH4": (0, "2023Natur.614..659R", "2211.10487", "non-detection", "Ahrer+2023 (2211.10489) NIRCam upper limit"),
        "CO": (1, "2023Natur.614..659R", "2211.10487", "7 sigma (NIRSpec PRISM)", "Carter+2024 (2407.13893) confirms"),
        "SO2": (1, "2023Natur.614..664A", "2211.10488", "4.8 sigma (NIRSpec G395H)", "confirmed by Powell+2024 MIRI (2407.07965) and Carter+2024; PRISM alone gave 2.7 sigma"),
    },
    "WASP-107 b": {
        "H2O": (1, "2024Natur.630..836W", "2405.11018", "21 sigma", "Dyrek+2024 MIRI (2311.12515) ~12 sigma"),
        "CO2": (1, "2024Natur.630..836W", "2405.11018", "29 sigma", ""),
        "CH4": (1, "2024Natur.630..836W", "2405.11018", "5 sigma", "Dyrek+2024 MIRI-only (5-12 um, no strong CH4 band) reported no CH4; Sing+2024 NIRSpec also reports CH4"),
        "CO": (1, "2024Natur.630..836W", "2405.11018", "7 sigma", ""),
        "SO2": (1, "2024Natur.630..836W", "2405.11018", "9 sigma", "Dyrek+2024 MIRI 9 sigma"),
    },
    "HD 189733 b": {
        "H2O": (1, "2024Natur.632..752F", "2407.06163", "13.4 sigma", "Zhang+2025 (2410.22398) also detects"),
        "CO2": (1, "2024Natur.632..752F", "2407.06163", "11.2 sigma", "Zhang+2025 also detects"),
        "CH4": (0, "2024Natur.632..752F", "2407.06163", "5 sigma upper limit 0.1 ppm", "Zhang+2025: depleted"),
        "CO": (1, "2024Natur.632..752F", "2407.06163", "5 sigma", "Zhang+2025 also detects"),
        "SO2": (U, "", "", "not addressed in abstract", ""),
    },
    "HD 209458 b": {
        "H2O": (1, "2024ApJ...963L...5X", "2310.03245", "strong features (sigma not stated in abstract)", ""),
        "CO2": (1, "2024ApJ...963L...5X", "2310.03245", "strong features (sigma not stated in abstract)", ""),
        "CH4": (0, "2024ApJ...963L...5X", "2310.03245", "upper limit", "title: no evidence of CH4"),
        "CO": (U, "", "", "not addressed in abstract", ""),
        # Updated 2026-09-17 from a user-supplied literature check (not re-verified here).
        "SO2": (0, "2024ApJ...963L...5X", "2310.03245", "non-detection at 4.05 micron",
                "Same paper (Xue+2024) explicitly states non-detection of SO2 at 4.05 micron."),
    },
    "GJ 3470 b": {
        "H2O": (1, "2024ApJ...970L..10B", "2406.04450", ">3 sigma", "NIRCam + archival HST/Spitzer"),
        "CO2": (1, "2024ApJ...970L..10B", "2406.04450", ">3 sigma", ""),
        "CH4": (1, "2024ApJ...970L..10B", "2406.04450", ">3 sigma", ""),
        "CO": (U, "", "", "not addressed in abstract", ""),
        "SO2": (1, "2024ApJ...970L..10B", "2406.04450", ">3 sigma", ""),
    },
    "K2-18 b": {
        "H2O": (0, "2023ApJ...956L..13M", "2309.05566", "95% upper limit", "limit applies to the stratosphere (<~100 mbar); earlier HST water feature attributed to CH4"),
        # Updated 2026-09-17 from a user-supplied literature check (not re-verified here).
        "CO2": (0, "arXiv:2501.18477", "2501.18477", "2 sigma upper limit log10 CO2 < -1.58",
                "Original CO2=1 from Madhusudhan+2023 (single-instrument). Superseded by a 2025 reanalysis combining "
                "NIRISS+NIRSpec (arXiv:2501.18477), which finds neither CO2 nor DMS robustly detected - 2 sigma upper "
                "limit log10 CO2 < -1.58. Treated as a non-detection."),
        "CH4": (1, "2023ApJ...956L..13M", "2309.05566", "4.7-5.0 sigma", ""),
        "CO": (0, "2023ApJ...956L..13M", "2309.05566", "not detected", "despite expected features in 0.9-5.2 um"),
        "SO2": (U, "", "", "no observational claim", ""),
    },
    "HAT-P-18 b": {
        "H2O": (1, "2024MNRAS.528.3354F", "2310.14950", "12.5 sigma", "Fu+2022 (2211.13761) also detects water"),
        "CO2": (1, "2024MNRAS.528.3354F", "2310.14950", "7.3 sigma", ""),
        "CH4": (U, "2024MNRAS.528.3354F", "2310.14950", "conflicting", "Fournier-Tondreau+2024: not detected (log CH4 < -6, 2 sigma); Fu+2022: moderate evidence (Bayes factor 3.79) on the same NIRISS data"),
        "CO": (U, "", "", "not addressed; NIRISS 0.6-2.8 um", ""),
        "SO2": (U, "", "", "not addressed; NIRISS 0.6-2.8 um", ""),
    },
    "HAT-P-26 b": {
        "H2O": (1, "2025AJ....170..292G", "2509.16082", "ln B = 4.1", "authors: detected with high confidence"),
        "CO2": (1, "2025AJ....170..292G", "2509.16082", "ln B = 85.6", ""),
        "CH4": (U, "", "", "not addressed in abstract", ""),
        "CO": (U, "2025AJ....170..292G", "2509.16082", "marginal (ln B < 0.5)", ""),
        "SO2": (1, "2025AJ....170..292G", "2509.16082", "ln B = 13.5", ""),
    },
    "TRAPPIST-1 c": {
        "H2O": (0, "2025ApJ...979L...5R", "2409.19333", "partial pressure > ~10 mbar disfavoured at 2 sigma", "stellar contamination modelled jointly"),
        "CO2": (U, "", "", "not constrained (NIRISS 0.6-2.8 um)", ""),
        "CH4": (0, "2025ApJ...979L...5R", "2409.19333", "partial pressure > ~10 mbar disfavoured at 2 sigma", ""),
        "CO": (0, "2025ApJ...979L...5R", "2409.19333", "partial pressure > ~10 mbar disfavoured at 2 sigma", ""),
        "SO2": (U, "", "", "not addressed", ""),
    },
    "TOI-270 d": {
        "H2O": (U, "2024A&A...683L...2H", "2403.03244", "1.6-4.4 sigma depending on analysis", "tentative -> not labelled"),
        "CO2": (U, "2024A&A...683L...2H", "2403.03244", "2.9-3.9 sigma depending on analysis", "range dips below 3 sigma -> not labelled 1"),
        "CH4": (1, "2024A&A...683L...2H", "2403.03244", "3.8-4.9 sigma", ""),
        "CO": (U, "", "", "not addressed in abstract", ""),
        "SO2": (U, "", "", "not addressed in abstract", ""),
    },
    "WASP-80 b": {
        "H2O": (U, "", "", "not addressed in abstract", ""),
        "CO2": (U, "", "", "not addressed in abstract", ""),
        "CH4": (1, "2023Natur.623..709B", "2309.04042", "> 6 sigma", "NIRCam 2.4-4.0 um transmission and emission"),
        "CO": (U, "", "", "not addressed in abstract", ""),
        "SO2": (U, "", "", "not addressed in abstract", ""),
    },
    "WASP-52 b": {
        "H2O": (1, "2025MNRAS.539..422F", "2412.17072", "10.8 sigma", "NIRISS 0.6-2.8 um"),
        "CO2": (U, "", "", "not addressed in abstract", ""),
        "CH4": (U, "", "", "not addressed in abstract", ""),
        "CO": (U, "", "", "not addressed in abstract", ""),
        "SO2": (U, "", "", "not addressed in abstract", ""),
    },
    "WASP-17 b": {
        "H2O": (1, "2025AJ....169...86L", "2412.03675", "multiple absorption features (sigma not stated in abstract)", "Grant+2023 MIRI (2310.08637): H2O depleted, not absent"),
        "CO2": (U, "", "", "not addressed in abstract", ""),
        "CH4": (U, "", "", "not addressed in abstract", ""),
        "CO": (U, "", "", "not addressed in abstract", ""),
        "SO2": (U, "", "", "not addressed in abstract", ""),
    },
}

EXCLUDED = {
    "TRAPPIST-1 b": "no per-molecule result: Rathcke+2025 (2412.16541) says only 'bare rock or thin atmosphere'; Greene+2023 is a 15 um eclipse point",
    "GJ 1132 b": "ambiguous: May+2023 (2310.10711) visits disagree; Bennett+2025 (2508.10579) best fit is a flat line, no per-molecule limits",
    "GJ 486 b": "water in planet vs unocculted starspots indistinguishable (Moran+2023, 2305.00868)",
    "GJ 1214 b": "CO2/CH4 only 'possible' at 3.3-3.6 sigma pending more data (Schlawin+2024, 2410.10183); Kempton+2023 (2305.06240) H2O from emission",
    "LHS 475 b": "only hydrogen-dominated and cloudless pure-methane atmospheres ruled out (Lustig-Yaeger+2023, 2301.04191); no per-molecule limits",
    "WASP-121 b": "abstract reports H2O dissociation and SiO, no explicit detection of the five target molecules (Gapp+2025, 2506.02199)",
    "WASP-127 b": "Fu+2025 (2025ApJ...986....1F) could not be found/verified",
    "WASP-43 b": "evidence is from MIRI emission/phase curve, not transmission (Bell+2024, 2401.13027)",
    "WASP-96 b": "Radica+2023 (2305.17001) abstract makes no molecule claims; Wang+2026 not verified",
}


def main() -> None:
    rows = []
    for planet, mols in LABELS.items():
        row = {"planet": planet}
        for m in MOLECULES:
            value, bibcode, arxiv, sig, note = mols[m]
            row[m] = value
            row[f"{m}_bibcode"] = bibcode
            row[f"{m}_arxiv"] = arxiv
            row[f"{m}_significance"] = sig
            row[f"{m}_note"] = note
        rows.append(row)
    table = pd.DataFrame(rows)
    # every non-unknown label must carry a bibcode and arXiv id
    for m in MOLECULES:
        known = table[m] != U
        assert (table.loc[known, f"{m}_bibcode"] != "").all() and (table.loc[known, f"{m}_arxiv"] != "").all(), m
    table.to_csv(OUT, index=False)
    pd.DataFrame([{"planet": p, "reason": r} for p, r in EXCLUDED.items()]).to_csv(EXCLUDED_OUT, index=False)

    print(f"wrote {OUT.name}: {len(table)} planets")
    counts = pd.DataFrame({m: table[m].astype(str).value_counts() for m in MOLECULES}).reindex(["1", "0", U]).fillna(0).astype(int)
    print(counts.to_string())
    print(f"wrote {EXCLUDED_OUT.name}: {len(EXCLUDED)} planets excluded")


if __name__ == "__main__":
    main()
