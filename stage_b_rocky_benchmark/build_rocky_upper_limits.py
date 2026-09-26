"""Write rocky_benchmark/rocky_planet_upper_limits.csv: literature per-molecule constraints for the 11 rocky
planets in the Kreidberg & Stevenson (2025) compilation.

SEPARATE from labels/verified_gas_labels.csv and never merged into it. Nothing here is a training label.
There are no positive (1) entries: no rocky-planet molecule detection is claimed by any source paper.

Sources were read on arXiv (HTML/abstract pages) on 2026-09-19. Quotes are the papers' words as retrieved;
numbers should be checked against the journal PDFs before being cited in a publication (column `verify`).

Column meanings
  constraint_type  upper_limit_log_vmr | composition_ruled_out | composition_disfavoured |
                   excludes_low_fraction_mixture | conditional_limit | tentative_not_claimed |
                   consistent_not_constrained | not_stated | reviewed_no_per_molecule_result | not_checked
  nondetection_equivalent  "0" only for an explicit numerical per-molecule upper limit (the same bar that
                   turned TRAPPIST-1 c's 2-sigma partial-pressure limits into 0s in the verified set);
                   otherwise "n/a". Never "1".
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "rocky_benchmark" / "rocky_planet_upper_limits.csv"
VERIFY = "number retrieved from arXiv HTML; confirm against journal PDF before citing"

K24 = ("2024AJ....167...90K", "2401.06043", "Kirk et al. 2024")
S24 = ("2024AJ....168..276S", "2409.07552", "Scarsdale et al. 2024")
G24 = ("2024ApJ...975L..10G", "2408.15855", "Gressier et al. 2024")
C24 = ("2024ApJ...970L...2C", "2406.15136", "Cadieux et al. 2024")
A24 = ("2024AJ....167..216A", "2404.00093", "Alderson et al. 2024")


def row(planet, molecule, ctype, value, units, sigma, src, quote, note="", equiv="n/a", verify=VERIFY):
    bib, arx, ref = src if src else ("", "", "")
    return {"planet": planet, "molecule": molecule, "constraint_type": ctype, "limit_value": value,
            "limit_units": units, "sigma": sigma, "bibcode": bib, "arxiv": arx, "reference": ref,
            "quote": quote, "note": note, "nondetection_equivalent": equiv, "verify": verify}


ROWS = [
    # ---- already reviewed during the 13-planet label work (carried over from verified_gas_labels_excluded.csv) ----
    row("GJ 1132 b", "all five", "reviewed_no_per_molecule_result", "", "", "", None, "",
        "carried over, not re-checked: May+2023 (2310.10711) visits disagree; Bennett+2025 (2508.10579) best fit "
        "is a flat line; no per-molecule limits", verify=""),
    row("GJ 486 b", "all five", "reviewed_no_per_molecule_result", "", "", "", None, "",
        "carried over, not re-checked: water in the planet vs unocculted starspots indistinguishable "
        "(Moran+2023, 2305.00868)", verify=""),
    row("LHS 475 b", "all five", "reviewed_no_per_molecule_result", "", "", "", None, "",
        "carried over, not re-checked: only hydrogen-dominated and cloudless pure-methane atmospheres ruled out "
        "(Lustig-Yaeger+2023, 2301.04191); no per-molecule limits", verify=""),
    row("TRAPPIST-1 b", "all five", "reviewed_no_per_molecule_result", "", "", "", None, "",
        "carried over, not re-checked: Rathcke+2025 (2412.16541) 'bare rock or thin atmosphere' only. "
        "Planet also EXCLUDED from the benchmark (spectrum provenance unresolved)", verify=""),

    # ---- GJ 341 b ----
    row("GJ 341 b", "CH4", "composition_disfavoured", "1-bar CH4 atmosphere", "", "1-3 (reduction-dependent)", K24,
        "Methane atmospheres of 1 bar are disfavored at 3σ, 2.5σ, or 1σ",
        "bulk-composition statement, not an abundance limit"),
    row("GJ 341 b", "CO2", "tentative_not_claimed", "log VMR -2.1 (-1.9/+1.2) in one reduction", "dex", "3.1", K24,
        "the Tiberius reduction implies a weak detection of CO2 (3.1σ)",
        "UNEXPECTED >3 sigma retrieval result, explicitly rejected by the authors: 'the candidate features are seen at "
        "different wavelengths (4.3 μm vs 4.7 μm) ... suggesting that they are not real astrophysical signals'. "
        "Tswift disfavours a CO2 atmosphere at 2σ. NOT a detection."),
    row("GJ 341 b", "H2O", "consistent_not_constrained", "", "", "", K24,
        "water atmospheres ... are consistent with each reduction", "a water-dominated atmosphere is allowed"),
    row("GJ 341 b", "CO", "not_stated", "", "", "", K24, "", "no per-molecule statement found"),
    row("GJ 341 b", "SO2", "not_stated", "", "", "", K24, "", "no per-molecule statement found"),
    row("GJ 341 b", "bulk (H2-rich)", "composition_ruled_out", "<350x solar metallicity", "", ">=3", K24,
        "rules out a low mean molecular weight atmosphere (< 350× solar metallicity) to at least 3σ", ""),

    # ---- L 98-59 c ----
    row("L 98-59 c", "CH4", "composition_ruled_out", "pure CH4 atmosphere", "", "2.8", S24,
        "We rule out a pure CH4 atmosphere at ∼2.8σ", "also: 'the data disfavors a clear CH4 atmosphere (3.2-σ)'"),
    row("L 98-59 c", "CO2", "excludes_low_fraction_mixture", "CO2 fraction <~10% in H2 (MMW <~8 g/mol)", "", "3",
        S24, "the featureless spectrum rules out CO2 concentrations ≲10% (i.e., MMW ≲8 g mol⁻¹) to 3-σ confidence",
        "excludes LOW CO2 fractions (large-scale-height H2 mixtures); this is NOT an upper limit on CO2"),
    row("L 98-59 c", "H2O", "consistent_not_constrained", "", "", "", S24,
        "low water vapor concentrations (e.g. <10%) ... are permitted by the data ... so long as there is a high "
        "cloud deck or the atmosphere is thin", ""),
    row("L 98-59 c", "CO", "not_stated", "", "", "", S24, "", "no per-molecule statement found"),
    row("L 98-59 c", "SO2", "not_stated", "", "", "", S24, "", "no per-molecule statement found"),
    row("L 98-59 c", "bulk (H2-rich)", "composition_ruled_out", "<~300x solar metallicity (MMW <~10)", "", "3", S24,
        "the data rules out metallicities ≲300× solar to 3-σ confidence", ""),

    # ---- L 98-59 d (free-chemistry retrieval; the paper frames the atmosphere itself as tentative "hints") ----
    row("L 98-59 d", "H2O", "upper_limit_log_vmr", "-5.0", "log10 VMR", "not stated", G24, "<−5.0",
        "free-chemistry retrieval (H2-dominated background); sigma convention not given in the retrieved text",
        equiv="0"),
    row("L 98-59 d", "CH4", "upper_limit_log_vmr", "-6.0", "log10 VMR", "not stated", G24, "<−6.0",
        "free-chemistry retrieval; sigma convention not given in the retrieved text", equiv="0"),
    row("L 98-59 d", "CO2", "upper_limit_log_vmr", "-2.5", "log10 VMR", "not stated", G24, "<−2.5",
        "free-chemistry retrieval; sigma convention not given in the retrieved text", equiv="0"),
    row("L 98-59 d", "CO", "upper_limit_log_vmr", "-2.5", "log10 VMR", "not stated", G24, "<−2.5",
        "free-chemistry retrieval; sigma convention not given in the retrieved text", equiv="0"),
    row("L 98-59 d", "SO2", "tentative_not_claimed", "log VMR -5.64 (-4.12/+4.41); <-2.0 with N2", "log10 VMR",
        "not stated", G24, "log(SO2)=−5.64−4.12+4.41",
        "unconstrained in free chemistry (posterior spans ~8 dex); the sulfur 'hint' is carried mainly by H2S"),
    row("L 98-59 d", "H2S (not a target)", "tentative_not_claimed", "log VMR -0.74 (-0.49/+0.14)", "log10 VMR",
        "2.6-5.6 (atmosphere vs flat line; reduction/retrieval-dependent)", G24,
        "deviates from a flat line by 2.6 to 5.6σ, depending on the data reduction and retrieval setup",
        "UNEXPECTED relative to 'no atmospheres detected': the upper end exceeds 3σ, but it is reduction- and "
        "retrieval-dependent and titled 'Hints'; Banerjee+2024 (2408.15707) is an independent analysis. NOT a detection."),

    # ---- LHS 1140 b ----
    row("LHS 1140 b", "H2O", "upper_limit_log_vmr", "-2.94", "log10 VMR", "2", C24,
        "We rule out cloud-free H2O-rich, CH4-rich, and H2-rich atmospheres, with 2σ upper limits of log H₂O<−2.94",
        "NIRISS; cloud-free assumption", equiv="0"),
    row("LHS 1140 b", "CH4", "upper_limit_log_vmr", "-2.78", "log10 VMR", "2", C24, "log CH₄<−2.78",
        "NIRISS; cloud-free assumption", equiv="0"),
    row("LHS 1140 b", "CO2", "conditional_limit", "pure CO2 requires T < 233 K", "K", "2", C24,
        "Pure CO2 atmospheres must have T<233 K (2σ upper limit) to be compatible with our non-detection of CO2 absorption",
        "temperature condition on a pure-CO2 atmosphere, not an abundance limit. Damiano+2024 (NIRSpec) predicts a "
        "~20 ppm CO2 feature at 4.2 um that the data cannot yet detect"),
    row("LHS 1140 b", "CO", "not_stated", "", "", "", C24, "", "no per-molecule statement found (Cadieux+2024, Damiano+2024)"),
    row("LHS 1140 b", "SO2", "not_stated", "", "", "", C24, "", "no per-molecule statement found (Cadieux+2024, Damiano+2024)"),
    row("LHS 1140 b", "N2 (not a target)", "tentative_not_claimed", "N2-dominated favoured", "", "2.3", C24,
        "The former N2-rich atmospheric composition is favored by the data at 2.3 σ from a tentative detection of "
        "N2 Rayleigh scattering", "N2 has no band in 1-5 um; not testable with the band statistic"),
    row("LHS 1140 b", "bulk (H2-rich)", "composition_ruled_out", "H2-rich GCMs", "", ">10", C24,
        "All hydrogen-rich GCMs are formally rejected by the data (fitting for an offset) with a confidence level "
        "exceeding 10 σ", "Damiano+2024 NIRSpec: flat spectrum, no numerical per-molecule limit"),

    # ---- LHS 1140 c ----
    row("LHS 1140 c", "CH4", "upper_limit_log_vmr", "-2.25", "log10 VMR", "2", C24,
        "The data is incompatible with clear CH₄-rich and H₂-rich atmospheres with 2σ upper limits of log CH₄<−2.25",
        "clear-atmosphere assumption", equiv="0"),
    row("LHS 1140 c", "H2O", "not_stated", "", "", "", C24, "", "'A flat solution is favored for LHS 1140 c (2.1 σ)'"),
    row("LHS 1140 c", "CO2", "not_stated", "", "", "", C24, "", "NIRISS only (to 2.8 um): CO2/CO/SO2 bands not covered"),
    row("LHS 1140 c", "CO", "not_stated", "", "", "", C24, "", "NIRISS only (to 2.8 um)"),
    row("LHS 1140 c", "SO2", "not_stated", "", "", "", C24, "", "NIRISS only (to 2.8 um)"),

    # ---- TOI-836 b ----
    row("TOI-836 b", "all five", "not_stated", "", "", "", A24, "",
        "no per-molecule limit: the paper constrains bulk metallicity / mean molecular weight only"),
    row("TOI-836 b", "bulk (H2-rich)", "composition_ruled_out", "<250x solar metallicity at 0.1 bar (MMW <~6)", "",
        "not stated in retrieved text", A24,
        "we are able to rule out metallicities <250×Solar for an opaque pressure level of 0.1 bar, corresponding to "
        "mean molecular weights of ≲6 g mol⁻¹", ""),

    # ---- TRAPPIST-1 h ----
    row("TRAPPIST-1 h", "all five", "not_checked", "", "", "", None, "",
        "EXCLUDED from the benchmark: the Zenodo spectrum's source is unknown, so there is no source paper to check. "
        "The only paper the compilation cites for this planet is Garcia+2022 (HST/WFC3), a different dataset", verify=""),
]


def main() -> None:
    df = pd.DataFrame(ROWS)
    assert not (df.nondetection_equivalent == "1").any(), "no positive entries are expected"
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT.name}: {len(df)} rows, {df.planet.nunique()} planets")
    print(df.groupby("constraint_type").size().to_string())
    print("nondetection-equivalent 0s:", df[df.nondetection_equivalent == "0"][["planet", "molecule", "limit_value", "sigma"]].to_string(index=False))


if __name__ == "__main__":
    main()
