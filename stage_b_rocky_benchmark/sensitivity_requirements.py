"""Forward-looking result: what precision would be needed to detect an atmospheric band in these systems?

For each benchmark spectrum and evaluable molecule band, from rocky_band_statistics.csv:
  current band precision sigma_delta (error-inflated), in ppm and in scale heights (mu = 18 / 28 / 44)
  required sigma to detect a band of N_H scale heights at 5 sigma: N_H * A_H / 5
  precision improvement factor = sigma_now / sigma_required; transits needed scale as factor^2
    (photon-limited assumption; systematics floors make this a lower bound)
Writes rocky_benchmark/rocky_sensitivity_requirements.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "rocky_benchmark"  # stage_b_rocky_benchmark/ -> root
N_SIGMA = 5.0
FEATURE_H = (2.0, 5.0)  # modest (cloudy / weak band) and strong (clear, strong band) cases, in scale heights


def main() -> None:
    st = pd.read_csv(OUT / "rocky_band_statistics.csv")
    sh = pd.read_csv(OUT / "rocky_scale_heights.csv").set_index("planet")
    st = st[st.evaluable == True].copy()  # noqa: E712
    rows = []
    for r in st.itertuples():
        p = sh.loc[r.planet]
        row = {"file": r.file, "planet": r.planet, "molecule": r.molecule,
               "sigma_band_ppm": r.sigma_delta_inflated_ppm}
        for mu in (18, 28, 44):
            a_h = p[f"A_H_ppm_mu{mu}"]
            row[f"sigma_band_in_H_mu{mu}"] = round(r.sigma_delta_inflated_ppm / a_h, 2)
            for nh in FEATURE_H:
                need = nh * a_h / N_SIGMA
                f = r.sigma_delta_inflated_ppm / need
                row[f"improve_x_{int(nh)}H_mu{mu}"] = round(f, 1)
                row[f"transits_x_{int(nh)}H_mu{mu}"] = round(f ** 2, 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rocky_sensitivity_requirements.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["file", "molecule", "sigma_band_ppm", "sigma_band_in_H_mu28", "improve_x_5H_mu28", "transits_x_5H_mu28",
            "improve_x_2H_mu28", "transits_x_2H_mu28", "transits_x_5H_mu18", "transits_x_5H_mu44"]
    print(df[show].to_string(index=False))
    best = df.loc[df.groupby("file").sigma_band_in_H_mu28.idxmin()]
    print("\nbest band per spectrum (mu=28):")
    print(best[show].to_string(index=False))
    co2 = df[df.molecule == "CO2"]
    print("\nCO2 band (4.05-4.55 um) per spectrum (mu=28):")
    print(co2[show].to_string(index=False))


if __name__ == "__main__":
    main()
