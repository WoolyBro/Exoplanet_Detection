"""
eda_habitability.py
===================

Read-only EDA on the Stage 2 (habitability / atmosphere) data, run before any
labeling rule or preprocessing is written.

Data access goes through data_manifest.py only:
  * PyATMOS + INARA  -> load_stage2_catalogs()
  * Habitable World Catalogue -> its STAGE_2_REFERENCE_ONLY manifest entry

Outputs (Research/eda/):
  pyatmos_gas_distributions.png   gas mixing-ratio distributions (Steps 1-2)
  pyatmos_vs_inara_gases.png      INARA vs PyATMOS comparison     (Step 3)
  mixing_ratio_sums.png           per-run gas sums                (Steps 4-5)
  pyatmos_o2_co2_scatter.png      O2 vs CO2 coloured by T         (Steps 6-8)
  hwc_habitability.png            catalogue habitability flags    (Step 9)
Printed output is appended to run_log.txt. No input file is modified.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root: shared modules
import data_manifest as dm  # noqa: E402
from research_log import banner, run_logged  # noqa: E402

try:
    from scipy.stats import ks_2samp, spearmanr
except ImportError:  # scipy is optional; the KS test is then skipped
    ks_2samp = spearmanr = None

RESEARCH_DIR = dm.RESEARCH_DIR
EDA_DIR = RESEARCH_DIR / "eda"

# Illustrative Earth reference (not a rule): modern O2 ~21%, CO2 ~0.04%.
EARTH = {"O2": 0.21, "CO2": 4.2e-4}
SUM_TOLERANCE = 0.05  # flag rows whose gas sum deviates from 1.0 by more than this

# ---- chart styling (reference palette, light mode) ------------------------ #
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"  # PyATMOS, INARA
# Sequential ramp starts at step 250 so the coolest cells stay visible on the surface.
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False,
})


def needs_log_scale(s: pd.Series) -> bool:
    """Log x-axis when strictly positive and spanning >= 2 orders of magnitude."""
    return bool(s.min() > 0 and s.max() / s.min() >= 100)


def save(fig, name: str) -> None:
    path = EDA_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(RESEARCH_DIR)}")


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"\n\n{'#' * 88}\n# STAGE 2 HABITABILITY EDA "
          f"(eda_habitability.py, run {datetime.now():%Y-%m-%d %H:%M:%S})\n{'#' * 88}")
    EDA_DIR.mkdir(exist_ok=True)

    banner("LOAD via data_manifest.load_stage2_catalogs()")
    frames = dm.load_stage2_catalogs()
    pyatmos, inara = frames["pyatmos_final.csv"], frames["psg_models.csv"]
    print(f"\n  PyATMOS columns: {list(pyatmos.columns)}")
    print(f"  INARA columns  : {list(inara.columns)}")

    # Gas columns are taken from the printed names above: PyATMOS stores surface
    # mixing ratios as concentration_<gas>; INARA stores bare gas names.
    py_gas_cols = [c for c in pyatmos.columns if c.startswith("concentration_")]
    py_gases = [c.removeprefix("concentration_") for c in py_gas_cols]
    inara_gases = [c for c in ["H2O", "CO2", "O2", "N2", "CH4", "N2O", "CO", "O3",
                               "SO2", "NH3", "C2H6", "NO2"] if c in inara.columns]
    print(f"\n  PyATMOS gas columns ({len(py_gases)}): {py_gas_cols}")
    print(f"  INARA gas columns   ({len(inara_gases)}): {inara_gases}")
    print(f"  requested gases missing from PyATMOS: {[g for g in ['O2', 'CO2', 'H2O', 'N2', 'CH4'] if g not in py_gases]}")

    # ------------------------------------------------------------------ 3.1 #
    banner("STEP 1: PyATMOS gas mixing-ratio summary statistics")
    py = pyatmos[py_gas_cols].rename(columns=dict(zip(py_gas_cols, py_gases)))
    stats = py.agg(["min", "max", "mean", "median", "std", "skew"]).T
    stats["distinct values"] = py.nunique()
    stats["log x?"] = [needs_log_scale(py[g]) for g in py_gases]
    print(stats.to_string(float_format=lambda v: f"{v:.4g}"))

    print("\n  Are concentration_* just the grid inputs (input_*)?")
    for g in py_gases:
        same = int((pyatmos[f"concentration_{g}"] == pyatmos[f"input_{g}"]).sum())
        print(f"    {g:<4} identical in {same:>6} / {len(pyatmos)} rows; "
              f"input_{g} distinct values: {pyatmos[f'input_{g}'].nunique()}")

    banner("STEP 2: Gas distribution figure")
    fig, axes = plt.subplots(1, len(py_gases), figsize=(3.1 * len(py_gases), 3.0))
    for ax, g in zip(axes, py_gases):
        counts = py[g].value_counts().sort_index()
        # Values sit on a discrete grid, so each distinct value gets its own bar
        # (a histogram at the data's native resolution).
        ax.vlines(counts.index, 0, counts.values, color=SERIES_1, linewidth=2.5, capstyle="round")
        if needs_log_scale(py[g]):
            ax.set_xscale("log")
        ax.set_title(g)
        ax.set_xlabel("surface mixing ratio" + (" (log)" if needs_log_scale(py[g]) else ""))
        ax.set_ylim(bottom=0)
        if not needs_log_scale(py[g]):
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    axes[0].set_ylabel("runs")
    fig.suptitle(f"PyATMOS surface gas mixing ratios, {len(py):,} runs (one bar per distinct grid value)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "pyatmos_gas_distributions.png")

    banner("STEP 3: INARA (30 rows) vs PyATMOS, shared gases")
    shared = [g for g in py_gases if g in inara_gases]
    print(f"  shared gas columns: {shared}  (PyATMOS-only: {sorted(set(py_gases) - set(shared))}, "
          f"INARA-only: {sorted(set(inara_gases) - set(shared))})")
    rows = []
    for g in shared:
        a, b = py[g], inara[g]
        inside = ((b >= a.min()) & (b <= a.max())).mean()
        # Percentile of each INARA value within PyATMOS; a representative draw
        # would spread these roughly evenly around 50.
        pct = np.array([(a <= v).mean() * 100 for v in b])
        row = {
            "gas": g,
            "PyATMOS median": a.median(), "PyATMOS range": f"{a.min():.3g}-{a.max():.3g}",
            "INARA median": b.median(), "INARA range": f"{b.min():.3g}-{b.max():.3g}",
            "INARA in PyATMOS range": f"{inside:.0%}",
            "median INARA percentile": np.median(pct),
        }
        if ks_2samp is not None:
            row["KS p-value"] = ks_2samp(a, b).pvalue
        rows.append(row)
    comp = pd.DataFrame(rows).set_index("gas")
    print(comp.to_string(float_format=lambda v: f"{v:.3g}"))

    print("\n  Physical context (not gases):")
    ctx = pd.DataFrame({
        "PyATMOS": [f"{pyatmos.temperature_kelvin.min():.0f}-{pyatmos.temperature_kelvin.max():.0f}",
                    f"{pyatmos.pressure_bar.min():.2f}-{pyatmos.pressure_bar.max():.2f}"],
        "INARA": [f"{inara.planet_surface_temperature_Kelvin.min():.0f}-{inara.planet_surface_temperature_Kelvin.max():.0f}",
                  f"{inara.planet_surface_pressure_bars.min():.2f}-{inara.planet_surface_pressure_bars.max():.2f}"],
    }, index=["surface temperature (K)", "surface pressure (bar)"])
    print(ctx.to_string())

    fig, axes = plt.subplots(1, len(shared), figsize=(3.2 * len(shared), 3.1), sharey=True)
    for ax, g in zip(axes, shared):
        for series, color, label, marker in [(py[g], SERIES_1, "PyATMOS", None),
                                             (inara[g], SERIES_2, "INARA (30)", "o")]:
            x = np.sort(series.to_numpy())
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(x, y, where="post", color=color, linewidth=2, label=label)
            if marker:
                ax.plot(x, y, marker, color=color, markersize=4,
                        markeredgecolor=SURFACE, markeredgewidth=1)
        combined = pd.concat([py[g], inara[g]])
        if needs_log_scale(combined):
            ax.set_xscale("log")
        ax.set_title(g)
        ax.set_xlabel("mixing ratio" + (" (log)" if needs_log_scale(combined) else ""))
    axes[0].set_ylabel("cumulative fraction of rows")
    axes[0].legend(loc="upper left")
    fig.suptitle("INARA vs PyATMOS: cumulative distributions of shared gases",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "pyatmos_vs_inara_gases.png")

    # ------------------------------------------------------------------ 3.2 #
    banner("STEP 4: PyATMOS - do gas mixing ratios sum to ~1?")
    py_sum = py.sum(axis=1)
    py_input_sum = pyatmos[[f"input_{g}" for g in py_gases]].sum(axis=1)
    for label, s in [("sum of concentration_* (5 gases)", py_sum),
                     ("sum of input_* (5 gases, input H2O)", py_input_sum)]:
        dev = (s - 1).abs() > SUM_TOLERANCE
        print(f"\n  {label}")
        print("    " + s.describe(percentiles=[.01, .25, .5, .75, .99]).to_string(float_format=lambda v: f"{v:.4f}")
              .replace("\n", "\n    "))
        print(f"    rows deviating from 1.0 by > {SUM_TOLERANCE:.0%}: {int(dev.sum())} / {len(s)} ({dev.mean():.2%})")
        print(f"    rows with sum > 1.0: {int((s > 1).sum())}")
    print(f"\n  unaccounted remainder (1 - concentration sum): median {1 - py_sum.median():.3f}, "
          f"range {1 - py_sum.max():.3f}-{1 - py_sum.min():.3f}")
    flagged = pyatmos.loc[(py_sum - 1).abs() > SUM_TOLERANCE, ["hash"] + py_gas_cols].assign(gas_sum=py_sum)
    print("  sample of flagged rows:")
    print(flagged.head(5).to_string(index=False))

    banner("STEP 5: INARA - do gas mixing ratios sum to ~1?")
    in_sum = inara[inara_gases].sum(axis=1)
    in_dev = (in_sum - 1).abs() > SUM_TOLERANCE
    print(f"  sum over {len(inara_gases)} gases: min {in_sum.min():.6f}, max {in_sum.max():.6f}, "
          f"mean {in_sum.mean():.6f}")
    print(f"  rows deviating from 1.0 by > {SUM_TOLERANCE:.0%}: {int(in_dev.sum())} / {len(inara)}")
    top = inara[inara_gases].mean().sort_values(ascending=False)
    print("  mean mixing ratio by gas: " + ", ".join(f"{g} {v:.3f}" for g, v in top.items()))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    axes[0].hist(py_sum, bins=60, color=SERIES_1, edgecolor=SURFACE, linewidth=0.5)
    axes[0].set_title(f"PyATMOS: sum of {len(py_gases)} gas columns")
    axes[0].set_ylabel("runs")
    axes[1].hist(in_sum, bins=np.linspace(0.9, 1.1, 41), color=SERIES_2, edgecolor=SURFACE, linewidth=0.5)
    axes[1].set_title(f"INARA: sum of {len(inara_gases)} gas columns")
    axes[1].set_ylabel("rows")
    for ax in axes:
        ax.axvspan(1 - SUM_TOLERANCE, 1 + SUM_TOLERANCE, color=GRID, alpha=0.6, zorder=0, linewidth=0)
        ax.axvline(1, color=INK_2, linewidth=1, linestyle="--")
        ax.set_xlabel("sum of mixing ratios (band = 1.0 ± 5%)")
    axes[0].set_xlim(0, 1.1)
    fig.tight_layout()
    save(fig, "mixing_ratio_sums.png")

    # ------------------------------------------------------------------ 3.3 #
    banner("STEP 6: O2 vs CO2 scatter, coloured by surface temperature")
    print(f"  colour candidates present: temperature_kelvin={'temperature_kelvin' in pyatmos}, "
          f"pressure_bar={'pressure_bar' in pyatmos}")
    print(f"  temperature_kelvin: {pyatmos.temperature_kelvin.min():.1f}-{pyatmos.temperature_kelvin.max():.1f} K "
          f"(std {pyatmos.temperature_kelvin.std():.1f});  pressure_bar: {pyatmos.pressure_bar.min():.3f}-"
          f"{pyatmos.pressure_bar.max():.3f} (std {pyatmos.pressure_bar.std():.3f}) -> colour by temperature")
    if spearmanr is not None:
        drivers = {c: spearmanr(pyatmos[c], pyatmos.temperature_kelvin).statistic
                   for c in [f"input_{g}" for g in py_gases] + ["pressure_bar"]}
        print("  Spearman correlation with temperature_kelvin: "
              + ", ".join(f"{k} {v:+.2f}" for k, v in drivers.items()))

    cells = (pyatmos.groupby(["concentration_CO2", "concentration_O2"])
             .agg(runs=("hash", "size"), t_median=("temperature_kelvin", "median"),
                  t_min=("temperature_kelvin", "min"), t_max=("temperature_kelvin", "max"))
             .reset_index())
    print(f"  distinct (CO2, O2) grid cells: {len(cells)}; runs per cell: "
          f"median {cells.runs.median():.0f}, max {cells.runs.max()}")
    print(f"  within-cell temperature spread (max-min): median {(cells.t_max - cells.t_min).median():.1f} K")

    # ---- Step 7: candidate threshold rules (illustrative only) ----------- #
    banner("STEP 7: Candidate threshold rules (illustrative, NOT final)")
    o2, co2, t = pyatmos.concentration_O2, pyatmos.concentration_CO2, pyatmos.temperature_kelvin
    rules = {
        "R1  O2 >= 0.10 AND CO2 <= 0.01": (o2 >= 0.10) & (co2 <= 0.01),
        "R1' O2 >= 0.10 AND CO2 <  0.01 (strict)": (o2 >= 0.10) & (co2 < 0.01),
        "R2  0.15 <= O2 <= 0.30 AND CO2 <= 0.05": o2.between(0.15, 0.30) & (co2 <= 0.05),
        "R3  R1 AND 273 <= T <= 323 K": (o2 >= 0.10) & (co2 <= 0.01) & t.between(273, 323),
    }
    rule_table = pd.DataFrame({
        "runs": {k: int(v.sum()) for k, v in rules.items()},
        "% of runs": {k: round(100 * v.mean(), 2) for k, v in rules.items()},
    })
    print(rule_table.to_string())
    print(f"\n  CO2 grid values near the R1 boundary: {sorted(co2.unique())[:4]}")
    print(f"  runs at CO2 == 0.0004 (Earth-like): {int((co2 == 4e-4).sum())} ({(co2 == 4e-4).mean():.1%}); "
          f"at CO2 == 0.01: {int((co2 == 0.01).sum())} ({(co2 == 0.01).mean():.1%})")
    print(f"  runs with O2 >= 0.10: {(o2 >= 0.10).mean():.1%};  temperature 273-323 K: {t.between(273, 323).mean():.1%}")

    # ---- Step 8: scatter figure ------------------------------------------ #
    banner("STEP 8: Save scatter")
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    sizes = 8 + 60 * np.sqrt(cells.runs / cells.runs.max())
    sc = ax.scatter(cells.concentration_CO2, cells.concentration_O2, c=cells.t_median, s=sizes,
                    cmap=cmap, edgecolors=SURFACE, linewidths=0.6, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("CO2 surface mixing ratio (log)")
    ax.set_ylabel("O2 surface mixing ratio")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("median surface temperature in cell (K)", color=INK_2)
    cbar.outline.set_visible(False)

    x_lo = co2.min() * 0.7
    for (x0, x1, y0, y1, style, label) in [
        (x_lo, 0.01, 0.10, o2.max() * 1.02, "--", f"R1: O2 ≥ 0.10, CO2 ≤ 0.01  ({rule_table.iloc[0, 1]:.1f}% of runs)"),
        (x_lo, 0.05, 0.15, 0.30, ":", f"R2: 0.15 ≤ O2 ≤ 0.30, CO2 ≤ 0.05  ({rule_table.iloc[2, 1]:.1f}% of runs)"),
    ]:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=INK,
                               linestyle=style, linewidth=1.3, zorder=4))
    ax.plot(EARTH["CO2"], EARTH["O2"], marker="*", markersize=13, color=SERIES_2,
            markeredgecolor=INK, markeredgewidth=0.8, zorder=5, linestyle="none")
    ax.annotate("modern Earth\n(reference only)", (EARTH["CO2"], EARTH["O2"]), xytext=(10, -26),
                textcoords="offset points", fontsize=8, color=INK)
    handles = [
        Line2D([], [], color=INK, linestyle="--", label=f"R1: O2 ≥ 0.10, CO2 ≤ 0.01 ({rule_table.iloc[0, 1]:.1f}% of runs)"),
        Line2D([], [], color=INK, linestyle=":", label=f"R2: 0.15 ≤ O2 ≤ 0.30, CO2 ≤ 0.05 ({rule_table.iloc[2, 1]:.1f}% of runs)"),
        Line2D([], [], marker="o", linestyle="none", color=BLUE_RAMP[3], markersize=6,
               label="grid cell (size = run count)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(f"PyATMOS O2 vs CO2: {len(cells)} grid cells covering all {len(pyatmos):,} runs "
                 f"(illustrative rules, not final)", loc="left")
    fig.tight_layout()
    save(fig, "pyatmos_o2_co2_scatter.png")

    # ------------------------------------------------------------------ 3.4 #
    banner("STEP 9: Habitable World Catalogue (STAGE_2_REFERENCE_ONLY)")
    hwc_entry = dm.STAGE_2_REFERENCE_ONLY[0]
    dm.assert_role(hwc_entry, "STAGE_2_REFERENCE_ONLY")
    hwc = pd.read_csv(dm.resolve(hwc_entry), low_memory=False)
    print(f"  loaded {hwc_entry}: {hwc.shape}")
    print(f"  columns: {list(hwc.columns)}")
    print("  NOTE: real confirmed exoplanets classified by orbital/physical parameters; PyATMOS is a")
    print("        synthetic atmosphere grid. No row-level join is meaningful - reference comparison only.")
    gas_like = [c for c in hwc.columns if any(k in c.upper() for k in ["O2", "CO2", "ATMOS", "H2O"])]
    print(f"  atmosphere/gas columns in the catalogue: {gas_like or 'none'}")

    hab_cols = ["P_HABITABLE", "P_HABZONE_OPT", "P_HABZONE_CON", "P_ESI"]
    for c in hab_cols[:3]:
        vc = hwc[c].value_counts(dropna=False).sort_index()
        print(f"\n  {c}: " + ", ".join(f"{k}: {v} ({v / len(hwc):.1%})" for k, v in vc.items()))
    print("\n  P_HABITABLE x P_HABZONE_CON / P_HABZONE_OPT:")
    print(pd.crosstab(hwc.P_HABITABLE, [hwc.P_HABZONE_CON, hwc.P_HABZONE_OPT],
                      rownames=["P_HABITABLE"], colnames=["HZ_CON", "HZ_OPT"]).to_string())
    esi = hwc.groupby("P_HABITABLE")["P_ESI"].describe()
    print("\n  P_ESI by P_HABITABLE:")
    print(esi.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"  P_ESI blank: {int(hwc.P_ESI.isna().sum())}; P_ESI >= 0.8: {int((hwc.P_ESI >= 0.8).sum())}")
    if "P_TYPE_TEMP" in hwc:
        print("\n  P_TYPE_TEMP counts: " + ", ".join(f"{k}: {v}" for k, v in
                                                     hwc.P_TYPE_TEMP.value_counts(dropna=False).items()))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), gridspec_kw={"width_ratios": [1, 1.6]})
    vc = hwc.P_HABITABLE.value_counts().sort_index()
    bars = axes[0].bar([str(k) for k in vc.index], vc.values, color=SERIES_1, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vc.values):
        axes[0].annotate(f"{v:,}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                         textcoords="offset points", ha="center", fontsize=8, color=INK)
    axes[0].set_title("P_HABITABLE")
    axes[0].set_xlabel("P_HABITABLE value")
    axes[0].set_ylabel("planets")
    bins = np.linspace(0, 1, 41)
    axes[1].hist([hwc.loc[hwc.P_HABITABLE == 0, "P_ESI"].dropna(),
                  hwc.loc[hwc.P_HABITABLE > 0, "P_ESI"].dropna()],
                 bins=bins, stacked=True, color=[SERIES_1, SERIES_2], edgecolor=SURFACE,
                 linewidth=0.5, label=["P_HABITABLE = 0", "P_HABITABLE = 1 or 2"])
    axes[1].set_yscale("log")
    axes[1].set_title("Earth Similarity Index (P_ESI), stacked, log counts")
    axes[1].set_xlabel("P_ESI")
    axes[1].legend(loc="upper right")
    fig.suptitle(f"Habitable World Catalogue ({len(hwc):,} planets): reference only, not joined to PyATMOS",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "hwc_habitability.png")

    banner("DONE")
    print("  Inputs were only read. Figures in eda/; written summary in eda/EDA_HABITABILITY.md.")


if __name__ == "__main__":
    run_logged(main)
