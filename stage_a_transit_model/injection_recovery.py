#!/usr/bin/env python3
"""
injection_recovery.py
=====================

Measures the trained model's detection completeness: inject synthetic transits of known
depth into real Kepler photometry, push them through the exact pipeline the model was
trained on, and count how many it recovers.

Why this rather than the DR25 INJ1 set
    INJ1 exists in two forms and neither answers this question. Its injected light curves
    cover ~160,000 targets, which is an enormous download. Its recovery table records what
    the *Kepler pipeline* recovered, not what this model recovers. Injecting into real
    photometry here measures this model directly, and mirrors what
    stage_b_rocky_benchmark/rocky_benchmark.py already does for spectra.
    The trade-off is honest: this is a bespoke test, not the official INJ1 set, so the
    numbers are not directly comparable with papers that quote INJ1 completeness.

What makes it a fair test
    The injected signal goes into the raw flux BEFORE detrending, and its ephemeris is
    handed to load_and_flatten so the Savitzky-Golay filter masks it exactly as it masks
    real transits. Skipping that would let the detrender eat the injection and the model
    would be blamed for the pipeline's damage. The star's real KOIs are masked out too, so
    a recovery cannot be a real planet showing through.

Transit shape
    Quadratic limb darkening (u1=0.4, u2=0.25, typical for Kepler bandpass), impact
    parameter 0.3. A flat-bottomed box would be easier to detect than any real transit and
    would flatter the model.

Usage
    python stage_a_transit_model/injection_recovery.py --stars 50
    python stage_a_transit_model/injection_recovery.py --stars 50 --score-only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))

import preprocessing_pipeline as pp  # noqa: E402

OUT_DIR = RESEARCH / "outputs" / "injection_recovery"
EDA = RESEARCH / "eda"
RUN = RESEARCH / "stage_a_transit_model" / "runs" / "bn_aug_sched"

# Depths reach far below an Earth-analogue (~84 ppm) on purpose. The views are DEPTH
# NORMALISED - the deepest bin is set to -1 - so absolute depth is erased before the model
# sees anything. Folding four years of Kepler data averages the noise down so far that even
# 100 ppm is already the deepest coherent feature, and every depth above that produces an
# identical view. The completeness floor therefore sits at the depth where the injection
# stops dominating the fold, which is well below 100 ppm. Recovery is reported against the
# measured folded SNR as well, because that is the axis the model can actually respond to.
# 0 ppm is the NULL CONTROL and the most important row in the table: the same star, the same
# random period, the same pipeline, but nothing injected. Whatever the model scores there is
# what it scores for pure folded noise, and no recovery claim means anything above it. Without
# this row a 100% recovery rate at 5 ppm looks like a detection instead of a baseline.
DEPTHS_PPM = [0, 5, 10, 20, 40, 80, 160, 320]
U1, U2, IMPACT = 0.4, 0.25, 0.3


def limb_darkened_profile(x: np.ndarray, b: float = IMPACT) -> np.ndarray:
    """Relative transit depth across the crossing, x in [-1, 1] spanning first to last contact.

    Quadratic limb darkening: the star is brighter at its centre, so a planet blocks more
    light mid-transit and the floor is curved rather than flat. Returned peak-normalised, so
    the caller controls absolute depth.
    """
    inside = np.abs(x) <= 1.0
    r = np.sqrt(b ** 2 + (1.0 - b ** 2) * np.clip(x, -1, 1) ** 2)   # projected separation
    mu = np.sqrt(np.clip(1.0 - r ** 2, 0.0, 1.0))
    intensity = 1.0 - U1 * (1.0 - mu) - U2 * (1.0 - mu) ** 2
    prof = np.where(inside, intensity, 0.0)
    peak = prof.max()
    return prof / peak if peak > 0 else prof


def inject(time: np.ndarray, flux: np.ndarray, period: float, epoch: float,
           duration_d: float, depth_ppm: float) -> np.ndarray:
    """Multiply a synthetic transit into the flux at the given ephemeris."""
    phase = (time - epoch + 0.5 * period) % period - 0.5 * period
    x = 2.0 * phase / duration_d                      # -1 .. +1 across the transit
    model = 1.0 - (depth_ppm * 1e-6) * limb_darkened_profile(x)
    return flux * model


def star_sample(n: int, seed: int) -> pd.DataFrame:
    """Kepler test-split stars, one row per star, with their real KOI ephemerides."""
    koi = pp.catalog_for("Kepler")
    test = koi[koi.split == "test"]
    rng = np.random.default_rng(seed)
    stars = test.kepid.dropna().unique()
    pick = rng.permutation(stars)[:n]
    return test[test.kepid.isin(pick)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stars", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--score-only", action="store_true",
                    help="reuse light curves already in the cache")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    from stage_a_transit_model.cnn_lstm import PLANET_LIKE_IDX
    from stage_a_transit_model.failure_analysis import load_run_model

    import torch

    model, arch = load_run_model(args.run)
    print("=" * 88)
    print(f"Injection-recovery | model {args.run.name} {arch}")
    print("=" * 88)

    part = star_sample(args.stars, args.seed)
    stars = sorted(part.kepid.unique())
    print(f"\n[1] {len(stars)} Kepler test-split stars, {len(DEPTHS_PPM)} depths each "
          f"= {len(stars) * len(DEPTHS_PPM)} injections")

    rows = []
    for si, star in enumerate(stars, 1):
        star_rows = part[part.kepid == star].to_dict("records")
        real = [pp.ephemeris_from_row(r, "Kepler") for r in star_rows]
        try:
            csv = pp.fetch_lightcurve_via_api(int(star), "Kepler",
                                              cache_dir=pp.LK_CACHE, use_s3=True)
            raw = pd.read_csv(csv)
        except Exception as exc:
            print(f"  [{si}/{len(stars)}] KIC {star}: fetch failed ({type(exc).__name__})")
            continue

        # A fresh ephemeris, deliberately away from any real KOI period on this star so the
        # injection cannot be confused with (or reinforced by) the real signal.
        real_periods = [e.period for e in real if np.isfinite(e.period)]
        for _ in range(40):
            period = float(rng.uniform(3.0, 40.0))
            if all(abs(period - p) / p > 0.15 for p in real_periods):
                break
        t = raw["time"].to_numpy(float)
        epoch = float(np.nanmin(t) + rng.uniform(0, period))
        # Duration from a central circular transit of a Sun-like star, in days.
        duration = float(period ** (1 / 3) * 0.0716)

        n_ok = 0
        for depth in DEPTHS_PPM:
            inj = raw.copy()
            inj["flux"] = inject(t, raw["flux"].to_numpy(float), period, epoch, duration, depth)
            tmp = pp.LK_CACHE / "csv" / f"_inj_{star}_{depth}.csv"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            inj.to_csv(tmp, index=False)
            eph = pp.Ephemeris(object_id=f"INJ{star}_{depth}", star_id=int(star),
                               mission="Kepler", period=period, epoch=epoch,
                               duration=duration, label="INJECTED")
            try:
                # The injected ephemeris AND the real ones are masked from the trend fit.
                flat = pp.load_and_flatten(tmp, "Kepler", ephemerides=[eph] + real)
                res = pp.process_koi(flat, eph, real)     # real KOIs removed from the fold
            except Exception as exc:
                rows.append({"star": star, "depth_ppm": depth, "period": period,
                             "status": f"failed: {type(exc).__name__}"})
                continue
            finally:
                tmp.unlink(missing_ok=True)

            # .values is the normalised view array (View.flux does not exist).
            g = torch.from_numpy(res["global"].values.astype(np.float32)[None, :])
            l = torch.from_numpy(res["local"].values.astype(np.float32)[None, :])
            with torch.no_grad():
                proba = torch.softmax(model(g, l), dim=1).numpy()[0]
            score = float(proba[PLANET_LIKE_IDX].sum())
            rows.append({"star": star, "depth_ppm": depth, "period": period,
                         "duration_d": duration, "planet_like_score": score,
                         "depth_snr": res.get("depth_snr"),
                         "measured_depth_ppm": res.get("depth_ppm"), "status": "ok"})
            n_ok += 1
        print(f"  [{si}/{len(stars)}] KIC {star}: {n_ok}/{len(DEPTHS_PPM)} injections scored "
              f"(P={period:.2f} d)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "injections.csv", index=False)
    ok = df[df.status == "ok"]
    if ok.empty:
        print("\nno injections scored; nothing to report")
        return 1

    # Recovery threshold: the model's own 90 %-recall operating point on Kepler val.
    thr_file = args.run / "report.json"
    thr = 0.5
    if thr_file.is_file():
        rep = json.loads(thr_file.read_text(encoding="utf-8"))
        for r in (rep.get("val_metrics") or {}).get("precision_at_recall", []):
            if r.get("target_recall") == 0.90 and r.get("achievable"):
                thr = float(r["threshold"])
    ok = ok.assign(recovered=ok.planet_like_score >= thr)

    # ---- the null control decides how any of this reads ------------------ #
    null = ok[ok.depth_ppm == 0].set_index("star").planet_like_score
    print("\n[2] NULL CONTROL (0 ppm: nothing injected, same star, same random period)")
    print(f"  n={len(null)}  median score {null.median():.3f}")
    print(f"  scored above the {thr:.3f} threshold: {int((null >= thr).sum())}/{len(null)}"
          f" = {(null >= thr).mean():.0%}")
    print("  The model calls most arbitrary folds planet-like, so a raw recovery fraction is")
    print("  almost entirely baseline and says little about detection.")

    # Paired statistic: each injection against ITS OWN star's null, which removes the large
    # per-star offset that swamps the unpaired numbers.
    wide = ok.pivot_table(index="star", columns="depth_ppm", values="planet_like_score")
    print("\n[3] paired excess over each star's own null - the informative metric")
    print(f"  {'depth (ppm)':>12} {'median excess':>14} {'stars responding':>18} {'recovered':>10}")
    summary = []
    for d in sorted(ok.depth_ppm.unique()):
        grp = ok[ok.depth_ppm == d]
        delta = (wide[d] - wide[0]).dropna() if (0 in wide.columns and d in wide.columns)             else pd.Series(dtype=float)
        rec = float(grp.recovered.mean())
        summary.append({"depth_ppm": int(d), "n": int(len(grp)),
                        "median_score": float(grp.planet_like_score.median()),
                        "median_excess_over_null": float(delta.median()) if len(delta) else None,
                        "stars_responding": int((delta > 0).sum()) if len(delta) else None,
                        "n_paired": int(len(delta)),
                        "recovered_fraction": rec})
        ex = f"{delta.median():+.4f}" if len(delta) else "-"
        resp = f"{int((delta > 0).sum())}/{len(delta)}" if len(delta) else "-"
        print(f"  {int(d):>12} {ex:>14} {resp:>18} {rec:>9.0%}")

    sm = pd.DataFrame(summary)
    sm.to_csv(OUT_DIR / "completeness.csv", index=False)

    # Detection floor: lowest depth whose paired excess is positive AND carried by a clear
    # majority of stars. Below it the injection is indistinguishable from folded noise.
    floor = None
    for r in summary:
        if (r["median_excess_over_null"] or 0) > 0.01 and                 (r["stars_responding"] or 0) >= 0.65 * max(r["n_paired"], 1):
            floor = r["depth_ppm"]
            break
    print(f"\n  lowest depth the model measurably responds to: "
          f"{str(floor) + ' ppm' if floor else 'none in this grid'}")
    half = floor

    # ---- figure ---------------------------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot = sm[sm.depth_ppm > 0]          # the null is the reference line, not a data point
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: the paired statistic. Plotted instead of recovery fraction because the null
    # already sits above the threshold, so raw recovery is nearly all baseline.
    ax = axes[0]
    ax.plot(plot.depth_ppm, plot.median_excess_over_null, "o-", color="#2d6fa8", lw=1.8)
    ax.axhline(0.0, color="grey", lw=1.0)
    if floor:
        ax.axvline(floor, color="crimson", ls=":", lw=1.2)
        ax.text(floor * 1.08, ax.get_ylim()[1] * 0.85,
                f"responds from\n~{floor:.0f} ppm", color="crimson", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("injected transit depth (ppm)")
    ax.set_ylabel("median score excess over the star's own null")
    ax.set_title(f"Response to injected depth, paired per star\n"
                 f"{len(ok)} injections into {ok.star.nunique()} real Kepler light curves",
                 fontsize=9)
    ax.grid(alpha=0.3)

    # Right: absolute scores, with the null band drawn so the baseline is impossible to miss.
    ax = axes[1]
    for d, grp in ok.groupby("depth_ppm"):
        x = max(d, 2.5)                  # 0 ppm has no place on a log axis; drawn at the left edge
        ax.scatter([x] * len(grp), grp.planet_like_score, s=12, alpha=0.5,
                   color="#999999" if d == 0 else "#2d6fa8")
    ax.axhline(float(null.median()), color="#c23b22", ls="-", lw=1.4,
               label=f"null median ({null.median():.3f}) - nothing injected")
    ax.axhline(thr, color="crimson", ls="--", lw=1.0,
               label=f"90 %-recall threshold ({thr:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("injected transit depth (ppm); grey points at left are the 0 ppm null")
    ax.set_ylabel("model planet-like score")
    ax.set_title("Absolute score vs depth\nthe null already sits above the threshold",
                 fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle("Injection-recovery: synthetic transits in real Kepler photometry, "
                 "through the training pipeline", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    EDA.mkdir(parents=True, exist_ok=True)
    out = EDA / "injection_recovery.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    (OUT_DIR / "summary.json").write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_run": args.run.name, "architecture": arch,
        "n_stars": int(ok.star.nunique()), "n_injections": int(len(ok)),
        "depths_ppm": DEPTHS_PPM, "recovery_threshold": thr,
        "limb_darkening": {"u1": U1, "u2": U2, "impact_parameter": IMPACT},
        "completeness": summary, "detection_floor_ppm": floor,
        "null_median_score": float(null.median()),
        "null_above_threshold_fraction": float((null >= thr).mean()),
        "caveat": ("Bespoke injection test, not the DR25 INJ1 set, so not comparable with "
                   "published INJ1 completeness. Read the PAIRED excess, not the recovery "
                   "fraction: the 0 ppm null already scores above the threshold, so raw "
                   "recovery is nearly all baseline. Injection happens before detrending "
                   "with the injected ephemeris masked, so the filter treats it exactly "
                   "as it treats a real transit."),
    }, indent=2), encoding="utf-8")
    print(f"\nsaved {OUT_DIR.relative_to(RESEARCH)}/ and {out.relative_to(RESEARCH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
