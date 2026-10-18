"""
build_dr25_simulated_views.py
=============================

Views for the Kepler DR25 simulated-data benchmark (eda/DR25_SIMULATED_PROTOCOL.md), through the
unchanged Kepler pipeline: preprocessing_pipeline.load_and_flatten + process_koi, error channel included.

  INJ1  the archive's injected PDC light curves (one file per quarter), stitched exactly like real data.
  INV   the real Kepler PDCSAP light curve (AWS mirror), normalised per quarter and inverted: 2 - f.

Every TCE of the set on a star is masked in detrending and removed from the other TCEs' folds, as real
KOIs are. Resumable (progress.jsonl); each star's downloads are deleted once its views are written.

    python tools/build_dr25_simulated_views.py --set INJ1 --workers 4
    python tools/build_dr25_simulated_views.py --set INV --workers 4
Progress and time left:  tail -1 detection_views_err/dr25_inj1/build_progress.txt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
SIM = R / "external_data" / "kepler_dr25_simulated"
# Long-cadence file timestamp of each quarter (the quarter's END), from MAST's own product names.
QUARTER_TS = {0: "2009131105131", 1: "2009166043257", 2: "2009259160929", 3: "2009350155506", 4: "2010078095331",
              5: "2010174085026", 6: "2010265121752", 7: "2010355172524", 8: "2011073133259", 9: "2011177032512",
              10: "2011271113734", 11: "2012004120508", 12: "2012088054726", 13: "2012179063303",
              14: "2012277125453", 15: "2013011073258", 16: "2013098041711", 17: "2013131215648"}
INJ_URL = ("https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/Simulated///light_curves//"
           "injected-light-curves-dr25-{s}/q{q}/kplr{kic:09d}-{ts}_INJECTED-{s}_llc.fits.gz")
# Truth by construction: only INJ1 (on-target planets) are planets. INJ2 transits sit on a neighbouring star,
# INJ3 are eclipsing-binary mimics, INV are false alarms; OBS rows here are real non-KOI TCEs (NTP).
LABEL = {"INJ1": "CONFIRMED", "INJ2": "FALSE POSITIVE", "INJ3": "FALSE POSITIVE", "INV": "FALSE POSITIVE", "OBS": "FALSE POSITIVE"}


def _get(url: str, dest: Path, attempts: int = 3) -> bool:
    """Download url to dest; False on 404 (the star was not observed that quarter)."""
    import ssl
    import urllib.error
    import urllib.request

    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    for a in range(attempts):
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research"}),
                                          timeout=120, context=ctx).read()
            dest.write_bytes(data)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            if a == attempts - 1:
                raise
        except Exception:
            if a == attempts - 1:
                raise
        time.sleep(5 * (a + 1))
    return False


def _star_job(job: dict) -> dict:
    """One star: fetch, (invert), flatten, and write one view per TCE. Never raises."""
    sys.path.insert(0, str(R))
    socket.setdefaulttimeout(120)
    import lightkurve as lk
    import preprocessing_pipeline as p

    kic, rows, sset, out, work, cache = job["kic"], job["rows"], job["set"], Path(job["out"]), Path(job["work"]), Path(job["cache"])
    started = time.time()
    label = LABEL[sset]
    ephs = [p.Ephemeris(object_id=f"{r['TCE_ID']}_{sset}", star_id=kic, mission="Kepler", period=float(r["period"]),
                        epoch=float(r["epoch"]), duration=float(r["duration"]) / 24.0, label=label,
                        period_err=np.nan, epoch_err=np.nan) for r in rows]
    # the star's real DR25 TCEs are masked in detrending too (they are in every injected light curve)
    masks = ephs + [p.Ephemeris(object_id=f"real_{m['TCE_ID']}", star_id=kic, mission="Kepler", period=float(m["period"]),
                                epoch=float(m["epoch"]), duration=float(m["duration"]) / 24.0, label=None)
                    for m in job.get("real", [])
                    # for OBS the sampled TCE IS a catalogue TCE (already in ephs); injected TCE numbers are unrelated
                    if not (sset == "OBS" and f"{m['TCE_ID']}_OBS" in {e.object_id for e in ephs})]
    star_dir = work / f"kic{kic}"
    star_dir.mkdir(parents=True, exist_ok=True)
    try:
        if sset.startswith("INJ"):
            from concurrent.futures import ThreadPoolExecutor
            def one(item):
                q, ts = item
                dest = star_dir / f"q{q}.fits.gz"
                return dest if _get(INJ_URL.format(s=sset.lower(), q=q, kic=kic, ts=ts), dest) else None
            with ThreadPoolExecutor(max_workers=6) as pool:      # the 18 quarter files, in parallel
                files = [f for f in pool.map(one, sorted(QUARTER_TS.items())) if f is not None]
            if not files:
                raise LookupError("no injected quarters")
            csv = p._write_stitched_csv((lk.read(str(f), quality_bitmask="none") for f in files), star_dir / "lc.csv", kic)
        else:
            src = p.fetch_lightcurve_via_api(kic, "Kepler", cache_dir=cache, use_s3=True)
            d = pd.read_csv(src)
            if sset == "INV":
                d["flux"] = 2.0 - d.flux                 # inverted about the per-quarter median of 1
            csv = star_dir / "lc.csv"
            d.to_csv(csv, index=False)
        flat = p.load_and_flatten(csv, mission="Kepler", ephemerides=masks)
    except Exception as exc:
        return {"kic": kic, "records": [{"object_id": e.object_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]}
                                        for e in ephs], "seconds": time.time() - started}
    finally:
        if sset in ("INV", "OBS"):
            p.discard_raw_files(kic, "Kepler", cache)
    records = []
    for e, r in zip(ephs, rows):
        try:
            res = p.process_koi(flat, e, masks)
            meta = {"object_id": e.object_id, "star_id": kic, "set": sset, "tce_id": r["TCE_ID"], "period_days": e.period,
                    "epoch": e.epoch, "duration_days": e.duration, "label": label, "MES": r.get("MES", r.get("mes")),
                    "expected_mes": r.get("Expected_MES"), "robovetter_disp": r.get("Disp"), "robovetter_score": r.get("Score"),
                    "split": r.get("split"),
                    "depth_ppm": res["depth_ppm"], "depth_snr": res["depth_snr"], "transit_observed": res["transit_observed"],
                    "global_coverage": res["global"].coverage, "local_coverage": res["local"].coverage,
                    "flatten": flat.meta, "has_errors": True}
            npz = out / p._npz_name(e.object_id)
            tmp = npz.with_name(npz.stem + f".tmp{os.getpid()}.npz")
            np.savez_compressed(tmp, global_view=res["global"].values.astype(np.float32),
                                local_view=res["local"].values.astype(np.float32),
                                global_err=res["global"].errors.astype(np.float32),
                                local_err=res["local"].errors.astype(np.float32),
                                label=np.array(label), label_index=np.int8(p.LABEL_INDEX[label]),
                                metadata=np.array(json.dumps(meta, default=float)))
            os.replace(tmp, npz)
            records.append({"object_id": e.object_id, "status": "ok"})
        except Exception as exc:
            records.append({"object_id": e.object_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]})
    shutil.rmtree(star_dir, ignore_errors=True)
    return {"kic": kic, "records": records, "seconds": time.time() - started}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["INJ1", "INJ2", "INJ3", "INV", "OBS"], required=True)
    ap.add_argument("--sample", type=Path, default=None, help="sample CSV (default: the 2026-10-18 DR25 benchmark sample)")
    ap.add_argument("--out", default=None, help="output folder name under detection_views_err/")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    sample = pd.read_csv(a.sample or SIM / f"sample_{a.set.lower()}.csv")
    out = R / "detection_views_err" / (a.out or f"dr25_{a.set.lower()}")
    from astropy.io import ascii
    rv = SIM / "robovetter_inputs" / "kplr_dr25_obs_robovetter_input.txt"
    real = ascii.read(rv, format="ipac").to_pandas()[["TCE_ID", "KIC", "period", "epoch", "duration"]] if rv.is_file() else None
    out.mkdir(parents=True, exist_ok=True)
    work = Path(os.path.expanduser("~/lightkurve_cache_dr25")) / f"work_{(a.out or a.set).lower()}"
    cache = Path(os.path.expanduser("~/lightkurve_cache_dr25"))
    prog = out / "progress.jsonl"
    done = set()
    if prog.is_file():
        # only stars whose every record succeeded count as done; failures are retried on the next run
        done = {r["kic"] for r in map(json.loads, filter(str.strip, prog.read_text().splitlines()))
                if all(x["status"] == "ok" for x in r["records"])}
    stars = [k for k in sample.KIC.drop_duplicates() if k not in done][: a.limit]
    jobs = [{"kic": int(k), "rows": sample[sample.KIC == k].to_dict("records"), "set": a.set, "out": str(out),
             "work": str(work), "cache": str(cache),
             "real": [] if real is None else real[real.KIC == k].to_dict("records")} for k in stars]
    log = out / "build_progress.txt"
    t0, n_done, counts = time.time(), 0, {}
    with open(log, "a") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] start: {len(jobs)} stars to do, {len(done)} already done\n")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_star_job, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            with open(prog, "a") as fh:
                fh.write(json.dumps(r, default=str) + "\n")
            n_done += 1
            for rec in r["records"]:
                counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            if n_done % 25 == 0 or n_done == len(jobs):
                el = time.time() - t0
                eta = el / n_done * (len(jobs) - n_done) / 3600
                line = (f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] stars {n_done}/{len(jobs)} | TCEs {counts} | "
                        f"elapsed {el / 3600:.2f} h | ETA {eta:.2f} h")
                with open(log, "a") as fh:
                    fh.write(line + "\n")
                print(line, flush=True)
    print(f"done: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
