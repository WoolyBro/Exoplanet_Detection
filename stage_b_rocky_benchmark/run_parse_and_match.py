"""Rocky benchmark Parts 1-2 (Steps 1-6): parse all 12 Zenodo files and cross-check each against the local
JWST .tbl spectra for the same planet. Writes, in rocky_benchmark/:
    zenodo_parse_index.csv      per-file units, layout, fixes, wavelength range, grid coverage
    zenodo_vs_local_matches.csv every (Zenodo file, local .tbl or per-paper combination) comparison
    zenodo_best_match.csv       best local counterpart per Zenodo file and its verdict

Read-only on the .tbl files, the labels and every Stage B output.
"""
from __future__ import annotations

import json

import pandas as pd

import zenodo_io as z

OUT = z.OUT_DIR


def main() -> None:
    params = z.planet_params()
    infos, data = [], {}
    for f in sorted(z.FILES):
        df, info = z.parse_zenodo(z.ZEN_DIR / f, params)
        infos.append(info)
        data[f] = df
    idx = pd.DataFrame(infos)
    idx["before_after_first3"] = idx.get("before_after_first3", pd.Series(dtype=object)).map(
        lambda v: json.dumps(v) if isinstance(v, list) else "")
    idx.to_csv(OUT / "zenodo_parse_index.csv", index=False)

    rows, best = [], []
    for f, df in data.items():
        cands = z.local_candidates(z.FILES[f])
        res = []
        for c in cands:
            m = z.compare(df, c["df"])
            m.update({"zenodo_file": f, "planet": z.FILES[f], "local": c["label"], "local_files": ";".join(c["files"]),
                      "bibcode": c["bibcode"], "authors": c["authors"], "instrument": c["instrument"]})
            m["verdict"] = z.classify_match(m)
            res.append(m)
        rows.extend(res)
        ok = [m for m in res if m.get("n_compared", 0) > 0]
        if not ok:
            best.append({"zenodo_file": f, "planet": z.FILES[f], "verdict": "NO LOCAL JWST TRANSMISSION SPECTRUM"})
            continue
        b = min(ok, key=lambda m: (m["norm_diff"], -m["coverage"]))
        best.append({k: b.get(k) for k in ("zenodo_file", "planet", "local", "local_files", "bibcode", "authors",
                                           "instrument", "coverage", "median_offset_ppm", "norm_diff",
                                           "norm_diff_after_offset", "corr", "verdict")})
    pd.DataFrame(rows).to_csv(OUT / "zenodo_vs_local_matches.csv", index=False)
    b = pd.DataFrame(best)
    b.to_csv(OUT / "zenodo_best_match.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(b[["zenodo_file", "local", "coverage", "median_offset_ppm", "norm_diff", "corr", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
