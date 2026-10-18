"""
refresh_archive_spectra.py
==========================

Refresh the NASA Exoplanet Archive Atmospheric Spectroscopy holdings. It downloads the current
`spectra` table (TAP) and only the spectrum files that are new since the research pack's snapshot.
The mentor's pack (exoplanet_research_data/) is read-only and is not modified. New material goes to
external_data/archive_spectra_<date>/.

    python tools/refresh_archive_spectra.py                # table + new files
    python tools/refresh_archive_spectra.py --table-only

File URLs follow the archive's own bulk-download service, as in the pack's
scripts/download_spectra.py: https://exoplanetarchive.ipac.caltech.edu/workspace/{SESSION}/atmospheres/tab1/data/{spec_path}
SESSION is a temporary id issued by the archive's table page. The script first re-downloads one
file the pack already has and requires it to be byte-identical, so the method is verified each run.
"""

from __future__ import annotations

import argparse
import re
import ssl
import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1]
PACK = R / "exoplanet_research_data" / "04_atmospheric_spectra"
HOST = "https://exoplanetarchive.ipac.caltech.edu"
UA = {"User-Agent": "Mozilla/5.0 (research project; exoplanet atmospheres)"}


def _ssl_context():
    """Verified TLS. python.org Python builds ship without CA certificates, so use certifi's bundle."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL = _ssl_context()


def fetch(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout, context=SSL) as r:
        return r.read()


def session_id() -> str:
    html = fetch(HOST + "/cgi-bin/atmospheres/nph-firefly?atmospheres").decode("utf-8", "ignore")
    m = re.search(r"/(?:workspace|work)/(TMP_[A-Za-z0-9_]+)", html)
    if not m:
        raise RuntimeError("could not obtain an archive workspace session id")
    return m.group(1)


def safe_name(pl_name: str, spec_type: str, spec_path: str) -> str:
    """The pack's file-naming convention, so existing loaders find the files."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{pl_name.strip()}_{spec_type.strip()}_{Path(spec_path).name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-only", action="store_true")
    a = ap.parse_args()
    out = R / "external_data" / f"archive_spectra_{date.today().isoformat()}"
    (out / "spectra_tbl").mkdir(parents=True, exist_ok=True)
    table = out / "spectra_meta.csv"
    table.write_bytes(fetch(HOST + "/TAP/sync?query=select+*+from+spectra&format=csv", timeout=120))
    new = pd.read_csv(table, low_memory=False)
    old = pd.read_csv(PACK / "metadata" / "spectra_metadata.csv", low_memory=False)
    added = new[~new.spec_path.isin(old.spec_path)].copy()
    added.to_csv(out / "added_since_pack.csv", index=False)
    print(f"archive now: {len(new)} spectra; pack snapshot: {len(old)}; new since snapshot: {len(added)}")
    if a.table_only:
        return 0

    sid = session_id()
    # verify the method on a file the pack already has
    ref = old.iloc[0]
    ref_local = next((p for p in (PACK / "spectra_tbl").glob("*.tbl") if p.name.endswith(Path(ref.spec_path).name)), None)
    probe = fetch(f"{HOST}/workspace/{sid}/atmospheres/tab1/data/{ref.spec_path}")
    if ref_local is None or probe != ref_local.read_bytes():
        raise SystemExit("verification failed: a re-downloaded pack file is not byte-identical; not downloading")
    print(f"verified: re-downloaded {ref_local.name} is byte-identical to the pack copy")

    ok, failed = 0, []
    for r in added.itertuples():
        dest = out / "spectra_tbl" / safe_name(r.pl_name, r.spec_type, r.spec_path)
        if dest.is_file() and dest.stat().st_size > 0:
            ok += 1
            continue
        for attempt in range(3):
            try:
                data = fetch(f"{HOST}/workspace/{sid}/atmospheres/tab1/data/{r.spec_path}")
                if len(data) < 50 or b"Not Found" in data[:500]:
                    raise IOError(f"bad payload ({len(data)} bytes)")
                dest.write_bytes(data)
                ok += 1
                break
            except Exception as e:  # session expiry: refresh and retry
                if attempt == 2:
                    failed.append(f"{r.spec_path}: {e}")
                sid = session_id()
                time.sleep(1 + attempt)
    print(f"downloaded {ok}/{len(added)}; failed {len(failed)}")
    for f in failed:
        print("  FAIL", f)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
