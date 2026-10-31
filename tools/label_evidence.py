"""
label_evidence.py
=================

eda/POST_REVIEW_PROTOCOL.md, analysis M (post hoc): pull the sentences about each gas from the paper cited for each label, classify them
by fixed rules, and flag rows for a person to read. This speeds up the human check and does not replace it.

    python -I tools/label_evidence.py <folder for downloaded pages>
Output: outputs/post_review/label_evidence.csv, label_evidence_summary.json
"""

from __future__ import annotations

import csv
import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[1]
SHEET = R / "paper" / "Label_Audit_Sheet.csv"
OUT = R / "outputs" / "post_review"
NAMES = {"CH4": r"(?:CH\s*_?\{?\s*4|CH₄|methane)", "CO2": r"(?:CO\s*_?\{?\s*2|CO₂|carbon dioxide)"}
POS = r"(?:detect(?:ed|ion|s)?|evidence (?:for|of)|significance|[3-9](?:\.\d+)?\s*[-]?\s*(?:σ|sigma)|\d\d(?:\.\d+)?\s*[-]?\s*(?:σ|sigma)|identif|confirm|presence of|ln\s*B)"
NEG = r"(?:non-?detect|no (?:evidence|detection|significant|sign|clear|strong)|not detect|undetect|upper limit|absence of|featureless|flat spectrum|rule[sd]? out|lack of|cannot confirm|does not show)"
TENT = r"(?:tentative|hint|marginal|possibl|weak evidence|inconclusive|low significance|2(?:\.\d+)?\s*[-]?\s*(?:σ|sigma))"


def fetch(arxiv_id: str, folder: Path) -> str:
    f = folder / (arxiv_id.replace("/", "_") + ".html")
    if not f.is_file():
        for url in (f"https://arxiv.org/html/{arxiv_id}", f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}", f"https://arxiv.org/abs/{arxiv_id}"):
            r = subprocess.run(["curl", "-sL", "--max-time", "40", "-A", "label-audit", "-o", str(f), "-w", "%{http_code}", url], capture_output=True, text=True)
            time.sleep(3)
            if r.stdout.strip() == "200" and f.stat().st_size > 20000:
                break
    t = f.read_text(encoding="utf-8", errors="ignore") if f.is_file() else ""
    t = re.sub(r"(?is)<(script|style|annotation)[^>]*>.*?</\1>", " ", t)
    t = html.unescape(re.sub(r"(?s)<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t)


def main() -> int:
    folder = Path(sys.argv[1])
    rows, cache = list(csv.DictReader(SHEET.open(encoding="utf-8"))), {}
    out = []
    for r in rows:
        m = re.search(r"(\d{4}\.\d{4,5})", r["paper"])
        if not m:
            out.append({**r, "rule_label": "no link", "n_sentences": 0, "snippet": ""})
            continue
        t = cache.setdefault(m.group(1), fetch(m.group(1), folder))
        sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if re.search(NAMES[r["gas"]], s) and 30 < len(s) < 500]
        p = sum(bool(re.search(POS, s, re.I)) and not re.search(NEG, s, re.I) and not re.search(TENT, s, re.I) for s in sents)
        n = sum(bool(re.search(NEG, s, re.I)) for s in sents)
        q = sum(bool(re.search(TENT, s, re.I)) for s in sents)
        lab = "1" if p >= 3 and p >= 2 * max(n, 1) else "0" if n >= 2 and n >= p else "unclear"
        key = sorted(sents, key=lambda s: -(bool(re.search(POS + "|" + NEG, s, re.I)) + bool(re.search(r"\d\s*[-]?\s*(?:σ|sigma)|ppm", s))))[:1]
        snip = " ".join(key[0].split()[:25]) if key else ""
        out.append({**r, "rule_label": lab, "n_sentences": len(sents), "n_pos": p, "n_neg": n, "n_tentative": q, "snippet": snip})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "label_evidence.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["planet", "gas", "source", "paper", "rule_label", "n_sentences", "n_pos", "n_neg", "n_tentative", "snippet", "your_label", "notes"], extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(len(out), "rows", len(cache), "papers", sum(bool(v) for v in cache.values()), "with text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
