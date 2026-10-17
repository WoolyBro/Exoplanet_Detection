"""Append one CH4 label row: python tools/add_ch4_label.py "planet" 1|0|unknown featureless(0/1) "source" "significance" "note" """
import csv, sys
from pathlib import Path
import os
p = Path(__file__).resolve().parents[1] / "outputs" / "labels" / os.environ.get("CH4_LABEL_FILE", "ch4_labels_extended.csv")
planet, lab, feat, src, sig, note = sys.argv[1:7]
assert lab in ("1", "0", "unknown") and feat in ("0", "1")
done = {r["planet"] for r in csv.DictReader(p.open())}
assert planet not in done, f"{planet} already labelled"
with p.open("a", newline="") as f:
    csv.writer(f).writerow([planet, lab, feat, src, sig, note])
print(f"{len(done) + 1:2d}  {planet}: {lab}")
