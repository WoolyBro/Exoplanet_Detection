# JWST Rocky Exoplanet Spectra (Kreidberg & Stevenson 2025) — external archive, NOT integrated

- **Source:** Zenodo, concept DOI 10.5281/zenodo.15084225, which resolves to record 10.5281/zenodo.15084226 (published 2025-03-25), by Stevenson, K. and Kreidberg, L.
- **Licence:** CC-BY-4.0, open access.
- **Paper:** Kreidberg & Stevenson 2025, PNAS, "A first look at rocky exoplanets with JWST" (arXiv:2507.00933, doi:10.1073/pnas.2416190122).
- **Downloaded:** 2026-09-19. 12 files, 50 KB total. All md5 checksums match the Zenodo record.
- **Status:** a scoping download only. These files are not in the data manifest, not used by Stage B, and not labelled. See `eda/STAGE_B_ROCKY_ZENODO_SCOPING.md`.

## Format

The record says every file has 4 columns: wavelength (µm), half-width of the channel (µm), transit depth (ppm) and 1σ uncertainty (ppm). Lines starting with `#` are comments. Three files differ from that:

| File | Quirk |
|---|---|
| `LHS1140b-NIRSpec.txt` | Columns are **bin start (µm), bin end (µm)**, depth, error, pasted from the paper's journal tables 8–9 and tab-separated. The file holds the G235H NRS1/NRS2 and G395H NRS1/NRS2 segments, which overlap between 2.88 and 3.07 µm. The centre wavelength is (start + end) / 2. |
| `GJ1132b.txt`, `L98-59c.txt` | The half-width column is all zeros. |
| `GJ486b.txt` | Has a 20-line CDS byte-by-byte header. Its data is still 4 numeric columns. |
