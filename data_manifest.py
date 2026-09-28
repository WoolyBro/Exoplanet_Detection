"""
data_manifest.py
================

Single source of truth for which data file plays which role in the two
modelling stages, so they cannot share training data by accident:

    Stage 1  transit detection            (CNN+LSTM on light curves)
    Stage 2  habitability / atmospheres   (O2, CO2 detection)

Path conventions: relative paths are relative to this Research folder; absolute
paths point at raw data that lives outside it (read-only, never written to).
A trailing "/" marks a directory.

Enforcement:
  * The manifest is validated on import: an entry may appear in only one role,
    no entry may sit inside another role's directory, and nothing may point
    into the excluded 05_final_ML_dataset_DO_NOT_USE folder.
  * load_stage1_catalogs() / load_stage2_catalogs() load ONLY their stage's
    TRAINING files and raise if any is missing.
  * role_of(path) / assert_role(path, ...) let other scripts check a file's
    role before using it.

Run `python data_manifest.py` to print every entry, its role, provenance and
whether it exists on disk.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parent
PACK = "exoplanet_research_data"  # mentor-approved pack (read-only)

# --------------------------------------------------------------------------- #
# Roles
# --------------------------------------------------------------------------- #
STAGE_1_TRAINING = [
    "splits/koi_cumulative_split.csv",
    "splits/k2_planets_candidates_split.csv",
    "splits/toi_tess_candidates_split.csv",
]
STAGE_1_SUPPLEMENTARY = [
    f"{PACK}/01_candidate_catalogs/pscomppars_confirmed_planets.csv",
    f"{PACK}/01_candidate_catalogs/ps_transiting_default.csv",
    f"{PACK}/03_stellar_parameters/stellarhosts.csv",
    # 36 mentor light curves (22 Kepler Q9, 14 TESS single sectors) + lightcurve_manifest.csv.
    # Used for local-file pipeline tests and per-KOI folding checks, not as the primary
    # training source: preprocessing_pipeline.build_dataset pulls full baselines via the API.
    f"{PACK}/02_lightcurves/",
]
STAGE_1_VALIDATION_ONLY: list[str] = []
STAGE_2_TRAINING = [
    # PyATMOS simulation runs. Previously D:/Files/Dataset/run_summary_final.csv, which
    # was missing dir_0 entirely (7,828 runs) and had blank temperature/pressure for
    # 62,040 runs. pyatmos_final.csv (built by build_pyatmos_final.py) is the corrected,
    # complete-where-possible version: 108,839 runs that exist on disk. It uses
    # surface-level gas composition as a reference population rather than generating
    # spectra from vertical profiles.
    "datasets/pyatmos_final.csv",
    "D:/Files/inara_earthlike_subset/psg_models.csv",     # INARA (PSG models)
]
STAGE_2_VALIDATION_ONLY = [
    f"{PACK}/04_atmospheric_spectra/",  # real observed spectra, from mentor's package
]
STAGE_2_REFERENCE_ONLY = [
    "D:/Files/Habitable World Catalogue.csv",
]

# Never to be used by any stage (the user renamed it to say so).
EXCLUDED = [
    f"{PACK}/05_final_ML_dataset_DO_NOT_USE/",
]

# Derived outputs (not manifest entries, so role_of() does not recognise them):
#   habitability_prepared/{train,val,test}.csv + label_rule.json
#     written by prepare_habitability_data.py from STAGE_2_TRAINING psg_models.csv (INARA),
#     loaded via load_stage2_catalogs(). Labels use an exploratory PLACEHOLDER rule, and the
#     outputs are not meaningful for training until INARA is expanded; re-run the script then.

ROLES = {
    "STAGE_1_TRAINING": STAGE_1_TRAINING,
    "STAGE_1_SUPPLEMENTARY": STAGE_1_SUPPLEMENTARY,
    "STAGE_1_VALIDATION_ONLY": STAGE_1_VALIDATION_ONLY,
    "STAGE_2_TRAINING": STAGE_2_TRAINING,
    "STAGE_2_VALIDATION_ONLY": STAGE_2_VALIDATION_ONLY,
    "STAGE_2_REFERENCE_ONLY": STAGE_2_REFERENCE_ONLY,
}

# A directory entry that a loader must read is represented by one table inside it.
# (Currently none: PyATMOS is now listed directly as datasets/pyatmos_final.csv.)
DIRECTORY_TABLES: dict[str, str] = {}

# Which archive each entry comes from. Printed by the manifest report so any table can
# be traced back to its source; never branched on.
PROVENANCE = {
    "splits/koi_cumulative_split.csv": "research pack",
    "splits/k2_planets_candidates_split.csv": "research pack",
    "splits/toi_tess_candidates_split.csv": "research pack",
    f"{PACK}/01_candidate_catalogs/pscomppars_confirmed_planets.csv": "research pack",
    f"{PACK}/01_candidate_catalogs/ps_transiting_default.csv": "research pack",
    f"{PACK}/03_stellar_parameters/stellarhosts.csv": "research pack",
    f"{PACK}/02_lightcurves/": "research pack",
    "datasets/pyatmos_final.csv": "PyATMOS (VPL)",
    "D:/Files/inara_earthlike_subset/psg_models.csv": "INARA / PSG",
    f"{PACK}/04_atmospheric_spectra/": "research pack",
    "D:/Files/Habitable World Catalogue.csv": "PHL Habitable Worlds Catalogue",
}


# --------------------------------------------------------------------------- #
# Path helpers and manifest validation
# --------------------------------------------------------------------------- #
def resolve(entry: str) -> Path:
    """Absolute path for a manifest entry."""
    p = Path(entry)
    return (p if p.is_absolute() else RESEARCH_DIR / p).resolve()


def _contains(parent: Path, child: Path) -> bool:
    return child == parent or parent in child.parents


def _validate_manifest() -> None:
    problems = []
    seen: dict[Path, str] = {}
    for role, entries in ROLES.items():
        for entry in entries:
            path = resolve(entry)
            if path in seen:
                problems.append(f"{entry} is in both {seen[path]} and {role}")
            seen[path] = role
            if entry not in PROVENANCE:
                problems.append(f"{entry} ({role}) has no PROVENANCE tag")
            for excluded in EXCLUDED:
                if _contains(resolve(excluded), path):
                    problems.append(f"{entry} ({role}) points into excluded {excluded}")

    # No entry may live inside a directory that belongs to a different role.
    for dir_path, dir_role in seen.items():
        for path, role in seen.items():
            if path != dir_path and role != dir_role and _contains(dir_path, path):
                problems.append(f"{path} ({role}) is inside {dir_path} ({dir_role})")

    if problems:
        raise ValueError("data_manifest.py is inconsistent:\n  " + "\n  ".join(problems))


_validate_manifest()


def role_of(path: str | Path) -> str:
    """Return the manifest role that owns `path` (a file or anything inside a listed directory).

    Raises ValueError for excluded or unlisted paths, so unlisted data can't be
    used without first being given a role here.
    """
    target = resolve(str(path))
    for excluded in EXCLUDED:
        if _contains(resolve(excluded), target):
            raise ValueError(f"{path} is EXCLUDED ({excluded}) and must not be used")
    for role, entries in ROLES.items():
        if any(_contains(resolve(e), target) for e in entries):
            return role
    raise ValueError(f"{path} is not in data_manifest.py; add it to a role before using it")


def assert_role(path: str | Path, *allowed_roles: str) -> None:
    """Raise unless `path` belongs to one of `allowed_roles`."""
    role = role_of(path)
    if role not in allowed_roles:
        raise PermissionError(f"{path} is {role}, not one of {allowed_roles}")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _table_path(entry: str) -> Path:
    path = resolve(entry)
    if entry.endswith("/"):
        if entry not in DIRECTORY_TABLES:
            raise ValueError(f"directory entry {entry} has no table in DIRECTORY_TABLES")
        path = path / DIRECTORY_TABLES[entry]
    return path


def _load_role(role: str) -> dict[str, pd.DataFrame]:
    """Load every entry of `role`; raise (listing all of them) if any file is missing."""
    paths = {entry: _table_path(entry) for entry in ROLES[role]}
    missing = [f"{entry} -> {p}" for entry, p in paths.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{role}: manifest files not found on disk:\n  " + "\n  ".join(missing))

    frames = {}
    for entry, path in paths.items():
        if path.name in frames:
            raise ValueError(f"{role}: two entries share the file name {path.name}")
        frames[path.name] = pd.read_csv(path, low_memory=False)
        print(f"  [{role}] loaded {path.name}: {frames[path.name].shape} "
              f"(provenance: {PROVENANCE[entry]})")
    return frames


def load_stage1_catalogs() -> dict[str, pd.DataFrame]:
    """Stage 1 (transit detection) training catalogs only, keyed by file name."""
    return _load_role("STAGE_1_TRAINING")


def load_stage2_catalogs() -> dict[str, pd.DataFrame]:
    """Stage 2 (habitability / atmospheres) training tables only, keyed by file name.

    Both Stage 2 training sources come from external archives (PyATMOS and INARA/PSG).
    """
    return _load_role("STAGE_2_TRAINING")


# --------------------------------------------------------------------------- #
# Status report
# --------------------------------------------------------------------------- #
def manifest_status() -> pd.DataFrame:
    rows = []
    for role, entries in {**ROLES, "EXCLUDED": EXCLUDED}.items():
        for entry in entries:
            path = resolve(entry)
            if entry.endswith("/"):
                kind = "directory"
                exists = path.is_dir()
                if exists and entry in DIRECTORY_TABLES:
                    table = path / DIRECTORY_TABLES[entry]
                    detail = f"loads {table.name} ({'found' if table.is_file() else 'MISSING'})"
                else:
                    detail = ""
            else:
                kind = "file"
                exists = path.is_file()
                size = path.stat().st_size if exists else 0
                detail = "" if not exists else (
                    f"{size / 1e6:.1f} MB" if size >= 1e5 else f"{size / 1e3:.1f} KB")
                if exists and role.endswith("_TRAINING"):
                    n_rows = len(pd.read_csv(path, usecols=[0], low_memory=False))
                    detail += f", {n_rows:,} rows"
            rows.append({
                "role": role,
                "entry": entry,
                "provenance": PROVENANCE.get(entry, "-"),
                "kind": kind,
                "exists": "yes" if exists else "NO",
                "detail": detail,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 80)
    print("Manifest validated: no entry has two roles, none sits inside another role's folder.\n")
    print(manifest_status().to_string(index=False))
