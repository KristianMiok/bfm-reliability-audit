"""Paths, model constants, and the species list.

Everything here that is not read from the model's own artefacts is marked
UNVERIFIED. Do not let an unverified default silently become an assumption in
a result: run scripts/01_extract_model_grid.py, which overwrites the grid
extent with values taken from the model's batch metadata.
"""

from __future__ import annotations

import json
import dataclasses
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"              # GBIF download archives, model checkpoints
INTERIM = DATA / "interim"      # parsed / joined intermediates
REFERENCE = DATA / "reference"  # small committed tables

SPECIES_TABLE = REFERENCE / "bfm_species_keys.csv"
GRID_JSON = REFERENCE / "model_grid.json"

for _p in (RAW, INTERIM, REFERENCE):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# model grid
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelGrid:
    """The spatial grid BioAnalyst predicts on.

    H, W and the 0.25 degree resolution are read directly from
    bfm_model/bfm/configs/train_config.yaml and are VERIFIED.

    lat_max / lon_min are NOT in that config. In the model repo the coordinate
    vectors arrive through ``batch['batch_metadata']['latitudes']`` and
    ``['longitudes']``, i.e. they come from the data, not the config. The
    values below are a plausible European extent consistent with H=160, W=280
    at 0.25 deg, and nothing more.
    """

    height: int = 160          # VERIFIED: train_config.yaml `H`
    width: int = 280           # VERIFIED: train_config.yaml `W`
    resolution: float = 0.25   # VERIFIED: 160 x 0.25 = 40 deg lat, 280 x 0.25 = 70 deg lon
    # Edge values below follow from the release code of both repos (see
    # scripts/01_extract_model_grid.py and DECISIONS.md 2026-08-10): cell
    # centres lat 32.00..71.75, lon -25.00..44.75, padded by half a cell.
    # The earlier guess (lat 34..74) was wrong by 2 degrees; the verified
    # flag still gates every use, so these defaults decide nothing.
    lat_max: float = 71.875    # derived, not yet batch-confirmed
    lon_min: float = -25.125   # derived, not yet batch-confirmed
    verified: bool = False     # set True only by 01_extract_model_grid.py

    @property
    def lat_min(self) -> float:
        return self.lat_max - self.height * self.resolution

    @property
    def lon_max(self) -> float:
        return self.lon_min + self.width * self.resolution

    @property
    def bbox_wkt(self) -> str:
        """POLYGON for a GBIF geometry predicate, counter-clockwise."""
        x0, x1 = self.lon_min, self.lon_max
        y0, y1 = self.lat_min, self.lat_max
        return (f"POLYGON(({x0} {y0}, {x1} {y0}, {x1} {y1}, "
                f"{x0} {y1}, {x0} {y0}))")

    def save(self, path: Path = GRID_JSON) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = GRID_JSON) -> "ModelGrid":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @staticmethod
    def provenance(path: Path = GRID_JSON) -> dict:
        """The derivation record written next to the grid, {} if absent."""
        if not path.exists():
            return {}
        return json.loads(path.read_text()).get("provenance", {})


def require_verified_grid() -> ModelGrid:
    """Grid loader that refuses to hand back guessed coordinates.

    Any script whose output depends on where the grid actually sits must call
    this rather than ``ModelGrid()``.
    """
    g = ModelGrid.load()
    if not g.verified:
        raise RuntimeError(
            "Model grid extent is unverified. Run "
            "scripts/01_extract_model_grid.py against a real BioAnalyst batch "
            "before any step whose result depends on grid position."
        )
    return g


# --------------------------------------------------------------------------
# species
# --------------------------------------------------------------------------

def load_species() -> pd.DataFrame:
    """The 28 GBIF taxonKeys BioAnalyst is trained on.

    Source: `species_vars` in bfm_model/bfm/configs/train_config.yaml.
    Columns beyond key/name/class come from the R screening scripts
    (12_bfm_species_quality.R, 13_reporting_controlled.R) and are carried here
    so this project does not depend on the R repo at runtime.
    """
    df = pd.read_csv(SPECIES_TABLE)
    assert len(df) == 28, f"expected 28 species, found {len(df)}"
    return df


TAXON_KEYS: list[int] = load_species()["taxon_key"].tolist()


# --------------------------------------------------------------------------
# screening thresholds carried over from the R work
# --------------------------------------------------------------------------

# A record is "reporting" if it states any coordinateUncertaintyInMeters.
# The controlled stratum used in 13_ was basisOfRecord == HUMAN_OBSERVATION
# and year >= 2015; the reporting gap survived it (spread 92.7 points,
# Spearman(raw, controlled) = +0.93).
CONTROL_BASIS_OF_RECORD = "HUMAN_OBSERVATION"
CONTROL_YEAR_FROM = 2015

# Placeholder radii known to be software defaults rather than measurements.
# Documented in the GBIF data blog and used in the earlier crayfish work.
FAKE_RADII = {301, 999, 3036, 9999}
