"""Construct and retrieve the GBIF download for BioAnalyst's training species.

Counts only were used for screening (the R scripts 10_ through 13_). This step
needs actual records, because the audit stratum is spatial: which grid cells
are dominated by records that state no coordinate precision.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import ModelGrid, TAXON_KEYS, RAW


def build_predicate(grid: ModelGrid, year_from: int, year_to: int) -> dict:
    """The occ_download predicate.

    Deliberately does NOT filter on coordinateUncertaintyInMeters. Records
    without a radius are the object of study, not noise to be removed -- the
    standard `r < 1000` filter drops honest reporters while keeping silent
    publishers and fake-precise records, which is the bias this project is
    about.
    """
    return {
        "type": "and",
        "predicates": [
            {"type": "in", "key": "TAXON_KEY",
             "values": [str(k) for k in TAXON_KEYS]},
            {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
            {"type": "equals", "key": "HAS_GEOSPATIAL_ISSUE", "value": "false"},
            {"type": "within", "geometry": grid.bbox_wkt},
            {"type": "greaterThanOrEquals", "key": "YEAR", "value": str(year_from)},
            {"type": "lessThanOrEquals", "key": "YEAR", "value": str(year_to)},
        ],
    }


REQUESTED_FIELDS = [
    "gbifID", "speciesKey", "datasetKey", "publishingOrgKey",
    "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters",
    "coordinatePrecision", "year", "month", "basisOfRecord",
    "issue", "informationWithheld",
]


def request(grid: ModelGrid, year_from: int, year_to: int) -> str:
    """Submit the download. Requires GBIF credentials in the environment.

    Returns the download key. GBIF prepares large downloads asynchronously;
    expect minutes to hours for a request of this size.
    """
    from pygbif import occurrences as occ

    for var in ("GBIF_USER", "GBIF_PWD", "GBIF_EMAIL"):
        if not os.environ.get(var):
            raise RuntimeError(
                f"{var} not set. Export GBIF_USER, GBIF_PWD and GBIF_EMAIL, "
                "or put them in a .env file that is NOT committed."
            )

    pred = build_predicate(grid, year_from, year_to)
    res = occ.download(pred, user=os.environ["GBIF_USER"],
                       pwd=os.environ["GBIF_PWD"], email=os.environ["GBIF_EMAIL"])
    return res[0]


def fetch(download_key: str, dest: Path = RAW) -> Path:
    """Download the prepared archive once GBIF reports it ready."""
    from pygbif import occurrences as occ

    dest.mkdir(parents=True, exist_ok=True)
    occ.download_get(download_key, path=str(dest))
    return dest / f"{download_key}.zip"
