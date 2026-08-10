"""Construct, submit, and retrieve the GBIF download for the 28 species.

Counts only were used for screening (R scripts 10_ through 13_). This step
needs actual records, because the audit stratum is spatial: which grid cells
are dominated by records that state no coordinate precision.

All requests go straight to the GBIF API with the standard library. An earlier
version routed submission through pygbif, whose ``occurrences.download()``
expects query *strings* and does not reliably accept a prebuilt predicate
dict; a malformed submission would have been discovered only at run time.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
from pathlib import Path

from .config import ModelGrid, TAXON_KEYS, RAW

API = "https://api.gbif.org/v1/occurrence/download"


def build_predicate(grid: ModelGrid, year_from: int, year_to: int) -> dict:
    """The occurrence-download predicate.

    Deliberately does NOT filter on coordinateUncertaintyInMeters. Records
    without a radius are the object of study, not noise to be removed -- the
    standard ``r < 1000`` filter drops honest reporters while keeping silent
    publishers and fake-precise records, which is the bias this project is
    about.

    DOES filter OCCURRENCE_STATUS = PRESENT, mirroring the WHERE clause of
    the documented GBIF cube template the training data went through. Done
    server-side because SIMPLE_CSV's fixed column set does not guarantee an
    occurrenceStatus column for a local filter. Measured absence share under
    this predicate without the clause: 97,281 / 12,611,544 = 0.77 %
    (2026-08-10, see DECISIONS.md).
    """
    return {
        "type": "and",
        "predicates": [
            {"type": "in", "key": "TAXON_KEY",
             "values": [str(k) for k in TAXON_KEYS]},
            {"type": "equals", "key": "OCCURRENCE_STATUS", "value": "PRESENT"},
            {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
            {"type": "equals", "key": "HAS_GEOSPATIAL_ISSUE", "value": "false"},
            {"type": "within", "geometry": grid.bbox_wkt},
            {"type": "greaterThanOrEquals", "key": "YEAR", "value": str(year_from)},
            {"type": "lessThanOrEquals", "key": "YEAR", "value": str(year_to)},
        ],
    }


def _auth_header() -> dict:
    for var in ("GBIF_USER", "GBIF_PWD", "GBIF_EMAIL"):
        if not os.environ.get(var):
            raise RuntimeError(
                f"{var} not set. Export GBIF_USER, GBIF_PWD and GBIF_EMAIL "
                "(e.g. `set -a; source .env; set +a`)."
            )
    token = base64.b64encode(
        f"{os.environ['GBIF_USER']}:{os.environ['GBIF_PWD']}".encode()
    ).decode()
    return {"Authorization": f"Basic {token}",
            "Content-Type": "application/json"}


def request(grid: ModelGrid, year_from: int, year_to: int) -> str:
    """Submit the download; returns the download key.

    SIMPLE_CSV fixed columns include everything scripts downstream need:
    gbifID, datasetKey, taxon/speciesKey, decimalLatitude/Longitude,
    coordinateUncertaintyInMeters, coordinatePrecision, eventDate,
    day/month/year, basisOfRecord, issue, license.
    """
    body = {
        "creator": os.environ.get("GBIF_USER", ""),
        "notificationAddresses": [os.environ.get("GBIF_EMAIL", "")],
        "sendNotification": True,
        "format": "SIMPLE_CSV",
        "predicate": build_predicate(grid, year_from, year_to),
    }
    req = urllib.request.Request(
        f"{API}/request", data=json.dumps(body).encode(),
        headers=_auth_header(), method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        key = r.read().decode().strip()
    if not key:
        raise RuntimeError("GBIF returned an empty download key")
    return key


def status(download_key: str) -> dict:
    """Current state of a download: status, totalRecords, size (bytes)."""
    with urllib.request.urlopen(f"{API}/{download_key}", timeout=60) as r:
        meta = json.loads(r.read())
    return {"status": meta.get("status"),
            "totalRecords": meta.get("totalRecords"),
            "size": meta.get("size"),
            "doi": meta.get("doi")}


def fetch(download_key: str, dest: Path = RAW,
          poll_seconds: int = 0) -> Path:
    """Stream the prepared archive to ``dest`` once GBIF reports SUCCEEDED.

    With ``poll_seconds`` > 0, waits and re-checks until ready.
    """
    dest.mkdir(parents=True, exist_ok=True)
    while True:
        st = status(download_key)
        if st["status"] == "SUCCEEDED":
            break
        if st["status"] in {"FAILED", "KILLED", "CANCELLED"}:
            raise RuntimeError(f"download {download_key}: {st['status']}")
        if not poll_seconds:
            raise RuntimeError(
                f"download {download_key} not ready ({st['status']}); "
                "re-run with poll_seconds>0 or try later")
        print(f"  {st['status']} ... waiting {poll_seconds}s", flush=True)
        time.sleep(poll_seconds)

    out = dest / f"{download_key}.zip"
    with urllib.request.urlopen(f"{API}/request/{download_key}",
                                timeout=300) as r, open(out, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    print(f"  archive: {out} ({out.stat().st_size/1e9:.2f} GB)")
    return out
