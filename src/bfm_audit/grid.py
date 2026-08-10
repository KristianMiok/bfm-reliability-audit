"""Assign occurrence records to BioAnalyst grid cells and build the stratum.

The audit stratum is per-CELL, not per-species. C12 screened species and the
gradient turned out to be continuous rather than bimodal; C13 then showed the
reporting gap is real and survives an era/record-type control, but is almost
certainly driven by publisher convention rather than by the species itself
(Sturnus vulgaris 55.0% vs Pieris brassicae 96.4%, neither sensitive, both
mass citizen science). What a group-conditional coverage test conditions on is
therefore the composition of each grid cell's training records, which is what
this module computes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ModelGrid, FAKE_RADII


def assign_cells(
    df: pd.DataFrame,
    grid: ModelGrid,
    lat_col: str = "decimalLatitude",
    lon_col: str = "decimalLongitude",
) -> pd.DataFrame:
    """Add integer row/col indices into the model grid.

    Row 0 is the SOUTHERNMOST band. This deliberately breaks the usual
    raster convention because it matches the model: the release writer
    reindexes every array onto an ascending latitude vector and the reader
    crops from index 0 (G1), so tensor row 0 sits at lat 32.00. The E2 join
    of stratum cells with model outputs is index-for-index only under this
    convention; north-up here would silently flip the map vertically.
    Rows and columns falling outside the grid are set to -1 rather than
    clipped, so out-of-extent records are droppable rather than silently
    piled onto the edge -- edge pile-up is exactly the kind of artefact
    that produces a fake spatial pattern.
    """
    out = df.copy()
    col = np.floor((out[lon_col] - grid.lon_min) / grid.resolution)
    row = np.floor((out[lat_col] - grid.lat_min) / grid.resolution)

    col = col.to_numpy()
    row = row.to_numpy()
    inside = (col >= 0) & (col < grid.width) & (row >= 0) & (row < grid.height)

    out["grid_row"] = np.where(inside, row, -1).astype(int)
    out["grid_col"] = np.where(inside, col, -1).astype(int)
    out["in_grid"] = inside
    return out


def flag_reporting(
    df: pd.DataFrame,
    radius_col: str = "coordinateUncertaintyInMeters",
    drop_fake: bool = True,
) -> pd.DataFrame:
    """Classify each record into the three states used throughout this work.

    reporting   : states a radius that is a plausible measurement
    fake        : states a known software placeholder (301, 999, 3036, 9999)
    unknown     : states nothing at all

    `fake` is kept as its own class rather than folded into either side. A
    placeholder is not a measurement, but it is also not silence -- the
    publisher populated the field. Collapsing it either way would beg the
    question this project is about.
    """
    out = df.copy()
    r = out[radius_col]
    is_fake = r.isin(FAKE_RADII) if drop_fake else pd.Series(False, index=r.index)

    out["state"] = np.select(
        [r.isna(), is_fake],
        ["unknown", "fake"],
        default="reporting",
    )
    return out


def training_window_mask(
    df: pd.DataFrame,
    year_to: int = 2020,
    month_to: int = 6,
    year_col: str = "year",
    month_col: str = "month",
) -> pd.Series:
    """True for records inside BioAnalyst's training window (<= 2020-06).

    The GBIF YEAR predicate cannot express "January-June of the final year
    only", so the download runs to the end of 2020 and the tail is cut here.
    Final-year records with no month are excluded: they cannot be shown to
    precede the cutoff (decision 2026-08-10, DECISIONS.md).
    """
    y = pd.to_numeric(df[year_col], errors="coerce")
    m = pd.to_numeric(df[month_col], errors="coerce")
    return (y < year_to) | ((y == year_to) & (m <= month_to))


def cell_stratum(df: pd.DataFrame, min_records: int = 20) -> pd.DataFrame:
    """Per-cell composition: the grouping variable for the calibration audit.

    Returns one row per occupied grid cell with the share of its records in
    each state, plus publisher concentration. Cells below `min_records` are
    returned but flagged: a cell with three records has a reporting share of
    0, 33, 67 or 100 by construction, and treating that as a stratum value
    would manufacture a gradient out of small-sample noise.
    """
    d = df[df["in_grid"]]

    states = (
        d.groupby(["grid_row", "grid_col", "state"], observed=True)
        .size().unstack("state", fill_value=0)
    )
    for c in ("reporting", "unknown", "fake"):
        if c not in states:
            states[c] = 0
    out = states.rename(columns={"reporting": "n_reporting",
                                 "unknown": "n_unknown",
                                 "fake": "n_fake"})
    out["n_records"] = out[["n_reporting", "n_unknown", "n_fake"]].sum(axis=1)
    extra = d.groupby(["grid_row", "grid_col"]).agg(
        n_species=("speciesKey", "nunique"),
        n_datasets=("datasetKey", "nunique"),
    )
    out = out.join(extra).reset_index()

    out["pct_reporting"] = 100 * out["n_reporting"] / out["n_records"]
    out["pct_unknown"] = 100 * out["n_unknown"] / out["n_records"]
    out["reliable_estimate"] = out["n_records"] >= min_records

    # Publisher concentration. If one dataset dominates a cell, that cell's
    # reporting share is a property of that publisher, which is the mechanism
    # C13 pointed at. Herfindahl index over datasets, 1 = single publisher.
    hhi = (
        d.groupby(["grid_row", "grid_col", "datasetKey"])
        .size()
        .rename("n")
        .reset_index()
    )
    tot = hhi.groupby(["grid_row", "grid_col"])["n"].transform("sum")
    hhi["share2"] = (hhi["n"] / tot) ** 2
    hhi = hhi.groupby(["grid_row", "grid_col"])["share2"].sum().rename("dataset_hhi")

    return out.merge(hhi, on=["grid_row", "grid_col"], how="left")


def to_raster(cells: pd.DataFrame, value_col: str, grid: ModelGrid) -> np.ndarray:
    """Rasterise a per-cell value onto the full H x W model grid.

    Unoccupied cells are NaN, not zero. A cell with no occurrence records is
    not a cell with poor data quality; it is a cell the audit has nothing to
    say about, and the distinction must survive into any figure.

    Row 0 of the returned array is the SOUTHERN edge (model convention, G1).
    Plot with ``imshow(..., origin="lower")`` or the map is upside down.
    """
    arr = np.full((grid.height, grid.width), np.nan, dtype=float)
    arr[cells["grid_row"].to_numpy(), cells["grid_col"].to_numpy()] = cells[value_col]
    return arr
