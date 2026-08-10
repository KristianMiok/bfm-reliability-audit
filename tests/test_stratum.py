"""Regression locks on cell assignment, window cut, states, and stratum math."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfm_audit.config import ModelGrid
from bfm_audit.grid import (assign_cells, cell_stratum, flag_reporting,
                            training_window_mask)

GRID = ModelGrid(lat_max=71.875, lon_min=-25.125, verified=True)


def test_assign_cells_is_south_up():
    df = pd.DataFrame({
        "decimalLatitude":  [32.00, 71.75, 51.90, 71.90, 31.80, 32.00],
        "decimalLongitude": [-25.00, 44.75, 14.50, 0.00, 0.00, 45.00],
    })
    out = assign_cells(df, GRID)
    # SW corner cell centre -> row 0, col 0 (model tensor convention, G1)
    assert (out.loc[0, "grid_row"], out.loc[0, "grid_col"]) == (0, 0)
    # NE corner cell centre -> row 159, col 279
    assert (out.loc[1, "grid_row"], out.loc[1, "grid_col"]) == (159, 279)
    # interior sanity: row index grows northward
    assert out.loc[2, "grid_row"] == int((51.90 - 31.875) // 0.25)
    # outside: north of 71.875, south of 31.875, east of 44.875
    assert not out.loc[3, "in_grid"]
    assert not out.loc[4, "in_grid"]
    assert not out.loc[5, "in_grid"]
    assert (out.loc[~out["in_grid"], ["grid_row", "grid_col"]] == -1).all().all()


def test_assign_cells_edges_half_open():
    df = pd.DataFrame({"decimalLatitude": [31.875, 71.875],
                       "decimalLongitude": [-25.125, 44.875]})
    out = assign_cells(df, GRID)
    assert out.loc[0, "in_grid"] and out.loc[0, "grid_row"] == 0
    assert not out.loc[1, "in_grid"]  # top/right edges exclusive


def test_training_window_mask():
    df = pd.DataFrame({
        "year":  [2019, 2020, 2020, 2020, 2018, 2020],
        "month": [12,   6,    7,    np.nan, np.nan, 1],
    })
    m = training_window_mask(df)
    assert m.tolist() == [True, True, False, False, True, True]


def test_flag_reporting_three_states():
    df = pd.DataFrame({"coordinateUncertaintyInMeters":
                       [np.nan, 999, 50, 3036, 10000]})
    out = flag_reporting(df)
    assert out["state"].tolist() == ["unknown", "fake", "reporting",
                                     "fake", "reporting"]


def test_cell_stratum_math():
    n = 24
    df = pd.DataFrame({
        "decimalLatitude": [32.0] * n,
        "decimalLongitude": [-25.0] * n,
        "coordinateUncertaintyInMeters": [50] * 12 + [np.nan] * 8 + [999] * 4,
        "speciesKey": [1.0] * 20 + [2.0] * 4,
        "datasetKey": ["A"] * 18 + ["B"] * 6,
    })
    df = flag_reporting(df)
    df = assign_cells(df, GRID)
    cells = cell_stratum(df, min_records=20)
    assert len(cells) == 1
    row = cells.iloc[0]
    assert row["n_records"] == 24 and row["reliable_estimate"]
    assert row["pct_reporting"] == 100 * 12 / 24
    assert row["n_fake"] == 4 and row["n_species"] == 2
    # hhi over datasets: (18/24)^2 + (6/24)^2
    assert abs(row["dataset_hhi"] - (0.75**2 + 0.25**2)) < 1e-9
