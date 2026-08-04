"""Cell assignment must not silently pile out-of-extent records on the edge."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bfm_audit.config import ModelGrid
from bfm_audit.grid import assign_cells, flag_reporting, cell_stratum

GRID = ModelGrid(height=160, width=280, resolution=0.25,
                 lat_max=74.0, lon_min=-25.0, verified=True)


def test_grid_extent():
    assert GRID.lat_min == pytest.approx(34.0)
    assert GRID.lon_max == pytest.approx(45.0)


def test_corner_cells():
    df = pd.DataFrame({
        "decimalLatitude":  [73.9, 34.1, 60.0],
        "decimalLongitude": [-24.9, 44.9, 10.0],
    })
    out = assign_cells(df, GRID)
    assert out.loc[0, "grid_row"] == 0 and out.loc[0, "grid_col"] == 0
    assert out.loc[1, "grid_row"] == GRID.height - 1
    assert out.loc[1, "grid_col"] == GRID.width - 1
    assert out["in_grid"].all()


def test_outside_is_marked_not_clipped():
    df = pd.DataFrame({
        "decimalLatitude":  [80.0, 20.0, 50.0],
        "decimalLongitude": [10.0, 10.0, 100.0],
    })
    out = assign_cells(df, GRID)
    assert not out["in_grid"].any()
    assert (out["grid_row"] == -1).sum() + (out["grid_col"] == -1).sum() >= 3


def test_three_states():
    df = pd.DataFrame({"coordinateUncertaintyInMeters": [10.0, np.nan, 301.0, 5000.0]})
    out = flag_reporting(df)
    assert list(out["state"]) == ["reporting", "unknown", "fake", "reporting"]


def test_cell_stratum_hhi():
    df = pd.DataFrame({
        "decimalLatitude": [60.0] * 4,
        "decimalLongitude": [10.0] * 4,
        "coordinateUncertaintyInMeters": [10.0, np.nan, 10.0, np.nan],
        "speciesKey": [1, 1, 2, 2],
        "datasetKey": ["a", "a", "a", "a"],
    })
    cells = cell_stratum(assign_cells(flag_reporting(df), GRID), min_records=1)
    assert len(cells) == 1
    assert cells.loc[0, "pct_reporting"] == pytest.approx(50.0)
    assert cells.loc[0, "dataset_hhi"] == pytest.approx(1.0)  # single publisher
