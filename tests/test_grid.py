"""Cell assignment must not silently pile out-of-extent records on the edge."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bfm_audit.config import ModelGrid
from bfm_audit.grid import assign_cells, flag_reporting, cell_stratum

# Verified extent (G1): cell edges, centres 32.00..71.75 x -25.00..44.75
GRID = ModelGrid(height=160, width=280, resolution=0.25,
                 lat_max=71.875, lon_min=-25.125, verified=True)


def test_grid_extent():
    assert GRID.lat_min == pytest.approx(31.875)
    assert GRID.lon_max == pytest.approx(44.875)


def test_corner_cells_south_up():
    """Row 0 = SOUTH (model tensor convention, G1); see test_stratum.py."""
    df = pd.DataFrame({
        "decimalLatitude":  [31.9, 71.8, 60.0],
        "decimalLongitude": [-25.1, 44.8, 10.0],
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
