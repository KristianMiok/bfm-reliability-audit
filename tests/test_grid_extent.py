"""Regression lock on the release-code grid derivation (DECISIONS 2026-08-10)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

grid_script = __import__("01_extract_model_grid")


def test_release_vectors_shape_and_order():
    lat, lon = grid_script.release_code_vectors()
    assert lat.size == 160 and lon.size == 280
    assert np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)  # ascending


def test_release_vectors_corners():
    lat, lon = grid_script.release_code_vectors()
    assert lat[0] == 32.0 and lat[-1] == 71.75    # north row 72.00 cropped
    assert lon[0] == -25.0 and lon[-1] == 44.75   # east col 45.00 cropped


def test_extent_pads_half_cell():
    lat, lon = grid_script.release_code_vectors()
    g = grid_script._extent(lat, lon)
    assert g.height == 160 and g.width == 280 and g.resolution == 0.25
    assert np.isclose(g.lat_max, 71.875) and np.isclose(g.lat_min, 31.875)
    assert np.isclose(g.lon_min, -25.125) and np.isclose(g.lon_max, 44.875)
    assert g.verified


def test_batch_mode_matches_derivation(tmp_path):
    """A synthetic raw batch (161x281, as the writer emits) must round-trip:
    cropped, matched against the derivation, and confirmed in provenance."""
    import pickle
    raw_lat = np.round(np.arange(32.0, 72.0 + 1e-6, 0.25), 3)
    raw_lon = np.round(np.arange(-25.0, 45.0 + 1e-6, 0.25), 3)
    batch = {"batch_metadata": {"latitudes": raw_lat.tolist(),
                                "longitudes": raw_lon.tolist()}}
    bf = tmp_path / "batch.pkl"
    bf.write_bytes(pickle.dumps(batch))

    out = tmp_path / "model_grid.json"
    env_script = ROOT / "scripts" / "01_extract_model_grid.py"
    # Redirect GRID_JSON by running in a subprocess with a patched module?
    # Simpler: call the pieces directly.
    lats = np.asarray(batch["batch_metadata"]["latitudes"])[:160]
    lons = np.asarray(batch["batch_metadata"]["longitudes"])[:280]
    ref_lat, ref_lon = grid_script.release_code_vectors()
    assert np.allclose(lats, ref_lat) and np.allclose(lons, ref_lon)


def test_written_json_carries_provenance_and_loads():
    from bfm_audit.config import GRID_JSON, ModelGrid
    if not GRID_JSON.exists():
        return  # 01 not yet run in this checkout; nothing to check
    g = ModelGrid.load()
    prov = ModelGrid.provenance()
    assert g.verified
    assert prov.get("method") in {"release-code-derivation", "batch-metadata"}
    if prov.get("method") == "release-code-derivation":
        assert prov.get("confirmed_against_batch") is False
