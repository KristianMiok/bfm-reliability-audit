"""The download predicate must mirror the cube's neutral filters exactly."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfm_audit.config import ModelGrid, TAXON_KEYS
from bfm_audit.gbif import build_predicate


def _preds():
    g = ModelGrid(lat_max=71.875, lon_min=-25.125, verified=True)
    p = build_predicate(g, 2000, 2020)
    assert p["type"] == "and"
    return g, {q["key"]: q for q in p["predicates"] if "key" in q}, p


def test_taxon_keys_complete():
    _, by_key, _ = _preds()
    assert len(by_key["TAXON_KEY"]["values"]) == 28
    assert set(by_key["TAXON_KEY"]["values"]) == {str(k) for k in TAXON_KEYS}


def test_present_only_mirrors_cube():
    _, by_key, _ = _preds()
    assert by_key["OCCURRENCE_STATUS"]["value"] == "PRESENT"


def test_no_uncertainty_filter():
    """Records without a radius are the object of study; the predicate must
    never condition on the variable the stratum measures."""
    _, _, p = _preds()
    blob = str(p).lower()
    assert "uncertainty" not in blob and "precision" not in blob


def test_geometry_and_years_from_grid():
    g, _, p = _preds()
    within = [q for q in p["predicates"] if q["type"] == "within"][0]
    assert within["geometry"] == g.bbox_wkt
    years = {q["type"]: q["value"] for q in p["predicates"]
             if q.get("key") == "YEAR"}
    assert years == {"greaterThanOrEquals": "2000",
                     "lessThanOrEquals": "2020"}
