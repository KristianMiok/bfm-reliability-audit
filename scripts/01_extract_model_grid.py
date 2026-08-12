"""Establish BioAnalyst's actual lat/lon extent.

The model's grid position is NOT in train_config.yaml. H=160, W=280 and the
0.25 degree resolution are, but the coordinate vectors arrive at runtime via
``batch['batch_metadata']['latitudes']`` and ``['longitudes']``.

Two ways to establish the extent, in order of directness:

1. ``--batch-file``  : read the vectors from a real batch pickle. Gold
   standard; sets ``confirmed_against_batch: true``.
2. ``--from-release-code`` : reproduce the vectors from the public release
   code of BOTH repos. The derivation chain (verified 2026-08-10, see
   DECISIONS.md for file:line references):

     writer  bfm-data/.../scan_biocube.py
              GRID_LAT = round(arange(32.0, 72.0+1e-6, 0.25), 3)  # ascending, 161
              GRID_LON = round(arange(-25.0, 45.0+1e-6, 0.25), 3) # ascending, 281
     writer  bfm-data/.../build_batches_monthly.py
              ds.sel(latitude=GRID_LAT, longitude=GRID_LON)   # arrays reindexed
              batch_metadata["latitudes"] = GRID_LAT.tolist() # metadata written
     reader  bfm-model/.../dataloader_monthly.py  (patch_size = 4)
              new_H = (161//4)*4 = 160 ; new_W = (281//4)*4 = 280
              tensors[..., :160, :280] ; latitudes[:160] ; longitudes[:280]

   Result: cell centres lat 32.00..71.75, lon -25.00..44.75, both ascending;
   the writer's north row (72.00) and east column (45.00) are cropped away.
   Sets ``verified: true`` but ``confirmed_against_batch: false`` -- the gate
   in DECISIONS.md blocks any model-facing inference until a real batch has
   confirmed this via mode 1.

Do NOT use the legacy plotting constants (152x320, lat 34.25..72 descending,
lon -30..40) found in bfm-model utils.py / batch_visualisation.ipynb: that is
a pre-release grid with 22 species and does not match the released model.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bfm_audit.config import GRID_JSON, ModelGrid  # noqa: E402

# Constants copied verbatim from the release code, then cropped as the model
# reader crops. Single source of truth for mode 2 and for the mode-1 check.
WRITER_LAT_START, WRITER_LAT_END = 32.0, 72.0     # scan_biocube.py
WRITER_LON_START, WRITER_LON_END = -25.0, 45.0    # scan_biocube.py
RESOLUTION = 0.25                                  # scan_biocube.py
MODEL_PATCH_SIZE = 4                               # train_config.yaml


def release_code_vectors():
    """Cell-centre vectors exactly as the released model sees them."""
    grid_lat = np.round(
        np.arange(WRITER_LAT_START, WRITER_LAT_END + 1e-6, RESOLUTION), 3)
    grid_lon = np.round(
        np.arange(WRITER_LON_START, WRITER_LON_END + 1e-6, RESOLUTION), 3)
    assert grid_lat.size == 161 and grid_lon.size == 281, (
        grid_lat.size, grid_lon.size)
    new_h = (grid_lat.size // MODEL_PATCH_SIZE) * MODEL_PATCH_SIZE
    new_w = (grid_lon.size // MODEL_PATCH_SIZE) * MODEL_PATCH_SIZE
    return grid_lat[:new_h], grid_lon[:new_w]


def _extent(lats, lons) -> ModelGrid:
    lats = np.asarray(lats, dtype=float).ravel()
    lons = np.asarray(lons, dtype=float).ravel()

    res_lat = float(np.median(np.abs(np.diff(lats))))
    res_lon = float(np.median(np.abs(np.diff(lons))))
    print(f"  latitudes : n={lats.size}  {lats.min():.3f} .. {lats.max():.3f}"
          f"  step {res_lat:.4f}")
    print(f"  longitudes: n={lons.size}  {lons.min():.3f} .. {lons.max():.3f}"
          f"  step {res_lon:.4f}")

    if not np.isclose(res_lat, 0.25, atol=1e-3) or not np.isclose(res_lon, 0.25, atol=1e-3):
        print("  WARNING: step is not 0.25 deg. Do not proceed until this is understood.")

    # Cell edges, not centres: GBIF coordinates are points and must be binned.
    # Batch metadata gives cell centres, so pad by half a cell.
    return ModelGrid(
        height=int(lats.size),
        width=int(lons.size),
        resolution=round((res_lat + res_lon) / 2, 4),
        lat_max=float(lats.max()) + res_lat / 2,
        lon_min=float(lons.min()) - res_lon / 2,
        verified=True,
    )


def _save_with_provenance(grid: ModelGrid, provenance: dict) -> None:
    payload = {**asdict(grid), "provenance": provenance}
    GRID_JSON.write_text(json.dumps(payload, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-file", type=Path,
                    help="pickle with batch_metadata latitudes/longitudes; "
                         "gold standard, sets confirmed_against_batch")
    ap.add_argument("--from-release-code", action="store_true",
                    help="derive the extent from the public release code of "
                         "bfm-data (writer) + bfm-model (reader crop)")
    ap.add_argument("--bfm-repo", type=Path, help="path to bfm-model checkout")
    args = ap.parse_args()

    ref_lat, ref_lon = release_code_vectors()

    if args.batch_file:
        if args.batch_file.suffix == ".pt":
            import torch
            try:
                obj = torch.load(args.batch_file, map_location="cpu",
                                 weights_only=True)
            except Exception:
                obj = torch.load(args.batch_file, map_location="cpu",
                                 weights_only=False)
        else:
            with open(args.batch_file, "rb") as fh:
                obj = pickle.load(fh)
        meta = obj.get("batch_metadata", obj)
        lats = np.asarray(meta["latitudes"], dtype=float).ravel()
        lons = np.asarray(meta["longitudes"], dtype=float).ravel()
        # Raw batches carry 161/281; the model reader crops to 160/280.
        # Apply the same crop before comparing, but refuse silent surprises.
        if lats.size == 161 and lons.size == 281:
            lats, lons = lats[:160], lons[:280]
        elif not (lats.size == 160 and lons.size == 280):
            print(f"UNEXPECTED batch shape {lats.size}x{lons.size}; "
                  "expected 161x281 (raw) or 160x280 (cropped). Stop.")
            return 1
        grid = _extent(lats, lons)
        match = (np.allclose(lats, ref_lat, atol=1e-6)
                 and np.allclose(lons, ref_lon, atol=1e-6))
        if not match:
            print("\nMISMATCH between batch vectors and the release-code "
                  "derivation. The batch wins, but DECISIONS.md must be "
                  "amended before anything downstream runs.")
        _save_with_provenance(grid, {
            "method": "batch-metadata",
            "batch_file": str(args.batch_file),
            "confirmed_against_batch": True,
            "matches_release_code_derivation": bool(match),
            "date": date.today().isoformat(),
        })
    elif args.from_release_code:
        grid = _extent(ref_lat, ref_lon)
        _save_with_provenance(grid, {
            "method": "release-code-derivation",
            "writer": "BioDT/bfm-data scan_biocube.py + build_batches_monthly.py",
            "reader": "BioDT/bfm-model dataloader_monthly.py, patch_size=4",
            "confirmed_against_batch": False,
            "gate": "no model-facing inference until --batch-file confirms",
            "date": date.today().isoformat(),
        })
    elif args.bfm_repo:
        print("Produce one batch with the model dataloader and pass its "
              "pickle via --batch-file.\nSee bfm_model/bfm/dataset_basics.py, "
              "which already prints the vectors.")
        return 1
    else:
        ap.error("give --batch-file, --from-release-code, or --bfm-repo")

    print("\nVerified grid written to data/reference/model_grid.json")
    print(f"  lat {grid.lat_min} .. {grid.lat_max}")
    print(f"  lon {grid.lon_min} .. {grid.lon_max}")
    print(f"  bbox WKT: {grid.bbox_wkt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
