"""Read BioAnalyst's actual lat/lon extent from a real batch.

The model's grid position is NOT in train_config.yaml. H=160, W=280 and the
0.25 degree resolution are, but the coordinate vectors arrive at runtime via
``batch['batch_metadata']['latitudes']`` and ``['longitudes']``. Everything
downstream that places a record in a cell depends on where the grid actually
sits, so this must be measured rather than assumed.

Usage:
    python scripts/01_extract_model_grid.py --bfm-repo /path/to/bfm-model-main

Requires one prepared batch from the model's dataloader. If you cannot produce
a batch yet, an alternative is to read the coordinate vectors out of any
single NetCDF/pickle in the model's dataset directory -- pass --batch-file.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bfm_audit.config import ModelGrid  # noqa: E402


def _extent(lats, lons) -> ModelGrid:
    lats = np.asarray(lats, dtype=float).ravel()
    lons = np.asarray(lons, dtype=float).ravel()

    res_lat = float(np.median(np.abs(np.diff(lats))))
    res_lon = float(np.median(np.abs(np.diff(lons))))
    print(f"  latitudes : n={lats.size}  {lats.min():.3f} .. {lats.max():.3f}  step {res_lat:.4f}")
    print(f"  longitudes: n={lons.size}  {lons.min():.3f} .. {lons.max():.3f}  step {res_lon:.4f}")

    if not np.isclose(res_lat, 0.25, atol=1e-3) or not np.isclose(res_lon, 0.25, atol=1e-3):
        print("  WARNING: step is not 0.25 deg. Do not proceed until this is understood.")

    # Cell edges, not centres: GBIF coordinates are points and must be binned.
    # Batch metadata conventionally gives cell centres, so pad by half a cell.
    grid = ModelGrid(
        height=int(lats.size),
        width=int(lons.size),
        resolution=round((res_lat + res_lon) / 2, 4),
        lat_max=float(lats.max()) + res_lat / 2,
        lon_min=float(lons.min()) - res_lon / 2,
        verified=True,
    )
    return grid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfm-repo", type=Path, help="path to bfm-model-main")
    ap.add_argument("--batch-file", type=Path,
                    help="pickle with 'latitudes'/'longitudes', bypasses the dataloader")
    args = ap.parse_args()

    if args.batch_file:
        with open(args.batch_file, "rb") as fh:
            obj = pickle.load(fh)
        meta = obj.get("batch_metadata", obj)
        grid = _extent(meta["latitudes"], meta["longitudes"])
    elif args.bfm_repo:
        sys.path.insert(0, str(args.bfm_repo))
        print("Import the model dataloader and produce one batch, then pass its\n"
              "batch_metadata to _extent(). See bfm_model/bfm/dataset_basics.py,\n"
              "which already prints these vectors.")
        return 1
    else:
        ap.error("give --batch-file or --bfm-repo")

    grid.save()
    print("\nVerified grid written to data/reference/model_grid.json")
    print(f"  lat {grid.lat_min} .. {grid.lat_max}")
    print(f"  lon {grid.lon_min} .. {grid.lon_max}")
    print(f"  bbox WKT: {grid.bbox_wkt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
