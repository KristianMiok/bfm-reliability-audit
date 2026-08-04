"""Turn the GBIF download into a per-cell stratum on the model grid.

Output: one row per occupied 0.25 degree cell with the share of its occurrence
records that state any coordinate precision, plus publisher concentration.
That per-cell share is the grouping variable a group-conditional coverage test
conditions on -- the analogue of language in the BAN hate-speech work, where
BERT's own confidence separated correct from incorrect predictions for English
(p = 1.4e-11) and was worthless for Croatian (p = 1).

Usage:
    python scripts/03_build_stratum.py --download-key 0012345-260804000000000
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bfm_audit.config import require_verified_grid, RAW, INTERIM  # noqa: E402
from bfm_audit.grid import assign_cells, flag_reporting, cell_stratum  # noqa: E402

USE_COLS = [
    "gbifID", "speciesKey", "datasetKey", "decimalLatitude", "decimalLongitude",
    "coordinateUncertaintyInMeters", "year", "basisOfRecord",
]


def read_archive(path: Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Read the occurrence table from a GBIF simple/DwC-A zip in chunks."""
    with zipfile.ZipFile(path) as z:
        member = next((n for n in z.namelist()
                       if n.endswith(("occurrence.txt", ".csv"))), None)
        if member is None:
            raise RuntimeError(f"no occurrence table in {path}: {z.namelist()[:10]}")
        print(f"reading {member}")
        parts = []
        with z.open(member) as fh:
            for i, chunk in enumerate(pd.read_csv(
                fh, sep="\t", usecols=lambda c: c in USE_COLS,
                chunksize=chunksize, low_memory=False, on_bad_lines="warn",
            )):
                parts.append(chunk)
                print(f"  chunk {i}: {len(chunk):,} rows")
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-key", required=True)
    ap.add_argument("--min-records", type=int, default=20,
                    help="cells below this are flagged, not dropped")
    args = ap.parse_args()

    grid = require_verified_grid()
    archive = RAW / f"{args.download_key}.zip"
    if not archive.exists():
        raise SystemExit(f"{archive} not found. Fetch it first (gbif.fetch).")

    df = read_archive(archive)
    print(f"\n{len(df):,} records read")

    df = flag_reporting(df)
    print("\nrecord states:")
    print(df["state"].value_counts().to_string())

    df = assign_cells(df, grid)
    outside = (~df["in_grid"]).sum()
    print(f"\noutside the model grid: {outside:,} ({100*outside/len(df):.2f}%)")
    if outside / len(df) > 0.05:
        print("  WARNING: >5% outside. Check the grid extent before trusting this.")

    cells = cell_stratum(df, min_records=args.min_records)
    print(f"\noccupied cells: {len(cells):,} of {grid.height*grid.width:,}")
    ok = cells[cells["reliable_estimate"]]
    print(f"cells with >= {args.min_records} records: {len(ok):,}")

    if len(ok):
        q = ok["pct_reporting"].quantile([0, .1, .25, .5, .75, .9, 1])
        print("\npct_reporting across reliable cells:")
        print(q.round(1).to_string())
        print(f"\nspread p10-p90: {q[0.9] - q[0.1]:.1f} points")
        print("A wide spread here is what makes a group-conditional test possible.")
        print("A narrow one means the species-level gradient does not survive")
        print("aggregation to cells, and the audit has nothing to condition on.")

    out = INTERIM / "cell_stratum.parquet"
    cells.to_parquet(out, index=False)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
