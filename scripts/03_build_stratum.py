"""Turn the GBIF download into a per-cell stratum on the model grid.

Output: one row per occupied 0.25 degree cell with the share of its occurrence
records that state any coordinate precision, plus publisher concentration.
That per-cell share is the grouping variable a group-conditional coverage test
conditions on -- the analogue of language in the BAN hate-speech work, where
BERT's own confidence separated correct from incorrect predictions for English
(p = 1.4e-11) and was worthless for Croatian (p = 1).

Evaluates the pre-registered criterion from DECISIONS.md ("Open criterion --
the spatial stratum") and prints an explicit verdict. The verdict can kill
the project; if it does, that outcome gets written up rather than worked
around.

Usage:
    python scripts/03_build_stratum.py --download-key 0012345-260804000000000
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bfm_audit.config import require_verified_grid, RAW, INTERIM  # noqa: E402
from bfm_audit.grid import (  # noqa: E402
    assign_cells, cell_stratum, flag_reporting, training_window_mask,
)

USE_COLS = [
    "speciesKey", "datasetKey", "decimalLatitude", "decimalLongitude",
    "coordinateUncertaintyInMeters", "year", "month", "basisOfRecord",
]

DTYPES = {
    "speciesKey": "float64",
    "decimalLatitude": "float64",
    "decimalLongitude": "float64",
    "coordinateUncertaintyInMeters": "float64",
    "year": "float32",
    "month": "float32",
}

# Pre-registered criterion (DECISIONS.md, fixed before this script ran).
SPREAD_PASS = 30.0
SPREAD_FAIL = 15.0
TERCILE_MIN_CELLS = 200


def read_archive(path: Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Read the occurrence table from a GBIF SIMPLE_CSV zip in chunks.

    GBIF's TSV is unquoted; stray double quotes inside free-text columns
    derail pandas' default quote handling, hence QUOTE_NONE.
    """
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
                dtype=DTYPES, quoting=csv.QUOTE_NONE,
                chunksize=chunksize, low_memory=False, on_bad_lines="warn",
            )):
                chunk["basisOfRecord"] = chunk["basisOfRecord"].astype("category")
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

    # Training-window tail cut (decision 2, 2026-08-10): the download runs to
    # end-2020, the model saw data to 2020-06.
    keep = training_window_mask(df)
    n_tail = (~keep).sum()
    n_nomonth = ((df["year"] == 2020) & df["month"].isna()).sum()
    print(f"outside training window (>2020-06): {n_tail:,} dropped "
          f"(of which year-2020 with no month: {n_nomonth:,})")
    df = df[keep]

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
    ok = cells[cells["reliable_estimate"]].copy()
    print(f"cells with >= {args.min_records} records: {len(ok):,}")

    out = INTERIM / "cell_stratum.parquet"
    cells.to_parquet(out, index=False)
    print(f"saved {out}")

    if not len(ok):
        print("\nVERDICT: no reliable cells -- nothing to evaluate.")
        return 1

    q = ok["pct_reporting"].quantile([0, .1, .25, .5, .75, .9, 1])
    spread = q[0.9] - q[0.1]
    print("\npct_reporting across reliable cells:")
    print(q.round(1).to_string())
    print(f"\nspread p10-p90: {spread:.1f} points")

    # Rank-based terciles: robust to heavy ties (many cells at exactly 0 or
    # 100). Ties on a boundary split arbitrarily; the E2 grouping definition
    # is finalised in the pre-registration, this is the registered count check.
    ok["tercile"] = pd.qcut(ok["pct_reporting"].rank(method="first"), 3,
                            labels=["bottom", "middle", "top"])
    tct = ok["tercile"].value_counts()
    n_bot = int(tct.get("bottom", 0))
    n_top = int(tct.get("top", 0))
    print(f"tercile cell counts: bottom={n_bot:,}  "
          f"middle={int(tct.get('middle', 0)):,}  top={n_top:,}")

    # Publisher-map check (C13 mechanism): is the stratum just an hhi map?
    # Spearman = Pearson on ranks; computed directly to avoid a scipy
    # dependency (pandas delegates method="spearman" to scipy).
    r_hhi = ok["pct_reporting"].rank().corr(ok["dataset_hhi"].rank())
    mono_bot = (ok.loc[ok["tercile"] == "bottom", "dataset_hhi"] > 0.5).mean()
    mono_top = (ok.loc[ok["tercile"] == "top", "dataset_hhi"] > 0.5).mean()
    print(f"\npublisher concentration: Spearman(pct_reporting, hhi) = {r_hhi:+.3f}")
    print(f"share of near-single-publisher cells (hhi>0.5): "
          f"bottom {100*mono_bot:.0f}%  vs  top {100*mono_top:.0f}%")

    print("\n--- pre-registered criterion ---")
    if spread >= SPREAD_PASS and min(n_bot, n_top) >= TERCILE_MIN_CELLS:
        print(f"VERDICT: PASS  (spread {spread:.1f} >= {SPREAD_PASS}, "
              f"extreme terciles >= {TERCILE_MIN_CELLS} cells)")
        print("Proceed to E2 pre-registration. The hhi numbers above decide how")
        print("the stratum may be described, not whether it exists.")
    elif spread < SPREAD_FAIL:
        print(f"VERDICT: FAIL  (spread {spread:.1f} < {SPREAD_FAIL})")
        print("The species-level gradient does not survive aggregation to cells.")
        print("Write up the negative result. Do not touch the model.")
    else:
        print(f"VERDICT: BETWEEN  (spread {spread:.1f}, "
              f"terciles bottom={n_bot}, top={n_top})")
        print("Report the attenuation. Do not proceed to the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
