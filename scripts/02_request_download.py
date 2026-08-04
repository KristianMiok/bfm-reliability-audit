"""Submit the GBIF download for the 28 BioAnalyst training species.

Refuses to run against an unverified grid: a download built on a guessed
bounding box would be silently wrong at the edges and expensive to redo.

Usage:
    export GBIF_USER=... GBIF_PWD=... GBIF_EMAIL=...
    python scripts/02_request_download.py --year-from 2000 --year-to 2020
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bfm_audit.config import require_verified_grid, INTERIM  # noqa: E402
from bfm_audit import gbif  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-from", type=int, required=True,
                    help="match BioAnalyst's training period")
    ap.add_argument("--year-to", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the predicate without submitting")
    args = ap.parse_args()

    grid = require_verified_grid()
    pred = gbif.build_predicate(grid, args.year_from, args.year_to)

    print(json.dumps(pred, indent=2))
    if args.dry_run:
        print("\n(dry run, nothing submitted)")
        return 0

    key = gbif.request(grid, args.year_from, args.year_to)
    (INTERIM / "download_key.txt").write_text(key + "\n")
    print(f"\nDownload key: {key}")
    print("GBIF prepares large downloads asynchronously. Check status at")
    print(f"  https://www.gbif.org/occurrence/download/{key}")
    print("Then: python scripts/03_build_stratum.py --download-key", key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
