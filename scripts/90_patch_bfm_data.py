"""Guard for BioDT/bfm-data build_batches_monthly: skip ERA5 files that
cannot yield a complete 2-month window (the released static auxiliary files
cvh/cvl/tvh/tvl carry a single 1996/2013 timestamp and crash the released
builder; none of their variables appear in the model's train_config).
Recorded in DECISIONS.md 2026-08-12. Idempotent."""
from pathlib import Path

p = Path.home() / ("Desktop/Papers/bfm-data/bfm_data/dataset_creation/"
                   "batch_creation/build_batches_monthly.py")
s = p.read_text()
GUARD = "except ValueError as e:  # bfm-reliability-audit guard"
if GUARD in s:
    print("already patched")
else:
    old = """        ds = _load_era5([Path(row.path)], t0, t1)
        slot = row.planned_slot"""
    new = """        try:
            ds = _load_era5([Path(row.path)], t0, t1)
        except ValueError as e:  # bfm-reliability-audit guard
            log.warning("skipping %s (no complete 2-month window: %s)",
                        row.path, e)
            continue
        slot = row.planned_slot"""
    assert old in s, "builder drifted; do not patch blind"
    p.write_text(s.replace(old, new, 1))
    print("builder patched")
