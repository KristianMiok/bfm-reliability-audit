"""G3 item 3 — K pilot on ONE window OUTSIDE the final-13 evaluation set.

Window pairs are (file_i, file_{i+1}), i = 0..17; the registered final-13
are i = 5..17. Pilot uses i = 0 (x = 2018-11/12). K = 64 MC samples with
stochastic depth enabled surgically; sigma stability for K in {8,16,32,64}
is Spearman rank correlation with the K=64 sigma over land cells, pooled
across the 28 species channels. Smallest K with rho >= 0.99 is frozen."""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bfm_audit.bfm_loader import (  # noqa: E402
    AUDIT, load_model, make_dataset, custom_collate, batch_to_device,
    enable_stochastic_depth,
)

PILOT_WINDOW = 0
K_MAX = 64
K_GRID = [8, 16, 32, 64]

model, cfg = load_model()
device = next(model.parameters()).device.type
n_dp = enable_stochastic_depth(model)
print(f"stochastic depth enabled on {n_dp} DropPath modules; device={device}")
assert n_dp == 6

ds = make_dataset(cfg)
x, _y = custom_collate([ds[PILOT_WINDOW]])
x = batch_to_device(x, device)
lt = getattr(model, "lead_time", 1)

with open(cfg.data.land_mask_path, "rb") as fh:
    land = np.asarray(pickle.load(fh)).astype(bool)
print(f"land cells: {land.sum():,} / {land.size:,}")

samples = np.empty((K_MAX, 28, 160, 280), dtype=np.float32)
times = []
for k in range(K_MAX):
    torch.manual_seed(1000 + k)
    if device == "mps":
        torch.mps.manual_seed(1000 + k)
    t0 = time.time()
    with torch.no_grad():
        out = model(x, lt)
    times.append(time.time() - t0)
    sp = (out.species_variables if hasattr(out, "species_variables")
          else out["species_variables"])
    for c, key in enumerate(sorted(sp)):
        t = sp[key]
        samples[k, c] = t.reshape(160, 280).float().cpu().numpy()
    if k in (0, 1) or (k + 1) % 8 == 0:
        print(f"  sample {k+1}/{K_MAX}  {times[-1]:.1f}s", flush=True)

d01 = float(np.abs(samples[0] - samples[1]).max())
print(f"stochasticity check: max|s0-s1| = {d01:.6f}")
assert d01 > 0, "DropPath produced identical samples — mechanism inactive"

def ranks(a):
    order = np.argsort(a, kind="mergesort")
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(a.size)
    return r

sigma64 = samples.std(axis=0, ddof=1)          # (28,160,280)
flat64 = sigma64[:, land].ravel()
report = {}
print("\nK-stability (Spearman of per-cell sigma vs K=64, land only):")
for K in K_GRID:
    sig = samples[:K].std(axis=0, ddof=1)
    flat = sig[:, land].ravel()
    rho = float(np.corrcoef(ranks(flat), ranks(flat64))[0, 1])
    per_sp = [float(np.corrcoef(ranks(sig[c][land]),
                                ranks(sigma64[c][land]))[0, 1])
              for c in range(28)]
    report[K] = dict(rho_pooled=round(rho, 4),
                     rho_species_min=round(min(per_sp), 4))
    print(f"  K={K:>2}: pooled rho={rho:.4f}  per-species min={min(per_sp):.4f}")

chosen = next(K for K in K_GRID if report[K]["rho_pooled"] >= 0.99)
med_t = float(np.median(times[5:]))
print(f"\nCHOSEN K = {chosen}  (registered rule: smallest K with pooled rho >= 0.99)")
print(f"steady-state forward: {med_t:.1f}s  ->  main run "
      f"~{13 * chosen * med_t / 60:.0f} min for 13 windows")

np.savez_compressed(AUDIT / "data/interim/kpilot_sigma.npz",
                    sigma64=sigma64, land=land)
(AUDIT / "data/interim/kpilot_report.json").write_text(json.dumps(dict(
    pilot_window=PILOT_WINDOW, k_grid=K_GRID, chosen_k=chosen,
    stability=report, droppath_modules=n_dp,
    forward_seconds_median=round(med_t, 2),
    stochasticity_max_abs_diff=d01)))
print("K-PILOT DONE — report written")
