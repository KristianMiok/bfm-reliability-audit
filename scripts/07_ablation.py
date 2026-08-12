"""Systematic input-pathway ablation across windows (paper-grade evidence).

For each tested window: zero each input group in turn, plus two adversarial
species perturbations (channel permutation, x100 amplification), and measure
the response of the species OUTPUT over land. Deterministic: DropPath in
eval, no MC sampling. Also records output spatial structure, so a constant
map cannot masquerade as a prediction."""
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bfm_audit.bfm_loader import (  # noqa: E402
    AUDIT, load_model, load_config, make_dataset, custom_collate,
    batch_to_device,
)

WINDOWS = [0, 4, 9, 14, 17]      # spread across seasons; 5..17 are final-13
GROUPS = ["species_variables", "atmospheric_variables", "climate_variables",
          "surface_variables", "edaphic_variables", "vegetation_variables",
          "land_variables", "agriculture_variables", "forest_variables",
          "redlist_variables", "misc_variables"]

cfg = load_config()
ds = make_dataset(cfg)
model, _ = load_model()
device = next(model.parameters()).device.type
lt = getattr(model, "lead_time", 1)
with open(cfg.data.land_mask_path, "rb") as fh:
    land = np.asarray(pickle.load(fh)).astype(bool)
print(f"device={device}  windows={WINDOWS}  land cells={land.sum():,}")

def run(batch):
    b = batch_to_device(copy.deepcopy(batch), device)
    with torch.no_grad():
        out = model(b, lt)
    sp = (out.species_variables if hasattr(out, "species_variables")
          else out["species_variables"])
    return np.stack([sp[k].reshape(160, 280).float().cpu().numpy()
                     for k in sorted(sp)])

rows = []
for w in WINDOWS:
    xb, yb = custom_collate([ds[w]])
    base = run(xb)
    scale = float(base[:, land].std())
    # spatial structure of the prediction itself
    per_cell = base[:, land]
    struct = float(per_cell.std(axis=1).mean() / max(abs(per_cell.mean()), 1e-30))
    print(f"\nwindow {w}: output std(land)={scale:.4e}  "
          f"per-species spatial CV={struct:.3f}")

    variants = {}
    for g in GROUPS:
        xa = copy.deepcopy(xb)
        d = getattr(xa, g)
        for k in d:
            d[k] = torch.zeros_like(d[k])
        variants[f"zero:{g}"] = xa
    xp = copy.deepcopy(xb)
    keys = sorted(xp.species_variables)
    shuffled = keys[7:] + keys[:7]
    orig = {k: xp.species_variables[k].clone() for k in keys}
    for k, src in zip(keys, shuffled):
        xp.species_variables[k] = orig[src]
    variants["permute:species"] = xp
    xamp = copy.deepcopy(xb)
    for k in xamp.species_variables:
        xamp.species_variables[k] = xamp.species_variables[k] * 100.0
    variants["x100:species"] = xamp

    for name, xa in variants.items():
        o = run(xa)
        md = float(np.abs(o - base)[:, land].mean())
        rows.append(dict(window=w, variant=name, mean_abs_d=md,
                         max_abs_d=float(np.abs(o - base).max()),
                         relative=md / max(scale, 1e-30),
                         out_std=scale, spatial_cv=struct))
        print(f"  {name:<30}{md:>12.4e}{md/max(scale,1e-30):>10.2%}")

out = AUDIT / "data/interim/ablation.json"
out.write_text(json.dumps(rows, indent=1))
print(f"\nwritten {out}")

import statistics as st
print("\n=== summary across windows (relative mean|d|) ===")
for name in sorted({r["variant"] for r in rows}):
    v = [r["relative"] for r in rows if r["variant"] == name]
    print(f"{name:<30} median={st.median(v):>9.4%}  max={max(v):>9.4%}")
