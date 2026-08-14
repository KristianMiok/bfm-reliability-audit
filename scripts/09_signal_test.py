"""Decisive test: is the species modality predictable at all, and does the
model beat trivial baselines that use it?

Input = months (m, m+1); target = month m+2 (see compute_loss: target is
gt[:,1] of the NEXT batch). Baselines, all in the model's own scaled space,
over land cells:

  persistence : predict month m+2 = month m+1   (uses species(t) only)
  zero        : predict 0 everywhere            (exploits sparsity)
  climatology : predict the per-cell mean over the evaluation period
  model       : the released checkpoint

If persistence beats zero by a wide margin, species(t) carries real
predictive signal and discarding it is a defect. If the model loses to
persistence, it has failed the task in the most basic way available."""
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

cfg = load_config()
ds = make_dataset(cfg)
model, _ = load_model()
dev = next(model.parameters()).device.type
lt = getattr(model, "lead_time", 1)
with open(cfg.data.land_mask_path, "rb") as f:
    land = np.asarray(pickle.load(f)).astype(bool)

def sp_stack(b, t):
    sp = b.species_variables
    return np.stack([sp[k].reshape(-1, 160, 280)[t].float().numpy()
                     for k in sorted(sp)])

def run(b):
    bb = batch_to_device(b, dev)
    with torch.no_grad():
        o = model(bb, lt)
    sp = (o.species_variables if hasattr(o, "species_variables")
          else o["species_variables"])
    return np.stack([sp[k].reshape(160, 280).float().cpu().numpy()
                     for k in sorted(sp)])

P, T, PERS = [], [], []
n = len(ds)
print(f"windows: {n}")
for i in range(n):
    x, y = custom_collate([ds[i]])
    P.append(run(x))
    T.append(sp_stack(y, 1))      # target month m+2
    PERS.append(sp_stack(x, 1))   # last input month m+1
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{n}", flush=True)

P, T, PERS = np.stack(P), np.stack(T), np.stack(PERS)
L = land[None, None]
clim = T.mean(axis=0, keepdims=True)

def mae(a):
    return float(np.abs(a - T)[:, :, land].mean())

res = {"model": mae(P), "persistence": mae(PERS), "zero": mae(np.zeros_like(T)),
       "climatology": mae(np.broadcast_to(clim, T.shape))}
print("\n=== MAE on species targets (scaled space, land cells) ===")
for k, v in sorted(res.items(), key=lambda kv: kv[1]):
    print(f"  {k:<14}{v:.6e}")
best_triv = min(res["persistence"], res["zero"], res["climatology"])
print(f"\nmodel / best trivial baseline = {res['model']/best_triv:.3f}x "
      f"({'model wins' if res['model'] < best_triv else 'MODEL LOSES'})")
print(f"persistence / zero = {res['persistence']/res['zero']:.3f}x "
      f"({'species(t) carries signal' if res['persistence'] < res['zero'] else 'no persistence signal'})")

# how much signal is there, independent of any model?
t_flat = T[:, :, land].ravel()
p_flat = PERS[:, :, land].ravel()
nz = (t_flat > 0) | (p_flat > 0)
if nz.sum() > 10:
    r = np.corrcoef(t_flat[nz], p_flat[nz])[0, 1]
    print(f"\ncorr(species(t), species(t+1)) on cells ever occupied: {r:.4f} "
          f"(n={int(nz.sum()):,})")
print(f"target sparsity: {100*(t_flat == 0).mean():.1f}% of land cell-months are zero")
print(f"model output: min={P.min():.3e} max={P.max():.3e} "
      f"mean={P[:, :, land].mean():.3e}")
print(f"target      : min={T.min():.3e} max={T.max():.3e} "
      f"mean={T[:, :, land].mean():.3e}")

per_sp = []
for c in range(28):
    m = float(np.abs(P[:, c][:, land] - T[:, c][:, land]).mean())
    pz = float(np.abs(PERS[:, c][:, land] - T[:, c][:, land]).mean())
    per_sp.append((m, pz))
wins = sum(1 for m, pz in per_sp if m < pz)
print(f"\nper-species: model beats persistence on {wins}/28 channels")

np.savez_compressed(AUDIT / "data/interim/signal_test.npz",
                    mae=np.array([res[k] for k in sorted(res)]),
                    names=np.array(sorted(res)))
print("saved data/interim/signal_test.npz")
