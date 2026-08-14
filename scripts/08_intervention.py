"""Intervention test + signal localisation.

Part A — does a different input-scaling convention wake the species pathway?
The release scales each channel by its per-channel max, so sparse count
channels arrive at ~1/136 the amplitude of atmospheric ones. We rebuild the
species inputs under three conventions and, WITHIN each convention, measure
the model's sensitivity to species presence vs absence. This separates
"input convention problem" (fixable without training) from "learned
attenuation" (needs retraining).

Part B — where does the species signal die? Forward hooks at the encoder
embedding, the perceiver fusion, the backbone and the decoder projection;
for each, the relative change between the species-present and species-absent
runs. The first stage where the difference collapses is the culprit.
"""
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bfm_audit.bfm_loader import (  # noqa: E402
    AUDIT, BFM_REPO, load_model, load_config, make_dataset, custom_collate,
    batch_to_device,
)

WINDOW = 0
cfg = load_config()
ds = make_dataset(cfg)
model, _ = load_model()
dev = next(model.parameters()).device.type
lt = getattr(model, "lead_time", 1)
with open(cfg.data.land_mask_path, "rb") as f:
    land = np.asarray(pickle.load(f)).astype(bool)
stats = json.loads(
    (BFM_REPO / "batch_statistics/monthly_batches_stats_splitted_channels.json"
     ).read_text())["species_variables"]

x0, _ = custom_collate([ds[WINDOW]])

def raw_species(batch):
    """Undo the release's per-channel max scaling -> raw counts."""
    return {k: v.float() * float(stats[k]["max"])
            for k, v in batch.species_variables.items()}

def apply_convention(batch, raw, mode, present=True):
    b = copy.deepcopy(batch)
    for k, r in raw.items():
        r = r if present else torch.zeros_like(r)
        s = stats[k]
        if mode == "max":
            v = r / float(s["max"])
        elif mode == "std":
            v = r / float(s["std"])
        elif mode == "zscore":
            v = (r - float(s["mean"])) / float(s["std"])
        b.species_variables[k] = v.to(batch.species_variables[k].dtype)
    return b

def run(batch, hooks=None):
    b = batch_to_device(copy.deepcopy(batch), dev)
    with torch.no_grad():
        out = model(b, lt)
    sp = (out.species_variables if hasattr(out, "species_variables")
          else out["species_variables"])
    return np.stack([sp[k].reshape(160, 280).float().cpu().numpy()
                     for k in sorted(sp)])

raw = raw_species(x0)
print("=== Part A: sensitivity to species presence, per scaling convention ===")
print(f"{'convention':<12}{'input std':>12}{'mean|d|':>14}{'relative':>12}")
resA = {}
for mode in ("max", "std", "zscore"):
    bp = apply_convention(x0, raw, mode, present=True)
    ba = apply_convention(x0, raw, mode, present=False)
    istd = float(np.stack([v.float().numpy().ravel()
                           for v in bp.species_variables.values()]).std())
    op, oa = run(bp), run(ba)
    scale = float(op[:, land].std())
    d = float(np.abs(op - oa)[:, land].mean())
    resA[mode] = dict(input_std=istd, mean_abs_d=d, relative=d / max(scale, 1e-30),
                      out_std=scale)
    print(f"{mode:<12}{istd:>12.4e}{d:>14.4e}{d/max(scale,1e-30):>11.4%}")
print("reference: zeroing atmospheric inputs moves the output 5.87% (median)")

print("\n=== Part B: where does the species signal die? ===")
TARGETS = ["encoder.species_token_embeds", "encoder.pre_perceiver_norm",
           "encoder.perceiver_io", "backbone", "decoder.perceiver_io",
           "decoder.species_token_proj"]
mods = dict(model.named_modules())
hooked = [t for t in TARGETS if t in mods]
print(f"hook points found: {hooked}")

captured = {}
def mk(name):
    def hook(_m, _i, o):
        t = o
        while isinstance(t, (tuple, list)) and t:
            t = t[0]
        if torch.is_tensor(t):
            captured[name] = t.detach().float().cpu().numpy()
    return hook

handles = [mods[t].register_forward_hook(mk(t)) for t in hooked]
bp = apply_convention(x0, raw, "max", present=True)
ba = apply_convention(x0, raw, "max", present=False)
run(bp); act_present = dict(captured); captured.clear()
run(ba); act_absent = dict(captured)
for h in handles:
    h.remove()

print(f"\n{'stage':<34}{'shape':>18}{'rel. change':>14}")
for t in hooked:
    if t not in act_present or t not in act_absent:
        continue
    a, b = act_present[t], act_absent[t]
    if a.shape != b.shape:
        print(f"{t:<34}{'shape mismatch':>18}")
        continue
    rel = float(np.abs(a - b).mean() / max(np.abs(a).std(), 1e-30))
    print(f"{t:<34}{str(a.shape):>18}{rel:>13.4%}")

Path(AUDIT / "data/interim/intervention.json").write_text(json.dumps(
    dict(window=WINDOW, part_a=resA,
         part_b={t: float(np.abs(act_present[t] - act_absent[t]).mean()
                          / max(np.abs(act_present[t]).std(), 1e-30))
                 for t in hooked
                 if t in act_present and act_present[t].shape == act_absent[t].shape}),
    indent=1))
print("\nwritten data/interim/intervention.json")
