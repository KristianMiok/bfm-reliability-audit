"""First contact with the released model: weights census, exact-match
instantiation, G3 item-3 pre-checks, one forward pass, timing.
Writes data/interim/smoke_report.json; refuses the forward on any mismatch."""
import inspect, json, sys, time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

AUDIT = Path(__file__).resolve().parents[1]
BFM_REPO = AUDIT.parent / "bfm-model"
sys.path.insert(0, str(BFM_REPO))
from bfm_model.bfm.model import BFM  # noqa: E402
from bfm_model.bfm.dataloader_monthly import (  # noqa: E402
    LargeClimateDataset, custom_collate, batch_to_device,
)

cfg = OmegaConf.load(BFM_REPO / "bfm_model/bfm/configs/train_config.yaml")
cfg.data.scaling.stats_path = str(
    BFM_REPO / "batch_statistics/monthly_batches_stats_splitted_channels.json")
cfg.data.land_mask_path = str(
    BFM_REPO / "batch_statistics/europe_Land_2020_grid.pkl")

sd = load_file(AUDIT / "data/external/weights/bfm-pretrained-small.safetensors")
n_params = sum(v.numel() for v in sd.values())
print(f"weights: {len(sd)} tensors, {n_params/1e6:.1f}M params")
print("sample keys:", list(sd)[:4])

def build(size):
    c = cfg.model_swin_backbone[size]
    return BFM(
        surface_vars=cfg.model.surface_vars, edaphic_vars=cfg.model.edaphic_vars,
        atmos_vars=cfg.model.atmos_vars, climate_vars=cfg.model.climate_vars,
        species_vars=cfg.model.species_vars,
        vegetation_vars=cfg.model.vegetation_vars,
        land_vars=cfg.model.land_vars,
        agriculture_vars=cfg.model.agriculture_vars,
        forest_vars=cfg.model.forest_vars, redlist_vars=cfg.model.redlist_vars,
        misc_vars=cfg.model.misc_vars, atmos_levels=cfg.data.atmos_levels,
        species_num=cfg.data.species_number, H=cfg.model.H, W=cfg.model.W,
        num_latent_tokens=cfg.model.num_latent_tokens,
        backbone_type=cfg.model.backbone, patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim, num_heads=cfg.model.num_heads,
        head_dim=cfg.model.head_dim, depth=cfg.model.depth,
        learning_rate=cfg.training.lr, weight_decay=cfg.training.wd,
        batch_size=1, td_learning=cfg.training.td_learning,
        land_mask_path=cfg.data.land_mask_path, use_mask=cfg.training.use_mask,
        partially_masked_groups=cfg.training.partially_masked_groups,
        swin_encoder_depths=tuple(c.encoder_depths),
        swin_encoder_num_heads=tuple(c.encoder_num_heads),
        swin_decoder_depths=tuple(c.decoder_depths),
        swin_decoder_num_heads=tuple(c.decoder_num_heads),
        swin_window_size=tuple(c.window_size), swin_mlp_ratio=c.mlp_ratio,
        swin_qkv_bias=c.qkv_bias, swin_drop_rate=c.drop_rate,
        swin_attn_drop_rate=c.attn_drop_rate,
        swin_drop_path_rate=c.drop_path_rate, use_lora=c.use_lora,
    )

sizes = [cfg.model.swin_backbone_size] + [
    s for s in cfg.model_swin_backbone if s != cfg.model.swin_backbone_size]
chosen, model = None, None
for size in sizes:
    m = build(size)
    msd = m.state_dict()
    mk, wk = set(msd), set(sd)
    bad = [k for k in (mk & wk) if msd[k].shape != sd[k].shape]
    print(f"[{size}] missing={len(wk - mk)} unexpected={len(mk - wk)} "
          f"shape_mismatch={len(bad)}")
    if not (wk - mk) and not (mk - wk) and not bad:
        chosen, model = size, m
        break
if model is None:
    print("NO exact architecture match — refusing to load. STOP.")
    raise SystemExit(1)
model.load_state_dict(sd, strict=True)
print(f"loaded strict=True into swin '{chosen}'")

bn = [n for n, mm in model.named_modules()
      if isinstance(mm, torch.nn.modules.batchnorm._BatchNorm)]
dp = [(n, getattr(mm, "drop_prob", None)) for n, mm in model.named_modules()
      if "droppath" in type(mm).__name__.lower()]
print(f"G3 pre-check: BatchNorm = {len(bn)}  DropPath = {len(dp)}  "
      f"rates = {sorted({round(p,3) for _, p in dp if p is not None})}")
if bn:
    print("BatchNorm present — G3 item 3 needs an isolation strategy. STOP.")
    raise SystemExit(1)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device).eval()

sig = inspect.signature(LargeClimateDataset.__init__).parameters
want = dict(data_dir=str(AUDIT / "data/external/batches"),
            scaling_settings=cfg.data.scaling,
            num_species=cfg.data.species_number,
            atmos_levels=cfg.data.atmos_levels,
            mode="pretrain", model_patch_size=cfg.model.patch_size)
kwargs = {k: v for k, v in want.items() if k in sig}
print("dataset kwargs:", sorted(kwargs))
ds = LargeClimateDataset(**kwargs)
print(f"dataset samples: {len(ds)}")
dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=custom_collate,
                                 shuffle=False)
x, y = next(iter(dl))
x = batch_to_device(x, device)

with torch.no_grad():
    t0 = time.time(); out = model(x, None); dt = time.time() - t0
print(f"forward OK on {device} in {dt:.1f}s  out={type(out).__name__}")
sp = out.species_variables if hasattr(out, "species_variables") else out["species_variables"]
k0 = next(iter(sp)); t = sp[k0].float()
print(f"species out: {len(sp)} channels  {k0} shape={tuple(t.shape)}  "
      f"finite%={100*float(torch.isfinite(t).float().mean()):.1f}  "
      f"mean={float(t[torch.isfinite(t)].mean()):.4f}")

(AUDIT / "data/interim/smoke_report.json").write_text(json.dumps(dict(
    n_params_m=round(n_params/1e6, 1), backbone_size=chosen,
    batchnorm=len(bn), droppath=len(dp),
    droppath_rates=sorted({round(p,3) for _, p in dp if p is not None}),
    device=device, forward_seconds=round(dt, 1), species_channels=len(sp),
    dataset_samples=len(ds))))
print("SMOKE PASS — report written")
