"""Load the released small checkpoint under forensically verified conditions
and run one forward pass. Architecture pinned by state-dict matching:
embed 256, heads 16, head_dim 64, PERCEIVER DEPTH 3, latents 8, swin (2,2)x8.
Controlled strict=False: the ONLY tolerated missing keys are (a) three
dead-branch PatchMerging fallback weights that postdate the checkpoint and
cannot execute at 40x70 patches, and (b) eleven ParameterList aliases of the
named latents, proven to share storage. Anything else aborts."""
import contextlib, inspect, json, os, sys, time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

AUDIT = Path(os.getcwd()).resolve()
BFM_REPO = AUDIT.parent / "bfm-model"
sys.path.insert(0, str(BFM_REPO))
from bfm_model.bfm.model import BFM  # noqa: E402
from bfm_model.bfm.dataloader_monthly import (  # noqa: E402
    LargeClimateDataset, custom_collate, batch_to_device,
)

# --- MPS float64 shim v2 ----------------------------------------------------
# FourierExpansion.forward casts to fp64 internally and returns .float() at
# the end; MPS has no fp64. Running the fp64 part on CPU and moving the fp32
# result back is numerically identical to the CUDA path (IEEE double either
# way). Class-level patch covers the module-level lead_time_expansion
# instance and any other use.
import bfm_model.swin_transformer.helpers.fourier_expansion as _fe  # noqa: E402

_orig_fourier_forward = _fe.FourierExpansion.forward

def _mps_safe_fourier_forward(self, x, d):
    if torch.is_tensor(x) and x.device.type == "mps":
        return _orig_fourier_forward(self, x.detach().cpu(), d).to(x.device)
    return _orig_fourier_forward(self, x, d)

_fe.FourierExpansion.forward = _mps_safe_fourier_forward
print("MPS fp64 shim v2: FourierExpansion.forward -> CPU fp64, fp32 back on device")
# ---------------------------------------------------------------------------

cfg = OmegaConf.load(BFM_REPO / "bfm_model/bfm/configs/train_config.yaml")
cfg.data.scaling.stats_path = str(
    BFM_REPO / "batch_statistics/monthly_batches_stats_splitted_channels.json")
cfg.data.land_mask_path = str(
    BFM_REPO / "batch_statistics/europe_Land_2020_grid.pkl")
sd = load_file(AUDIT / "data/external/weights/bfm-pretrained-small.safetensors")

ARCH = dict(embed_dim=256, num_heads=16, head_dim=64, depth=3,
            num_latent_tokens=8, swin_heads=8, swin_depths=(2, 2))
with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
    model = BFM(
        surface_vars=cfg.model.surface_vars, edaphic_vars=cfg.model.edaphic_vars,
        atmos_vars=cfg.model.atmos_vars, climate_vars=cfg.model.climate_vars,
        species_vars=cfg.model.species_vars,
        vegetation_vars=cfg.model.vegetation_vars, land_vars=cfg.model.land_vars,
        agriculture_vars=cfg.model.agriculture_vars,
        forest_vars=cfg.model.forest_vars, redlist_vars=cfg.model.redlist_vars,
        misc_vars=cfg.model.misc_vars, atmos_levels=cfg.data.atmos_levels,
        species_num=cfg.data.species_number, H=cfg.model.H, W=cfg.model.W,
        num_latent_tokens=ARCH["num_latent_tokens"], backbone_type="swin",
        patch_size=cfg.model.patch_size, embed_dim=ARCH["embed_dim"],
        num_heads=ARCH["num_heads"], head_dim=ARCH["head_dim"],
        depth=ARCH["depth"],
        learning_rate=cfg.training.lr, weight_decay=cfg.training.wd,
        batch_size=1, td_learning=cfg.training.td_learning,
        land_mask_path=cfg.data.land_mask_path, use_mask=cfg.training.use_mask,
        partially_masked_groups=cfg.training.partially_masked_groups,
        swin_encoder_depths=ARCH["swin_depths"],
        swin_encoder_num_heads=(ARCH["swin_heads"],) * 2,
        swin_decoder_depths=ARCH["swin_depths"],
        swin_decoder_num_heads=(ARCH["swin_heads"],) * 2,
        swin_window_size=tuple(cfg.model_swin_backbone.medium.window_size),
        swin_mlp_ratio=cfg.model_swin_backbone.medium.mlp_ratio,
        swin_qkv_bias=cfg.model_swin_backbone.medium.qkv_bias,
        swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.1,
        use_lora=False,
    )

msd = model.state_dict()
mk, wk = set(msd), set(sd)
bad = [k for k in (mk & wk) if msd[k].shape != sd[k].shape]
missing = mk - wk
unexpected = wk - mk
DEAD = {k for k in missing if "downsample.identity" in k}
ALIAS = {k for k in missing if "_latent_parameter_list" in k}
print(f"diff: unexpected={len(unexpected)} shape_mismatch={len(bad)} "
      f"missing={len(missing)} (dead-branch {len(DEAD)}, aliases {len(ALIAS)})")
assert not unexpected and not bad, "checkpoint has keys/shapes we cannot place"
assert missing == DEAD | ALIAS and len(DEAD) == 3 and len(ALIAS) == 11, \
    f"unexplained missing keys: {sorted(missing - DEAD - ALIAS)}"
assert cfg.model.H // cfg.model.patch_size > 1 and \
       cfg.model.W // cfg.model.patch_size > 1  # dead branch stays dead

res = model.load_state_dict(sd, strict=False)
assert set(res.missing_keys) == missing and not res.unexpected_keys

named = [n for n, _ in model.encoder.named_parameters(recurse=False)
         if n.endswith("latents")]
ptrs_named = {getattr(model.encoder, n).data_ptr() for n in named}
ptrs_list = {p.data_ptr() for p in model.encoder._latent_parameter_list}
assert ptrs_list <= ptrs_named, "ParameterList entries are NOT aliases"
chk = torch.equal(model.encoder.species_latents.cpu(),
                  sd["encoder.species_latents"])
print(f"alias check: list_ptrs subset of named_ptrs = True; "
      f"species_latents equals checkpoint = {chk}")
assert chk

bn = [n for n, mm in model.named_modules()
      if isinstance(mm, torch.nn.modules.batchnorm._BatchNorm)]
dp = [(n, getattr(mm, "drop_prob", None)) for n, mm in model.named_modules()
      if "droppath" in type(mm).__name__.lower()]
rates = sorted({round(p, 3) for _, p in dp if p is not None})
print(f"G3 pre-check: BatchNorm={len(bn)} DropPath={len(dp)} rates={rates}")
assert not bn

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device).eval()

sig = inspect.signature(LargeClimateDataset.__init__).parameters
want = dict(data_dir=str(AUDIT / "data/external/batches"),
            scaling_settings=cfg.data.scaling,
            num_species=cfg.data.species_number,
            atmos_levels=cfg.data.atmos_levels,
            mode="pretrain", model_patch_size=cfg.model.patch_size)
ds = LargeClimateDataset(**{k: v for k, v in want.items() if k in sig})
print(f"dataset samples (window pairs): {len(ds)}")
dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=custom_collate,
                                 shuffle=False)
x, y = next(iter(dl))
x = batch_to_device(x, device)
lt = getattr(model, "lead_time", 1)

with torch.no_grad():
    t0 = time.time(); out = model(x, lt); dt = time.time() - t0
sp = out.species_variables if hasattr(out, "species_variables") \
    else out["species_variables"]
k0 = next(iter(sp)); t = sp[k0].float()
print(f"forward OK on {device} in {dt:.1f}s  species={len(sp)} ch  "
      f"{k0} shape={tuple(t.shape)}  "
      f"finite%={100*float(torch.isfinite(t).float().mean()):.1f}")

Path("data/reference/model_arch.json").write_text(json.dumps(
    {**ARCH, "swin_depths": list(ARCH["swin_depths"]),
     "source": "state-dict forensics vs released small checkpoint"}))
Path("data/interim/smoke_report.json").write_text(json.dumps(dict(
    n_params_m=round(sum(v.numel() for v in sd.values()) / 1e6, 1),
    arch=ARCH | {"swin_depths": list(ARCH["swin_depths"])},
    dead_branch_keys=len(DEAD), alias_keys=len(ALIAS),
    batchnorm=len(bn), droppath=len(dp), droppath_rates=rates,
    device=device, mps_fp64_shim=True,
    forward_seconds=round(dt, 1),
    species_channels=len(sp), dataset_samples=len(ds))))
print("SMOKE PASS — arch + report written")
