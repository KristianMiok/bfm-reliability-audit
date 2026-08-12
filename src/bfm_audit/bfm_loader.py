"""Load the released BioAnalyst small checkpoint under the forensically
verified conditions established by scripts/05_smoke_test.py (DECISIONS
2026-08-12). Single source of truth for every model-facing script."""
from __future__ import annotations

import contextlib
import inspect
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

AUDIT = Path(__file__).resolve().parents[2]
BFM_REPO = AUDIT.parent / "bfm-model"
sys.path.insert(0, str(BFM_REPO))

from bfm_model.bfm.model import BFM  # noqa: E402
from bfm_model.bfm.dataloader_monthly import (  # noqa: E402
    LargeClimateDataset, custom_collate, batch_to_device,
)
import bfm_model.swin_transformer.helpers.fourier_expansion as _fe  # noqa: E402

ARCH = dict(embed_dim=256, num_heads=16, head_dim=64, depth=3,
            num_latent_tokens=8, swin_heads=8, swin_depths=(2, 2))

_orig_fourier = _fe.FourierExpansion.forward

def _mps_safe_fourier(self, x, d):
    if torch.is_tensor(x) and x.device.type == "mps":
        return _orig_fourier(self, x.detach().cpu(), d).to(x.device)
    return _orig_fourier(self, x, d)

_fe.FourierExpansion.forward = _mps_safe_fourier


def load_config():
    cfg = OmegaConf.load(BFM_REPO / "bfm_model/bfm/configs/train_config.yaml")
    cfg.data.scaling.stats_path = str(
        BFM_REPO / "batch_statistics/monthly_batches_stats_splitted_channels.json")
    cfg.data.land_mask_path = str(
        BFM_REPO / "batch_statistics/europe_Land_2020_grid.pkl")
    return cfg


def load_model(device: str | None = None) -> tuple[torch.nn.Module, dict]:
    cfg = load_config()
    sd = load_file(AUDIT / "data/external/weights/bfm-pretrained-small.safetensors")
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        model = BFM(
            surface_vars=cfg.model.surface_vars,
            edaphic_vars=cfg.model.edaphic_vars,
            atmos_vars=cfg.model.atmos_vars, climate_vars=cfg.model.climate_vars,
            species_vars=cfg.model.species_vars,
            vegetation_vars=cfg.model.vegetation_vars,
            land_vars=cfg.model.land_vars,
            agriculture_vars=cfg.model.agriculture_vars,
            forest_vars=cfg.model.forest_vars,
            redlist_vars=cfg.model.redlist_vars,
            misc_vars=cfg.model.misc_vars, atmos_levels=cfg.data.atmos_levels,
            species_num=cfg.data.species_number, H=cfg.model.H, W=cfg.model.W,
            num_latent_tokens=ARCH["num_latent_tokens"], backbone_type="swin",
            patch_size=cfg.model.patch_size, embed_dim=ARCH["embed_dim"],
            num_heads=ARCH["num_heads"], head_dim=ARCH["head_dim"],
            depth=ARCH["depth"], learning_rate=cfg.training.lr,
            weight_decay=cfg.training.wd, batch_size=1,
            td_learning=cfg.training.td_learning,
            land_mask_path=cfg.data.land_mask_path,
            use_mask=cfg.training.use_mask,
            partially_masked_groups=cfg.training.partially_masked_groups,
            swin_encoder_depths=ARCH["swin_depths"],
            swin_encoder_num_heads=(ARCH["swin_heads"],) * 2,
            swin_decoder_depths=ARCH["swin_depths"],
            swin_decoder_num_heads=(ARCH["swin_heads"],) * 2,
            swin_window_size=tuple(cfg.model_swin_backbone.medium.window_size),
            swin_mlp_ratio=cfg.model_swin_backbone.medium.mlp_ratio,
            swin_qkv_bias=cfg.model_swin_backbone.medium.qkv_bias,
            swin_drop_rate=0.0, swin_attn_drop_rate=0.0,
            swin_drop_path_rate=0.1, use_lora=False,
        )
    msd = model.state_dict()
    mk, wk = set(msd), set(sd)
    bad = [k for k in (mk & wk) if msd[k].shape != sd[k].shape]
    missing = mk - wk
    dead = {k for k in missing if "downsample.identity" in k}
    alias = {k for k in missing if "_latent_parameter_list" in k}
    assert not (wk - mk) and not bad, "checkpoint/instance mismatch"
    assert missing == dead | alias and len(dead) == 3 and len(alias) == 11
    res = model.load_state_dict(sd, strict=False)
    assert set(res.missing_keys) == missing and not res.unexpected_keys
    assert torch.equal(model.encoder.species_latents.cpu(),
                       sd["encoder.species_latents"])
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return model.to(device).eval(), cfg


def droppath_modules(model):
    return [m for m in model.modules()
            if "droppath" in type(m).__name__.lower()]


def enable_stochastic_depth(model):
    """model stays in eval(); ONLY DropPath modules go to train mode."""
    model.eval()
    mods = droppath_modules(model)
    for m in mods:
        m.train()
    return len(mods)


def make_dataset(cfg):
    sig = inspect.signature(LargeClimateDataset.__init__).parameters
    want = dict(data_dir=str(AUDIT / "data/external/batches"),
                scaling_settings=cfg.data.scaling,
                num_species=cfg.data.species_number,
                atmos_levels=cfg.data.atmos_levels,
                mode="pretrain", model_patch_size=cfg.model.patch_size)
    return LargeClimateDataset(**{k: v for k, v in want.items() if k in sig})


__all__ = ["AUDIT", "BFM_REPO", "ARCH", "load_model", "load_config",
           "make_dataset", "custom_collate", "batch_to_device",
           "enable_stochastic_depth", "droppath_modules"]
