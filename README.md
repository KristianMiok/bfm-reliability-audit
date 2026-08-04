# bfm-reliability-audit

A group-conditional reliability audit of **BioAnalyst**, a multimodal
biodiversity foundation model for Europe at 0.25° resolution
([`bfm-model`](https://github.com/BioDT), Trantas et al. 2025).

## The question

BioAnalyst ships with **no uncertainty estimate of any kind**. Its loss is mean
absolute error on a deterministic point prediction; grepping the model source
for `uncertain|variance|ensemble|posterior|confidence|calibrat|quantile|conformal`
returns nothing. Meanwhile a 2026 perspective from an overlapping community
(Herzschuh et al., *Front. Ecol. Evol.*, with Zurell as senior author) argues
that the next generation of such models must offer transparent uncertainty,
disequilibrium-aware evaluation, and benchmarks reporting *discrimination,
calibration, trajectory coherence and uncertainty coverage*.

This project does three things:

1. **Add** an uncertainty estimate to a model that has none, by sampling with
   stochastic depth left active at inference (`drop_path_rate: 0.1` is the only
   stochastic component the config leaves on — `drop_rate` and
   `attn_drop_rate` are both `0.0`, so classical MC-Dropout is unavailable
   without retraining).
2. **Measure** whether that uncertainty is calibrated.
3. **Measure it conditionally** on the quality of the training data in each
   grid cell.

Step 3 is the point. A model can look well calibrated marginally and be
worthless on a subgroup — which is what the same author found for BERT on
Croatian hate speech, where the model's own confidence separated correct from
incorrect predictions at *p* = 1.4 × 10⁻¹¹ for English and *p* = 1 for
Croatian (Miok et al., *Cognitive Computation* 14:353–371, 2022).

## Why this model can be audited at all

`species_vars` in `bfm_model/bfm/configs/train_config.yaml` lists 28 GBIF
taxonKeys. The model's training targets are therefore directly traceable in
GBIF, and the quality of the records it learned from can be measured
independently of the model.

Screening (in R, `KristianMiok/gbif-coordinate-uncertainty`, scripts `10_`–`13_`)
established that those 28 species differ enormously in whether their records
state any coordinate precision at all — from *Gulo gulo* at 4.8% to *Pieris
brassicae* at 96.9% — and that the gap survives controlling for record type and
era (spread 92.7 points, Spearman ρ = 0.93 between raw and controlled).

The mechanism is **not** sensitive-species withholding: *Sturnus vulgaris*
(55.0%) and *Pieris brassicae* (96.4%) are both mass citizen-science taxa,
neither sensitive, and sit 41 points apart. The likely driver is publisher
convention. See [`DECISIONS.md`](DECISIONS.md) for the full history, including
three designs that were killed against pre-registered criteria.

## Layout

```
src/bfm_audit/     config, grid assignment, GBIF download construction
scripts/           numbered, run in order
data/reference/    small committed tables (species keys, verified grid)
data/raw/          GBIF archives and model weights (gitignored)
data/interim/      derived tables (gitignored except the download key)
tests/             pytest
DECISIONS.md       pre-registered criteria and their outcomes
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
cp .env.example .env      # then fill in GBIF credentials; .env is gitignored
```

## Running

```bash
# 1. read the model's real grid extent -- lat/lon are NOT in train_config.yaml,
#    they arrive via batch['batch_metadata']. Everything downstream depends on
#    this, so scripts refuse to run against a guessed extent.
python scripts/01_extract_model_grid.py --batch-file /path/to/batch.pkl

# 2. request the GBIF download for all 28 species inside that box
python scripts/02_request_download.py --year-from 2000 --year-to 2020 --dry-run
python scripts/02_request_download.py --year-from 2000 --year-to 2020

# 3. build the per-cell stratum
python scripts/03_build_stratum.py --download-key <key>
```

Step 3 has a pre-registered pass/fail criterion in `DECISIONS.md`. It can kill
the project, and if it does, that outcome gets written up rather than worked
around.

## Not yet in this repository

Nothing that touches the model. Loading the released weights (Small; Medium was
not public as of the repo snapshot), sampling with DropPath active, and the
coverage tests all come after the stratum passes. Building them now would mean
writing code against an unvalidated premise, which is the failure mode this
project has already paid for three times.

One design question is deferred rather than open: `td_learning: True` means the
model predicts *change*, not level, so "calibration" has to be defined against
a delta target. That must be settled before any coverage number is computed.

## Related work by the same author

- Miok, Škrlj, Zaharie & Robnik-Šikonja (2022). *To BAN or Not to BAN: Bayesian
  Attention Networks for Reliable Hate Speech Detection.* Cognitive Computation
  14:353–371. — MC-Dropout inside transformer attention; the group-conditional
  reliability test this project transfers.
- Miok et al. (2026). *Environmentally grounded pseudo-absence sampling for
  species distribution models.* Diversity and Distributions 32:e70199.
- Miok, Škrlj, Robnik-Šikonja & Pârvulescu (2026). *Interpretable by design:
  language model–derived ecological rules for SDM.* Ecological Informatics
  97:103883.
