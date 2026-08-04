# Reference tables

`bfm_species_keys.csv` — the 28 GBIF taxonKeys from `species_vars` in
BioAnalyst's `train_config.yaml`, joined to the screening results from the R
scripts `12_bfm_species_quality.R` and `13_reporting_controlled.R`.

| column | meaning |
|---|---|
| `taxon_key` | GBIF taxonKey, verbatim from the model config |
| `n_europe` | European georeferenced records |
| `reported_pct` | share stating any `coordinateUncertaintyInMeters` |
| `ctrl_n` | records in the controlled stratum (HUMAN_OBSERVATION, year ≥ 2015) |
| `ctrl_reported_pct` | reporting share within that stratum |

Blank `ctrl_*` means fewer than 1,000 records in the controlled stratum
(4 species: *Bombus hyperboreus*, *Monachus monachus*, *Lynx pardinus*,
*Callosciurus erythraeus*).

`model_grid.json` is written by `scripts/01_extract_model_grid.py` and is not
committed until it carries `verified: true`.
