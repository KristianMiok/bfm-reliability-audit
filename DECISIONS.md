# Decisions log

Every design in this project is tested against a criterion fixed **before** the
test runs. Outcomes are recorded whether they pass or fail. Three designs died
this way before the current one survived; that history is kept here because the
failures constrain what can honestly be claimed later.

Screening was done in R, in `KristianMiok/gbif-coordinate-uncertainty`
(scripts `10_` through `13_`). This repository begins where that left off.

---

## Background: what makes this auditable

BioAnalyst (`bfm-model`, Trantas et al. 2025) lists its 28 target species in
`bfm_model/bfm/configs/train_config.yaml` as `species_vars`. Those strings are
GBIF taxonKeys, so the model's training targets are directly auditable in GBIF.

Three facts established by reading the model source:

- **No uncertainty of any kind.** Grepping `bfm_model/` for
  `uncertain|variance|ensemble|posterior|confidence|calibrat|quantile|conformal`
  returns nothing in the model itself. The loss is MAE
  (`torch.abs(prediction_for_loss - target_tensor)`) — a deterministic point
  prediction.
- **MC-Dropout is unavailable as-is, MC-DropPath is available.** Config sets
  `drop_rate: 0.0` and `attn_drop_rate: 0.0`, so `nn.Dropout(0.0)` is a no-op.
  But `drop_path_rate: 0.1` is active via `timm.layers.DropPath`, applied on
  both residual branches of every Swin block. Caveat:
  `dpr = torch.linspace(0, 0.1, sum(depths))` ramps the rate from 0 in the
  first layer, so variance will come almost entirely from late layers. Whether
  that yields usable spread is an open empirical question.
- **`td_learning: True`** means the model predicts *change*, not level
  (`target = gt[:,1] - gt[:,0]`). What "calibration" means must be defined
  against that before anything is measured.

---

## C10 — coordinate obscuring in European Orchidaceae · **FAILED**

Hypothesis: sensitive-species generalization (iNaturalist places obscured
records at a random point in a 0.2° cell and sets the radius to that cell's
diagonal, ~24–29 km in Europe) would give a contamination axis operating at the
model's 0.25° grain, unlike ordinary 45–3000 m positional error.

Criterion: ≥2% of radius-reporting records in the 20–35 km window, **and** a
density excess over adjacent bands.

Result: 1.75% — **failed on prevalence**. Density ratio 10.65 — passed.

Two notes:

- **Script bug.** GBIF range predicates are inclusive at both ends, so adjacent
  disjoint bands double-count boundary values. Bands summed to 8,168,844
  against a denominator of 6,204,053 (132%). Earlier scripts (`04_`, `06_`)
  used cumulative thresholds and differences, which is correct. **All future
  band work must be cumulative.**
- **Process.** The gate thresholded *prevalence*. A group-conditional audit
  needs a clean partition and enough records per group, not a common one. The
  gate was mis-specified, not merely unlucky.

## C11 — is the 26–29 km spike a point mass? · **FAILED, honoured**

Criterion: ≥50% of the spike in one 10 m bin → machine fingerprint, design
lives; <20% → design dies.

Result: 4.6%. **Design killed.**

Observation recorded but *not acted on*: within 26700–26800 m, three 10 m bins
hold 10,545 records and the other seven hold 4, 3 and ~0. The values are
quantized, so these are machine-assigned, plausibly one per latitude — the
printed verdict "coarse georeferencing" is probably the wrong label. It changes
nothing: the design needed **one** value so there would be no cut to defend in
review. Many discrete values reinstate exactly that problem.

## C12 — do the 28 species separate on `usable`? · **FAILED on shape**

Criterion: ≥8 species below 60% usable **and** ≥8 above 85%, spread ≥15 points,
each with ≥5,000 European records.

Result: fails at all three thresholds (1 km: 21/0; 5 km: 10/5; 10 km: 5/7).

**But the spread is 95.5 points** (Gulo gulo 0.5% vs Pieris brassicae 95.0% at
5 km). The species separate enormously. What failed is that the criterion
demanded *bimodality* around two absolute cuts while the distribution is a
continuous gradient with mass in the middle. The cuts were arbitrary and the
wrong shape.

Five species ineligible for n<5000: *Monachus monachus* (242), *Bombus
hyperboreus* (617), *Callosciurus erythraeus* (1018), *Lynx pardinus* (1814),
*Canis aureus* (2588). The most threatened taxa drop out for having too little
data.

## C13 — does the reporting gap survive an era/type control? · **PASSED**

The other column separated sharply: `reported` (share of records stating any
radius) runs from Gulo gulo 4.8% to Pieris brassicae 96.9%.

Confound to exclude: large carnivores might be documented by older museum and
atlas records predating the `coordinateUncertaintyInMeters` convention.

Control: `basisOfRecord = HUMAN_OBSERVATION` and `year ≥ 2015`.

Criterion (deliberately not a threshold on prevalence — that shape failed
twice): controlled spread ≥40 points **and** Spearman(raw, controlled) ≥0.5.

Result: spread 92.7 points, ρ = +0.93. **Passed.**

Two things must be carried forward honestly:

- **The confound does not exist.** Spearman(reported, %post-2015) = −0.01;
  with %HUMAN_OBSERVATION = +0.29. The legacy-data story was never viable and
  one query would have shown it. Retained as a documented negative control.
- **Sensitivity does NOT explain the gradient.** *Canis lupus* 67.4% (the most
  persecuted European carnivore, mid-table); *Caretta caretta* 55.3% (strictly
  protected, mid-table); and decisively **Sturnus vulgaris 55.0% vs Pieris
  brassicae 96.4%** — both mass citizen science, neither sensitive, 41 points
  apart. **Do not write the withholding story.** The likely mechanism is
  publisher convention, i.e. the dataset effect already measured in `08_`
  (U = 0.593, permutation null 0.000).

Open and interesting: *Lynx lynx* delta −36.2, *Aquila fasciata* −23.8. The
control made these *worse* — their modern human observations report precision
less often than their other records. Opposite of any legacy-data story.
Probably one dominant monitoring publisher; the download will say.

## G1 — where does the model grid actually sit? · **RESOLVED from release code, 2026-08-10**

The scaffold shipped with a guessed extent (lat 34–74, lon −25..45) marked
UNVERIFIED, and scripts 02/03 refused to run against it. The guess was
**wrong by 2°**: the real window is lat 32–72. Running the download against
the guess would have silently dropped the 32–34°N band — southern Iberia,
Sicily, Crete, Cyprus, the Maghreb coast — and spent quota on an empty
72–74°N band. The guard earned its keep.

Resolution chain, all from public release code at current `main`
(BioDT/bfm-data and BioDT/bfm-model, read 2026-08-10):

1. **Writer constants** — `bfm-data/bfm_data/dataset_creation/batch_creation/scan_biocube.py`:
   `GRID_LAT = round(arange(32.0, 72.0+1e-6, 0.25), 3)` (ascending, 161 pts),
   `GRID_LON = round(arange(-25.0, 45.0+1e-6, 0.25), 3)` (ascending, 281 pts),
   `EXPECTED_LAT, EXPECTED_LON = 161, 281`.
2. **Writer reindex + metadata** — `build_batches_monthly.py`:
   `ds.sel(latitude=GRID_LAT, longitude=GRID_LON)` (line ~282) puts every
   source array on the ascending grid; `batch_metadata["latitudes"] =
   GRID_LAT.tolist()` (line ~402) writes the same vectors. Arrays and
   metadata therefore share one order: **ascending, row 0 = south**.
3. **Reader crop** — `bfm-model/bfm_model/bfm/dataloader_monthly.py` with
   `patch_size: 4` (train_config.yaml): `new_H = (161//4)*4 = 160`,
   `new_W = (281//4)*4 = 280`; tensors `[..., :160, :280]` and metadata
   `latitudes[:160]`, `longitudes[:280]` — same end, index 0.
4. **Model grid**: cell centres lat 32.00..71.75, lon −25.00..44.75; the
   writer's north row (72.00) and east column (45.00) are cropped away.
   Cell-edge bbox: lat 31.875..71.875, lon −25.125..44.875.

Independent checks: committed land mask
`bfm-model/batch_statistics/europe_Land_2020_grid.pkl` is exactly 160×280;
the 28 taxonKeys hard-coded in `build_batches_monthly.py` match
`data/reference/bfm_species_keys.csv` **28/28**.

**Trap documented**: `bfm-model/bfm_model/bfm/utils.py` (lat 34.25–72
descending, lon −30..40, 152×320) and `documentation/batch_visualisation.ipynb`
(152×320 shapes, 22 species) describe a **pre-release grid**. An older
`bfm-data` helper (`get_lat_lon_ranges`, with `lat_range[::-1]`) belongs to
the same legacy path. Never source coordinates from those files.

**Residual risk and gate**: the batches on Hugging Face could have been
produced by an earlier writer than today's `main`. Probability low (shapes,
land mask, and species list all match), consequence bounded (≤ one row/column
at the window edge for cell assignment). Therefore: `model_grid.json` carries
`verified: true` with `provenance.method = "release-code-derivation"` and
`confirmed_against_batch: false`. Scripts 02 (download bbox is a covering
box; a half-cell excess is harmless) and 03 (assignment; worst case one edge
row/column, re-checked later) may run. **No script that loads model weights
or reads model predictions may run until `01 --batch-file` against a real
batch has flipped `confirmed_against_batch` — which costs nothing, because
inference requires downloading batches anyway.**

---

## Standing rules

1. **Scan before code.** Three directions in the parent project lost time to
   coding before a prior-art scan.
2. **Controls before claims.** Controls have repeatedly knocked findings down;
   the ones that survived are trusted precisely because they were attacked.
3. **A criterion that is reinterpreted after firing is not a criterion.** C10's
   gate was argued post-hoc to be the wrong quantity — defensible once, and
   flagged as post-hoc. C11's was honoured. C12's was wrong-shaped. Two of
   three proposed thresholds did not measure what was intended; weight future
   proposals accordingly.
4. **No fourth axis.** If the current design fails, write up what was learned.
   A fourth attempt after three failures is fishing, not screening.

---

## Open criterion — the spatial stratum

Fixed here, before `03_build_stratum.py` is run.

The species-level gradient must survive aggregation to grid cells, otherwise
there is nothing to condition on:

- **Pass**: p10–p90 spread of `pct_reporting` across cells with ≥20 records is
  ≥30 points, and ≥200 cells fall in each of the bottom and top terciles.
- **Fail**: spread <15 points → the gradient washes out spatially; write up the
  negative result and stop.
- **Between**: report the attenuation; do not proceed to the model.

A second requirement, independent of the above: cells must not be separable by
`dataset_hhi` alone. If low-reporting cells are simply single-publisher cells,
the stratum is a publisher map rather than a data-quality map, and the claim
must be stated that way.
