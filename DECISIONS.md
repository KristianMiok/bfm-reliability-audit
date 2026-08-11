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

---

## 2026-08-10 — pre-flight against the release paper; download decisions

**G1 addendum — published confirmation.** BioAnalyst paper v2 (arXiv
2507.09080, Table 5) states the pre-training window explicitly: Europe grid
280 (lon) x 160 (lat), latitude bounds [32, 72], longitude bounds [-25, 45],
time range 01-01-2000 to 01-06-2020, 233 monthly batches. The release-code
derivation of G1 now has a published source as well.

**D1 — the release paper's occurrence inventory is partly wrong.** Table 3 of
the paper lists approximate 2000-2020 occurrence totals per species. Against
live GBIF counts under the release's own window and bbox (script-02 predicate
without the PRESENT clause; frozen in
`data/reference/preflight_counts_2026-08-10.csv`):

- 17 of 28 entries sit in a plausible band (ratio 0.65-1.2; the mild deficit
  is consistent with our tighter bbox + coordinate filters).
- Five entries are off by three to four orders of magnitude, in both
  directions: Bombus hyperboreus 206 actual vs 325,000 claimed; Monachus
  monachus 112 vs 137,000; Lynx pardinus 578 vs 436,000; Callosciurus
  erythraeus 850 vs 900,000; Episyrphus balteatus 102,546 vs 10.
- Four more are off by factors 4-14: Gulo gulo x6.2 (undercounted in the
  paper), Caretta caretta x9.4, Canis aureus x14, Ursus arctos x3.9.

Accurate and wildly wrong entries coexist side by side, which rules out a
uniform methodological difference (different counting units would shift all
ratios together). This is a documentation error, not — as far as counts can
show — a training-data error: the cube archives cannot contain records that
do not exist. Consequence for the audit: four species channels are
ultra-sparse in reality (112-850 records over 44,800 cells x 233 months);
per-species endpoints for these are degenerate and must be handled explicitly
in the E2 pre-registration.

**Decisions taken:**

1. `OCCURRENCE_STATUS = PRESENT` added to the download predicate, mirroring
   the WHERE clause of the documented GBIF cube template the training data
   went through. Server-side because SIMPLE_CSV's fixed columns do not
   guarantee occurrenceStatus for a local filter. Measured absence share
   without the clause: 97,281 / 12,611,544 = 0.77 %.
2. Year-2020 records past the training window (after 2020-06-01; year-2020
   total 1,277,511) are filtered in script 03 via eventDate, since GBIF's
   MONTH predicate cannot express "July-December of one year only".
3. SIMPLE_CSV format approved: ~12.5 M records, ~3.2 GB zipped, locally
   processable.
4. `gbif.py` rewritten against the GBIF API directly (stdlib). The previous
   pygbif path passed a prebuilt predicate dict to `occurrences.download()`,
   which expects query strings — submission would have failed at run time.
   pygbif removed from dependencies.

---

## 2026-08-10 — G2 closed as bounded; row convention fixed before first stratum run

**G2 — cube assignment variant: bounded unknown.** Public trail exhausted:
the BioCube HF card ships the multimodal parquet branch, not the 28-species
cube zips; no GBIF download DOI exists in either paper, the repositories, or
GBIF literature tracking. A direct author query was considered and declined.
The documented GBIF SQL cube service randomises each record inside
COALESCE(coordinateUncertaintyInMeters, 1000 m) before cell assignment; a
fixed-assignment variant also exists in the UDF README. The audit's
measurements are invariant to which variant BioDT used — the stratum is
built from raw records, and the model is audited as released — so the paper
states both variants and confines the difference to the mechanism narrative
(differential jitter vs fictitious precision for silent publishers).

**Row convention fixed before first use.** `assign_cells` placed row 0 at
the NORTH edge ("usual raster convention"); G1 established the model's
tensors are ascending with row 0 at the SOUTH edge (writer reindexes onto
ascending GRID_LAT; reader crops from index 0). Harmless for the stratum in
isolation, fatal for the E2 index-for-index join with model outputs — the
map would have flipped vertically, silently. Fixed to south-up; regression
tests pin the corners; `to_raster` figures now require origin="lower".

Implementation notes from the same review: the 2020-07..12 tail cut
(decision 2) is implemented as `training_window_mask` (final-year records
with no month are excluded and counted); GBIF TSV is parsed with QUOTE_NONE
(the format is unquoted, stray quotes in free-text fields derail default
parsing); script 03 now prints the pre-registered thresholds and an explicit
PASS / BETWEEN / FAIL verdict with tercile cell counts and hhi publisher-map
diagnostics. Thresholds themselves unchanged from the open criterion.
End-to-end verified on a synthetic archive with planted tail, out-of-grid,
and fake-radius records before first contact with real data.

---

## 2026-08-10 — G2 closed as bounded; row convention fixed before first stratum run

**G2 — cube assignment variant: bounded unknown.** Public trail exhausted:
the BioCube HF card ships the multimodal parquet branch, not the 28-species
cube zips; no GBIF download DOI exists in either paper, the repositories, or
GBIF literature tracking. A direct author query was considered and declined.
The documented GBIF SQL cube service randomises each record inside
COALESCE(coordinateUncertaintyInMeters, 1000 m) before cell assignment; a
fixed-assignment variant also exists in the UDF README. The audit's
measurements are invariant to which variant BioDT used — the stratum is
built from raw records, and the model is audited as released — so the paper
states both variants and confines the difference to the mechanism narrative
(differential jitter vs fictitious precision for silent publishers).

**Row convention fixed before first use.** `assign_cells` placed row 0 at
the NORTH edge ("usual raster convention"); G1 established the model's
tensors are ascending with row 0 at the SOUTH edge (writer reindexes onto
ascending GRID_LAT; reader crops from index 0). Harmless for the stratum in
isolation, fatal for the E2 index-for-index join with model outputs — the
map would have flipped vertically, silently. Fixed to south-up; regression
tests pin the corners; `to_raster` figures now require origin="lower".

Implementation notes from the same review: the 2020-07..12 tail cut
(decision 2) is implemented as `training_window_mask` (final-year records
with no month are excluded and counted); GBIF TSV is parsed with QUOTE_NONE
(the format is unquoted, stray quotes in free-text fields derail default
parsing); script 03 now prints the pre-registered thresholds and an explicit
PASS / BETWEEN / FAIL verdict with tercile cell counts and hhi publisher-map
diagnostics. Thresholds themselves unchanged from the open criterion.
End-to-end verified on a synthetic archive with planted tail, out-of-grid,
and fake-radius records before first contact with real data.

---

## 2026-08-10 — E1 outcome: the spatial stratum criterion PASSED

Download: GBIF.org (10 August 2026) Occurrence Download
https://doi.org/10.15468/dl.ntnmuv — 12,514,271 records (index drifted +8
from the pre-flight count taken hours earlier), SIMPLE_CSV, 1.14 GB.
Training-window tail cut removed 541,866 records (10,534 of them year-2020
with no month). States on the kept 11,972,405: reporting 9,879,112 (82.5%),
unknown 2,091,212 (17.5%), fake 2,081 (0.017% — the third class is preserved
by design but quantitatively negligible in this species set, in contrast to
the crayfish work). Out-of-grid: 0 of 12.5M — the server-side polygon and the
local south-up assignment agree exactly, closing the G1 loop on real data.

Stratum: 18,870 occupied cells; 10,206 with >= 20 records. pct_reporting
p10 = 1.9, p90 = 98.8, spread 96.8 vs registered 30; terciles
3,402 / 3,402 / 3,402 vs registered 200. **VERDICT: PASS.** The distribution
is near-bimodal (p25 19.8 -> p50 70.1 -> p75 93.9).

Publisher-map check: overall Spearman(pct_reporting, hhi) = -0.067, but the
association is U-SHAPED: median hhi bottom 0.658 (68% of cells > 0.5),
middle 0.372 (28%), top 0.711 (64%). Both extremes are publisher-
concentrated. Required description, per the registration: the stratum is the
spatial projection of publisher reporting convention — the C13 mechanism made
spatial — not a publisher-independent "quality" field. Consequence: an
hhi < 0.5 sensitivity analysis enters G3.

Density: median n_records 63 (bottom tercile) vs 422 (top), a 6.7x gradient
aligned with the stratum. Consequence: the density control moves from a
listed robustness check into the primary endpoint itself (stratified
estimator, G3 item 5).

---

## G3 — E2 pre-registration (fixed before any model weight is loaded)

Scope: group-conditional reliability audit of the released BioAnalyst
checkpoint (HF `BioDT/bfm-pretrained`; released config has drop_rate 0.0,
attn_drop_rate 0.0, drop_path_rate 0.1).

1. **Evaluation window.** Test months only. The release splits batches
   chronologically (`scripts/split_dataset.sh`, lexicographic, first-N
   train) and its ablations report 13 prediction windows in the test set;
   we register the FINAL 13 prediction windows of the 233 as evaluation
   targets. The exact filename boundary is pinned once batches exist; any
   deviation is logged here before a single coverage number is computed.
   Batches are not shipped; they are rebuilt from BioCube (HF) with
   `BioDT/bfm-data` `build_batches_monthly`, restricted to the months the
   evaluation needs. The G1 batch-confirmation gate is discharged by
   `01 --batch-file` on the first rebuilt batch, before inference.

2. **Target and space.** The model's own training target: the temporal-
   difference increment (td_learning), species channels, evaluated in the
   model's normalised space (per-channel centre/scale of the release
   statistics). Unscaled deltas and level reconstructions: descriptive only.

3. **Uncertainty mechanism.** K MC forward passes with stochastic depth
   active at inference. Registered pre-check: assert the architecture
   contains no BatchNorm (LayerNorm has no train/eval difference and the
   other dropout rates are 0.0, so train() touches DropPath only); if
   BatchNorm is found, isolate DropPath modules instead. K is chosen by a
   pilot on ONE batch outside the evaluation window: the smallest
   K in {8, 16, 32, 64} whose per-cell sigma ranks correlate >= 0.99 with
   K = 64; K is then frozen. The pilot is exploratory and reported as such.

4. **Intervals and per-cell coverage.** Gaussian mu +/- z*sigma at nominal
   90% (primary) and 80% (secondary), per cell x species x step. Cell
   coverage = share of that cell's species-step predictions inside the
   interval, species weighted equally.

5. **PRIMARY endpoint (confirmatory).** Difference in mean cell coverage at
   nominal 90% between TOP and BOTTOM stratum terciles (E1 rank-based
   terciles), density-stratified: cells binned by quintiles of
   log10(n_records); the gap computed within each bin; the primary statistic
   is the bin-size-weighted mean gap. Inference: spatial block permutation —
   10x10-cell blocks (2.5 deg), tercile labels permuted at block level,
   10,000 permutations, two-sided, alpha = 0.05.

6. **Secondary endpoints.** (a) spread-skill: Spearman(sigma, |error|),
   overall and per tercile; (b) BAN-style discrimination: AUROC of sigma for
   top-decile |error|, per tercile; (c) the 80% analogue of the primary;
   (d) per-species heterogeneity across the 24 non-degenerate channels.

7. **Registered sensitivity analyses.** (i) mixed-publisher cells only
   (dataset_hhi < 0.5) — required by the E1 U-shape; (ii) cells with
   n_records >= 50 only; (iii) fake recoded to unknown (n = 2,081,
   negligible, registered for completeness); (iv) 5-degree latitude bands
   replacing the density bins.

8. **Ultra-sparse channels.** Monachus monachus, Lynx pardinus, Bombus
   hyperboreus, Callosciurus erythraeus (112-850 in-window records) are
   excluded from per-species endpoints, retained in pooled cell-level
   endpoints, and tabulated.

9. **Named outcomes, fixed now.** (1) bottom-tercile undercoverage -> the
   reliability-gap story; (2) no significant gap -> quality-blind
   uncertainty, publishable as a uniformity finding with the stratum and D1
   as contributions; (3) reversed sign -> stop and investigate; no claim
   without a mechanism.

10. **Stop rules.** If sigma is degenerate (median 0, or rank instability
    persisting at K = 64), MC-DropPath fails as a mechanism and THAT becomes
    the finding ("the released model exposes no usable internal
    uncertainty"); no post-hoc switch of UQ method within this paper. No
    fourth design if the audit dies here (standing rule 4).

---

## 2026-08-11 — G3 pins: public checkpoints, primary weights, RedList gap

HF weights repo ships TWO checkpoints as of today:
`bfm-pretrained-small.safetensors` (0.78 GB) and
`bfm-pretrain-large.safetensors` (2.84 GB). The earlier snapshot note that
the larger model was not public is superseded. **Primary audit checkpoint:
SMALL**, because the released training config (`train_config.yaml`:
drop_rate 0.0, attn_drop_rate 0.0, drop_path_rate 0.1) makes stochastic
depth the ONLY inference-time stochastic component — exactly the registered
mechanism (G3 item 3). The large checkpoint (paper Table 7: 0.1/0.1/0.1)
would mix ordinary and attention dropout into the samples; registered as an
optional replication with a mixed mechanism, never the primary.

BioCube on HF carries `Species/europe_species.parquet` (5.1 MB), the
candidate distribution ground truth; schema verification logged in the repo
history. The RedList folder listed in the BioCube README is ABSENT from the
HF tree; whether RLI input channels can be reconstructed is determined at
scan_biocube time and will be logged here before any inference runs.

---

## 2026-08-11 — species ground truth verified; the parquet extends to 2025-05

`Species/europe_species.parquet` (BioCube HF) is confirmed as the
distribution ground-truth table: 2,427,403 rows; columns Species (the 28
GBIF taxonKeys exactly), Latitude/Longitude on quarter-degree cell centres
matching the verified model grid, monthly Timestamp, integer Distribution
(occurrence count), plus taxonomy. Rows on the cropped writer edge
(lat 72.00 / lon 45.00) do not exist in the model tensor and are dropped at
join time; count logged in the run output.

**Finding: the table runs 2000-01 through 2025-05** — 59 months past the
model's training window (2000-01..2020-06). This creates a genuinely
out-of-sample evaluation period the model has never seen. Handled as a
REGISTERED AMENDMENT CANDIDATE (A1), not a silent change: G3 item 1 keeps
the final-13-windows holdout as registered; A1 (post-2020-06 evaluation)
is adopted only if the ERA5 inputs on HF cover the same months, and the
adoption (or its impossibility) is logged here before any inference runs.

Cross-check of cube occurrence sums against our raw-GBIF pre-flight counts
per species is in the run output committed with this entry; ratios below 1
are expected (cube-side filters and aggregation), and the four ultra-sparse
channels (D1) remain ultra-sparse here.

---

## 2026-08-11 — A1 closed; archive truncation incident; D2 is multi-mechanism

**A1 (post-window evaluation): CLOSED — impossible from public artifacts.**
Species ground truth runs to 2025-05, but the released inputs stop earlier:
all seven ERA5 files end 2020-12 (n=252 months each), and monthly NDVI ends
2020-06 — exactly at the training-window end, which is presumably WHY the
window ends there. No post-2020-06 batch can be assembled without imputing
an input channel, and imputed inputs would contaminate the attribution of
any coverage result to the model. The registered final-13 holdout (G3 item
1) stands; its months sit inside every input's coverage. Recorded for
completeness: ERA5 carries six extra months (2020-07..12) and the yearly
indicator CSVs run to 2021.

**Archive truncation incident.** The GBIF archive read successfully in full
by script 03 (12.5M records) was found next day at 63,963,136 of
1,139,734,500 bytes, ZIP magic intact, central directory gone. Cause
undetermined; leading hypothesis is macOS "Optimize Mac Storage" eviction
(repository lives under ~/Desktop, which is iCloud-managed). Healed by
ranged resume against the GBIF endpoint to the exact API-reported size;
is_zipfile passes. Guard going forward: compare on-disk size to the GBIF
API size before any read; if it recurs, relocate data stores outside
~/Desktop.

**D2 — cube shortfall is multi-mechanism.** Evidence table
`data/reference/basis_by_species_2026-08-11.csv` (commit ffd8976).
Spearman(occ_ratio, %HUMAN_OBSERVATION) = +0.615 across the 28 species.
Three named mechanisms, none sufficient alone: (i) machine-observation
exclusion — Caretta caretta at ratio 0.06 with 78% MACHINE_OBSERVATION is
the clean case; (ii) uncertainty-based exclusion hitting sensitive-species
coordinate generalisation — the collapse cluster (Canis aureus 0.03, Gulo
0.12, Ursus 0.17, Monachus 0.34, both lynxes, wolf) is precisely the set
whose coordinates portals obscure for protection, which would make the
cube's quality filter THE SAME VARIABLE as the audit stratum; (iii) cube
snapshot age (post-snapshot publication growth). Next falsification step:
per-species coordinate-uncertainty profile vs occ_ratio, from our own
download.

---

## 2026-08-11 — D2 closed as composite depletion; ERA5 acquisition decision

**D2 CLOSED: training-side depletion is real, nonrandom, and composite.**
Per-species correlations of cube retention (occ_ratio) over the 28 species:
%HUMAN_OBSERVATION +0.618, pct_unknown -0.580, median radius -0.509,
pct>=10km -0.397. No single rule survives the table: high-unknown carnivores
collapse (Canis aureus 95.2% unknown -> 0.03 retained; Ursus 87% -> 0.17;
wolf 47% -> 0.49, near one-to-one loss of silent records), the
machine-observation case collapses independently (Caretta 78% MO -> 0.06),
yet Aquila fasciata with 41% of records at r>=10 km (median 50 km,
nest-protection obscuring) retains 0.72 — excluding any single hard radius
filter. Conclusion recorded for the paper: cube ingestion depletes training
data along the same metadata axes the audit stratum measures (missing
precision, non-human modalities, large radii), hardest for
conservation-critical carnivores. Exact SQL attribution remains bounded by
G2 (query declined). Evidence: data/reference/basis_by_species_2026-08-11.csv.
No further D2 forensics inside this paper (priority discipline).

**ERA5 acquisition: full snapshot, lazy slicing rejected on measurement.**
One month of one pressure variable over HTTP took 3,030 s (chunk layout
42x1x121x240 forces near-full reads). Decision: snapshot the whole
Copernicus folder (~31 GB) via huggingface_hub with resume; files are
global 721x1440, Europe is cropped by the batch builder. RedList remains
absent on HF; resolved at scan_biocube time as already logged.
