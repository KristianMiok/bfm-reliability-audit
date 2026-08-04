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
