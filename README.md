# REFIT data cleaning and washing-machine disaggregation

Take-home assignment for Flock Energy: a reproducible pipeline that cleans raw REFIT smart-meter
data, analyses washing-machine use across 14 UK homes, and estimates washing-machine power from
the household aggregate at 1-minute resolution.

| Document | Contents |
|---|---|
| [DATA.md](DATA.md) | Dataset, raw-data quality findings, cleaning steps and decisions, validation of the cleaning, remaining concerns |
| [RESULTS.md](RESULTS.md) | Washing-machine EDA (Section 2), model, validation design, results and failure analysis (Section 3), papers used |
| [Recommendation.md](Recommendation.md) | One-page, utility-facing summary of suitable uses and limits, plus the Section 4 answers |

## What I built

1. **Raw House 1 cleaning** ([`src/section1_raw.py`](src/section1_raw.py)). The two raw files overlap by
   three months and **use different clocks** (UK local time vs UTC); they also contain a million
   duplicate timestamps and sensor error codes up to 98 kW. A seven-step documented pipeline
   (clock → overlap → sort → impossible values → duplicates → 1-minute → gaps) fixes these.
   **Validation:** the result matches the authors' official cleaned House 1 with **r = 0.999**
   (aggregate MAE 4 W).
2. **Washing-machine EDA** ([`src/section2_eda.py`](src/section2_eda.py)) on 14 homes, 6,801 house-days.
   A data-derived cycle rule (> 20 W; pauses ≤ 15 min merged; ≥ 10 min and ≥ 0.05 kWh) finds
   4,951 washes. It reports availability, off time, timing by hour and weekday, and a cross-house
   comparison. Along the way it found **frozen sensors** in the official cleaned data (House 5: a
   constant 2,129 W for 13 h).
3. **Disaggregation models** ([`src/section3_model.py`](src/section3_model.py)). Each model reads a
   ±60-minute window of the aggregate (the Seq2Point framing) and predicts the washing machine's
   power for every minute. Compared on the same features and split: a one-stage LightGBM regressor;
   two-stage models (classifier for "running?", regressor for watts) with **LightGBM** and with a
   **random forest**; a **multi-appliance** LightGBM whose classifier also learns dishwashers, tumble
   dryers and washer-dryers as a separate class; and each with or without a **cycle filter** (the
   Section 2 rule applied to the predictions). The final model is the one with the lowest
   validation MAE: **LightGBM multi-appliance + cycle filter**. All models are trained on 9 homes,
   tuned on 2, **tested on 3 unseen homes**, and compared with always-off and usage-profile baselines.

**Headline test results (3 unseen homes, ~1,400 house-days):**

| | Always off | Usage profile | **Final model** |
|---|---|---|---|
| Total energy error | −100 % | −12 % | **−15 %** (−7 % without the cycle filter) |
| Daily energy MAE | 0.46 kWh | 0.48 kWh | **0.28 kWh** |
| On/off F1 | 0 | 0.08 | **0.51** |
| Washes found / detected washes that are real | 0 % / – | 55 % / 16 % | **80 % / 53 %** |
| Per-minute MAE | **19.0 W** | 34.0 W | 19.2 W |

Always-off still has a marginally lower per-minute MAE, a known artefact for rare, spiky loads
(explained in RESULTS.md §3.4). The random forest was worse than LightGBM. The multi-appliance
classifier gave a small, consistent gain, and the cycle filter traded total energy for per-minute
accuracy. The main failure remains confusion with **dishwashers and tumble dryers** (28–46 % and
31 % of false-alarm energy in the homes that have them).

## Setup

Tested with **Python 3.13.5** on Windows 11 (4-core laptop, 16 GB RAM, no GPU).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The data is not in the repository (≈ 14 GB extracted, CC BY 4.0).

1. Download both archives **in a browser**. The portal blocks scripted downloads.
   - Raw: <https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z>
   - Cleaned: <https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z>
2. Extract them into `data/archives/`, with 7-Zip ("Extract to folder") or the bundled extractor:
   ```bash
   python -m py7zr x REFIT_RAW_081116.7z data/archives/REFIT_RAW_081116
   python -m py7zr x CLEAN_REFIT_081116.7z data/archives/CLEAN_REFIT_081116
   ```
3. Check that these files exist. Only House 1 raw and the CLEAN files are read:
   ```
   data/archives/REFIT_RAW_081116/RAW_House1_Part1.csv
   data/archives/REFIT_RAW_081116/RAW_House1_Part2.csv
   data/archives/CLEAN_REFIT_081116/CLEAN_House1.csv ... CLEAN_House21.csv
   ```
   One extra nested folder level (e.g. `REFIT_RAW_081116/REFIT_RAW_081116/...`) is also found.

The official cleaned-data readme is included as [`docs/CLEAN_READ_ME_081116.txt`](docs/CLEAN_READ_ME_081116.txt).

## Run

```bash
python run_all.py
```

This runs everything in order, from the CSVs to every table in [`results/`](results/) and figure in
[`figures/`](figures/), in about 15 minutes. The steps can also be run one at a time:

| Command | Does | Time |
|---|---|---|
| `python -m src.prepare_clean` | CLEAN houses → 1-minute parquet in `data/processed/`, screening table | ~40 s |
| `python -m src.section1_raw` | Section 1: inspection, cleaning, validation vs official data, plots | ~1 min |
| `python -m src.section2_eda` | Section 2: cycle detection, availability, patterns, plots | ~10–50 s |
| `python -m src.section3_model` | Section 3: features, baselines, 4 models × cycle filter, metrics, failure analysis, plots | ~13 min |

Sections 2 and 3 need `prepare_clean` first. Section 3 also reads `results/s2_cycles.csv`.

**Reproducibility check:** after deleting `results/`, `figures/` and `data/processed/`, a fresh
`python run_all.py` (12.7 min) regenerated all 20 tables value-for-value and all 12 figures
byte-for-byte identical to the previous run's. LightGBM and the random forest use a fixed seed.

### Code layout

```
src/config.py         paths, house/channel metadata, study houses
src/refit_io.py       CSV readers (RAW parts, CLEAN houses)
src/resample.py       1-minute averaging and the shared gap rule (fill <= 5 min, flag the rest)
src/plotting.py       one figure style for the whole report
src/prepare_clean.py  CLEAN -> 1-minute parquet + house screening
src/section1_raw.py   Section 1
src/section2_eda.py   Section 2 (cycle rule lives here)
src/section3_model.py Section 3
run_all.py            the whole pipeline
```

## Assumptions and preprocessing decisions

Each is justified with evidence in DATA.md or RESULTS.md.

- **Timestamps.** RAW Part1 is UK local time and RAW Part2 is UTC from Oct 2014. This was verified
  month by month against the official data; the repeated clock-change hour (27 Oct 2013) is dropped.
  **CLEAN is UTC**, so all hour-of-day analysis converts to Europe/London.
- **Overlap.** Where both raw parts cover Jun–Oct 2014, Part1 is kept: the official data matches it
  (MAE 3 W vs 58 W for Part2).
- **Duplicates.** Exact duplicate rows are dropped; otherwise the last non-missing reading per
  channel is kept.
- **Impossible values → missing, not zero.** Aggregate > 23 kW (100 A × 230 V supply) or any
  appliance > 4 kW (the official REFIT threshold).
- **Gaps.** Gaps ≤ 5 minutes are forward-filled (REFIT logs on change, so carrying the last value is
  faithful). Longer gaps are flagged as outages and never filled.
- **Houses 3, 11, 21** are excluded (solar PV distorts the aggregate), as are **4** (two washing
  machines) and **13** (machine replaced mid-study).
- **Frozen sensors.** An identical washing-machine reading above 20 W for ≥ 30 min is treated as
  missing. The same rule applies to the dishwasher, tumble-dryer and washer-dryer channels used as
  look-alike labels in Section 3.
- **`Issues` flag.** It fires at 11–46 % of washing minutes (an IAM timing artefact at switch-on),
  so it is not used to discard data. Training drops minutes whose *1-minute* appliance sum exceeds
  the aggregate; evaluation keeps every observed minute.

## Validation design

- **Split by household:** train {1, 5, 6, 7, 9, 10, 16, 18, 19}, validation {15, 17}, test {2, 8, 20}.
  No home is in two sets, so neither near-duplicate neighbouring minutes nor a household's habits
  can leak. This mirrors deploying to new customers.
- **Validation** is used for early stopping, each classifier's threshold τ, and the choice of final
  model by a rule fixed in the code: **lowest validation MAE** among the learned models. The models
  were built in stages (one-stage, then two-stage, then random forest, multi-appliance and the cycle
  filter), and earlier test results had been seen before later models existed. Every choice was
  made on validation numbers, and **every model is reported on test**.
- **Metrics:** MAE, MAE while running, total and daily energy error, on/off precision/recall/F1, and
  cycle-level recall/precision. The last ones are needed because always-off scores well on MAE.

## Trade-offs

| Choice | Benefit | Cost |
|---|---|---|
| Simple, auditable cleaning rules | easy to defend; reproduces official data (r = 0.999) | no per-appliance limits or rolling spike filter (a few residual spikes) |
| Tree models (LightGBM, random forest), not a CNN | train in minutes on CPU, inspectable feature importance | may learn cycle shapes less well than Seq2Point |
| Multi-appliance as an extra *class*, not separate watts models | one small change to the classifier; uses look-alike labels the data already has | small gain only; it does not share out energy between appliances |
| Cycle filter on the predictions | removes short false alarms with an already-justified rule; MAE falls | removes some real running minutes: energy error −7 % → −15 % |
| Final model chosen by validation MAE | one rule, fixed in advance | MAE is not the metric a utility cares about most (energy, washes found) |
| Centred ±60-min window | uses the whole wash shape | offline only: needs an hour of "future" data |
| Split by house (3 test homes) | honest estimate for new customers | small test set: per-home F1 ranges 0.33–0.72 |
| τ chosen on 2 validation homes | no test leakage | validation F1 (0.79) overstates test F1 (0.51) |
| Every 3rd minute for training | 3× faster, unbiased (no class re-balancing) | discards near-duplicate rows only |

## AI tools used

I used **Claude Code** (Anthropic's AI coding assistant, Claude Opus 5.5 model) as a pair
programmer in this repository, working through the assignment one section at a time.

- **My role:**
  - set the order and scope of the work;
  - downloaded and set up the data;
  - asked for an explanation of what was done at each step and why, and reviewed the results;
  - decided to add the extra Section 3 models (random forest, multi-appliance, cycle filter);
  - requested final checks against the brief;
  - committed and pushed the work.
- **The assistant's role:** it generated the code and first drafts of the documents, ran the
  analyses, proposed cleaning and modelling options with the evidence for each, and found and read
  the cited papers.
- **How the output was checked:** every number in the documents is produced by the committed code
  and can be regenerated with `python run_all.py`. A from-scratch run in an empty folder reproduced
  all 20 tables and 12 figures. Where a first choice was contradicted by evidence, the evidence won,
  and both are documented: the Part1/Part2 overlap rule, the `Issues` handling and the
  one-stage → two-stage model. The later models were judged on validation like the rest, and all
  are reported on test.

## Possible improvements

1. **Leave-one-house-out cross-validation** over all 14 homes, for confidence intervals on per-home
   performance.
2. **Full multi-appliance disaggregation:** predict the watts of the washing machine, dishwasher and
   tumble dryer together and share the aggregate's energy between them. The current model only
   adds a look-alike *class*, which helped a little; sharing energy goes further at the main failure.
3. **Seq2Point CNN** on the same split as a direct comparison.
4. **Resolution study:** retrain and evaluate at 15 and 30 minutes to quantify Recommendation.md §3.
5. **Per-appliance plausibility limits** and a rolling-median spike filter in Section 1.
6. **Few-shot calibration** of a new home from a handful of customer-confirmed washes.

## References

**Dataset:** REFIT Electrical Load Measurements, University of Strathclyde (sources as given in the
assignment).

- Dataset page and documentation: <https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/>
- Raw REFIT data (.7z): <https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z>
- Cleaned REFIT data (.7z): <https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z>
- Cleaned REFIT readme: <https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt>
  (a copy is in [`docs/CLEAN_READ_ME_081116.txt`](docs/CLEAN_READ_ME_081116.txt))

**Research papers used** (the idea taken from each is listed in RESULTS.md §3.7):

- Murray, D., Stankovic, L., Stankovic, V. (2017). An electrical load measurements dataset of United
  Kingdom households from a two-year longitudinal study. *Scientific Data* 4, 160122.
- Zhang, C., Zhong, M., Wang, Z., Goddard, N., Sutton, C. (2018). Sequence-to-point learning with
  neural networks for non-intrusive load monitoring. *AAAI*. arXiv:1612.09106.
- Kelly, J., Knottenbelt, W. (2015). Neural NILM: Deep neural networks applied to energy
  disaggregation. *ACM BuildSys*.
- Precioso, D., Gómez-Ullate, D. (2020). NILM as a regression versus classification problem: the
  importance of thresholding. arXiv:2010.16050.
- D'Incecco, M., Squartini, S., Zhong, M. (2020). Transfer learning for non-intrusive load
  monitoring. *IEEE Transactions on Smart Grid*.
