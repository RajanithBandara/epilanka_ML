# EpiLanka ML

> Disease-surveillance data pipeline for Sri Lanka — case forecasting, risk classification, and rainfall correlation analytics across **3 diseases**, **26 districts**, and **52 ISO weeks**.

EpiLanka ML ingests historical weekly case counts (Dysentery, Meningitis, Tuberculosis) together with district-level population and national rainfall data, trains a single `RandomForestRegressor` over the joined feature set, and produces a full district × disease × week forecast for a chosen future year. Incidence-per-100k thresholds are derived per district and persisted alongside historical, predicted, and risk records into an Aiven-hosted PostgreSQL instance that powers the separate `epilanka` web front-end.

---

## Table of contents

- [Highlights](#highlights)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Interactive console](#interactive-console)
- [Pipeline overview](#pipeline-overview)
- [Data inputs](#data-inputs)
- [Generated outputs](#generated-outputs)
- [Model bundle format](#model-bundle-format)
- [Conventions and shared primitives](#conventions-and-shared-primitives)
- [Analytics module](#analytics-module)
- [Running individual steps](#running-individual-steps)
- [Dependencies](#dependencies)

---

## Highlights

| | |
|---|---|
| **Scope** | 3 diseases × 26 districts × 52 weeks × multiple years |
| **Model** | `RandomForestRegressor` (300 trees, depth-unbounded, all cores) |
| **Features** | year, ISO week, derived month, annual rainfall (area), national monthly rainfall, one-hot district + disease |
| **Risk model** | per-district incidence-per-100k bands — mean ± 1σ for *Normal*, +2σ flagged as *Outbreak* |
| **Persistence** | Aiven PostgreSQL — `historicaldata`, `reports`, `risk_levels`, `perdistrictpopulation`, `rainfall_data` |
| **Entry point** | `python main.py` — single 10-option console covering every step |
| **Testing/CI** | None — pipeline is idempotent and reruns are safe |

---

## Repository layout

```
epilanka_ML/
├── core/                      Shared primitives — imported everywhere
│   ├── config.py              DB credentials, DATASETS_DIR / OUTPUTS_DIR, model + CSV paths
│   ├── districts.py           26-district canonical IDs, aliases, display + rainfall-area resolvers
│   ├── diseases.py            Disease IDs (Dysentery=1, Meningitis=2, Tuberculosis=3)
│   ├── db.py                  connect() context manager + ensure_unique_constraint() helper
│   ├── rainfall.py            load_rainfall_features() / load_rainfall_lookups()
│   └── time_utils.py          week_to_month(), MONTH_ORDER, MONTH_TO_NUM
│
├── pipeline/                  One module per pipeline step (each is executable)
│   ├── store_population.py    -> perdistrictpopulation
│   ├── store_rainfall.py      -> rainfall_data
│   ├── store_historical.py    -> historicaldata (all 3 diseases in one pass)
│   ├── compute_risk.py        -> risk_levels (+ thresholds CSVs in outputs/)
│   ├── train_model.py         -> datasets/disease_prediction_model.pkl
│   ├── generate_predictions.py -> datasets/predictions_{year}.csv
│   └── store_predictions.py   -> reports
│
├── analytics/                 Exploratory reports (charts + summary text)
│   ├── data.py                Joined long-format frame builder
│   ├── disease_patterns.py    Year-on-year totals + district hotspots
│   ├── rain_patterns.py       Rainfall extremes
│   ├── correlations.py        Same-month and lagged Pearson r
│   └── run.py                 Orchestrator — writes outputs/analytics/summary.txt
│
├── datasets/                  Input CSVs + model bundle + generated predictions
├── outputs/                   Risk thresholds, analytics charts, summary text
├── main.py                    Interactive 10-option console (primary entry point)
├── population_loader_check.py Dry-run population loader (no DB writes)
└── requirements.txt
```

---

## Quick start

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the interactive console
python main.py
```

> **Database credentials** live in `core/config.py`. The password is in source — rotate on Aiven before sharing the repo publicly.

---

## Interactive console

`python main.py` opens a numbered menu. Each option is also runnable standalone (see [Running individual steps](#running-individual-steps)).

```
EpiLanka ML Console
-------------------
 1. Store district population data
 2. Store annual rainfall data (CSV -> rainfall_data)
 3. Store historical data for all diseases (Dysentery, Meningitis, Tuberculosis)
 4. Calculate thresholds and store risk levels
 5. Train prediction model
 6. Generate prediction CSV
 7. Store prediction CSV in database
 8. Train model, generate predictions, and store them
 9. Run full pipeline end-to-end
10. Run analytics (disease patterns, rain patterns, correlations)
 0. Exit
```

---

## Pipeline overview

The natural end-to-end ordering (matches option **9** in the console):

| # | Step | Module | Reads | Writes |
|---|------|--------|-------|--------|
| 1 | Store population | `pipeline.store_population` | `datasets/srilanka_population.csv` | `perdistrictpopulation` |
| 2 | Store rainfall | `pipeline.store_rainfall` | `datasets/rainfall_annual.csv` | `rainfall_data` |
| 3 | Store historical cases | `pipeline.store_historical` | `datasets/{year}_{Disease}.csv` × 9 | `historicaldata` |
| 4 | Compute risk levels | `pipeline.compute_risk` | `historicaldata`, `perdistrictpopulation` | `risk_levels`, `outputs/*.csv` |
| 5 | Train model | `pipeline.train_model` | 9 disease×year CSVs + rainfall features | `datasets/disease_prediction_model.pkl` |
| 6 | Generate predictions | `pipeline.generate_predictions` | model bundle + rainfall features | `datasets/predictions_{year}.csv` |
| 7 | Store predictions | `pipeline.store_predictions` | `datasets/predictions_{year}.csv` | `reports` |

Each `store_*` module idempotently provisions the necessary unique constraint via `core.db.ensure_unique_constraint()` before upserting, so a fresh database is bootstrapped automatically.

### Risk classification

`pipeline.compute_risk` derives weekly incidence per 100k per district from `historicaldata + perdistrictpopulation`, then bands it against per-district baselines:

| Band | Condition |
|------|-----------|
| **Below Expected** | `incidence < mean − 1σ` |
| **Normal**         | `mean − 1σ ≤ incidence ≤ mean + 1σ` |
| **Warning**        | `mean + 1σ < incidence ≤ mean + 2σ` |
| **High Risk**      | `incidence > mean + 2σ` |
| **Unknown**        | missing population, cases, or thresholds |

---

## Data inputs

All CSVs live in `datasets/`.

| File | Purpose |
|------|---------|
| `{year}_Dysentery.csv` / `{year}_Meningitis.csv` / `{year}_Tuberculosis.csv` (2023–2025) | District × ISO-week case counts |
| `srilanka_population.csv` | 26-district population (drives incidence rates) |
| `rainfall_annual.csv` | Annual rainfall per area + national monthly averages |
| `disease_prediction_model.pkl` | Generated — model bundle (see below) |
| `predictions_{year}.csv` | Generated — forecast grid |

> **Kalmunai note:** rainfall data has no Kalmunai row, so `core.districts.rainfall_area_name()` remaps district 26 to Ampara for the merge.

---

## Generated outputs

| Path | Produced by | Contents |
|------|-------------|----------|
| `datasets/disease_prediction_model.pkl` | `pipeline.train_model` | Pickled bundle dict |
| `datasets/predictions_{year}.csv`       | `pipeline.generate_predictions` | District × disease × week forecasts |
| `outputs/district_thresholds_{year}.csv` | `pipeline.compute_risk` | Per-district incidence bands |
| `outputs/weekly_risk_levels_{year}.csv`  | `pipeline.compute_risk` | Weekly risk labels |
| `outputs/analytics/summary.txt`          | `analytics.run`         | Human-readable analytics report |
| `outputs/analytics/charts/*.png`         | `analytics.run`         | Time-series and correlation plots |

---

## Model bundle format

`datasets/disease_prediction_model.pkl` is **not a bare estimator** — it's a dict with everything needed to re-encode prediction inputs to match the training schema:

```python
{
    "model":              RandomForestRegressor,
    "feature_columns":    list[str],   # full one-hot column order from training
    "area_categories":    list[str],   # 26 districts (canonical display names)
    "disease_categories": list[str],   # ['Dysentery', 'Meningitis', 'Tuberculosis']
}
```

`pipeline.generate_predictions` uses `feature_columns` with `DataFrame.reindex(columns=..., fill_value=0)` so the inference grid always lines up with training, regardless of which one-hot levels appear in the prediction year. There is no version field — if `train_model.py` changes feature engineering, delete the old `.pkl` before regenerating.

---

## Conventions and shared primitives

- **Districts: one source of truth.** `core/districts.py` exposes:
  - `canonical_key(value)` — lowercase, no-space key (`"nuwaraeliya"`); use for dict lookups
  - `district_id(value)` — `int` 1–26 or `None`
  - `display_name(value)` — spaced display form (`"Nuwara Eliya"`)
  - `rainfall_area_name(value)` — display form, with Kalmunai → Ampara remap
- **Diseases.** `core/diseases.py` — `dysentery=1, meningitis=2, tuberculosis=3`; `disease_id()` is case-insensitive.
- **DB access.** `core.db.connect()` is a context manager yielding `(conn, cursor)`. UUID adapter is registered once at import time. Always wrap writes with `try / except / conn.rollback()`.
- **Unique constraints.** `core.db.ensure_unique_constraint(cursor, table, [cols])` provisions on first run. `compute_risk` additionally drops a legacy pre-`disease_id` constraint — keep that migration intact.
- **Week → month.** Always `core.time_utils.week_to_month` — do not reimplement.

---

## Analytics module

`analytics/` is an exploratory layer that joins the same source data into a long-format frame and produces:

- **Year-on-year cases per disease** (`disease_patterns.run_disease_patterns`)
- **Top hotspot districts** (2023–2025 totals)
- **Rainfall extremes** — wettest and driest districts
- **Same-month and lagged Pearson correlation** between rainfall and weekly cases (`correlations.run_correlations`)
- **Human-readable summary** written to `outputs/analytics/summary.txt`

Run via menu option **10** or directly:

```powershell
python -m analytics.run
```

---

## Running individual steps

Every pipeline module is executable on its own:

```powershell
python -m pipeline.store_population
python -m pipeline.store_rainfall
python -m pipeline.store_historical
python -m pipeline.compute_risk
python -m pipeline.train_model
python -m pipeline.generate_predictions
python -m pipeline.store_predictions
python -m analytics.run

# Dry-run population loader without touching the DB
python population_loader_check.py
```

---

## Dependencies

Pinned only by name in `requirements.txt`:

- `pandas`, `numpy` — data wrangling
- `scikit-learn` — `RandomForestRegressor`, train/test split, metrics
- `joblib` — model bundle serialization
- `matplotlib`, `seaborn` — analytics charts
- `psycopg2-binary` — PostgreSQL driver

---

## Related repository

EpiLanka ML is the **data layer**. The Sri Lankan public-facing dashboard (`historicaldata`, `reports`, `risk_levels` consumers) lives in the separate `epilanka` web app and is fed exclusively by the tables this pipeline writes.
