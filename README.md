# EpiLanka ML

Disease-surveillance data pipeline for Sri Lanka covering three diseases (Dysentery, Meningitis, Tuberculosis) across 26 districts and ISO weeks 1-52. Trains a RandomForestRegressor on historical case + rainfall data, generates weekly forecasts, computes incidence-per-100k risk thresholds, and upserts results into a PostgreSQL database.

## Project Structure

```
core/        # Shared primitives
  config.py          # DB configuration
  districts.py       # District ID mappings and utilities
  diseases.py        # Disease ID mappings
  db.py              # Database connection helpers
  rainfall.py        # Rainfall data loaders
  time_utils.py      # Week/month utilities

pipeline/    # Pipeline modules (one per step)
  store_population.py    # Load district populations
  store_historical.py    # Store case data for all diseases
  compute_risk.py        # Calculate risk thresholds
  train_model.py         # Train RandomForest model
  generate_predictions.py # Generate forecast CSV
  store_predictions.py   # Store predictions in DB

datasets/    # Input CSVs, model bundle, predictions
outputs/     # Risk threshold outputs
```

## Quick Start

```powershell
# First-time setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Launch interactive console menu
python main.py
```

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `pipeline.store_population` | Store district population data |
| 2 | `pipeline.store_historical` | Store historical case data (all 3 diseases) |
| 3 | `pipeline.compute_risk` | Calculate risk thresholds and levels |
| 4 | `pipeline.train_model` | Train RandomForest model |
| 5 | `pipeline.generate_predictions` | Generate prediction CSV |
| 6 | `pipeline.store_predictions` | Store predictions in database |

## Running Individual Steps

```powershell
# Run modules directly
python -m pipeline.store_population
python -m pipeline.store_historical
python -m pipeline.compute_risk
python -m pipeline.train_model
python -m pipeline.generate_predictions
python -m pipeline.store_predictions

# Dry-run population loader without hitting DB
python population_loader_check.py
```

## Input Data

- Disease data: `datasets/{year}_{Disease}.csv` (Dysentery, Meningitis, Tuberculosis)
- Population: `datasets/srilanka_population.csv`
- Rainfall features: `datasets/rainfall_annual.csv` (loaded via `core/rainfall.py`)

## Analysis

`analysis.py` generates visualizations comparing disease trends with rainfall patterns for each district.

```powershell
python analysis.py
```

## Dependencies

- pandas, numpy
- scikit-learn
- joblib
- matplotlib, seaborn
- psycopg2-binary