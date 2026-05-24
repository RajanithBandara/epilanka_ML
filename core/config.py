from __future__ import annotations

from pathlib import Path


DB_CONFIG: dict[str, object] = {
    "host": "epilanka-epilanka.j.aivencloud.com",
    "port": 16878,
    "database": "epilanka",
    "user": "avnadmin",
    "password": "AVNS_PEp6c7CMZQHAPYVNCEX",
}

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

RAINFALL_FILE: Path = DATASETS_DIR / "rainfall_annual.csv"
POPULATION_FILE: Path = DATASETS_DIR / "srilanka_population.csv"
MODEL_FILE: Path = DATASETS_DIR / "disease_prediction_model.pkl"


def predictions_file(year: int) -> Path:
    return DATASETS_DIR / f"predictions_{year}.csv"
