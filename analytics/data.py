from __future__ import annotations

import pandas as pd

from core.config import DATASETS_DIR, OUTPUTS_DIR, POPULATION_FILE, RAINFALL_FILE
from core.diseases import ALL_DISEASES
from core.districts import display_name, rainfall_area_name
from core.time_utils import MONTH_ORDER, MONTH_TO_NUM, week_to_month


ANALYTICS_DIR = OUTPUTS_DIR / "analytics"
CHARTS_DIR = ANALYTICS_DIR / "charts"

YEARS = (2023, 2024, 2025)


def load_population() -> pd.DataFrame:
    df = pd.read_csv(POPULATION_FILE)
    df["district"] = df["Region"].map(display_name)
    df["population"] = pd.to_numeric(
        df["Population"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    return df[["district", "population"]].dropna().drop_duplicates("district")


def load_rainfall_long() -> pd.DataFrame:
    rainfall_wide = pd.read_csv(RAINFALL_FILE)
    district_column = "District" if "District" in rainfall_wide.columns else "Area"
    df = rainfall_wide.melt(
        id_vars=[district_column],
        value_vars=MONTH_ORDER,
        var_name="month_name",
        value_name="avg_rainfall_mm",
    )
    df["avg_rainfall_mm"] = pd.to_numeric(
        df["avg_rainfall_mm"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    df["month"] = df["month_name"].map(MONTH_TO_NUM).astype("Int64")
    df["rainfall_area"] = df[district_column].astype(str).str.strip()
    return df[["rainfall_area", "month", "avg_rainfall_mm"]].dropna()


def _load_disease_year(year: int, disease: str) -> pd.DataFrame | None:
    path = DATASETS_DIR / f"{year}_{disease}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["year"] = year
    df["disease"] = disease
    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce").astype("Int64")
    df["cases_reported"] = pd.to_numeric(df["cases_reported"], errors="coerce")
    df["district"] = df["area_reported"].map(display_name)
    df = df.dropna(subset=["week_number", "cases_reported"])
    df["month"] = df["week_number"].map(week_to_month).astype("Int64")
    return df[["year", "disease", "district", "week_number", "month", "cases_reported"]]


def load_all_diseases() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        for disease in ALL_DISEASES:
            sub = _load_disease_year(year, disease)
            if sub is not None:
                frames.append(sub)
    if not frames:
        raise FileNotFoundError("No disease CSVs found in datasets/")
    return pd.concat(frames, ignore_index=True)


def build_full_dataframe() -> pd.DataFrame:
    diseases = load_all_diseases()
    population = load_population()
    rainfall = load_rainfall_long()

    df = diseases.merge(population, on="district", how="left")
    df["rainfall_area"] = df["district"].map(rainfall_area_name)

    base = rainfall.rename(columns={"avg_rainfall_mm": "rainfall_mm"})
    df = df.merge(base, on=["rainfall_area", "month"], how="left")

    for lag in (1, 2):
        shifted = rainfall.copy()
        shifted["month"] = ((shifted["month"].astype(int) + lag - 1) % 12 + 1).astype("Int64")
        shifted = shifted.rename(columns={"avg_rainfall_mm": f"rainfall_lag{lag}_mm"})
        df = df.merge(shifted, on=["rainfall_area", "month"], how="left")

    df["incidence_per_100k"] = (df["cases_reported"] / df["population"]) * 100000
    return df


def ensure_dirs() -> None:
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
