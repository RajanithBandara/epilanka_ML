from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from core.config import DATASETS_DIR, MODEL_FILE
from core.districts import display_name as district_display_name
from core.districts import rainfall_area_name
from core.rainfall import load_rainfall_features
from core.time_utils import week_to_month


DATASETS: dict[str, str] = {
    "2023_Dysentery":    "2023_Dysentery.csv",
    "2024_Dysentery":    "2024_Dysentery.csv",
    "2025_Dysentery":    "2025_Dysentery.csv",
    "2023_Meningitis":   "2023_Meningitis.csv",
    "2024_Meningitis":   "2024_Meningitis.csv",
    "2025_Meningitis":   "2025_Meningitis.csv",
    "2023_Tuberculosis": "2023_Tuberculosis.csv",
    "2024_Tuberculosis": "2024_Tuberculosis.csv",
    "2025_Tuberculosis": "2025_Tuberculosis.csv",
}


def _build_training_dataframe() -> pd.DataFrame:
    annual_rainfall_by_area, national_rainfall_by_month = load_rainfall_features()
    rows = []

    for dataset_name, file_name in DATASETS.items():
        year, disease = dataset_name.split("_", 1)
        path = DATASETS_DIR / file_name
        if not path.exists():
            print(f"[WARN] Skipping missing file: {path}")
            continue

        df = pd.read_csv(path)
        df["year"] = int(year)
        df["disease"] = disease
        df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce")
        df["month"] = df["week_number"].map(week_to_month)
        df["area_reported"] = df["area_reported"].map(district_display_name)
        df["area_key"] = df["area_reported"].map(rainfall_area_name)

        df = df.merge(annual_rainfall_by_area, on="area_key", how="left")
        df = df.merge(national_rainfall_by_month, on="month", how="left")
        rows.append(df)

    if not rows:
        raise FileNotFoundError("No training CSV files found in datasets/")

    data = pd.concat(rows, ignore_index=True)
    data = data.dropna(subset=["cases_reported", "week_number"])
    data["cases_reported"] = pd.to_numeric(data["cases_reported"], errors="coerce")
    return data.dropna().copy()


def train_prediction_model(
    model_output: Path = MODEL_FILE,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    data = _build_training_dataframe()

    features = data[
        [
            "year",
            "week_number",
            "month",
            "avg_annual_rainfall_mm",
            "national_avg_rainfall_mm",
            "area_reported",
            "disease",
        ]
    ]
    target = data["cases_reported"]

    area_categories = sorted(features["area_reported"].dropna().unique().tolist())
    disease_categories = sorted(features["disease"].dropna().unique().tolist())

    features_encoded = pd.get_dummies(
        features,
        columns=["area_reported", "disease"],
        drop_first=True,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features_encoded,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    bundle = {
        "model": model,
        "feature_columns": features_encoded.columns.tolist(),
        "area_categories": area_categories,
        "disease_categories": disease_categories,
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_output)

    result = {
        "model_output": model_output,
        "rows": len(data),
        "mae": float(mae),
        "rmse": float(rmse),
    }

    print("Model Evaluation")
    print("----------------")
    print(f"Training rows: {result['rows']}")
    print(f"MAE : {result['mae']:.2f}")
    print(f"RMSE: {result['rmse']:.2f}")
    print(f"\nModel saved to: {model_output}")

    return result


if __name__ == "__main__":
    train_prediction_model()
