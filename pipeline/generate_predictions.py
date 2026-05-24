from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from core.config import MODEL_FILE, predictions_file
from core.diseases import ALL_DISEASES
from core.districts import rainfall_area_name
from core.rainfall import load_rainfall_lookups
from core.time_utils import week_to_month


WEEKS = range(1, 53)


def default_output_path(prediction_year: int) -> Path:
    return predictions_file(prediction_year)


def generate_predictions(
    prediction_year: int = 2026,
    model_file: Path = MODEL_FILE,
    output_file: Path | None = None,
) -> Path:
    if output_file is None:
        output_file = default_output_path(prediction_year)

    bundle = joblib.load(model_file)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    areas = bundle.get("area_categories")
    diseases = bundle.get("disease_categories")

    if not areas:
        areas = sorted({
            column.replace("area_reported_", "")
            for column in feature_columns
            if column.startswith("area_reported_")
        })
    if not diseases:
        diseases = list(ALL_DISEASES)

    annual_by_area, monthly_national = load_rainfall_lookups()
    rows = []

    for area in areas:
        area_key = rainfall_area_name(area)
        for disease in diseases:
            for week in WEEKS:
                month = week_to_month(week)
                rows.append(
                    {
                        "year": prediction_year,
                        "week_number": week,
                        "month": month,
                        "avg_annual_rainfall_mm": annual_by_area.get(area_key),
                        "national_avg_rainfall_mm": monthly_national.get(month),
                        "area_reported": area,
                        "disease": disease,
                    }
                )

    future_df = pd.DataFrame(rows)
    future_encoded = pd.get_dummies(
        future_df,
        columns=["area_reported", "disease"],
        drop_first=True,
    )
    future_encoded = future_encoded.reindex(columns=feature_columns, fill_value=0)

    future_df["predicted_cases"] = model.predict(future_encoded)
    future_df["predicted_cases"] = (
        future_df["predicted_cases"]
        .round()
        .clip(lower=0)
        .astype(int)
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    future_df.to_csv(output_file, index=False)
    print(f"{prediction_year} predictions saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    generate_predictions()
