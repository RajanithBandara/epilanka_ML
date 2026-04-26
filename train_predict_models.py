from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


DATASETS_DIR = Path("datasets")

DATASETS = {
    "2023_Dysentery": ("2023_Dysentery.csv",),
    "2024_Dysentery": ("2024_Dysentery.csv",),
    "2025_Dysentery": ("2025_Dysentery.csv",),
    "2023_Meningitis": ("2023_Meningitis.csv",),
    "2024_Meningitis": ("2024_Meningitis.csv",),
    "2025_Meningitis": ("2025_Meningitis.csv",),
    "2023_Tuberculosis": ("2023_Tuberculosis_harmonized.csv", "2023_Tuberculosis.csv"),
    "2024_Tuberculosis": ("2024_Tuberculosis_harmonized.csv", "2024_Tuberculosis.csv"),
    "2025_Tuberculosis": ("2025_Tuberculosis_harmonized.csv", "2025_Tuberculosis.csv"),
}

RAINFALL_FILE = DATASETS_DIR / "rainfall_annual.csv"
MODEL_OUTPUT = DATASETS_DIR / "disease_prediction_model.pkl"
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DISTRICT_NAME_MAP = {
    "ampara": "Ampara",
    "anuradhapura": "Anuradhapura",
    "badulla": "Badulla",
    "batticaloa": "Batticaloa",
    "colombo": "Colombo",
    "galle": "Galle",
    "gampaha": "Gampaha",
    "hambantota": "Hambantota",
    "jaffna": "Jaffna",
    "kalmune": "Kalmunai",
    "kalmunai": "Kalmunai",
    "kalutara": "Kalutara",
    "kandy": "Kandy",
    "kegalle": "Kegalle",
    "kilinochchi": "Kilinochchi",
    "kurunegala": "Kurunegala",
    "mannar": "Mannar",
    "matale": "Matale",
    "matara": "Matara",
    "monaragala": "Monaragala",
    "moneragala": "Monaragala",
    "mullaitivu": "Mullaitivu",
    "nuwaraeliya": "Nuwara Eliya",
    "polonnaruwa": "Polonnaruwa",
    "puttalam": "Puttalam",
    "ratnapura": "Ratnapura",
    "trincomalee": "Trincomalee",
    "vavuniya": "Vavuniya",
}


def week_to_month(week_number: int) -> int:
    wk = int(week_number)
    wk = max(1, min(53, wk))
    return int(((wk - 1) / 53) * 12) + 1


def normalize_district_name(value: object) -> str:
    key = str(value).strip().casefold().replace(" ", "")
    canonical = DISTRICT_NAME_MAP.get(key)
    if canonical is None:
        raise ValueError(f"Unknown district name: {value!r}")
    return canonical


def rainfall_area_key(value: object) -> str:
    district = normalize_district_name(value)
    if district == "Kalmunai":
        return "Ampara"
    return district


def load_rainfall_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    rainfall_wide = pd.read_csv(RAINFALL_FILE)
    district_column = "District" if "District" in rainfall_wide.columns else "Area"

    rainfall_long = rainfall_wide.melt(
        id_vars=[district_column],
        value_vars=MONTH_ORDER,
        var_name="month_name",
        value_name="avg_rainfall_mm",
    )
    rainfall_long["avg_rainfall_mm"] = pd.to_numeric(
        rainfall_long["avg_rainfall_mm"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    month_to_num = {month_name: index + 1 for index, month_name in enumerate(MONTH_ORDER)}
    rainfall_long["month"] = rainfall_long["month_name"].map(month_to_num)
    rainfall_long["area_key"] = rainfall_long[district_column].map(rainfall_area_key)

    if "Annual" in rainfall_wide.columns:
        annual_rainfall_by_area = (
            rainfall_wide.assign(
                area_key=rainfall_wide[district_column].map(rainfall_area_key),
                avg_annual_rainfall_mm=pd.to_numeric(
                    rainfall_wide["Annual"].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                ),
            )[["area_key", "avg_annual_rainfall_mm"]]
            .dropna(subset=["avg_annual_rainfall_mm"])
            .drop_duplicates(subset=["area_key"])
        )
    else:
        annual_rainfall_by_area = (
            rainfall_long.groupby("area_key")["avg_rainfall_mm"]
            .mean()
            .reset_index()
            .rename(columns={"avg_rainfall_mm": "avg_annual_rainfall_mm"})
        )

    national_rainfall_by_month = (
        rainfall_long.groupby("month")["avg_rainfall_mm"]
        .mean()
        .reset_index()
        .rename(columns={"avg_rainfall_mm": "national_avg_rainfall_mm"})
    )

    return annual_rainfall_by_area, national_rainfall_by_month


def resolve_dataset_path(file_names: tuple[str, ...]) -> Path:
    for file_name in file_names:
        candidate = DATASETS_DIR / file_name
        if candidate.exists():
            return candidate
    return DATASETS_DIR / file_names[0]


def build_training_dataframe() -> pd.DataFrame:
    annual_rainfall_by_area, national_rainfall_by_month = load_rainfall_features()
    rows = []

    for dataset_name, file_names in DATASETS.items():
        year, disease = dataset_name.split("_", 1)
        df = pd.read_csv(resolve_dataset_path(file_names))

        df["year"] = int(year)
        df["disease"] = disease
        df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce")
        df["month"] = df["week_number"].map(week_to_month)
        df["area_reported"] = df["area_reported"].map(normalize_district_name)
        df["area_key"] = df["area_reported"].map(rainfall_area_key)

        df = df.merge(annual_rainfall_by_area, on="area_key", how="left")
        df = df.merge(national_rainfall_by_month, on="month", how="left")
        rows.append(df)

    data = pd.concat(rows, ignore_index=True)
    data = data.dropna(subset=["cases_reported", "week_number"])
    data["cases_reported"] = pd.to_numeric(data["cases_reported"], errors="coerce")
    return data.dropna().copy()


def train_prediction_model(
    model_output: Path = MODEL_OUTPUT,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    data = build_training_dataframe()

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


def main() -> None:
    train_prediction_model()


if __name__ == "__main__":
    main()
