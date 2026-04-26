from pathlib import Path

import joblib
import pandas as pd


DATASETS_DIR = Path("datasets")
MODEL_FILE = DATASETS_DIR / "disease_prediction_model.pkl"
RAINFALL_FILE = DATASETS_DIR / "rainfall_annual.csv"
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
WEEKS = range(1, 53)


def default_output_path(prediction_year: int) -> Path:
    return DATASETS_DIR / f"predictions_{prediction_year}.csv"


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


def load_rainfall_features() -> tuple[dict, dict]:
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
    rainfall_long["month"] = rainfall_long["month_name"].map(
        {month_name: index + 1 for index, month_name in enumerate(MONTH_ORDER)}
    )
    rainfall_long["area_key"] = rainfall_long[district_column].map(rainfall_area_key)

    if "Annual" in rainfall_wide.columns:
        avg_rainfall_by_area = (
            rainfall_wide.assign(
                area_key=rainfall_wide[district_column].map(rainfall_area_key),
                avg_annual_rainfall_mm=pd.to_numeric(
                    rainfall_wide["Annual"].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                ),
            )
            .dropna(subset=["avg_annual_rainfall_mm"])
            .drop_duplicates(subset=["area_key"])
            .set_index("area_key")["avg_annual_rainfall_mm"]
            .to_dict()
        )
    else:
        avg_rainfall_by_area = (
            rainfall_long.groupby("area_key")["avg_rainfall_mm"]
            .mean()
            .to_dict()
        )

    avg_rainfall_by_month = (
        rainfall_long.groupby("month")["avg_rainfall_mm"]
        .mean()
        .to_dict()
    )
    return avg_rainfall_by_area, avg_rainfall_by_month


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
        diseases = ["Dysentery", "Meningitis", "Tuberculosis"]

    avg_rainfall_by_area, avg_rainfall_by_month = load_rainfall_features()
    rows = []

    for area in areas:
        area_key = rainfall_area_key(area)
        for disease in diseases:
            for week in WEEKS:
                month = week_to_month(week)
                rows.append(
                    {
                        "year": prediction_year,
                        "week_number": week,
                        "month": month,
                        "avg_annual_rainfall_mm": avg_rainfall_by_area.get(area_key),
                        "national_avg_rainfall_mm": avg_rainfall_by_month.get(month),
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


def main() -> None:
    generate_predictions()


if __name__ == "__main__":
    main()
