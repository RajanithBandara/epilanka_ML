import pandas as pd
from pathlib import Path
import joblib

DATASETS_DIR = Path("datasets")
MODEL_FILE = DATASETS_DIR / "disease_prediction_model.pkl"
RAINFALL_FILE = DATASETS_DIR / "rainfall_annual.csv"
OUTPUT_FILE = DATASETS_DIR / "predictions_2026.csv"

DISEASES = ["Dysentery", "Meningitis"]
WEEKS = range(1, 53)

def week_to_month(week_number: int) -> int:
    wk = int(week_number)
    wk = max(1, min(53, wk))
    return int(((wk - 1) / 53) * 12) + 1

bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
feature_columns = bundle["feature_columns"]

rainfall_wide = pd.read_csv(RAINFALL_FILE)

rainfall_long = rainfall_wide.melt(
    id_vars=["Area"],
    var_name="month_name",
    value_name="avg_rainfall_mm",
)

month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_to_num = {m: i + 1 for i, m in enumerate(month_order)}

rainfall_long["month"] = rainfall_long["month_name"].map(month_to_num)
rainfall_long["area_key"] = rainfall_long["Area"].astype(str).str.strip().str.casefold()

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

areas = sorted({
    col.replace("area_reported_", "")
    for col in feature_columns
    if col.startswith("area_reported_")
})

rows = []

for area in areas:
    area_key = area.strip().casefold()

    for disease in DISEASES:
        for week in WEEKS:
            month = week_to_month(week)

            rows.append({
                "year": 2026,
                "week_number": week,
                "month": month,
                "avg_annual_rainfall_mm": avg_rainfall_by_area.get(area_key),
                "national_avg_rainfall_mm": avg_rainfall_by_month.get(month),
                "area_reported": area,
                "disease": disease,
            })

future_df = pd.DataFrame(rows)

future_encoded = pd.get_dummies(
    future_df,
    columns=["area_reported", "disease"],
    drop_first=True
)

future_encoded = future_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

future_df["predicted_cases"] = model.predict(future_encoded)

# Ensure non-negative integers
future_df["predicted_cases"] = (
    future_df["predicted_cases"]
    .round()
    .clip(lower=0)
    .astype(int)
)

future_df.to_csv(OUTPUT_FILE, index=False)

print(f"2026 predictions saved to: {OUTPUT_FILE}")
