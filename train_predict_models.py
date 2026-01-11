import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


DATASETS_DIR = Path("datasets")

DATASETS = {
    "2023_Dysentery": "2023_Dysentery.csv",
    "2024_Dysentery": "2024_Dysentery.csv",
    "2025_Dysentery": "2025_Dysentery.csv",
    "2023_Meningitis": "2023_Meningitis.csv",
    "2024_Meningitis": "2024_Meningitis.csv",
    "2025_Meningitis": "2025_Meningitis.csv",
}

RAINFALL_FILE = DATASETS_DIR / "rainfall_annual.csv"
MODEL_OUTPUT = DATASETS_DIR / "disease_prediction_model.pkl"

def week_to_month(week_number: int) -> int:
    wk = int(week_number)
    wk = max(1, min(53, wk))
    return int(((wk - 1) / 53) * 12) + 1

# ===============================
# LOAD & PREPARE RAINFALL DATA
# ===============================
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


rows = []

for dataset_name, file_name in DATASETS.items():
    year, disease = dataset_name.split("_", 1)

    df = pd.read_csv(DATASETS_DIR / file_name)

    df["year"] = int(year)
    df["disease"] = disease
    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce")
    df["month"] = df["week_number"].map(week_to_month)

    df["area_key"] = df["area_reported"].astype(str).str.strip().str.casefold()

    # merge rainfall
    df = df.merge(annual_rainfall_by_area, on="area_key", how="left")
    df = df.merge(national_rainfall_by_month, on="month", how="left")

    rows.append(df)

data = pd.concat(rows, ignore_index=True)


data = data.dropna(subset=["cases_reported", "week_number"])
data["cases_reported"] = pd.to_numeric(data["cases_reported"], errors="coerce")
data = data.dropna()


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

# One-hot encoding
features_encoded = pd.get_dummies(
    features,
    columns=["area_reported", "disease"],
    drop_first=True
)


X_train, X_test, y_train, y_test = train_test_split(
    features_encoded,
    target,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Model Evaluation")
print("----------------")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(
    {
        "model": model,
        "feature_columns": features_encoded.columns.tolist(),
    },
    MODEL_OUTPUT
)

print(f"\nModel saved to: {MODEL_OUTPUT}")
