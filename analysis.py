import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

# -----------------------------
# Load rainfall (same logic as yours)
# -----------------------------

rainfall_wide = pd.read_csv(RAINFALL_FILE)

month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

rainfall_long = rainfall_wide.melt(
    id_vars=["District"],
    value_vars=month_order,
    var_name="month_name",
    value_name="avg_rainfall_mm"
)

rainfall_long["avg_rainfall_mm"] = pd.to_numeric(
    rainfall_long["avg_rainfall_mm"].astype(str).str.replace(",", ""),
    errors="coerce"
)

month_to_num = {m: i+1 for i, m in enumerate(month_order)}
rainfall_long["month"] = rainfall_long["month_name"].map(month_to_num)

rainfall_long["area_key"] = rainfall_long["District"].astype(str).str.strip().str.casefold()

# -----------------------------
# Helper: week to month
# -----------------------------

def week_to_month(week_number):
    wk = int(week_number)
    wk = max(1, min(53, wk))
    return int(((wk - 1) / 53) * 12) + 1


# =============================
# MAIN VISUALIZATION LOOP
# =============================

for dataset_name, file_name in DATASETS.items():

    print(f"Processing {dataset_name}")

    year, disease = dataset_name.split("_", 1)
    df = pd.read_csv(DATASETS_DIR / file_name)

    df["area_key"] = df["area_reported"].astype(str).str.strip().str.casefold()
    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce")
    df["cases_reported"] = pd.to_numeric(df["cases_reported"], errors="coerce")

    df["month"] = df["week_number"].dropna().astype(int).map(week_to_month)

    # Merge rainfall
    merged = df.merge(
        rainfall_long[["area_key", "month", "avg_rainfall_mm"]],
        on=["area_key", "month"],
        how="left"
    )

    # Get top 3 affected areas for clearer visualization
    top_areas = (
        merged.groupby("area_reported")["cases_reported"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
        .index
    )

    for area in top_areas:

        area_df = merged[merged["area_reported"] == area]
        area_df = area_df.sort_values("week_number")

        fig, ax1 = plt.subplots(figsize=(12,6))

        # Disease trend
        ax1.plot(
            area_df["week_number"],
            area_df["cases_reported"],
            color="red",
            marker="o",
            label="Cases Reported"
        )

        ax1.set_xlabel("Week Number")
        ax1.set_ylabel("Cases Reported", color="red")
        ax1.tick_params(axis='y', labelcolor='red')

        # Rainfall on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(
            area_df["week_number"],
            area_df["avg_rainfall_mm"],
            color="blue",
            linestyle="dashed",
            label="Rainfall (mm)"
        )

        ax2.set_ylabel("Rainfall (mm)", color="blue")
        ax2.tick_params(axis='y', labelcolor='blue')

        plt.title(f"{disease} Weekly Trend vs Rainfall\n{area} - {year}")

        fig.tight_layout()
        plt.show()
