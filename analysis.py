import pandas as pd
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
OUTPUT_FILE = DATASETS_DIR / "disease_analytics_summary.csv"

# --- Load + normalize rainfall (monthly long format) ---
rainfall_wide = pd.read_csv(RAINFALL_FILE)

rainfall_long = rainfall_wide.melt(
    id_vars=["Area"],
    var_name="month_name",
    value_name="avg_rainfall_mm",
)

month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_to_num = {m: i + 1 for i, m in enumerate(month_order)}
rainfall_long["month"] = rainfall_long["month_name"].map(month_to_num)

# normalize area keys a bit for safer joins
rainfall_long["area_key"] = rainfall_long["Area"].astype(str).str.strip().str.casefold()

# --- Helpers ---
def week_to_month(week_number: int) -> int:
    """
    Approximate mapping from epidemiological week number (1..52/53) to month (1..12).
    Uses equal-length month buckets; good enough for correlation/overlay summaries.
    """
    wk = int(week_number)
    wk = max(1, min(53, wk))
    return int(((wk - 1) / 53) * 12) + 1

results = []

for dataset_name, file_name in DATASETS.items():
    year, disease = dataset_name.split("_", 1)
    file_path = DATASETS_DIR / file_name

    df = pd.read_csv(file_path)

    # Normalize keys for joins
    df["area_key"] = df["area_reported"].astype(str).str.strip().str.casefold()
    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce")

    # Add month derived from week_number
    df["month"] = df["week_number"].dropna().astype(int).map(week_to_month)
    df["month"] = pd.to_numeric(df["month"], errors="coerce")

    # --- Most affected areas (top 5) with annual rainfall context (mean across months) ---
    area_summary = (
        df.groupby(["area_reported", "area_key"], dropna=False)["cases_reported"]
        .sum()
        .reset_index()
        .sort_values("cases_reported", ascending=False)
        .head(5)
    )

    annual_rainfall_by_area = (
        rainfall_long.groupby("area_key", dropna=False)["avg_rainfall_mm"].mean().reset_index()
    ).rename(columns={"avg_rainfall_mm": "avg_annual_rainfall_mm"})

    area_summary = area_summary.merge(annual_rainfall_by_area, on="area_key", how="left")

    for _, row in area_summary.iterrows():
        results.append(
            {
                "year": year,
                "disease": disease,
                "analysis_type": "Most Affected Area",
                "identifier": row["area_reported"],
                "value": row["cases_reported"],
                "avg_annual_rainfall_mm": row.get("avg_annual_rainfall_mm"),
            }
        )

    # --- Peak weeks (top 5) with rainfall for the peak month (nationally averaged) ---
    week_summary = (
        df.groupby("week_number", dropna=False)["cases_reported"]
        .sum()
        .reset_index()
        .dropna(subset=["week_number"])
        .sort_values("cases_reported", ascending=False)
        .head(5)
    )
    week_summary["week_number"] = week_summary["week_number"].astype(int)
    week_summary["month"] = week_summary["week_number"].map(week_to_month)

    # national monthly avg rainfall (avg across all areas for that month)
    national_rainfall_by_month = (
        rainfall_long.groupby("month", dropna=False)["avg_rainfall_mm"].mean().reset_index()
    ).rename(columns={"avg_rainfall_mm": "national_avg_rainfall_mm"})

    week_summary = week_summary.merge(national_rainfall_by_month, on="month", how="left")

    for _, row in week_summary.iterrows():
        results.append(
            {
                "year": year,
                "disease": disease,
                "analysis_type": "Peak Week",
                "identifier": f"Week {int(row['week_number'])}",
                "value": row["cases_reported"],
                "month": int(row["month"]),
                "national_avg_rainfall_mm": row.get("national_avg_rainfall_mm"),
            }
        )

analytics_df = pd.DataFrame(results)
analytics_df.to_csv(OUTPUT_FILE, index=False)

print(f"Analytics saved to {OUTPUT_FILE}")
