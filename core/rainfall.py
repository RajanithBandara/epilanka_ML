from __future__ import annotations

import pandas as pd

from core.config import RAINFALL_FILE
from core.districts import rainfall_area_name
from core.time_utils import MONTH_ORDER, MONTH_TO_NUM


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
    rainfall_long["month"] = rainfall_long["month_name"].map(MONTH_TO_NUM)
    rainfall_long["area_key"] = rainfall_long[district_column].map(rainfall_area_name)

    if "Annual" in rainfall_wide.columns:
        annual = (
            rainfall_wide.assign(
                area_key=rainfall_wide[district_column].map(rainfall_area_name),
                avg_annual_rainfall_mm=pd.to_numeric(
                    rainfall_wide["Annual"].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                ),
            )[["area_key", "avg_annual_rainfall_mm"]]
            .dropna(subset=["avg_annual_rainfall_mm"])
            .drop_duplicates(subset=["area_key"])
        )
    else:
        annual = (
            rainfall_long.groupby("area_key")["avg_rainfall_mm"]
            .mean()
            .reset_index()
            .rename(columns={"avg_rainfall_mm": "avg_annual_rainfall_mm"})
        )

    national_monthly = (
        rainfall_long.groupby("month")["avg_rainfall_mm"]
        .mean()
        .reset_index()
        .rename(columns={"avg_rainfall_mm": "national_avg_rainfall_mm"})
    )

    return annual, national_monthly


def load_rainfall_lookups() -> tuple[dict, dict]:
    annual, national_monthly = load_rainfall_features()
    annual_by_area = annual.set_index("area_key")["avg_annual_rainfall_mm"].to_dict()
    monthly_national = (
        national_monthly.set_index("month")["national_avg_rainfall_mm"].to_dict()
    )
    return annual_by_area, monthly_national
