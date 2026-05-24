from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analytics.data import ANALYTICS_DIR, CHARTS_DIR, load_rainfall_long
from core.time_utils import MONTH_ORDER


def rainfall_monthly_by_district() -> pd.DataFrame:
    rainfall = load_rainfall_long().rename(columns={"rainfall_area": "district"})
    rainfall["month"] = rainfall["month"].astype(int)
    rainfall["month_name"] = rainfall["month"].apply(lambda m: MONTH_ORDER[m - 1])
    return rainfall.sort_values(["district", "month"])[
        ["district", "month", "month_name", "avg_rainfall_mm"]
    ]


def rainfall_district_totals(monthly: pd.DataFrame) -> pd.DataFrame:
    totals = (
        monthly.groupby("district", as_index=False)["avg_rainfall_mm"]
        .sum()
        .rename(columns={"avg_rainfall_mm": "annual_rainfall_mm"})
        .sort_values("annual_rainfall_mm", ascending=False)
        .reset_index(drop=True)
    )
    totals["rank"] = totals.index + 1
    return totals


def rainfall_peak_months(monthly: pd.DataFrame) -> pd.DataFrame:
    idx = monthly.groupby("district")["avg_rainfall_mm"].idxmax()
    peaks = monthly.loc[idx, ["district", "month_name", "avg_rainfall_mm"]].rename(
        columns={"month_name": "peak_month", "avg_rainfall_mm": "peak_rainfall_mm"}
    )
    return peaks.sort_values("district").reset_index(drop=True)


def _plot_rainfall_heatmap(monthly: pd.DataFrame) -> None:
    pivot = monthly.pivot(index="district", columns="month_name", values="avg_rainfall_mm")
    pivot = pivot.reindex(columns=MONTH_ORDER).sort_index()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        annot=True,
        fmt=".0f",
        linewidths=0.3,
        ax=ax,
        cbar_kws={"label": "mm"},
    )
    ax.set_title("Average monthly rainfall by district (mm)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "rainfall_heatmap.png", dpi=120)
    plt.close(fig)


def _plot_district_totals(totals: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(totals["district"], totals["annual_rainfall_mm"], color="cornflowerblue")
    ax.invert_yaxis()
    ax.set_xlabel("Annual rainfall (mm, summed from monthly averages)")
    ax.set_title("Districts ranked by annual rainfall")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "rainfall_district_totals.png", dpi=120)
    plt.close(fig)


def run_rain_patterns() -> dict[str, pd.DataFrame]:
    monthly = rainfall_monthly_by_district()
    totals = rainfall_district_totals(monthly)
    peaks = rainfall_peak_months(monthly)

    monthly.to_csv(ANALYTICS_DIR / "rainfall_monthly_by_district.csv", index=False)
    totals.to_csv(ANALYTICS_DIR / "rainfall_district_totals.csv", index=False)
    peaks.to_csv(ANALYTICS_DIR / "rainfall_peak_months.csv", index=False)

    _plot_rainfall_heatmap(monthly)
    _plot_district_totals(totals)

    return {"monthly": monthly, "totals": totals, "peaks": peaks}
