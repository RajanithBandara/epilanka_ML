from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from analytics.data import ANALYTICS_DIR, CHARTS_DIR


def disease_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    hotspots = df.groupby(["disease", "year", "district"], as_index=False).agg(
        total_cases=("cases_reported", "sum"),
        population=("population", "first"),
    )
    hotspots["incidence_per_100k"] = (
        hotspots["total_cases"] / hotspots["population"] * 100000
    ).round(2)
    return hotspots.sort_values(
        ["disease", "year", "total_cases"], ascending=[True, True, False]
    )


def disease_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    seasonality = df.groupby(["disease", "year", "week_number"], as_index=False).agg(
        total_cases=("cases_reported", "sum"),
        mean_cases_per_district=("cases_reported", "mean"),
    )
    return seasonality.sort_values(["disease", "year", "week_number"])


def disease_monthly(df: pd.DataFrame) -> pd.DataFrame:
    monthly = df.groupby(["disease", "month"], as_index=False).agg(
        total_cases=("cases_reported", "sum"),
        mean_cases=("cases_reported", "mean"),
    )
    return monthly.sort_values(["disease", "month"])


def disease_yoy(df: pd.DataFrame) -> pd.DataFrame:
    by_disease = (
        df.groupby(["disease", "year"], as_index=False)["cases_reported"]
        .sum()
        .rename(columns={"cases_reported": "total_cases"})
    )
    by_disease["yoy_pct_change"] = by_disease.groupby("disease")["total_cases"].pct_change() * 100
    return by_disease.sort_values(["disease", "year"])


def _plot_hotspots(hotspots: pd.DataFrame, top_n: int = 10) -> None:
    for disease in sorted(hotspots["disease"].unique()):
        sub = (
            hotspots[hotspots["disease"] == disease]
            .groupby("district", as_index=False)["total_cases"].sum()
            .sort_values("total_cases", ascending=False)
            .head(top_n)
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(sub["district"], sub["total_cases"], color="steelblue")
        ax.invert_yaxis()
        ax.set_title(f"Top {top_n} districts by total cases — {disease} (2023–2025)")
        ax.set_xlabel("Total cases")
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / f"hotspots_{disease.lower()}.png", dpi=120)
        plt.close(fig)


def _plot_seasonality(seasonality: pd.DataFrame) -> None:
    for disease in sorted(seasonality["disease"].unique()):
        fig, ax = plt.subplots(figsize=(12, 6))
        sub_disease = seasonality[seasonality["disease"] == disease]
        for year in sorted(sub_disease["year"].unique()):
            sub = sub_disease[sub_disease["year"] == year]
            ax.plot(sub["week_number"], sub["total_cases"], label=str(year), linewidth=2)
        ax.set_xlabel("ISO week")
        ax.set_ylabel("Total cases (summed across districts)")
        ax.set_title(f"Weekly seasonality for {disease} by Year")
        ax.legend(title="Year")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / f"seasonality_{disease.lower()}.png", dpi=120)
        plt.close(fig)


def _plot_yoy(yoy: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for disease in sorted(yoy["disease"].unique()):
        sub = yoy[yoy["disease"] == disease].sort_values("year")
        ax.plot(sub["year"], sub["total_cases"], marker="o", label=disease, linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total cases")
    ax.set_title("Annual cases by disease")
    ax.set_xticks(sorted(yoy["year"].unique()))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "yoy_by_disease.png", dpi=120)
    plt.close(fig)


def run_disease_patterns(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    hotspots = disease_hotspots(df)
    seasonality = disease_seasonality(df)
    monthly = disease_monthly(df)
    yoy = disease_yoy(df)

    hotspots.to_csv(ANALYTICS_DIR / "disease_hotspots.csv", index=False)
    seasonality.to_csv(ANALYTICS_DIR / "disease_seasonality.csv", index=False)
    monthly.to_csv(ANALYTICS_DIR / "disease_monthly.csv", index=False)
    yoy.to_csv(ANALYTICS_DIR / "disease_yoy.csv", index=False)

    _plot_hotspots(hotspots)
    _plot_seasonality(seasonality)
    _plot_yoy(yoy)

    return {"hotspots": hotspots, "seasonality": seasonality, "monthly": monthly, "yoy": yoy}
