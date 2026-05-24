from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so 'analytics' module can be resolved when script is run directly
if __name__ == "__main__":
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from analytics.correlations import run_correlations
from analytics.data import ANALYTICS_DIR, CHARTS_DIR, build_full_dataframe, ensure_dirs
from analytics.disease_patterns import run_disease_patterns
from analytics.rain_patterns import run_rain_patterns


def _format_yoy(yoy_df) -> str:
    pivot = (
        yoy_df.pivot(index="disease", columns="year", values="total_cases")
        .fillna(0)
        .astype(int)
    )
    return pivot.to_string()


def _format_top_hotspots(hotspots_df, top_n: int = 5) -> str:
    lines = []
    by_district = (
        hotspots_df.groupby(["disease", "district"], as_index=False)["total_cases"].sum()
    )
    for disease in sorted(by_district["disease"].unique()):
        sub = by_district[by_district["disease"] == disease].sort_values(
            "total_cases", ascending=False
        ).head(top_n)
        lines.append(f"  {disease}:")
        for _, row in sub.iterrows():
            lines.append(f"    {row['district']:<20s} {int(row['total_cases']):>8,d} cases")
    return "\n".join(lines)


def _format_rainfall(totals_df, n: int = 5) -> str:
    lines = ["  Wettest:"]
    for _, row in totals_df.head(n).iterrows():
        lines.append(f"    {row['district']:<20s} {row['annual_rainfall_mm']:>7.0f} mm")
    lines.append("  Driest:")
    for _, row in totals_df.tail(n).iterrows():
        lines.append(f"    {row['district']:<20s} {row['annual_rainfall_mm']:>7.0f} mm")
    return "\n".join(lines)


def _format_correlation(per_district_df) -> str:
    lines = []
    grouped = per_district_df.groupby("disease")["correlation"]
    for disease in sorted(per_district_df["disease"].unique()):
        sub = grouped.get_group(disease)
        lines.append(
            f"  {disease:<15s} mean r = {sub.mean():+.3f}   "
            f"median = {sub.median():+.3f}   n_districts = {len(sub)}"
        )
    return "\n".join(lines)


def _format_lagged(lagged_df) -> str:
    summary = (
        lagged_df.groupby(["disease", "lag_months"])["correlation"]
        .mean()
        .unstack()
        .round(3)
    )
    summary.columns = [f"lag_{int(c)}mo" for c in summary.columns]
    return summary.to_string()


def write_summary(disease_results, rain_results, corr_results) -> str:
    lines = [
        "EpiLanka analytics summary",
        "=" * 60,
        "",
        "Annual cases by disease:",
        _format_yoy(disease_results["yoy"]),
        "",
        "Top 5 hotspot districts (2023-2025 total cases):",
        _format_top_hotspots(disease_results["hotspots"]),
        "",
        "Rainfall extremes:",
        _format_rainfall(rain_results["totals"]),
        "",
        "Rainfall vs disease correlation (Pearson r, same-month, weekly cases):",
        _format_correlation(corr_results["per_district"]),
        "",
        "Mean lagged correlation (months of rainfall lag):",
        _format_lagged(corr_results["lagged"]),
        "",
        "Artefacts:",
        f"  CSV tables:  {ANALYTICS_DIR}",
        f"  Charts:      {CHARTS_DIR}",
    ]
    text = "\n".join(lines)
    (ANALYTICS_DIR / "summary.txt").write_text(text, encoding="utf-8")
    return text


def run_analytics() -> None:
    ensure_dirs()

    print("Loading data...")
    df = build_full_dataframe()
    print(f"  {len(df):,} (year x disease x district x week) rows")

    print("Computing disease patterns...")
    disease_results = run_disease_patterns(df)

    print("Computing rainfall patterns...")
    rain_results = run_rain_patterns()

    print("Computing rainfall vs disease correlations...")
    corr_results = run_correlations(df)

    print()
    print(write_summary(disease_results, rain_results, corr_results))


if __name__ == "__main__":
    run_analytics()
