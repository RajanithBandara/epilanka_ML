from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from analytics.data import ANALYTICS_DIR, CHARTS_DIR


def _pearson(x: pd.Series, y: pd.Series) -> float | None:
    if x.nunique() < 2 or y.nunique() < 2:
        return None
    return float(x.corr(y))


def per_district_correlation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (disease, district), sub in df.groupby(["disease", "district"]):
        sub = sub.dropna(subset=["cases_reported", "rainfall_mm"])
        if len(sub) < 10:
            continue
        corr = _pearson(sub["cases_reported"], sub["rainfall_mm"])
        if corr is None:
            continue
        rows.append(
            {
                "disease": disease,
                "district": district,
                "correlation": round(corr, 4),
                "n_weeks": len(sub),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["disease", "correlation"], ascending=[True, False]
    )


def lagged_correlation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lag_cols = {0: "rainfall_mm", 1: "rainfall_lag1_mm", 2: "rainfall_lag2_mm"}
    for (disease, district), sub in df.groupby(["disease", "district"]):
        for lag, col in lag_cols.items():
            aligned = sub.dropna(subset=["cases_reported", col])
            if len(aligned) < 10:
                continue
            corr = _pearson(aligned["cases_reported"], aligned[col])
            if corr is None:
                continue
            rows.append(
                {
                    "disease": disease,
                    "district": district,
                    "lag_months": lag,
                    "correlation": round(corr, 4),
                    "n_weeks": len(aligned),
                }
            )
    return pd.DataFrame(rows)


def _plot_correlation_distribution(corr_df: pd.DataFrame) -> None:
    if corr_df.empty:
        return
    diseases = sorted(corr_df["disease"].unique())
    data = [corr_df[corr_df["disease"] == d]["correlation"].dropna().values for d in diseases]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, tick_labels=diseases, showmeans=True)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Pearson r  (weekly cases vs same-month rainfall)")
    ax.set_title("Distribution of district-level correlation by disease")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "correlation_by_disease.png", dpi=120)
    plt.close(fig)


def _plot_lagged_correlation(lagged: pd.DataFrame) -> None:
    if lagged.empty:
        return
    summary = (
        lagged.groupby(["disease", "lag_months"])["correlation"]
        .agg(mean_corr="mean", std_corr="std")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for disease in sorted(summary["disease"].unique()):
        sub = summary[summary["disease"] == disease].sort_values("lag_months")
        ax.errorbar(
            sub["lag_months"],
            sub["mean_corr"],
            yerr=sub["std_corr"],
            marker="o",
            label=disease,
            capsize=4,
            linewidth=2,
        )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Rainfall lag (months)")
    ax.set_ylabel("Mean district correlation (± 1 std)")
    ax.set_title("Lagged rainfall correlation with weekly cases")
    ax.set_xticks([0, 1, 2])
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "lagged_correlation.png", dpi=120)
    plt.close(fig)


def run_correlations(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    per_district = per_district_correlation(df)
    lagged = lagged_correlation(df)

    per_district.to_csv(ANALYTICS_DIR / "rain_disease_correlation.csv", index=False)
    lagged.to_csv(ANALYTICS_DIR / "rain_disease_lagged_correlation.csv", index=False)

    _plot_correlation_distribution(per_district)
    _plot_lagged_correlation(lagged)

    return {"per_district": per_district, "lagged": lagged}
