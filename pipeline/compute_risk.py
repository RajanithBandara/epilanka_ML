from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from core.config import DATASETS_DIR, OUTPUTS_DIR, POPULATION_FILE
from core.db import connect
from core.diseases import disease_id
from core.districts import district_id, display_name


TARGET_YEAR = 2024

DISEASE_FILES: dict[str, Path] = {
    "Dysentery":    DATASETS_DIR / "2024_Dysentery.csv",
    "Meningitis":   DATASETS_DIR / "2024_Meningitis.csv",
    "Tuberculosis": DATASETS_DIR / "2024_Tuberculosis.csv",
}

RISK_LEVEL_PRIORITY: dict[str, int] = {
    "Unknown": 0,
    "Below Expected": 1,
    "Normal": 2,
    "Warning": 3,
    "High Risk": 4,
}


def _classify_risk(incidence, lower, upper, outbreak) -> str:
    if pd.isna(incidence) or pd.isna(lower) or pd.isna(upper) or pd.isna(outbreak):
        return "Unknown"
    if incidence < lower:
        return "Below Expected"
    elif incidence <= upper:
        return "Normal"
    elif incidence <= outbreak:
        return "Warning"
    else:
        return "High Risk"


def _load_population(pop_file: Path) -> pd.DataFrame:
    pop_df = pd.read_csv(pop_file)

    expected_cols = {"Region", "Population"}
    missing = expected_cols - set(pop_df.columns)
    if missing:
        raise ValueError(f"Population file is missing columns: {missing}")

    pop_df["Region"] = pop_df["Region"].map(display_name)
    pop_df["Population"] = pd.to_numeric(
        pop_df["Population"].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )

    if pop_df["Population"].isna().any():
        bad_rows = pop_df[pop_df["Population"].isna()]
        print("\n[WARN] Some population values could not be converted:")
        print(bad_rows)

    return pop_df


def _load_disease_file(file_path: Path, disease_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    expected_cols = {"week_number", "area_reported", "cases_reported"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"{file_path.name} is missing columns: {missing}")

    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce")
    df["cases_reported"] = pd.to_numeric(df["cases_reported"], errors="coerce").fillna(0)
    df["area_reported"] = df["area_reported"].map(display_name)
    df["disease"] = disease_name

    df = df.dropna(subset=["week_number", "area_reported"])
    df["week_number"] = df["week_number"].astype(int)
    return df


def _calculate_thresholds(merged_df: pd.DataFrame) -> pd.DataFrame:
    threshold_df = (
        merged_df.groupby(["disease", "area_reported"], as_index=False)
        .agg(
            mean_incidence=("incidence_per_100k", "mean"),
            std_incidence=("incidence_per_100k", "std"),
            mean_cases=("cases_reported", "mean"),
            std_cases=("cases_reported", "std"),
            population=("Population", "first"),
            weeks=("week_number", "nunique"),
        )
    )
    threshold_df["std_incidence"] = threshold_df["std_incidence"].fillna(0)
    threshold_df["std_cases"] = threshold_df["std_cases"].fillna(0)
    threshold_df["lower_threshold"] = (
        threshold_df["mean_incidence"] - threshold_df["std_incidence"]
    ).clip(lower=0)
    threshold_df["upper_threshold"] = (
        threshold_df["mean_incidence"] + threshold_df["std_incidence"]
    )
    threshold_df["outbreak_threshold"] = (
        threshold_df["mean_incidence"] + 2 * threshold_df["std_incidence"]
    )
    return threshold_df


def _apply_thresholds(merged_df: pd.DataFrame, threshold_df: pd.DataFrame) -> pd.DataFrame:
    result_df = merged_df.merge(
        threshold_df[
            ["disease", "area_reported", "lower_threshold", "upper_threshold", "outbreak_threshold"]
        ],
        on=["disease", "area_reported"],
        how="left",
    )
    result_df["risk_level"] = result_df.apply(
        lambda row: _classify_risk(
            row["incidence_per_100k"],
            row["lower_threshold"],
            row["upper_threshold"],
            row["outbreak_threshold"],
        ),
        axis=1,
    )
    return result_df


def _ensure_risk_levels_constraint(cursor) -> None:
    cursor.execute(
        """
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        WHERE t.relname = 'risk_levels'
          AND c.contype = 'u'
          AND pg_get_constraintdef(c.oid) = 'UNIQUE (district_id, week_number, year, disease_id)'
        LIMIT 1
        """
    )
    if cursor.fetchone() is not None:
        return

    # Drop pre-disease_id constraint that used the same name; safe no-op when absent.
    cursor.execute(
        "ALTER TABLE risk_levels DROP CONSTRAINT IF EXISTS risk_levels_district_week_year_uniq"
    )
    cursor.execute(
        """
        ALTER TABLE risk_levels
        ADD CONSTRAINT risk_levels_district_week_year_uniq
        UNIQUE (district_id, week_number, year, disease_id)
        """
    )


def _prepare_risk_levels_rows(weekly_risk_df: pd.DataFrame, year: int) -> pd.DataFrame:
    db_df = weekly_risk_df.copy()
    db_df["district_id"] = db_df["area_reported"].map(district_id)
    db_df["disease_id"] = db_df["disease"].map(disease_id)
    db_df["year"] = int(year)
    db_df["risk_score"] = pd.to_numeric(db_df["incidence_per_100k"], errors="coerce")
    db_df["risk_priority"] = db_df["risk_level"].map(RISK_LEVEL_PRIORITY).fillna(0)

    skipped_district = int(db_df["district_id"].isna().sum())
    skipped_disease = int(db_df["disease_id"].isna().sum())
    if skipped_district:
        print(f"[WARN] Skipped {skipped_district} rows due to unknown district mappings")
    if skipped_disease:
        print(f"[WARN] Skipped {skipped_disease} rows due to unknown disease mappings")

    db_df = db_df.dropna(
        subset=[
            "district_id", "disease_id", "week_number", "risk_level",
            "lower_threshold", "upper_threshold", "outbreak_threshold", "risk_score",
        ]
    ).copy()

    db_df = db_df.sort_values(
        ["district_id", "week_number", "disease_id", "risk_priority", "risk_score"],
        ascending=[True, True, True, False, False],
    )
    db_df = db_df.drop_duplicates(subset=["district_id", "week_number", "disease_id"], keep="first")

    for col in ["lower_threshold", "upper_threshold", "outbreak_threshold"]:
        db_df[col] = pd.to_numeric(db_df[col], errors="coerce").round().astype("Int64")

    db_df = db_df.dropna(subset=["lower_threshold", "upper_threshold", "outbreak_threshold"])

    return db_df[
        [
            "district_id", "week_number", "year", "disease_id", "risk_level",
            "lower_threshold", "upper_threshold", "outbreak_threshold", "risk_score",
        ]
    ]


def _store_risk_levels(weekly_risk_df: pd.DataFrame, year: int) -> None:
    db_df = _prepare_risk_levels_rows(weekly_risk_df, year)

    with connect() as (conn, cursor):
        try:
            _ensure_risk_levels_constraint(cursor)

            upserted = 0
            for _, row in db_df.iterrows():
                cursor.execute(
                    """
                    INSERT INTO risk_levels
                        (risk_id, district_id, week_number, year, disease_id, risk_level,
                         lower_threshold, upper_threshold, outbreak_threshold, risk_score)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (district_id, week_number, year, disease_id)
                    DO UPDATE SET
                        risk_level = EXCLUDED.risk_level,
                        lower_threshold = EXCLUDED.lower_threshold,
                        upper_threshold = EXCLUDED.upper_threshold,
                        outbreak_threshold = EXCLUDED.outbreak_threshold,
                        risk_score = EXCLUDED.risk_score,
                        calculated_at = NOW()
                    """,
                    (
                        str(uuid.uuid4()),
                        int(row["district_id"]),
                        int(row["week_number"]),
                        int(row["year"]),
                        int(row["disease_id"]),
                        str(row["risk_level"]),
                        int(row["lower_threshold"]),
                        int(row["upper_threshold"]),
                        int(row["outbreak_threshold"]),
                        float(row["risk_score"]),
                    ),
                )
                upserted += 1

            conn.commit()
            print(f"[OK] Upsert attempted for {upserted} rows into risk_levels")

        except Exception as exc:
            conn.rollback()
            print("[ERROR] Failed to persist risk_levels:", exc)
            raise


def calculate_thresholds_and_store_risk_levels() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print("Loading population data...")
    population_df = _load_population(POPULATION_FILE)

    print("Loading disease datasets...")
    frames = []
    for disease_name, file_path in DISEASE_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")
        frames.append(_load_disease_file(file_path, disease_name))
    all_disease_df = pd.concat(frames, ignore_index=True)

    print("Merging disease data with population...")
    merged_df = all_disease_df.merge(
        population_df, left_on="area_reported", right_on="Region", how="left"
    )

    unmatched = (
        merged_df[merged_df["Population"].isna()]["area_reported"]
        .drop_duplicates()
        .tolist()
    )
    if unmatched:
        print("\n[WARN] These areas were not matched with population data:")
        for area in unmatched:
            print(f" - {area}")

    merged_df = merged_df.dropna(subset=["Population"]).copy()

    print("Calculating incidence per 100,000...")
    merged_df["incidence_per_100k"] = (
        merged_df["cases_reported"] / merged_df["Population"]
    ) * 100000

    print("Calculating thresholds...")
    threshold_df = _calculate_thresholds(merged_df)

    print("Applying thresholds and assigning risk levels...")
    weekly_risk_df = _apply_thresholds(merged_df, threshold_df)

    print("Saving risk levels to database...")
    _store_risk_levels(weekly_risk_df, TARGET_YEAR)

    threshold_df = threshold_df.sort_values(["disease", "area_reported"]).reset_index(drop=True)
    weekly_risk_df = weekly_risk_df.sort_values(
        ["disease", "week_number", "area_reported"]
    ).reset_index(drop=True)

    threshold_file = OUTPUTS_DIR / f"district_thresholds_{TARGET_YEAR}.csv"
    weekly_file = OUTPUTS_DIR / f"weekly_risk_levels_{TARGET_YEAR}.csv"

    try:
        threshold_df.to_csv(threshold_file, index=False)
        print(f"Threshold file saved to: {threshold_file}")
    except PermissionError:
        print(f"[WARN] Could not write {threshold_file} (file is open or locked)")

    try:
        weekly_risk_df.to_csv(weekly_file, index=False)
        print(f"Weekly risk file saved to: {weekly_file}")
    except PermissionError:
        print(f"[WARN] Could not write {weekly_file} (file is open or locked)")

    print("\nDone.")


if __name__ == "__main__":
    calculate_thresholds_and_store_risk_levels()
