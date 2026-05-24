from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from core.config import DATASETS_DIR
from core.db import connect, ensure_unique_constraint
from core.diseases import disease_id
from core.districts import district_id


HISTORICAL_FILES: dict[tuple[int, str], Path] = {
    (2023, "Dysentery"):    DATASETS_DIR / "2023_Dysentery.csv",
    (2024, "Dysentery"):    DATASETS_DIR / "2024_Dysentery.csv",
    (2025, "Dysentery"):    DATASETS_DIR / "2025_Dysentery.csv",
    (2023, "Meningitis"):   DATASETS_DIR / "2023_Meningitis.csv",
    (2024, "Meningitis"):   DATASETS_DIR / "2024_Meningitis.csv",
    (2025, "Meningitis"):   DATASETS_DIR / "2025_Meningitis.csv",
    (2023, "Tuberculosis"): DATASETS_DIR / "2023_Tuberculosis.csv",
    (2024, "Tuberculosis"): DATASETS_DIR / "2024_Tuberculosis.csv",
    (2025, "Tuberculosis"): DATASETS_DIR / "2025_Tuberculosis.csv",
}


def load_historical_dataframe() -> pd.DataFrame:
    frames = []
    for (year, disease), file_path in HISTORICAL_FILES.items():
        if not file_path.exists():
            print(f"[WARN] Skipping missing file: {file_path}")
            continue
        df = pd.read_csv(file_path)
        df["year"] = year
        df["disease"] = disease
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No historical CSV files found in datasets/")

    return pd.concat(frames, ignore_index=True)


def store_historical_data() -> None:
    df = load_historical_dataframe()

    required = {"year", "week_number", "area_reported", "disease", "cases_reported"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce").astype("Int64")
    df["cases_reported"] = pd.to_numeric(df["cases_reported"], errors="coerce")
    df["district_id"] = df["area_reported"].map(district_id)
    df["disease_id"] = df["disease"].map(disease_id)

    df = df.dropna(subset=["year", "week_number", "cases_reported"])

    skipped_district = int(df["district_id"].isna().sum())
    skipped_disease = int(df["disease_id"].isna().sum())
    df = df.dropna(subset=["district_id", "disease_id"])

    with connect() as (conn, cursor):
        try:
            ensure_unique_constraint(
                cursor,
                "historicaldata",
                ["week_number", "year", "district_id", "disease_id"],
            )

            inserted = 0
            for wk, yr, did, disid, count in zip(
                df["week_number"],
                df["year"],
                df["district_id"],
                df["disease_id"],
                df["cases_reported"],
            ):
                cursor.execute(
                    """
                    INSERT INTO historicaldata
                        (data_id, week_number, year, district_id, disease_id, case_count)
                    VALUES
                        (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (week_number, year, district_id, disease_id)
                    DO NOTHING
                    """,
                    (
                        uuid.uuid4(),
                        int(wk),
                        int(yr),
                        int(did),
                        int(disid),
                        int(round(float(count))),
                    ),
                )
                inserted += 1

            conn.commit()

            print(f"[OK] Insert attempted for {inserted} rows into historicaldata")
            for disease, count in df.groupby("disease").size().items():
                print(f"        - {disease}: {count} rows")
            if skipped_district:
                print(f"[WARN] Skipped {skipped_district} rows due to unknown districts")
            if skipped_disease:
                print(f"[WARN] Skipped {skipped_disease} rows due to unknown diseases")

        except Exception as exc:
            conn.rollback()
            print("[ERROR]", exc)
            raise


if __name__ == "__main__":
    store_historical_data()
