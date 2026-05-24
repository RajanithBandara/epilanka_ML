from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd

from core.config import predictions_file
from core.db import connect, ensure_unique_constraint
from core.diseases import disease_id
from core.districts import district_id


def store_predictions(csv_file: str | Path) -> None:
    df = pd.read_csv(csv_file)

    required_cols = {"year", "week_number", "area_reported", "disease", "predicted_cases"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce").astype("Int64")
    df["predicted_cases"] = pd.to_numeric(df["predicted_cases"], errors="coerce")
    df["district_id"] = df["area_reported"].map(district_id)
    df["disease_id"] = df["disease"].map(disease_id)

    df = df.dropna(subset=["year", "week_number", "predicted_cases"])

    skipped_unknown_district = int(df["district_id"].isna().sum())
    skipped_unknown_disease = int(df["disease_id"].isna().sum())
    df = df.dropna(subset=["district_id", "disease_id"])

    with connect() as (conn, cursor):
        try:
            ensure_unique_constraint(
                cursor,
                "reports",
                ["week_number", "year", "district_id", "disease_id"],
            )

            upserted = 0
            for wk, yr, did, disid, pc in zip(
                df["week_number"],
                df["year"],
                df["district_id"],
                df["disease_id"],
                df["predicted_cases"],
            ):
                cursor.execute(
                    """
                    INSERT INTO reports
                        (report_id, week_number, year, district_id, disease_id, case_count)
                    VALUES
                        (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (week_number, year, district_id, disease_id)
                    DO UPDATE SET case_count = EXCLUDED.case_count
                    """,
                    (
                        uuid.uuid4(),
                        int(wk),
                        int(yr),
                        int(did),
                        int(disid),
                        int(round(float(pc))),
                    ),
                )
                upserted += 1

            conn.commit()

            print(f"[OK] Upsert attempted for {upserted} rows into reports")
            if skipped_unknown_district:
                print(f"[WARN] Skipped {skipped_unknown_district} rows due to unknown districts")
            if skipped_unknown_disease:
                print(f"[WARN] Skipped {skipped_unknown_disease} rows due to unknown diseases")

        except Exception as exc:
            conn.rollback()
            print("[ERROR]", exc)
            raise


if __name__ == "__main__":
    store_predictions(predictions_file(2026))
