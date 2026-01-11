import uuid
import warnings

import pandas as pd
import psycopg2
import psycopg2.extras

warnings.filterwarnings("ignore")

# Register UUID adapter for psycopg2
psycopg2.extras.register_uuid()

DB_CONFIG = {
    "host": "epilanka-epilanka.j.aivencloud.com",
    "port": 16878,
    "database": "epilanka",
    "user": "avnadmin",
    "password": "AVNS_PEp6c7CMZQHAPYVNCEX",
}

DISTRICT_MAP = {
    "Colombo": 1,
    "Gampaha": 2,
    "Kalutara": 3,
    "Kandy": 4,
    "Matale": 5,
    "Nuwara Eliya": 6,
    "Galle": 7,
    "Hambantota": 8,
    "Matara": 9,
    "Jaffna": 10,
    "Kilinochchi": 11,
    "Mannar": 12,
    "Vavuniya": 13,
    "Mullaitivu": 14,
    "Batticaloa": 15,
    "Ampara": 16,
    "Trincomalee": 17,
    "Kurunegala": 18,
    "Puttalam": 19,
    "Anuradhapura": 20,
    "Polonnaruwa": 21,
    "Badulla": 22,
    "Monaragala": 23,
    "Ratnapura": 24,
    "Kegalle": 25,
    "Kalmunai": 26,
}

DISEASE_ID_MAP = {
    "dysentery": 1,
    "meningitis": 2,
}


def ensure_reports_conflict_constraint(cursor) -> None:
    cursor.execute(
        """
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        WHERE t.relname = 'reports'
          AND c.contype = 'u'
          AND pg_get_constraintdef(c.oid) = 'UNIQUE (week_number, year, district_id, disease_id)'
        LIMIT 1
        """
    )
    if cursor.fetchone() is not None:
        return

    cursor.execute(
        """
        ALTER TABLE reports
        ADD CONSTRAINT reports_week_year_district_disease_uniq
        UNIQUE (week_number, year, district_id, disease_id)
        """
    )


def store_predictions(csv_file: str) -> None:
    df = pd.read_csv(csv_file)

    required_cols = {"year", "week_number", "area_reported", "disease", "predicted_cases"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["area_reported"] = df["area_reported"].astype(str).str.strip()
    df["disease_key"] = df["disease"].astype(str).str.strip().str.casefold()

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce").astype("Int64")
    df["predicted_cases"] = pd.to_numeric(df["predicted_cases"], errors="coerce")

    df = df.dropna(subset=["year", "week_number", "area_reported", "disease_key", "predicted_cases"])

    df["district_id"] = df["area_reported"].map(DISTRICT_MAP)
    df["disease_id"] = df["disease_key"].map(DISEASE_ID_MAP)

    skipped_unknown_district = int(df["district_id"].isna().sum())
    skipped_unknown_disease = int(df["disease_id"].isna().sum())

    df = df.dropna(subset=["district_id", "disease_id"])

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        ensure_reports_conflict_constraint(cursor)

        inserted = 0

        for wk, yr, did, disid, pc in zip(
            df["week_number"],
            df["year"],
            df["district_id"],
            df["disease_id"],
            df["predicted_cases"],
        ):
            report_id = uuid.uuid4()

            cursor.execute(
                """
                INSERT INTO reports
                    (report_id, week_number, year, district_id, disease_id, case_count)
                VALUES
                    (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (week_number, year, district_id, disease_id)
                DO NOTHING
                """,
                (
                    report_id,
                    int(wk),
                    int(yr),
                    int(did),
                    int(disid),
                    int(round(float(pc))),
                ),
            )
            inserted += 1

        conn.commit()

        print(f"[OK] Insert attempted for {inserted} rows")
        if skipped_unknown_district:
            print(f"[WARN] Skipped {skipped_unknown_district} rows due to unknown districts")
        if skipped_unknown_disease:
            print(f"[WARN] Skipped {skipped_unknown_disease} rows due to unknown diseases")

    except Exception as e:
        conn.rollback()
        print("[ERROR]", e)
        raise
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    CSV_FILE = r"datasets\predictions_2026.csv"
    store_predictions(CSV_FILE)
