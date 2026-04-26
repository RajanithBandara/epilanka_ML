import uuid
import warnings
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

warnings.filterwarnings("ignore")

psycopg2.extras.register_uuid()

DB_CONFIG = {
    "host": "epilanka-epilanka.j.aivencloud.com",
    "port": 16878,
    "database": "epilanka",
    "user": "avnadmin",
    "password": "AVNS_PEp6c7CMZQHAPYVNCEX",
}

DATASETS_DIR = Path("datasets")

TUBERCULOSIS_FILES = {
    2023: DATASETS_DIR / "2023_Tuberculosis.csv",
    2024: DATASETS_DIR / "2024_Tuberculosis.csv",
    2025: DATASETS_DIR / "2025_Tuberculosis.csv",
}

DISTRICT_MAP = {
    "colombo": 1,
    "gampaha": 2,
    "kalutara": 3,
    "kandy": 4,
    "matale": 5,
    "nuwaraeliya": 6,
    "galle": 7,
    "hambantota": 8,
    "matara": 9,
    "jaffna": 10,
    "kilinochchi": 11,
    "mannar": 12,
    "vavuniya": 13,
    "mullaitivu": 14,
    "batticaloa": 15,
    "ampara": 16,
    "trincomalee": 17,
    "kurunegala": 18,
    "puttalam": 19,
    "anuradhapura": 20,
    "polonnaruwa": 21,
    "badulla": 22,
    "monaragala": 23,
    "ratnapura": 24,
    "kegalle": 25,
    "kalmunai": 26,
}

DISTRICT_ALIASES = {
    "nuwara eliya": "nuwaraeliya",
    "nuwaraeliya": "nuwaraeliya",
    "kalmune": "kalmunai",
    "kalmunai": "kalmunai",
    "moneragala": "monaragala",
    "monaragala": "monaragala",
}

TUBERCULOSIS_DISEASE_ID = 3


def normalize_district_name(value: str) -> str:
    district = str(value).strip().casefold()
    district = DISTRICT_ALIASES.get(district, district)
    return district.replace(" ", "")


def ensure_historicaldata_conflict_constraint(cursor) -> None:
    cursor.execute(
        """
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        WHERE t.relname = 'historicaldata'
          AND c.contype = 'u'
          AND pg_get_constraintdef(c.oid) = 'UNIQUE (week_number, year, district_id, disease_id)'
        LIMIT 1
        """
    )
    if cursor.fetchone() is not None:
        return

    cursor.execute(
        """
        ALTER TABLE historicaldata
        ADD CONSTRAINT historicaldata_week_year_district_disease_uniq
        UNIQUE (week_number, year, district_id, disease_id)
        """
    )


def load_tuberculosis_dataframe() -> pd.DataFrame:
    frames = []

    for year, file_path in TUBERCULOSIS_FILES.items():
        df = pd.read_csv(file_path)
        df["year"] = year
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def store_tuberculosis_data() -> None:
    df = load_tuberculosis_dataframe()

    required_cols = {"year", "week_number", "area_reported", "cases_reported"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce").astype("Int64")
    df["cases_reported"] = pd.to_numeric(df["cases_reported"], errors="coerce")
    df["district_key"] = df["area_reported"].map(normalize_district_name)

    df = df.dropna(subset=["year", "week_number", "area_reported", "cases_reported"])
    df["district_id"] = df["district_key"].map(DISTRICT_MAP)

    skipped_unknown_district = int(df["district_id"].isna().sum())
    df = df.dropna(subset=["district_id"])

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        ensure_historicaldata_conflict_constraint(cursor)

        inserted = 0

        for wk, yr, did, count in zip(
            df["week_number"],
            df["year"],
            df["district_id"],
            df["cases_reported"],
        ):
            data_id = uuid.uuid4()

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
                    data_id,
                    int(wk),
                    int(yr),
                    int(did),
                    TUBERCULOSIS_DISEASE_ID,
                    int(round(float(count))),
                ),
            )
            inserted += 1

        conn.commit()

        print(f"[OK] Insert attempted for {inserted} tuberculosis rows into historicaldata")
        if skipped_unknown_district:
            print(f"[WARN] Skipped {skipped_unknown_district} rows due to unknown districts")

    except Exception as exc:
        conn.rollback()
        print("[ERROR]", exc)
        raise
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    store_tuberculosis_data()
