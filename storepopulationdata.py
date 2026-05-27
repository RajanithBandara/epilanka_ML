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

DATA_FILE = Path("datasets") / "srilanka_population.csv"

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


def normalize_district_name(value: str) -> str:
    district = str(value).strip().casefold()
    district = DISTRICT_ALIASES.get(district, district)
    return district.replace(" ", "")


def ensure_population_conflict_constraint(cursor) -> None:
    cursor.execute(
        """
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        WHERE t.relname = 'perdistrictpopulation'
          AND c.contype = 'u'
          AND pg_get_constraintdef(c.oid) = 'UNIQUE (districts_id)'
        LIMIT 1
        """
    )
    if cursor.fetchone() is not None:
        return

    cursor.execute(
        """
        ALTER TABLE perdistrictpopulation
        ADD CONSTRAINT perdistrictpopulation_districts_id_uniq
        UNIQUE (districts_id)
        """
    )


def load_population_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)

    required_cols = {"Region", "Population"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["district_key"] = df["Region"].map(normalize_district_name)
    df["population"] = pd.to_numeric(
        df["Population"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    df = df.dropna(subset=["district_key", "population"])
    df["districts_id"] = df["district_key"].map(DISTRICT_MAP)

    return df


def store_population_data() -> None:
    df = load_population_dataframe()

    skipped_unknown_district = int(df["districts_id"].isna().sum())
    df = df.dropna(subset=["districts_id"])

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        ensure_population_conflict_constraint(cursor)

        upserted = 0

        for districts_id, population in zip(
            df["districts_id"],
            df["population"],
        ):
            pop_id = uuid.uuid4()

            cursor.execute(
                """
                INSERT INTO perdistrictpopulation (id, districts_id, population)
                VALUES (%s, %s, %s)
                ON CONFLICT (districts_id)
                DO UPDATE SET population = EXCLUDED.population
                """,
                (
                    pop_id,
                    int(districts_id),
                    int(round(float(population))),
                ),
            )
            upserted += 1

        conn.commit()

        print(f"[OK] Upsert attempted for {upserted} rows into perdistrictpopulation")
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
    store_population_data()

