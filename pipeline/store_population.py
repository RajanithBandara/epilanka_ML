from __future__ import annotations

import uuid

import pandas as pd
import psycopg2.sql

from core.config import POPULATION_FILE
from core.db import (
    connect,
    ensure_unique_constraint,
    get_column_type,
    pick_existing_column,
)
from core.districts import district_id


def load_population_dataframe() -> pd.DataFrame:
    df = pd.read_csv(POPULATION_FILE)

    required_cols = {"Region", "Population"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["population"] = pd.to_numeric(
        df["Population"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    df["districts_id"] = df["Region"].map(district_id)

    return df.dropna(subset=["population"])


def store_population_data() -> None:
    df = load_population_dataframe()

    skipped_unknown_district = int(df["districts_id"].isna().sum())
    df = df.dropna(subset=["districts_id"])

    with connect() as (conn, cursor):
        try:
            district_column = pick_existing_column(
                cursor, "perdistrictpopulation", ["districts_id", "district_id"]
            )
            if district_column is None:
                raise ValueError(
                    "perdistrictpopulation must have districts_id or district_id column"
                )

            id_type = get_column_type(cursor, "perdistrictpopulation", "id")
            if id_type is None:
                raise ValueError("perdistrictpopulation must have an id column")

            ensure_unique_constraint(
                cursor, "perdistrictpopulation", [district_column]
            )

            upserted = 0
            insert_with_id = psycopg2.sql.SQL(
                """
                INSERT INTO perdistrictpopulation (id, {district_column}, population)
                VALUES (%s, %s, %s)
                ON CONFLICT ({district_column})
                DO UPDATE SET population = EXCLUDED.population
                """
            ).format(district_column=psycopg2.sql.Identifier(district_column))

            insert_without_id = psycopg2.sql.SQL(
                """
                INSERT INTO perdistrictpopulation ({district_column}, population)
                VALUES (%s, %s)
                ON CONFLICT ({district_column})
                DO UPDATE SET population = EXCLUDED.population
                """
            ).format(district_column=psycopg2.sql.Identifier(district_column))

            integer_id = id_type in {"integer", "bigint", "smallint"}

            for districts_id, population in zip(df["districts_id"], df["population"]):
                if integer_id:
                    cursor.execute(
                        insert_without_id,
                        (int(districts_id), int(round(float(population)))),
                    )
                else:
                    cursor.execute(
                        insert_with_id,
                        (uuid.uuid4(), int(districts_id), int(round(float(population)))),
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


if __name__ == "__main__":
    store_population_data()
