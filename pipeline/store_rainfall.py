from __future__ import annotations

import pandas as pd

from core.config import RAINFALL_FILE
from core.db import connect, ensure_unique_constraint
from core.districts import DISTRICT_DISPLAY, DISTRICT_IDS, canonical_key, district_id


MONTH_COLUMNS: list[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

MONTH_TO_DB_COLUMN: dict[str, str] = {
    "Jan": "january",
    "Feb": "february",
    "Mar": "march",
    "Apr": "april",
    "May": "may",
    "Jun": "june",
    "Jul": "july",
    "Aug": "august",
    "Sep": "september",
    "Oct": "october",
    "Nov": "november",
    "Dec": "december",
}


def _to_int(value: object) -> int:
    text = str(value).replace(",", "").strip()
    return int(round(float(text)))


def load_rainfall_dataframe() -> pd.DataFrame:
    df = pd.read_csv(RAINFALL_FILE)

    district_column = "District" if "District" in df.columns else "Area"

    required = {district_column, *MONTH_COLUMNS, "Annual"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["district_id"] = df[district_column].map(district_id)
    df["__district_key"] = df[district_column].map(canonical_key)
    return df


def store_rainfall_data() -> None:
    df = load_rainfall_dataframe()

    skipped_unknown = int(df["district_id"].isna().sum())
    df = df.dropna(subset=["district_id"]).copy()

    rows: dict[int, dict[str, int]] = {}
    for _, row in df.iterrows():
        did = int(row["district_id"])
        monthly_values = {
            MONTH_TO_DB_COLUMN[month]: _to_int(row[month]) for month in MONTH_COLUMNS
        }
        monthly_values["annual_rainfall"] = _to_int(row["Annual"])
        rows[did] = monthly_values

    # Kalmunai (id 26) is administratively split out of Ampara (id 16) and has
    # no row in the CSV. Reuse Ampara's monthly profile so every district has a
    # row in rainfall_data.
    kalmunai_id = DISTRICT_IDS.get("kalmunai")
    ampara_id = DISTRICT_IDS.get("ampara")
    backfilled_kalmunai = False
    if kalmunai_id and ampara_id and kalmunai_id not in rows and ampara_id in rows:
        rows[kalmunai_id] = dict(rows[ampara_id])
        backfilled_kalmunai = True

    with connect() as (conn, cursor):
        try:
            ensure_unique_constraint(cursor, "rainfall_data", ["district_id"])

            upserted = 0
            for did, values in rows.items():
                cursor.execute(
                    """
                    INSERT INTO rainfall_data
                        (district_id,
                         january, february, march, april, may, june,
                         july, august, september, october, november, december,
                         annual_rainfall)
                    VALUES
                        (%s,
                         %s, %s, %s, %s, %s, %s,
                         %s, %s, %s, %s, %s, %s,
                         %s)
                    ON CONFLICT (district_id) DO UPDATE SET
                        january         = EXCLUDED.january,
                        february        = EXCLUDED.february,
                        march           = EXCLUDED.march,
                        april           = EXCLUDED.april,
                        may             = EXCLUDED.may,
                        june            = EXCLUDED.june,
                        july            = EXCLUDED.july,
                        august          = EXCLUDED.august,
                        september       = EXCLUDED.september,
                        october         = EXCLUDED.october,
                        november        = EXCLUDED.november,
                        december        = EXCLUDED.december,
                        annual_rainfall = EXCLUDED.annual_rainfall
                    """,
                    (
                        did,
                        values["january"], values["february"], values["march"],
                        values["april"], values["may"], values["june"],
                        values["july"], values["august"], values["september"],
                        values["october"], values["november"], values["december"],
                        values["annual_rainfall"],
                    ),
                )
                upserted += 1

            conn.commit()

            print(f"[OK] Upsert attempted for {upserted} rows into rainfall_data")
            if backfilled_kalmunai:
                print("[INFO] Kalmunai (id 26) filled from Ampara (id 16) rainfall profile")
            if skipped_unknown:
                print(f"[WARN] Skipped {skipped_unknown} rows due to unknown districts")

            unmapped = sorted(
                DISTRICT_DISPLAY[key]
                for key in DISTRICT_IDS
                if DISTRICT_IDS[key] not in rows
            )
            if unmapped:
                print(f"[WARN] No rainfall row inserted for: {', '.join(unmapped)}")

        except Exception as exc:
            conn.rollback()
            print("[ERROR]", exc)
            raise


if __name__ == "__main__":
    store_rainfall_data()
