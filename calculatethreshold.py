from pathlib import Path
import uuid
import pandas as pd
import psycopg2


# =========================
# CONFIG
# =========================
DATASET_DIR = Path("datasets")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_YEAR = 2024

DISEASE_FILES = {
    "Dysentery": DATASET_DIR / "2024_Dysentery.csv",
    "Meningitis": DATASET_DIR / "2024_Meningitis.csv",
    "Tuberculosis": DATASET_DIR / "2024_Tuberculosis.csv",
}

POPULATION_FILE = DATASET_DIR / "srilanka_population.csv"

DB_CONFIG = {
    "host": "epilanka-epilanka.j.aivencloud.com",
    "port": 16878,
    "database": "epilanka",
    "user": "avnadmin",
    "password": "AVNS_PEp6c7CMZQHAPYVNCEX",
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

RISK_LEVEL_PRIORITY = {
    "Unknown": 0,
    "Below Expected": 1,
    "Normal": 2,
    "Warning": 3,
    "High Risk": 4,
}

DISEASE_ID_MAP = {
    "dysentery": 1,
    "meningitis": 2,
    "tuberculosis": 3,
}


# =========================
# HELPER FUNCTIONS
# =========================
def clean_text(value: str) -> str:
    """Standardize text for matching district/area names."""
    if pd.isna(value):
        return value
    value = str(value).strip()
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = " ".join(value.split())
    return value


def normalize_area_name(name: str) -> str:
    """
    Fix common area naming differences between disease files and population file.
    Add or edit mappings here if your CSVs use different spellings.
    """
    if pd.isna(name):
        return name

    name = clean_text(name)

    mapping = {
        "NuwaraEliya": "Nuwara Eliya",
        "Nuwara Eliya": "Nuwara Eliya",
        "Kalmune": "Kalmunai",
        "Jaffna": "Jaffna",
        "Batticaloa": "Batticaloa",
        "Anuradhapura": "Anuradhapura",
        "Polonnaruwa": "Polonnaruwa",
        "Kurunegala": "Kurunegala",
        "Trincomalee": "Trincomalee",
        "Mullaitivu": "Mullaitivu",
        "Kilinochchi": "Kilinochchi",
        "Kegalle": "Kegalle",
        "Puttalam": "Puttalam",
        "Monaragala": "Monaragala",
        "Ratnapura": "Ratnapura",
        "Hambantota": "Hambantota",
        "Badulla": "Badulla",
        "Matara": "Matara",
        "Galle": "Galle",
        "Mannar": "Mannar",
        "Vavuniya": "Vavuniya",
        "Colombo": "Colombo",
        "Gampaha": "Gampaha",
        "Kalutara": "Kalutara",
        "Kandy": "Kandy",
        "Matale": "Matale",
        "Ampara": "Ampara",
    }

    return mapping.get(name, name)


def classify_risk(incidence, lower, upper, outbreak):
    """Assign risk level based on thresholds."""
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


def load_population(pop_file: Path) -> pd.DataFrame:
    """Load and clean the population file."""
    pop_df = pd.read_csv(pop_file)

    expected_cols = {"Region", "Population"}
    missing = expected_cols - set(pop_df.columns)
    if missing:
        raise ValueError(f"Population file is missing columns: {missing}")

    pop_df["Region"] = pop_df["Region"].apply(normalize_area_name)

    # Remove commas from population values and convert to numeric
    pop_df["Population"] = (
        pop_df["Population"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    pop_df["Population"] = pd.to_numeric(pop_df["Population"], errors="coerce")

    if pop_df["Population"].isna().any():
        bad_rows = pop_df[pop_df["Population"].isna()]
        print("\nWarning: Some population values could not be converted:")
        print(bad_rows)

    return pop_df


def load_disease_file(file_path: Path, disease_name: str) -> pd.DataFrame:
    """Load one disease dataset and standardize columns."""
    df = pd.read_csv(file_path)

    expected_cols = {"week_number", "area_reported", "cases_reported"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"{file_path.name} is missing columns: {missing}")

    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce")
    df["cases_reported"] = pd.to_numeric(df["cases_reported"], errors="coerce").fillna(0)
    df["area_reported"] = df["area_reported"].apply(normalize_area_name)
    df["disease"] = disease_name

    # Drop rows with missing week or area
    df = df.dropna(subset=["week_number", "area_reported"])
    df["week_number"] = df["week_number"].astype(int)

    return df


def calculate_thresholds(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate district-level thresholds for each disease using incidence per 100k.
    lower_threshold    = mean - 1*std
    upper_threshold    = mean + 1*std
    outbreak_threshold = mean + 2*std
    """
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

    # Replace NaN std with 0 for groups with very small data
    threshold_df["std_incidence"] = threshold_df["std_incidence"].fillna(0)
    threshold_df["std_cases"] = threshold_df["std_cases"].fillna(0)

    threshold_df["lower_threshold"] = (
        threshold_df["mean_incidence"] - threshold_df["std_incidence"]
    )
    threshold_df["upper_threshold"] = (
        threshold_df["mean_incidence"] + threshold_df["std_incidence"]
    )
    threshold_df["outbreak_threshold"] = (
        threshold_df["mean_incidence"] + (2 * threshold_df["std_incidence"])
    )

    # Prevent negative lower threshold
    threshold_df["lower_threshold"] = threshold_df["lower_threshold"].clip(lower=0)

    return threshold_df


def apply_thresholds(merged_df: pd.DataFrame, threshold_df: pd.DataFrame) -> pd.DataFrame:
    """Attach thresholds to each weekly row and assign risk levels."""
    result_df = merged_df.merge(
        threshold_df[
            [
                "disease",
                "area_reported",
                "lower_threshold",
                "upper_threshold",
                "outbreak_threshold",
            ]
        ],
        on=["disease", "area_reported"],
        how="left",
    )

    result_df["risk_level"] = result_df.apply(
        lambda row: classify_risk(
            row["incidence_per_100k"],
            row["lower_threshold"],
            row["upper_threshold"],
            row["outbreak_threshold"],
        ),
        axis=1,
    )

    return result_df


def classify_prediction(predicted_cases, population, lower, upper, outbreak):
    """
    Example helper for future ML predictions.
    Converts predicted cases to predicted incidence and returns risk label.
    """
    predicted_incidence = (predicted_cases / population) * 100000

    if predicted_incidence < lower:
        risk = "Below Expected"
    elif predicted_incidence <= upper:
        risk = "Normal"
    elif predicted_incidence <= outbreak:
        risk = "Warning"
    else:
        risk = "High Risk"

    return predicted_incidence, risk


def normalize_district_key(value: str) -> str:
    district = str(value).strip().casefold()
    district = DISTRICT_ALIASES.get(district, district)
    return district.replace(" ", "")


def ensure_risk_levels_conflict_constraint(cursor) -> None:
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

    cursor.execute(
        """
        ALTER TABLE risk_levels
        DROP CONSTRAINT IF EXISTS risk_levels_district_week_year_uniq
        """
    )

    cursor.execute(
        """
        ALTER TABLE risk_levels
        ADD CONSTRAINT risk_levels_district_week_year_uniq
        UNIQUE (district_id, week_number, year, disease_id)
        """
    )


def prepare_risk_levels_rows(weekly_risk_df: pd.DataFrame, year: int) -> pd.DataFrame:
    db_df = weekly_risk_df.copy()
    db_df["district_key"] = db_df["area_reported"].map(normalize_district_key)
    db_df["district_id"] = db_df["district_key"].map(DISTRICT_MAP)
    db_df["disease_key"] = db_df["disease"].astype(str).str.strip().str.casefold()
    db_df["disease_id"] = db_df["disease_key"].map(DISEASE_ID_MAP)
    db_df["year"] = int(year)
    db_df["risk_score"] = pd.to_numeric(db_df["incidence_per_100k"], errors="coerce")

    db_df["risk_priority"] = db_df["risk_level"].map(RISK_LEVEL_PRIORITY).fillna(0)

    skipped_unknown_district = int(db_df["district_id"].isna().sum())
    skipped_unknown_disease = int(db_df["disease_id"].isna().sum())
    if skipped_unknown_district:
        print(f"[WARN] Skipped {skipped_unknown_district} rows due to unknown district mappings")
    if skipped_unknown_disease:
        print(f"[WARN] Skipped {skipped_unknown_disease} rows due to unknown disease mappings")

    db_df = db_df.dropna(
        subset=[
            "district_id",
            "disease_id",
            "week_number",
            "risk_level",
            "lower_threshold",
            "upper_threshold",
            "outbreak_threshold",
            "risk_score",
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
            "district_id",
            "week_number",
            "year",
            "disease_id",
            "risk_level",
            "lower_threshold",
            "upper_threshold",
            "outbreak_threshold",
            "risk_score",
        ]
    ]


def store_risk_levels(weekly_risk_df: pd.DataFrame, year: int) -> None:
    db_df = prepare_risk_levels_rows(weekly_risk_df, year)

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        ensure_risk_levels_conflict_constraint(cursor)

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

    except Exception as e:
        conn.rollback()
        print("[ERROR] Failed to persist risk_levels:", e)
        raise
    finally:
        cursor.close()
        conn.close()


# =========================
# MAIN PIPELINE
# =========================
def main():
    print("Loading population data...")
    population_df = load_population(POPULATION_FILE)

    print("Loading disease datasets...")
    disease_frames = []
    for disease_name, file_path in DISEASE_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")
        disease_frames.append(load_disease_file(file_path, disease_name))

    all_disease_df = pd.concat(disease_frames, ignore_index=True)

    print("Merging disease data with population...")
    merged_df = all_disease_df.merge(
        population_df,
        left_on="area_reported",
        right_on="Region",
        how="left",
    )

    unmatched = merged_df[merged_df["Population"].isna()]["area_reported"].drop_duplicates().tolist()
    if unmatched:
        print("\nWarning: These areas were not matched with population data:")
        for area in unmatched:
            print(f" - {area}")

    # Drop rows without population because incidence cannot be calculated
    merged_df = merged_df.dropna(subset=["Population"]).copy()

    print("Calculating incidence per 100,000...")
    merged_df["incidence_per_100k"] = (
        merged_df["cases_reported"] / merged_df["Population"]
    ) * 100000

    print("Calculating thresholds...")
    threshold_df = calculate_thresholds(merged_df)

    print("Applying thresholds and assigning risk levels...")
    weekly_risk_df = apply_thresholds(merged_df, threshold_df)

    print("Saving risk levels to database...")
    store_risk_levels(weekly_risk_df, TARGET_YEAR)

    # Sort for clean outputs
    threshold_df = threshold_df.sort_values(["disease", "area_reported"]).reset_index(drop=True)
    weekly_risk_df = weekly_risk_df.sort_values(
        ["disease", "week_number", "area_reported"]
    ).reset_index(drop=True)

    # Save outputs
    threshold_file = OUTPUT_DIR / "district_thresholds_2024.csv"
    weekly_file = OUTPUT_DIR / "weekly_risk_levels_2024.csv"

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

    # Show a small sample
    print("\nSample thresholds:")
    print(threshold_df.head(10).to_string(index=False))

    print("\nSample weekly risk levels:")
    print(
        weekly_risk_df[
            [
                "week_number",
                "area_reported",
                "disease",
                "cases_reported",
                "incidence_per_100k",
                "risk_level",
            ]
        ].head(10).to_string(index=False)
    )

    # Example: how to classify a future predicted value
    print("\nExample prediction classification:")
    example_row = threshold_df.iloc[0]
    predicted_cases = 10

    predicted_incidence, predicted_risk = classify_prediction(
        predicted_cases=predicted_cases,
        population=example_row["population"],
        lower=example_row["lower_threshold"],
        upper=example_row["upper_threshold"],
        outbreak=example_row["outbreak_threshold"],
    )

    print(
        f"Disease: {example_row['disease']}, Area: {example_row['area_reported']}, "
        f"Predicted cases: {predicted_cases}, "
        f"Predicted incidence: {predicted_incidence:.4f}, "
        f"Risk: {predicted_risk}"
    )


if __name__ == "__main__":
    main()



