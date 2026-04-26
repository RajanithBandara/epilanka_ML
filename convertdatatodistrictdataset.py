from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


DATASETS_DIR = Path("datasets")
TARGET_COLUMNS = ["week_number", "area_reported", "cases_reported"]
PROVINCE_COLUMNS = ["W", "C", "S", "N", "E", "NW", "NC", "U", "Sab"]

PROVINCE_TO_DISTRICTS = {
    "W": ["Colombo", "Gampaha", "Kalutara"],
    "C": ["Kandy", "Matale", "NuwaraEliya"],
    "S": ["Galle", "Hambantota", "Matara"],
    "N": ["Jaffna", "Kilinochchi", "Mannar", "Vavuniya", "Mullaitivu"],
    "E": ["Batticaloa", "Ampara", "Trincomalee", "Kalmune"],
    "NW": ["Kurunegala", "Puttalam"],
    "NC": ["Anuradhapura", "Polonnaruwa"],
    "U": ["Badulla", "Monaragala"],
    "Sab": ["Ratnapura", "Kegalle"],
}

DISTRICT_ORDER = [
    district
    for province in PROVINCE_COLUMNS
    for district in PROVINCE_TO_DISTRICTS[province]
]

DISTRICT_TO_PROVINCE = {
    district: province
    for province, districts in PROVINCE_TO_DISTRICTS.items()
    for district in districts
}

CANONICAL_DISTRICT_NAMES = {
    "colombo": "Colombo",
    "gampaha": "Gampaha",
    "kalutara": "Kalutara",
    "kandy": "Kandy",
    "matale": "Matale",
    "nuwaraeliya": "NuwaraEliya",
    "galle": "Galle",
    "hambantota": "Hambantota",
    "matara": "Matara",
    "jaffna": "Jaffna",
    "kilinochchi": "Kilinochchi",
    "mannar": "Mannar",
    "vavuniya": "Vavuniya",
    "mullaitivu": "Mullaitivu",
    "batticaloa": "Batticaloa",
    "ampara": "Ampara",
    "trincomalee": "Trincomalee",
    "kurunegala": "Kurunegala",
    "puttalam": "Puttalam",
    "anuradhapura": "Anuradhapura",
    "polonnaruwa": "Polonnaruwa",
    "badulla": "Badulla",
    "monaragala": "Monaragala",
    "moneragala": "Monaragala",
    "ratnapura": "Ratnapura",
    "kegalle": "Kegalle",
    "kalmune": "Kalmune",
    "kalmunai": "Kalmune",
}

SOURCE_FILES = {
    "2023_province": DATASETS_DIR / "tuberculosis_2023.csv",
    "2024_province": DATASETS_DIR / "tuberculosis_2024.csv",
    "2024_district": DATASETS_DIR / "2024_Tuberculosis.csv",
    "2025_district": DATASETS_DIR / "2025_Tuberculosis.csv",
}

OUTPUT_FILES = {
    "2023_harmonized": DATASETS_DIR / "2023_Tuberculosis_harmonized.csv",
    "2024_harmonized": DATASETS_DIR / "2024_Tuberculosis_harmonized.csv",
    "2025_harmonized": DATASETS_DIR / "2025_Tuberculosis_harmonized.csv",
}


def normalize_district_name(value: object) -> str:
    key = str(value).strip().casefold().replace(" ", "")
    canonical = CANONICAL_DISTRICT_NAMES.get(key)
    if canonical is None:
        raise ValueError(f"Unknown district name: {value!r}")
    return canonical


def order_district_rows(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.copy()
    ordered["area_reported"] = pd.Categorical(
        ordered["area_reported"], categories=DISTRICT_ORDER, ordered=True
    )
    ordered = ordered.sort_values(["week_number", "area_reported"]).reset_index(drop=True)
    ordered["area_reported"] = ordered["area_reported"].astype(str)
    return ordered


def load_province_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["week_number"] = pd.to_numeric(
        df["Week"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    df = df.dropna(subset=["week_number"]).copy()
    df["week_number"] = df["week_number"].astype(int)

    for column in PROVINCE_COLUMNS + ["Total"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    return df[["week_number", *PROVINCE_COLUMNS, "Total"]].sort_values("week_number")


def clean_district_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.loc[:, TARGET_COLUMNS].copy()
    cleaned["week_number"] = pd.to_numeric(cleaned["week_number"], errors="coerce")
    cleaned = cleaned.dropna(subset=["week_number", "area_reported"])
    cleaned["week_number"] = cleaned["week_number"].astype(int)
    cleaned["area_reported"] = cleaned["area_reported"].map(normalize_district_name)
    cleaned["cases_reported"] = (
        pd.to_numeric(cleaned["cases_reported"], errors="coerce").round().astype("Int64")
    )
    return order_district_rows(cleaned)


def load_2024_district_dataset() -> pd.DataFrame:
    raw = pd.read_csv(
        SOURCE_FILES["2024_district"],
        header=None,
        names=TARGET_COLUMNS,
    )
    return clean_district_dataset(raw)


def load_2025_district_dataset() -> pd.DataFrame:
    raw = pd.read_csv(SOURCE_FILES["2025_district"])
    raw = raw.iloc[:, :3].copy()
    raw.columns = TARGET_COLUMNS
    return clean_district_dataset(raw)


def build_share_reference(observed_district_rows: pd.DataFrame) -> tuple[dict, dict]:
    observed = observed_district_rows.dropna(subset=["cases_reported"]).copy()
    observed["cases_reported"] = observed["cases_reported"].astype(int)
    observed["province"] = observed["area_reported"].map(DISTRICT_TO_PROVINCE)

    week_level = (
        observed.groupby(["province", "week_number", "area_reported"], as_index=False)[
            "cases_reported"
        ]
        .sum()
    )
    week_totals = (
        week_level.groupby(["province", "week_number"], as_index=False)["cases_reported"]
        .sum()
        .rename(columns={"cases_reported": "province_total"})
    )
    week_level = week_level.merge(week_totals, on=["province", "week_number"], how="left")
    week_level = week_level[week_level["province_total"] > 0].copy()
    week_level["share"] = week_level["cases_reported"] / week_level["province_total"]

    week_lookup: dict[tuple[str, int], dict[str, float]] = {}
    for (province, week_number), group in week_level.groupby(["province", "week_number"]):
        week_lookup[(province, int(week_number))] = {
            row["area_reported"]: float(row["share"]) for _, row in group.iterrows()
        }

    province_level = (
        observed.groupby(["province", "area_reported"], as_index=False)["cases_reported"].sum()
    )
    province_totals = (
        province_level.groupby("province", as_index=False)["cases_reported"]
        .sum()
        .rename(columns={"cases_reported": "province_total"})
    )
    province_level = province_level.merge(province_totals, on="province", how="left")
    province_level["share"] = province_level["cases_reported"] / province_level["province_total"]

    overall_lookup: dict[str, dict[str, float]] = {}
    for province, group in province_level.groupby("province"):
        overall_lookup[province] = {
            row["area_reported"]: float(row["share"]) for _, row in group.iterrows()
        }

    return week_lookup, overall_lookup


def get_shares_for_province(
    province: str,
    week_number: int,
    week_lookup: dict[tuple[str, int], dict[str, float]],
    overall_lookup: dict[str, dict[str, float]],
) -> list[float]:
    districts = PROVINCE_TO_DISTRICTS[province]
    lookup = week_lookup.get((province, week_number), overall_lookup.get(province, {}))
    shares = [float(lookup.get(district, 0.0)) for district in districts]
    share_total = sum(shares)

    if share_total <= 0:
        return [1.0 / len(districts)] * len(districts)

    return [share / share_total for share in shares]


def allocate_cases(total_cases: int, province: str, shares: list[float]) -> dict[str, int]:
    districts = PROVINCE_TO_DISTRICTS[province]
    total_cases = int(total_cases)

    if total_cases <= 0:
        return {district: 0 for district in districts}

    raw_values = [total_cases * share for share in shares]
    allocated = [math.floor(value) for value in raw_values]
    remainder = total_cases - sum(allocated)

    fractions = sorted(
        range(len(districts)),
        key=lambda index: (raw_values[index] - allocated[index], -index),
        reverse=True,
    )

    for index in fractions[:remainder]:
        allocated[index] += 1

    return dict(zip(districts, allocated))


def convert_province_to_district(
    province_df: pd.DataFrame,
    week_lookup: dict[tuple[str, int], dict[str, float]],
    overall_lookup: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []

    for source_row in province_df.itertuples(index=False):
        week_number = int(source_row.week_number)

        for province in PROVINCE_COLUMNS:
            province_total = int(getattr(source_row, province))
            shares = get_shares_for_province(province, week_number, week_lookup, overall_lookup)
            allocation = allocate_cases(province_total, province, shares)

            for district in PROVINCE_TO_DISTRICTS[province]:
                rows.append(
                    {
                        "week_number": week_number,
                        "area_reported": district,
                        "cases_reported": allocation[district],
                    }
                )

    converted = pd.DataFrame(rows, columns=TARGET_COLUMNS)
    converted["cases_reported"] = converted["cases_reported"].astype("Int64")
    return order_district_rows(converted)


def aggregate_to_province(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.assign(province=df["area_reported"].map(DISTRICT_TO_PROVINCE))
        .groupby(["week_number", "province"], as_index=False)["cases_reported"]
        .sum()
    )

    pivoted = grouped.pivot(index="week_number", columns="province", values="cases_reported")
    pivoted = pivoted.reindex(columns=PROVINCE_COLUMNS, fill_value=0).fillna(0).astype(int)
    pivoted["Total"] = pivoted.sum(axis=1)
    return pivoted.reset_index()


def validate_preserved_totals(label: str, converted_df: pd.DataFrame, source_df: pd.DataFrame) -> None:
    converted_province = aggregate_to_province(converted_df)
    source_sorted = source_df[["week_number", *PROVINCE_COLUMNS, "Total"]].sort_values(
        "week_number"
    )
    merged = source_sorted.merge(
        converted_province,
        on="week_number",
        suffixes=("_source", "_converted"),
        how="inner",
    )

    mismatches = []
    for column in PROVINCE_COLUMNS + ["Total"]:
        mismatch_rows = merged[
            merged[f"{column}_source"] != merged[f"{column}_converted"]
        ]["week_number"].tolist()
        if mismatch_rows:
            mismatches.append((column, mismatch_rows))

    if mismatches:
        details = ", ".join(f"{column}: {weeks}" for column, weeks in mismatches)
        raise ValueError(f"{label} totals were not preserved: {details}")

    print(
        f"[OK] {label}: preserved province totals for "
        f"{merged['week_number'].nunique()} week(s)"
    )


def write_output(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    week_min = int(df["week_number"].min())
    week_max = int(df["week_number"].max())
    missing_cases = int(df["cases_reported"].isna().sum())
    print(
        f"[OK] Wrote {path} with {len(df)} rows, weeks {week_min}-{week_max}, "
        f"missing cases: {missing_cases}"
    )


def main() -> None:
    province_2023 = load_province_dataset(SOURCE_FILES["2023_province"])
    province_2024 = load_province_dataset(SOURCE_FILES["2024_province"])

    district_2024 = load_2024_district_dataset()
    district_2025 = load_2025_district_dataset()

    observed_for_shares = pd.concat(
        [
            district_2024.dropna(subset=["cases_reported"]),
            district_2025.dropna(subset=["cases_reported"]),
        ],
        ignore_index=True,
    )

    week_lookup, overall_lookup = build_share_reference(observed_for_shares)

    harmonized_2023 = convert_province_to_district(province_2023, week_lookup, overall_lookup)

    converted_2024_early = convert_province_to_district(
        province_2024[province_2024["week_number"] < 10],
        week_lookup,
        overall_lookup,
    )
    harmonized_2024 = order_district_rows(
        pd.concat(
            [
                converted_2024_early,
                district_2024.dropna(subset=["cases_reported"]),
            ],
            ignore_index=True,
        )
    )

    harmonized_2025 = order_district_rows(district_2025)

    validate_preserved_totals("2023 harmonized dataset", harmonized_2023, province_2023)
    validate_preserved_totals(
        "2024 converted weeks 1-9",
        harmonized_2024[harmonized_2024["week_number"] < 10],
        province_2024[province_2024["week_number"] < 10],
    )

    write_output(harmonized_2023, OUTPUT_FILES["2023_harmonized"])
    write_output(harmonized_2024, OUTPUT_FILES["2024_harmonized"])
    write_output(harmonized_2025, OUTPUT_FILES["2025_harmonized"])


if __name__ == "__main__":
    main()
