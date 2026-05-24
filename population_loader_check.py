from pipeline.store_population import load_population_dataframe


def main() -> None:
    df = load_population_dataframe()
    print(f"Loaded {len(df)} population rows")
    print(df[["Region", "population"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
