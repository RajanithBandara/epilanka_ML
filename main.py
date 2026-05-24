from pathlib import Path

from analytics.run import run_analytics
from pipeline.compute_risk import calculate_thresholds_and_store_risk_levels
from pipeline.generate_predictions import default_output_path, generate_predictions
from pipeline.store_historical import store_historical_data
from pipeline.store_population import store_population_data
from pipeline.store_predictions import store_predictions
from pipeline.store_rainfall import store_rainfall_data
from pipeline.train_model import train_prediction_model


def prompt_text(message: str, default: str | None = None) -> str:
    if default:
        raw_value = input(f"{message} [{default}]: ").strip()
        return raw_value or default
    return input(f"{message}: ").strip()


def prompt_int(message: str, default: int) -> int:
    while True:
        raw_value = input(f"{message} [{default}]: ").strip()
        if not raw_value:
            return default
        try:
            return int(raw_value)
        except ValueError:
            print("Please enter a valid integer.")


def run_train_model() -> None:
    train_prediction_model()


def run_generate_predictions() -> Path:
    prediction_year = prompt_int("Prediction year", 2026)
    suggested_output = default_output_path(prediction_year)
    output_text = prompt_text("Output CSV path", str(suggested_output))
    return generate_predictions(
        prediction_year=prediction_year,
        output_file=Path(output_text),
    )


def run_store_predictions() -> None:
    csv_path = prompt_text("Predictions CSV path", str(default_output_path(2026)))
    store_predictions(csv_path)


def run_train_generate_and_store() -> None:
    prediction_year = prompt_int("Prediction year", 2026)
    output_text = prompt_text(
        "Output CSV path",
        str(default_output_path(prediction_year)),
    )

    train_prediction_model()
    output_file = generate_predictions(
        prediction_year=prediction_year,
        output_file=Path(output_text),
    )
    store_predictions(str(output_file))


def run_full_pipeline() -> None:
    prediction_year = prompt_int("Prediction year", 2026)

    print("\n[1/7] Storing district population data...")
    store_population_data()

    print("\n[2/7] Storing annual rainfall data...")
    store_rainfall_data()

    print("\n[3/7] Storing historical case data (all diseases)...")
    store_historical_data()

    print("\n[4/7] Calculating thresholds and risk levels...")
    calculate_thresholds_and_store_risk_levels()

    print("\n[5/7] Training prediction model...")
    train_prediction_model()

    print(f"\n[6/7] Generating {prediction_year} predictions...")
    output_file = generate_predictions(prediction_year=prediction_year)

    print(f"\n[7/7] Storing {prediction_year} predictions in database...")
    store_predictions(str(output_file))


MENU_OPTIONS = {
    "1": ("Store district population data", store_population_data),
    "2": ("Store annual rainfall data (CSV → rainfall_data)", store_rainfall_data),
    "3": ("Store historical data for all diseases (Dysentery, Meningitis, Tuberculosis)", store_historical_data),
    "4": ("Calculate thresholds and store risk levels", calculate_thresholds_and_store_risk_levels),
    "5": ("Train prediction model", run_train_model),
    "6": ("Generate prediction CSV", run_generate_predictions),
    "7": ("Store prediction CSV in database", run_store_predictions),
    "8": ("Train model, generate predictions, and store them", run_train_generate_and_store),
    "9": ("Run full pipeline end-to-end", run_full_pipeline),
    "10": ("Run analytics (disease patterns, rain patterns, correlations)", run_analytics),
}


def print_menu() -> None:
    print("\nEpiLanka ML Console")
    print("-------------------")
    for key, (label, _) in MENU_OPTIONS.items():
        print(f"{key}. {label}")
    print("0. Exit")


def main() -> None:
    while True:
        print_menu()
        choice = input("Select an option: ").strip()

        if choice == "0":
            print("Exiting.")
            return

        selected = MENU_OPTIONS.get(choice)
        if selected is None:
            print("Invalid option. Try again.")
            continue

        label, action = selected
        print(f"\nRunning: {label}\n")

        try:
            action()
            print("\nCompleted successfully.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
        except Exception as exc:
            print(f"\nOperation failed: {exc}")


if __name__ == "__main__":
    main()
