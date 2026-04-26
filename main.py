from pathlib import Path

from calculatethreshold import main as calculate_thresholds_and_store_risk_levels
from convertdatatodistrictdataset import main as harmonize_tuberculosis_datasets
from predict_year import default_output_path, generate_predictions
from store_predictions import store_predictions
from storehistorydata import store_historical_data
from storetuberculosisdata import store_tuberculosis_data
from train_predict_models import train_prediction_model


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


MENU_OPTIONS = {
    "1": ("Convert tuberculosis datasets to district format", harmonize_tuberculosis_datasets),
    "2": ("Store dysentery and meningitis historical data", store_historical_data),
    "3": ("Store tuberculosis historical data", store_tuberculosis_data),
    "4": ("Calculate thresholds and store risk levels", calculate_thresholds_and_store_risk_levels),
    "5": ("Train prediction model", run_train_model),
    "6": ("Generate prediction CSV", run_generate_predictions),
    "7": ("Store prediction CSV in database", run_store_predictions),
    "8": ("Train model, generate predictions, and store them", run_train_generate_and_store),
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
