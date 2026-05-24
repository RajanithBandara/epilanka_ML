from __future__ import annotations


DISEASE_IDS: dict[str, int] = {
    "dysentery": 1,
    "meningitis": 2,
    "tuberculosis": 3,
}

DISEASE_DISPLAY: dict[str, str] = {
    "dysentery": "Dysentery",
    "meningitis": "Meningitis",
    "tuberculosis": "Tuberculosis",
}

ALL_DISEASES: tuple[str, ...] = ("Dysentery", "Meningitis", "Tuberculosis")


def disease_key(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().casefold()


def disease_id(value: object) -> int | None:
    return DISEASE_IDS.get(disease_key(value))


def display_name(value: object) -> str:
    return DISEASE_DISPLAY.get(disease_key(value), str(value))
