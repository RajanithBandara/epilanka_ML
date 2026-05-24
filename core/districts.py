from __future__ import annotations


DISTRICT_IDS: dict[str, int] = {
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

DISTRICT_DISPLAY: dict[str, str] = {
    "colombo": "Colombo",
    "gampaha": "Gampaha",
    "kalutara": "Kalutara",
    "kandy": "Kandy",
    "matale": "Matale",
    "nuwaraeliya": "Nuwara Eliya",
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
    "ratnapura": "Ratnapura",
    "kegalle": "Kegalle",
    "kalmunai": "Kalmunai",
}

DISTRICT_ALIASES: dict[str, str] = {
    "nuwara eliya": "nuwaraeliya",
    "kalmune": "kalmunai",
    "moneragala": "monaragala",
}


def canonical_key(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().casefold().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    text = DISTRICT_ALIASES.get(text, text)
    return text.replace(" ", "")


def district_id(value: object) -> int | None:
    return DISTRICT_IDS.get(canonical_key(value))


def display_name(value: object) -> str:
    key = canonical_key(value)
    return DISTRICT_DISPLAY.get(key, str(value))


def rainfall_area_name(value: object) -> str:
    key = canonical_key(value)
    if key == "kalmunai":
        return "Ampara"
    return DISTRICT_DISPLAY.get(key, str(value))


PROVINCE_TO_DISTRICTS: dict[str, list[str]] = {
    "W": ["Colombo", "Gampaha", "Kalutara"],
    "C": ["Kandy", "Matale", "Nuwara Eliya"],
    "S": ["Galle", "Hambantota", "Matara"],
    "N": ["Jaffna", "Kilinochchi", "Mannar", "Vavuniya", "Mullaitivu"],
    "E": ["Batticaloa", "Ampara", "Trincomalee", "Kalmunai"],
    "NW": ["Kurunegala", "Puttalam"],
    "NC": ["Anuradhapura", "Polonnaruwa"],
    "U": ["Badulla", "Monaragala"],
    "Sab": ["Ratnapura", "Kegalle"],
}

PROVINCE_COLUMNS: list[str] = list(PROVINCE_TO_DISTRICTS.keys())

DISTRICT_TO_PROVINCE: dict[str, str] = {
    district: province
    for province, districts in PROVINCE_TO_DISTRICTS.items()
    for district in districts
}

DISTRICT_ORDER: list[str] = [
    district
    for districts in PROVINCE_TO_DISTRICTS.values()
    for district in districts
]
