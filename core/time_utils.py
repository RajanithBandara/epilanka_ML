from __future__ import annotations


MONTH_ORDER: list[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

MONTH_TO_NUM: dict[str, int] = {name: index + 1 for index, name in enumerate(MONTH_ORDER)}


def week_to_month(week_number: int | float) -> int:
    wk = int(week_number)
    wk = max(1, min(53, wk))
    return int(((wk - 1) / 53) * 12) + 1
