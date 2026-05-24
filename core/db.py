from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Iterator

import psycopg2
import psycopg2.extras
import psycopg2.sql

from core.config import DB_CONFIG

warnings.filterwarnings("ignore")
psycopg2.extras.register_uuid()


@contextmanager
def connect() -> Iterator[tuple]:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    try:
        yield conn, cursor
    finally:
        cursor.close()
        conn.close()


def ensure_unique_constraint(
    cursor,
    table: str,
    columns: list[str],
    *,
    constraint_name: str | None = None,
) -> None:
    cols_csv = ", ".join(columns)
    cursor.execute(
        """
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        WHERE t.relname = %s
          AND c.contype = 'u'
          AND pg_get_constraintdef(c.oid) = %s
        LIMIT 1
        """,
        (table, f"UNIQUE ({cols_csv})"),
    )
    if cursor.fetchone() is not None:
        return

    name = constraint_name or f"{table}_{'_'.join(columns)}_uniq"
    cursor.execute(
        psycopg2.sql.SQL(
            "ALTER TABLE {table} ADD CONSTRAINT {name} UNIQUE ({cols})"
        ).format(
            table=psycopg2.sql.Identifier(table),
            name=psycopg2.sql.Identifier(name),
            cols=psycopg2.sql.SQL(", ").join(
                psycopg2.sql.Identifier(column) for column in columns
            ),
        )
    )


def has_single_column_unique(cursor, table: str, column: str) -> bool:
    cursor.execute(
        """
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY (c.conkey)
        WHERE t.relname = %s
          AND c.contype = 'u'
          AND a.attname = %s
        LIMIT 1
        """,
        (table, column),
    )
    return cursor.fetchone() is not None


def get_column_type(cursor, table: str, column: str) -> str | None:
    cursor.execute(
        """
        SELECT data_type
        FROM information_schema.columns
        WHERE table_name = %s AND column_name = %s
        """,
        (table, column),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def pick_existing_column(cursor, table: str, candidates: list[str]) -> str | None:
    cursor.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s AND column_name = ANY(%s)
        """,
        (table, candidates),
    )
    rows = {row[0] for row in cursor.fetchall()}
    for candidate in candidates:
        if candidate in rows:
            return candidate
    return None
