"""FMX Postgres access. `count` runs aggregate statements only; `rows` runs inside the
hospital network. DSN from FMX_DSN (never in git)."""
from __future__ import annotations

import os
import re
from typing import Iterator

_FORBIDDEN = re.compile(r"\b(text|display|presentedForm|url|markdown|data)\b", re.I)


def dsn() -> str:
    v = os.environ.get("FMX_DSN")
    if not v:
        raise RuntimeError("FMX_DSN not set")
    return v


def is_aggregate(sql: str) -> bool:
    outer = sql.strip().rsplit("SELECT", 1)[-1].split("FROM")[0]
    return "COUNT(" in outer.upper() and not _FORBIDDEN.search(outer) and "GROUP BY" in sql.upper()


def registered(statement: str) -> bool:
    """True when `statement` is byte-identical to one of the count_* queries in kevmed.fmx.sql for some ISO date pair."""
    from . import sql as _sql
    import re as _re
    dates = set(_re.findall(r"'(\d{4}-\d{2}-\d{2})'", statement))
    for fn in (_sql.count_i1, _sql.count_i2, _sql.count_i3, _sql.count_i4):
        for a in dates:
            for b in dates:
                try:
                    if fn(a, b) == statement:
                        return True
                except ValueError:
                    pass
    return False


def count(sql: str) -> list[tuple]:
    if not (registered(sql) and is_aggregate(sql)):
        raise ValueError("count() accepts only the registered count_* statements from kevmed.fmx.sql")
    import psycopg
    with psycopg.connect(dsn(), options="-c default_transaction_read_only=on") as conn:
        return conn.execute(sql).fetchall()


def rows(sql: str, batch: int = 500) -> Iterator[dict]:
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(dsn(), options="-c default_transaction_read_only=on", row_factory=dict_row) as conn:
        with conn.cursor(name="kevmed_extract") as cur:
            cur.itersize = batch
            cur.execute(sql)
            yield from cur
