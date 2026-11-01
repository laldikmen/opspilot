"""Append every prediction (timestamp, input hash, model version, probability) to a log table.

This is the groundwork for v2 monitoring: predicted risk can later be compared
with the outcomes that arrive.
"""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def input_hash(values: dict) -> str:
    """A stable hash of the feature values, so repeated inputs can be recognised."""
    canonical = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class PredictionLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute("""
                create table if not exists predictions (
                    logged_at text not null,
                    endpoint text not null,
                    order_id text,
                    input_hash text not null,
                    model_version text not null,
                    probability real not null
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def append(
        self, endpoint: str, order_id: str | None, values: dict, version: str, probability: float
    ) -> None:
        row = (
            datetime.now(UTC).isoformat(),
            endpoint,
            order_id,
            input_hash(values),
            version,
            probability,
        )
        with self._connect() as con:
            con.execute("insert into predictions values (?, ?, ?, ?, ?, ?)", row)
