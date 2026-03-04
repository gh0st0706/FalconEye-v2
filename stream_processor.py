from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = ("time", "engine_temp")
OPTIONAL_NUMERIC_COLUMNS = (
    "rpm",
    "vibration",
    "fuel_flow",
    "efficiency",
    "anomaly_score",
    "anomaly_confidence",
)


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_telemetry_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {col.strip().lower(): col for col in df.columns}
    missing = [col for col in REQUIRED_COLUMNS if col not in normalized]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    out = pd.DataFrame(
        {
            "time": pd.to_numeric(df[normalized["time"]], errors="coerce"),
            "engine_temp": pd.to_numeric(df[normalized["engine_temp"]], errors="coerce"),
        }
    )

    for col in OPTIONAL_NUMERIC_COLUMNS:
        if col in normalized:
            out[col] = pd.to_numeric(df[normalized[col]], errors="coerce")

    out = out.dropna(subset=["time", "engine_temp"]).sort_values("time").reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid rows after coercion")
    return out


@dataclass
class StreamingTelemetryProcessor:
    """Processes telemetry rows with resilience and bounded in-memory history."""

    max_buffer_size: int = 4000

    def __post_init__(self) -> None:
        self._buffer = deque(maxlen=self.max_buffer_size)
        self.total_processed = 0
        self.total_rejected = 0

    def ingest_record(self, record: dict[str, object]) -> dict[str, object] | None:
        time_v = _safe_float(record.get("time"))
        temp_v = _safe_float(record.get("engine_temp"))
        if time_v is None or temp_v is None:
            self.total_rejected += 1
            return None

        cleaned = {
            "time": time_v,
            "engine_temp": temp_v,
        }

        for col in OPTIONAL_NUMERIC_COLUMNS:
            if col in record:
                coerced = _safe_float(record[col])
                if coerced is not None:
                    cleaned[col] = coerced

        self._buffer.append(cleaned)
        self.total_processed += 1
        return cleaned

    def ingest_records(self, records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
        accepted: list[dict[str, object]] = []
        for record in records:
            cleaned = self.ingest_record(record)
            if cleaned is not None:
                accepted.append(cleaned)
        return accepted

    def snapshot(self) -> pd.DataFrame:
        if not self._buffer:
            return pd.DataFrame(columns=["time", "engine_temp"])
        return pd.DataFrame(self._buffer).sort_values("time").reset_index(drop=True)
