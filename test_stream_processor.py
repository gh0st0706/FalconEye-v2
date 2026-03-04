import unittest

import pandas as pd

from stream_processor import StreamingTelemetryProcessor, normalize_telemetry_frame


class TelemetryNormalizationTests(unittest.TestCase):
    def test_normalize_coerces_columns(self) -> None:
        df = pd.DataFrame(
            {
                "Time": [0, 1, 2],
                "Engine_Temp": ["100", "105.5", "bad"],
                "RPM": [1200, "1300", 1400],
            }
        )

        out = normalize_telemetry_frame(df)
        self.assertEqual(len(out), 2)
        self.assertIn("rpm", out.columns)

    def test_normalize_requires_engine_temp(self) -> None:
        df = pd.DataFrame({"time": [0, 1, 2], "rpm": [1000, 1100, 1200]})
        with self.assertRaises(ValueError):
            normalize_telemetry_frame(df)


class StreamingTelemetryProcessorTests(unittest.TestCase):
    def test_ingest_record_rejects_invalid(self) -> None:
        processor = StreamingTelemetryProcessor(max_buffer_size=5)
        bad = processor.ingest_record({"time": "x", "engine_temp": 150})
        self.assertIsNone(bad)
        self.assertEqual(processor.total_rejected, 1)

    def test_ingest_records_and_snapshot(self) -> None:
        processor = StreamingTelemetryProcessor(max_buffer_size=3)
        accepted = processor.ingest_records(
            [
                {"time": 2, "engine_temp": 170},
                {"time": 1, "engine_temp": 160},
                {"time": 3, "engine_temp": 180, "anomaly_score": 2.1},
                {"time": 4, "engine_temp": 190},
            ]
        )
        self.assertEqual(len(accepted), 4)
        snap = processor.snapshot()
        self.assertEqual(len(snap), 3)
        self.assertEqual(list(snap["time"]), [1.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
