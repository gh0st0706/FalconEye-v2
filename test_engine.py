import unittest

import numpy as np

from engine import EngineStateModel


class EngineStateModelTests(unittest.TestCase):
    def test_initial_state(self) -> None:
        model = EngineStateModel(rng_seed=7)
        snap = model.snapshot()

        self.assertEqual(snap["time"], 0.0)
        self.assertAlmostEqual(snap["throttle"], 0.0)
        self.assertGreaterEqual(snap["rpm"], 900.0)
        self.assertLessEqual(snap["rpm"], 2000.0)
        self.assertGreater(snap["engine_temp"], 40.0)
        self.assertGreater(snap["fuel_flow"], 100.0)
        self.assertGreater(snap["vibration"], 0.0)

    def test_state_evolves_over_time(self) -> None:
        model = EngineStateModel(rng_seed=42)
        first = model.step(0.3, dt=0.5)
        second = model.step(0.3, dt=0.5)

        self.assertAlmostEqual(second["time"], first["time"] + 0.5)
        self.assertNotEqual(first["rpm"], second["rpm"])
        self.assertNotEqual(first["engine_temp"], second["engine_temp"])

    def test_throttle_increase_raises_rpm_and_temp(self) -> None:
        model = EngineStateModel(rng_seed=13)

        for _ in range(25):
            low = model.step(0.15)

        low_rpm = low["rpm"]
        low_temp = low["engine_temp"]

        for _ in range(25):
            high = model.step(0.85)

        self.assertGreater(high["rpm"], low_rpm)
        self.assertGreater(high["engine_temp"], low_temp)
        self.assertGreater(high["fuel_flow"], low["fuel_flow"])

    def test_profile_output_shape_and_columns(self) -> None:
        model = EngineStateModel(rng_seed=5)
        throttles = np.linspace(0.0, 1.0, 40)
        df = model.run_profile(throttles, dt=0.2)

        expected = {
            "time",
            "throttle",
            "rpm",
            "engine_temp",
            "fuel_flow",
            "vibration",
            "anomaly_score",
        }

        self.assertEqual(len(df), 40)
        self.assertTrue(expected.issubset(set(df.columns)))

    def test_telemetry_stays_in_reasonable_bounds(self) -> None:
        model = EngineStateModel(rng_seed=123)

        for throttle in np.random.default_rng(123).uniform(0.0, 1.0, size=300):
            telemetry = model.step(float(throttle), dt=0.25)
            self.assertGreaterEqual(telemetry["rpm"], 900.0)
            self.assertLessEqual(telemetry["rpm"], 16000.0)
            self.assertGreaterEqual(telemetry["engine_temp"], 20.0)
            self.assertLessEqual(telemetry["engine_temp"], 980.0)
            self.assertGreaterEqual(telemetry["fuel_flow"], 120.0)
            self.assertLessEqual(telemetry["fuel_flow"], 2600.0)
            self.assertGreaterEqual(telemetry["vibration"], 0.03)
            self.assertLessEqual(telemetry["vibration"], 2.5)
            self.assertGreaterEqual(telemetry["anomaly_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
