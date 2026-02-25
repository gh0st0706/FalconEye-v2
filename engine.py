from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class EngineStateModel:
    """Stateful engine simulation with throttle-driven dynamics."""

    ambient_temp_c: float = 22.0
    idle_rpm: float = 1100.0
    max_rpm: float = 15500.0
    rng_seed: int | None = None

    # Dynamic state (initialized in __post_init__)
    time_s: float = field(init=False, default=0.0)
    throttle: float = field(init=False, default=0.0)
    rpm: float = field(init=False)
    engine_temp: float = field(init=False)
    fuel_flow: float = field(init=False)
    vibration: float = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.rng_seed)
        self.reset()

    def reset(self) -> None:
        self.time_s = 0.0
        self.throttle = 0.0
        self.rpm = self.idle_rpm
        self.engine_temp = self.ambient_temp_c + 65.0
        self.fuel_flow = 260.0
        self.vibration = 0.10

    def snapshot(self) -> dict[str, float]:
        anomaly_score = self._anomaly_score()
        return {
            "time": self.time_s,
            "throttle": self.throttle,
            "rpm": self.rpm,
            "engine_temp": self.engine_temp,
            "fuel_flow": self.fuel_flow,
            "vibration": self.vibration,
            "anomaly_score": anomaly_score,
        }

    def step(self, throttle: float, dt: float = 1.0) -> dict[str, float]:
        if dt <= 0:
            raise ValueError("dt must be > 0")

        self.time_s += dt
        self.throttle = _clamp(float(throttle), 0.0, 1.0)

        # RPM reacts quickly to throttle with first-order lag.
        rpm_target = self.idle_rpm + self.throttle * (self.max_rpm - self.idle_rpm)
        rpm_alpha = 1.0 - np.exp(-dt / 1.8)
        self.rpm += (rpm_target - self.rpm) * rpm_alpha
        self.rpm += self.rng.normal(0.0, 20.0 * np.sqrt(dt))
        self.rpm = _clamp(self.rpm, self.idle_rpm * 0.85, self.max_rpm * 1.02)

        load = self.rpm / self.max_rpm

        # Temperature rises with load and throttle, with slower thermal lag.
        temp_target = self.ambient_temp_c + 90.0 + 320.0 * load + 180.0 * self.throttle
        temp_alpha = 1.0 - np.exp(-dt / 9.0)
        self.engine_temp += (temp_target - self.engine_temp) * temp_alpha
        self.engine_temp += self.rng.normal(0.0, 0.8 * np.sqrt(dt))
        self.engine_temp = _clamp(self.engine_temp, self.ambient_temp_c, 980.0)

        # Fuel flow tracks throttle and RPM.
        fuel_target = 220.0 + 0.045 * self.rpm + 900.0 * self.throttle
        fuel_alpha = 1.0 - np.exp(-dt / 2.2)
        self.fuel_flow += (fuel_target - self.fuel_flow) * fuel_alpha
        self.fuel_flow += self.rng.normal(0.0, 6.0 * np.sqrt(dt))
        self.fuel_flow = _clamp(self.fuel_flow, 120.0, 2600.0)

        # Vibration increases with load and transients.
        rpm_transient = abs(rpm_target - self.rpm) / self.max_rpm
        vib_target = 0.08 + 0.22 * load + 0.45 * rpm_transient
        vib_alpha = 1.0 - np.exp(-dt / 3.5)
        self.vibration += (vib_target - self.vibration) * vib_alpha
        self.vibration += self.rng.normal(0.0, 0.01 * np.sqrt(dt))
        self.vibration = _clamp(self.vibration, 0.03, 2.5)

        return self.snapshot()

    def run_profile(self, throttles: Iterable[float], dt: float = 1.0) -> pd.DataFrame:
        rows = [self.step(throttle=t, dt=dt) for t in throttles]
        return pd.DataFrame(rows)

    def _anomaly_score(self) -> float:
        load = self.rpm / self.max_rpm
        expected_temp = self.ambient_temp_c + 90.0 + 320.0 * load + 180.0 * self.throttle
        expected_vib = 0.08 + 0.22 * load
        expected_fuel = 220.0 + 0.045 * self.rpm + 900.0 * self.throttle

        temp_dev = abs(self.engine_temp - expected_temp) / 35.0
        vib_dev = max(0.0, self.vibration - expected_vib) / 0.10
        fuel_dev = abs(self.fuel_flow - expected_fuel) / 140.0

        score = 1.0 + temp_dev + vib_dev + 0.5 * fuel_dev
        score += float(self.rng.normal(0.0, 0.04))
        return max(0.0, score)
