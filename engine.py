from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class EngineStateModel:
    """Stateful engine simulation and online anomaly scoring."""

    ambient_temp_c: float = 22.0
    idle_rpm: float = 1100.0
    max_rpm: float = 15500.0
    memory_window: int = 40
    rng_seed: int | None = None

    # Dynamic state (initialized in __post_init__)
    time_s: float = field(init=False, default=0.0)
    throttle: float = field(init=False, default=0.0)
    rpm: float = field(init=False)
    engine_temp: float = field(init=False)
    fuel_flow: float = field(init=False)
    vibration: float = field(init=False)
    health_score: float = field(init=False, default=1.0)
    sensor_health: float = field(init=False, default=1.0)
    _prev_rpm: float = field(init=False)
    _prev_temp: float = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.rng_seed)
        self._history_temp = deque(maxlen=max(10, self.memory_window))
        self._history_vibration = deque(maxlen=max(10, self.memory_window))
        self._history_rpm = deque(maxlen=max(10, self.memory_window))
        self.reset()

    def reset(self) -> None:
        self.time_s = 0.0
        self.throttle = 0.0
        self.rpm = self.idle_rpm
        self.engine_temp = self.ambient_temp_c + 65.0
        self.fuel_flow = 260.0
        self.vibration = 0.10
        self.health_score = 1.0
        self.sensor_health = 1.0
        self._prev_rpm = self.rpm
        self._prev_temp = self.engine_temp
        self._history_temp.clear()
        self._history_vibration.clear()
        self._history_rpm.clear()
        self._push_history()

    def snapshot(self) -> dict[str, float | str]:
        analysis = self._analyze_state(dt=1.0)
        return {
            "time": self.time_s,
            "throttle": self.throttle,
            "rpm": self.rpm,
            "engine_temp": self.engine_temp,
            "fuel_flow": self.fuel_flow,
            "vibration": self.vibration,
            "health_score": self.health_score,
            "sensor_health": self.sensor_health,
            **analysis,
        }

    def step(self, throttle: float, dt: float = 1.0) -> dict[str, float | str]:
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

        # Slow degradation and occasional sensor drift for realism.
        stress = _clamp((load - 0.55) / 0.45, 0.0, 1.0)
        self.health_score = _clamp(self.health_score - 0.0006 * stress * dt + self.rng.normal(0.0, 0.0002), 0.50, 1.00)
        self.sensor_health = _clamp(self.sensor_health - 0.0002 * dt + self.rng.normal(0.0, 0.00008), 0.70, 1.00)

        if self.sensor_health < 0.78:
            self.vibration += max(0.0, (0.80 - self.sensor_health) * 0.03)

        self._push_history()
        analysis = self._analyze_state(dt=dt)
        self._prev_rpm = self.rpm
        self._prev_temp = self.engine_temp

        return {
            "time": self.time_s,
            "throttle": self.throttle,
            "rpm": self.rpm,
            "engine_temp": self.engine_temp,
            "fuel_flow": self.fuel_flow,
            "vibration": self.vibration,
            "health_score": self.health_score,
            "sensor_health": self.sensor_health,
            **analysis,
        }

    def run_profile(self, throttles: Iterable[float], dt: float = 1.0) -> pd.DataFrame:
        rows = [self.step(throttle=t, dt=dt) for t in throttles]
        return pd.DataFrame(rows)

    def _expected_values(self) -> tuple[float, float, float]:
        load = self.rpm / self.max_rpm
        expected_temp = self.ambient_temp_c + 90.0 + 320.0 * load + 180.0 * self.throttle
        expected_vib = 0.08 + 0.22 * load
        expected_fuel = 220.0 + 0.045 * self.rpm + 900.0 * self.throttle
        return expected_temp, expected_vib, expected_fuel

    def _push_history(self) -> None:
        self._history_temp.append(self.engine_temp)
        self._history_vibration.append(self.vibration)
        self._history_rpm.append(self.rpm)

    def _history_stats(self, values: deque[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 1.0
        return mean, max(std, 1e-6)

    def _analyze_state(self, dt: float) -> dict[str, float | str]:
        expected_temp, expected_vib, expected_fuel = self._expected_values()
        temp_mean, temp_std = self._history_stats(self._history_temp)
        vib_mean, vib_std = self._history_stats(self._history_vibration)

        rpm_rate = (self.rpm - self._prev_rpm) / max(dt, 1e-6)
        temp_rate = (self.engine_temp - self._prev_temp) / max(dt, 1e-6)

        temp_dev = abs(self.engine_temp - expected_temp) / 35.0
        vib_dev = max(0.0, self.vibration - expected_vib) / 0.10
        fuel_dev = abs(self.fuel_flow - expected_fuel) / 140.0
        thermal_trend = max(0.0, temp_rate) / 8.0
        transient = abs(rpm_rate) / 2600.0

        temp_z = abs(self.engine_temp - temp_mean) / temp_std
        vib_z = abs(self.vibration - vib_mean) / vib_std
        confidence_penalty = max(0.0, 1.0 - self.sensor_health) * 0.9

        contributors = {
            "temp_deviation": temp_dev,
            "vibration_excess": vib_dev,
            "fuel_mismatch": 0.5 * fuel_dev,
            "thermal_trend": 0.6 * thermal_trend,
            "transient_instability": 0.5 * transient,
            "sensor_reliability_penalty": confidence_penalty,
            "historical_temp_z": 0.2 * temp_z,
            "historical_vib_z": 0.2 * vib_z,
        }

        anomaly_score = 1.0 + float(sum(contributors.values()))
        anomaly_score += float(self.rng.normal(0.0, 0.03))
        anomaly_score = max(0.0, anomaly_score)

        reason_codes: list[str] = []
        if temp_dev > 0.8 or temp_z > 2.5:
            reason_codes.append("THERMAL_DRIFT")
        if vib_dev > 0.7 or vib_z > 2.3:
            reason_codes.append("VIBRATION_GROWTH")
        if transient > 1.5:
            reason_codes.append("RPM_TRANSIENT")
        if self.health_score < 0.80:
            reason_codes.append("HEALTH_DEGRADATION")
        if self.sensor_health < 0.82:
            reason_codes.append("SENSOR_DRIFT")
        if not reason_codes:
            reason_codes.append("NOMINAL")

        top_reason = max(contributors.items(), key=lambda item: item[1])[0]
        confidence = _clamp(1.0 - 0.5 * max(0.0, temp_z - 2.0) / 4.0 - 0.5 * max(0.0, vib_z - 2.0) / 4.0, 0.25, 0.99)
        confidence *= _clamp(self.sensor_health, 0.7, 1.0)
        confidence = _clamp(confidence, 0.20, 0.99)

        return {
            "rpm_rate_of_change": rpm_rate,
            "temp_rate_of_change": temp_rate,
            "anomaly_score": anomaly_score,
            "anomaly_confidence": confidence,
            "primary_reason": top_reason,
            "reason_codes": "|".join(reason_codes),
            "temp_deviation_component": temp_dev,
            "vibration_component": vib_dev,
            "fuel_component": 0.5 * fuel_dev,
            "trend_component": 0.6 * thermal_trend,
            "transient_component": 0.5 * transient,
            "sensor_component": confidence_penalty,
            "estimated_temp_baseline": expected_temp,
            "estimated_vibration_baseline": expected_vib,
            "estimated_fuel_baseline": expected_fuel,
        }
