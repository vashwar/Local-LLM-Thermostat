#!/usr/bin/env python3
"""
Predictive Comfort Model — two-layer temperature decision engine.

Layer 1: ASHRAE-inspired adaptive baseline (deterministic)
Layer 2: Learned corrections from manual overrides (bucketed lookup)

No ML dependencies. Pure Python stdlib.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class ComfortInput:
    outdoor_temp: float          # Current outdoor temp (F)
    indoor_temp: float           # Current indoor temp (F)
    humidity: int                # Indoor humidity (%)
    hvac_mode: str               # "cooling" or "heating"
    hour: int                    # 0-23
    minute: int                  # 0-59
    zone: str                    # "upstairs" or "downstairs"
    is_vacation: bool = False
    current_target: Optional[float] = None  # Current thermostat target


@dataclass
class ComfortPrediction:
    target_temp: float           # Predicted target temperature (F)
    reasoning: str               # Human-readable explanation
    baseline_temp: float         # Layer 1 output (before corrections)
    correction: float = 0.0     # Layer 2 correction applied
    should_change: bool = True   # False if within deadband


# ── Time and temp bucketing ──────────────────────────────────────

def _time_bucket(hour: int) -> str:
    """Bucket hour into time period."""
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 22:
        return "evening"
    else:
        return "night"


def _outdoor_temp_bucket(temp: float) -> str:
    """Bucket outdoor temp into range."""
    if temp < 45:
        return "cold"
    elif temp < 75:
        return "mild"
    elif temp < 95:
        return "hot"
    else:
        return "extreme"


def _correction_key(mode: str, zone: str, hour: int, outdoor_temp: float) -> str:
    """Build a bucket key for the corrections lookup."""
    mode_key = "cool" if mode == "cooling" else "heat"
    zone_key = "up" if "upstairs" in zone.lower() or "bedroom" in zone.lower() else "down"
    time_key = _time_bucket(hour)
    temp_key = _outdoor_temp_bucket(outdoor_temp)
    return f"{mode_key}:{zone_key}:{time_key}:{temp_key}"


def _normalize_zone(zone_name: str) -> str:
    """Normalize zone name to 'upstairs' or 'downstairs'."""
    z = zone_name.lower()
    if "upstairs" in z or "bedroom" in z:
        return "upstairs"
    return "downstairs"


# ── Comfort Model ────────────────────────────────────────────────

class ComfortModel:
    """Two-layer predictive comfort model."""

    # Layer 1 defaults (overridable via config)
    VACATION_COOL = 82.0         # Vacation cooling ceiling
    VACATION_HEAT = 60.0         # Vacation heating floor

    # Layer 2
    EMA_ALPHA = 0.3              # Exponential moving average blend factor

    def __init__(self, deadband_f: float = 2.0, zone_offsets: Optional[dict] = None,
                 summer_range: Optional[list] = None, winter_range: Optional[list] = None,
                 sleep_time: str = "23:00", wake_time: str = "07:00",
                 precool_time: Optional[str] = None,
                 sleep_cool_temp: Optional[float] = None):
        self.deadband_f = deadband_f
        self.zone_offsets: dict = zone_offsets or {}
        self.corrections: dict = {}

        # Summer/winter targets from config
        sr = summer_range or [75, 80]
        wr = winter_range or [68, 72]
        self.COOL_DEFAULT = float(sr[1]) - 2.0     # Default cooling (78 from [75,80])
        self.COOL_PRECOOL = float(sr[0])            # Pre-cool for extreme heat (75)
        self.SLEEP_TARGET = float(sr[1]) - 2.0      # Sleep target (78)
        self.HEAT_MILD = float(wr[0])               # Heating mild (68)
        self.HEAT_COLD = float(wr[1])               # Heating extreme cold (72)

        # Pre-cool before bed temp
        self.COOL_PRECOOL_BED = float(sleep_cool_temp) if sleep_cool_temp is not None else self.COOL_PRECOOL

        # Sleep/wake schedule from config
        sleep_h, sleep_m = (int(x) for x in sleep_time.split(":"))
        wake_h, wake_m = (int(x) for x in wake_time.split(":"))

        # Pre-cool start: explicit config or default 1 hour before sleep
        if precool_time:
            pc_h, pc_m = (int(x) for x in precool_time.split(":"))
        else:
            precool_total = sleep_h * 60 + sleep_m - 60
            if precool_total < 0:
                precool_total += 24 * 60
            pc_h = precool_total // 60
            pc_m = precool_total % 60

        self.PRECOOL_START_HOUR = pc_h
        self.PRECOOL_START_MIN = pc_m
        self.PRECOOL_END_HOUR = sleep_h
        self.PRECOOL_END_MIN = sleep_m
        self.SLEEP_START_HOUR = sleep_h
        self.SLEEP_START_MIN = sleep_m
        self.WAKE_HOUR = wake_h

    # ── Layer 1: ASHRAE Adaptive Baseline ────────────────────────

    def _baseline(self, inp: ComfortInput) -> tuple:
        """Compute baseline target temp and reasoning."""

        # Vacation mode — simple energy saving
        if inp.is_vacation:
            if inp.hvac_mode == "cooling":
                return self.VACATION_COOL, "Vacation mode: cooling ceiling"
            else:
                return self.VACATION_HEAT, "Vacation mode: heating floor"

        # Heating mode
        if inp.hvac_mode == "heating":
            return self._baseline_heating(inp)

        # Cooling mode
        return self._baseline_cooling(inp)

    def _baseline_cooling(self, inp: ComfortInput) -> tuple:
        """Cooling mode baseline."""
        cur_minutes = inp.hour * 60 + inp.minute
        precool_start = self.PRECOOL_START_HOUR * 60 + self.PRECOOL_START_MIN  # 21:30
        precool_end = self.PRECOOL_END_HOUR * 60 + self.PRECOOL_END_MIN        # 22:30
        wake_minutes = self.WAKE_HOUR * 60                                      # 07:00

        # Pre-cool window: 9:30 PM - 10:30 PM
        if precool_start <= cur_minutes < precool_end:
            return self.COOL_PRECOOL_BED, "Pre-cooling before bed (9:30-10:30 PM)"

        # Sleep window: after 10:30 PM or before 7 AM
        if cur_minutes >= precool_end or cur_minutes < wake_minutes:
            return self.SLEEP_TARGET, "Sleep target"

        # Daytime: adaptive based on outdoor temp
        if inp.outdoor_temp >= 95:
            # Extreme heat — pre-cool aggressively
            return self.COOL_PRECOOL, f"Pre-cooling for extreme heat ({inp.outdoor_temp:.0f}F outdoor)"

        # Default: 78F for all normal cooling conditions
        return self.COOL_DEFAULT, "Normal cooling target"

    def _baseline_heating(self, inp: ComfortInput) -> tuple:
        """Heating mode baseline — linear interpolation between cold and mild."""
        outdoor = inp.outdoor_temp

        if outdoor <= 20:
            target = self.HEAT_COLD  # 72F
            reason = f"Extreme cold ({outdoor:.0f}F outdoor), maintaining {target:.0f}F"
        elif outdoor >= 65:
            target = self.HEAT_MILD  # 68F
            reason = f"Mild outdoor ({outdoor:.0f}F), minimal heating"
        else:
            # Linear interpolation: 20F->72F, 65F->68F
            t = (outdoor - 20) / (65 - 20)
            target = self.HEAT_COLD + t * (self.HEAT_MILD - self.HEAT_COLD)
            target = round(target)
            reason = f"Heating for {outdoor:.0f}F outdoor, target {target:.0f}F"

        return target, reason

    # ── Layer 2: Learned Corrections ─────────────────────────────

    def _get_correction(self, inp: ComfortInput) -> float:
        """Look up learned correction for current conditions."""
        key = _correction_key(inp.hvac_mode, inp.zone, inp.hour, inp.outdoor_temp)
        return self.corrections.get(key, 0.0)

    def learn_override(self, inp: ComfortInput, user_target: float):
        """Learn from a manual override using exponential moving average."""
        baseline, _ = self._baseline(inp)
        zone_offset = self.zone_offsets.get(_normalize_zone(inp.zone), 0.0)
        predicted = baseline + zone_offset

        error = user_target - predicted
        key = _correction_key(inp.hvac_mode, inp.zone, inp.hour, inp.outdoor_temp)

        old = self.corrections.get(key, 0.0)
        new = (1 - self.EMA_ALPHA) * old + self.EMA_ALPHA * error
        self.corrections[key] = round(new, 2)

        logger.info("Learned correction: %s = %.2f (was %.2f, error=%.1f)",
                     key, new, old, error)

    # ── Zone offset training ─────────────────────────────────────

    def train_zone_offsets(self, climate_data: list):
        """Learn zone temperature offsets from paired climate readings.

        Args:
            climate_data: list of dicts with 'zone', 'indoor_temp', 'timestamp'
        """
        # Group by timestamp (approximately)
        from collections import defaultdict
        by_time = defaultdict(dict)

        for row in climate_data:
            zone = _normalize_zone(row.get("zone", "default"))
            ts = row.get("timestamp", "")[:16]  # Group by minute
            temp = row.get("indoor_temp")
            if ts and temp is not None:
                by_time[ts][zone] = temp

        # Compute differential where both zones have data
        diffs = []
        for ts, zones in by_time.items():
            if "upstairs" in zones and "downstairs" in zones:
                diffs.append(zones["upstairs"] - zones["downstairs"])

        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            # Upstairs runs hotter → set target lower by half the differential
            self.zone_offsets["upstairs"] = round(-avg_diff / 2, 1)
            self.zone_offsets["downstairs"] = round(avg_diff / 2, 1)
            logger.info("Zone offsets trained from %d pairs: upstairs=%.1f, downstairs=%.1f",
                         len(diffs), self.zone_offsets["upstairs"], self.zone_offsets["downstairs"])
        else:
            logger.info("No paired zone data found — using default offsets")

    # ── Bulk training from overrides ─────────────────────────────

    def train_from_overrides(self, overrides: list):
        """Train correction layer from historical manual overrides.

        Args:
            overrides: list of dicts with keys: zone, detected_target,
                      indoor_temp, outdoor_temp, hvac_mode, timestamp
        """
        count = 0
        for row in overrides:
            try:
                ts = row.get("timestamp", "")
                hour = int(ts[11:13]) if len(ts) >= 13 else 12
                minute = int(ts[14:16]) if len(ts) >= 16 else 0

                inp = ComfortInput(
                    outdoor_temp=row.get("outdoor_temp") or 75.0,
                    indoor_temp=row.get("indoor_temp") or 72.0,
                    humidity=row.get("indoor_humidity") or 45,
                    hvac_mode=row.get("hvac_mode") or "cooling",
                    hour=hour,
                    minute=minute,
                    zone=_normalize_zone(row.get("zone") or "default"),
                )
                user_target = row.get("detected_target")
                if user_target is not None:
                    self.learn_override(inp, user_target)
                    count += 1
            except (ValueError, TypeError, KeyError) as e:
                logger.warning("Skipping override row: %s", e)

        logger.info("Trained from %d/%d overrides, %d corrections populated",
                     count, len(overrides), len(self.corrections))

    # ── Main prediction ──────────────────────────────────────────

    def predict(self, inp: ComfortInput) -> ComfortPrediction:
        """Predict target temperature for given conditions."""
        # Layer 1: baseline
        baseline, reasoning = self._baseline(inp)

        # Zone offset
        zone_key = _normalize_zone(inp.zone)
        zone_offset = self.zone_offsets.get(zone_key, 0.0)

        # Layer 2: learned correction
        correction = self._get_correction(inp)

        # Combined target
        target = baseline + zone_offset + correction
        target = round(target)  # Round to whole degrees

        # Build reasoning
        parts = [reasoning]
        if zone_offset != 0:
            parts.append(f"zone offset {zone_offset:+.1f}F")
        if correction != 0:
            parts.append(f"learned correction {correction:+.1f}F")
        full_reasoning = "; ".join(parts)

        # Deadband check
        should_change = True
        if inp.current_target is not None:
            if abs(target - inp.current_target) < self.deadband_f:
                should_change = False
                full_reasoning += f" [within {self.deadband_f}F deadband, no change]"

        return ComfortPrediction(
            target_temp=target,
            reasoning=full_reasoning,
            baseline_temp=baseline,
            correction=correction,
            should_change=should_change,
        )

    # ── Persistence ──────────────────────────────────────────────

    def save(self, path: str):
        """Save corrections and zone offsets to JSON."""
        data = {
            "corrections": self.corrections,
            "zone_offsets": self.zone_offsets,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Comfort model saved to %s (%d corrections)", path, len(self.corrections))

    def load(self, path: str):
        """Load corrections and zone offsets from JSON."""
        try:
            with open(path) as f:
                data = json.load(f)
            self.corrections = data.get("corrections", {})
            self.zone_offsets = data.get("zone_offsets", {})
            logger.info("Comfort model loaded from %s (%d corrections)", path, len(self.corrections))
        except FileNotFoundError:
            logger.info("No saved model at %s — starting fresh", path)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load model from %s: %s — starting fresh", path, e)
