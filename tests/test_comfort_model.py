"""
Tests for comfort_model.py — ASHRAE baseline, learned corrections, persistence.
"""

import json
import os
import tempfile
import pytest

from comfort_model import ComfortModel, ComfortInput, ComfortPrediction, _time_bucket, _outdoor_temp_bucket


# ── Helpers ──────────────────────────────────────────────────────

def _inp(outdoor=85, indoor=76, humidity=45, mode="cooling", hour=14, minute=0,
         zone="downstairs", vacation=False, current_target=None):
    return ComfortInput(
        outdoor_temp=outdoor, indoor_temp=indoor, humidity=humidity,
        hvac_mode=mode, hour=hour, minute=minute, zone=zone,
        is_vacation=vacation, current_target=current_target,
    )


# ── Bucketing Tests ──────────────────────────────────────────────

class TestBucketing:
    def test_time_buckets(self):
        assert _time_bucket(6) == "morning"
        assert _time_bucket(11) == "morning"
        assert _time_bucket(12) == "afternoon"
        assert _time_bucket(17) == "afternoon"
        assert _time_bucket(18) == "evening"
        assert _time_bucket(21) == "evening"
        assert _time_bucket(22) == "night"
        assert _time_bucket(3) == "night"

    def test_outdoor_temp_buckets(self):
        assert _outdoor_temp_bucket(30) == "cold"
        assert _outdoor_temp_bucket(60) == "mild"
        assert _outdoor_temp_bucket(85) == "hot"
        assert _outdoor_temp_bucket(100) == "extreme"


# ── Layer 1: ASHRAE Baseline Tests ───────────────────────────────

class TestBaseline:
    def setup_method(self):
        self.model = ComfortModel()

    def test_normal_cooling_default_78(self):
        """Normal cooling conditions → 78F."""
        pred = self.model.predict(_inp(outdoor=85, hour=14))
        assert pred.baseline_temp == 78

    def test_mild_day_cooling_78(self):
        """Mild outdoor temp → still 78F."""
        pred = self.model.predict(_inp(outdoor=70, hour=14))
        assert pred.baseline_temp == 78

    def test_extreme_heat_precool_75(self):
        """Extreme heat (95F+) → pre-cool to 75F."""
        pred = self.model.predict(_inp(outdoor=100, hour=14))
        assert pred.baseline_temp == 75
        assert "extreme heat" in pred.reasoning.lower() or "pre-cool" in pred.reasoning.lower()

    def test_precool_before_bed(self):
        """9:30-10:30 PM → pre-cool to 75F."""
        pred = self.model.predict(_inp(hour=21, minute=30))
        assert pred.baseline_temp == 75
        assert "pre-cool" in pred.reasoning.lower()

    def test_precool_at_10pm(self):
        """10:00 PM is still in pre-cool window."""
        pred = self.model.predict(_inp(hour=22, minute=0))
        assert pred.baseline_temp == 75

    def test_sleep_after_1030pm(self):
        """After 10:30 PM → sleep target 78F."""
        pred = self.model.predict(_inp(hour=22, minute=30))
        assert pred.baseline_temp == 78
        assert "sleep" in pred.reasoning.lower()

    def test_sleep_at_midnight(self):
        """Midnight → sleep target 78F."""
        pred = self.model.predict(_inp(hour=0, minute=0))
        assert pred.baseline_temp == 78

    def test_sleep_at_3am(self):
        """3 AM → sleep target 78F."""
        pred = self.model.predict(_inp(hour=3, minute=0))
        assert pred.baseline_temp == 78

    def test_sleep_ends_at_7am(self):
        """7 AM → normal daytime (no longer sleep)."""
        pred = self.model.predict(_inp(hour=7, minute=0, outdoor=85))
        assert pred.baseline_temp == 78  # Normal cooling, not sleep

    def test_heating_extreme_cold(self):
        """Extreme cold (20F) → 72F."""
        pred = self.model.predict(_inp(outdoor=15, mode="heating"))
        assert pred.baseline_temp == 72

    def test_heating_mild(self):
        """Mild outdoor (65F+) → 68F."""
        pred = self.model.predict(_inp(outdoor=65, mode="heating"))
        assert pred.baseline_temp == 68

    def test_heating_interpolation(self):
        """Mid-range outdoor → interpolated between 72 and 68."""
        pred = self.model.predict(_inp(outdoor=42, mode="heating"))
        # 42F is about halfway between 20 and 65 → ~70F
        assert 69 <= pred.baseline_temp <= 71

    def test_vacation_cooling(self):
        """Vacation + cooling → 82F."""
        pred = self.model.predict(_inp(vacation=True, mode="cooling"))
        assert pred.baseline_temp == 82
        assert "vacation" in pred.reasoning.lower()

    def test_vacation_heating(self):
        """Vacation + heating → 60F."""
        pred = self.model.predict(_inp(vacation=True, mode="heating", outdoor=30))
        assert pred.baseline_temp == 60
        assert "vacation" in pred.reasoning.lower()


# ── Deadband Tests ───────────────────────────────────────────────

class TestDeadband:
    def setup_method(self):
        self.model = ComfortModel(deadband_f=2.0)

    def test_within_deadband_no_change(self):
        """Target 78F, current 77F → within 2F deadband, no change."""
        pred = self.model.predict(_inp(current_target=77))
        assert pred.target_temp == 78
        assert pred.should_change is False
        assert "deadband" in pred.reasoning.lower()

    def test_outside_deadband_change(self):
        """Target 78F, current 75F → outside 2F deadband, should change."""
        pred = self.model.predict(_inp(current_target=75))
        assert pred.should_change is True

    def test_no_current_target_always_change(self):
        """No current target → always change."""
        pred = self.model.predict(_inp(current_target=None))
        assert pred.should_change is True

    def test_exact_deadband_boundary_no_change(self):
        """Difference of exactly 1F (< 2F deadband) → no change."""
        pred = self.model.predict(_inp(current_target=79))
        assert pred.should_change is False

    def test_at_deadband_threshold_change(self):
        """Difference of exactly 2F (>= deadband) → change."""
        pred = self.model.predict(_inp(current_target=76))
        assert pred.should_change is True


# ── Layer 2: Corrections Tests ───────────────────────────────────

class TestCorrections:
    def setup_method(self):
        self.model = ComfortModel()

    def test_learn_and_apply_correction(self):
        """After learning an override, correction is applied to predictions."""
        inp = _inp(outdoor=85, hour=14, zone="downstairs")
        # User wants 76F when baseline says 78F → correction should trend toward -2
        self.model.learn_override(inp, 76.0)

        pred = self.model.predict(inp)
        # After one learning: correction = 0.3 * -2 = -0.6, rounded target = 77
        assert pred.correction != 0
        assert pred.target_temp < 78

    def test_correction_converges(self):
        """Multiple identical overrides → correction converges to the error."""
        inp = _inp(outdoor=85, hour=14, zone="downstairs")
        for _ in range(20):
            self.model.learn_override(inp, 76.0)

        pred = self.model.predict(inp)
        # Should converge close to 76F
        assert 75 <= pred.target_temp <= 77

    def test_ema_blending(self):
        """EMA smooths out conflicting overrides."""
        inp = _inp(outdoor=85, hour=14, zone="downstairs")
        self.model.learn_override(inp, 76.0)  # wants lower
        self.model.learn_override(inp, 80.0)  # wants higher

        pred = self.model.predict(inp)
        # Should be between the two extremes
        assert 77 <= pred.target_temp <= 79

    def test_different_buckets_independent(self):
        """Corrections in one bucket don't affect another."""
        inp_hot = _inp(outdoor=85, hour=14, zone="downstairs")
        inp_cold = _inp(outdoor=30, hour=14, zone="downstairs", mode="heating")

        self.model.learn_override(inp_hot, 76.0)

        pred_cold = self.model.predict(inp_cold)
        assert pred_cold.correction == 0  # Different bucket, no correction


# ── Zone Offset Tests ────────────────────────────────────────────

class TestZoneOffsets:
    def setup_method(self):
        self.model = ComfortModel()

    def test_manual_zone_offsets(self):
        """Manually set zone offsets are applied."""
        self.model.zone_offsets = {"upstairs": -1.0, "downstairs": 0.0}

        pred_up = self.model.predict(_inp(zone="upstairs"))
        pred_down = self.model.predict(_inp(zone="downstairs"))

        assert pred_up.target_temp == pred_down.target_temp - 1

    def test_train_zone_offsets(self):
        """Zone offset training from paired data."""
        # Upstairs is 2F warmer on average
        climate_data = []
        for i in range(10):
            ts = f"2025-07-{15+i//5:02d} {10+i%5}:00"
            climate_data.append({"zone": "Upstairs Bedroom", "indoor_temp": 78.0, "timestamp": ts})
            climate_data.append({"zone": "Downstairs Kitchen", "indoor_temp": 76.0, "timestamp": ts})

        self.model.train_zone_offsets(climate_data)

        # Upstairs offset should be negative (target lower since it's hotter)
        assert self.model.zone_offsets.get("upstairs", 0) < 0
        assert self.model.zone_offsets.get("downstairs", 0) > 0


# ── Persistence Tests ────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load_roundtrip(self):
        """Save corrections and zone offsets, load them back."""
        model1 = ComfortModel()
        model1.zone_offsets = {"upstairs": -1.0, "downstairs": 0.5}
        model1.corrections = {"cool:up:afternoon:hot": -1.5, "heat:down:morning:cold": 0.8}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            model1.save(path)

            model2 = ComfortModel()
            model2.load(path)

            assert model2.zone_offsets == model1.zone_offsets
            assert model2.corrections == model1.corrections
        finally:
            os.unlink(path)

    def test_load_missing_file(self):
        """Loading from a missing file starts fresh."""
        model = ComfortModel()
        model.load("/nonexistent/path.json")
        assert model.corrections == {}
        assert model.zone_offsets == {}

    def test_load_corrupt_file(self):
        """Loading corrupt JSON starts fresh."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            path = f.name

        try:
            model = ComfortModel()
            model.load(path)
            assert model.corrections == {}
        finally:
            os.unlink(path)


# ── Bulk Training Tests ──────────────────────────────────────────

class TestBulkTraining:
    def test_train_from_overrides(self):
        """Train from a list of historical overrides."""
        model = ComfortModel()
        overrides = [
            {
                "timestamp": "2025-07-15 14:30:00",
                "zone": "Downstairs Kitchen",
                "detected_target": 76.0,
                "indoor_temp": 78.0,
                "outdoor_temp": 90.0,
                "hvac_mode": "cooling",
            },
            {
                "timestamp": "2025-07-15 15:00:00",
                "zone": "Downstairs Kitchen",
                "detected_target": 76.0,
                "indoor_temp": 78.0,
                "outdoor_temp": 88.0,
                "hvac_mode": "cooling",
            },
        ]
        model.train_from_overrides(overrides)
        assert len(model.corrections) > 0

    def test_train_skips_bad_rows(self):
        """Bad rows are skipped without crashing."""
        model = ComfortModel()
        overrides = [
            {"timestamp": "bad", "zone": "x"},
            {"timestamp": "2025-07-15 14:30:00", "zone": "Downstairs", "detected_target": 76.0,
             "outdoor_temp": 90.0, "hvac_mode": "cooling", "indoor_temp": 78.0},
        ]
        model.train_from_overrides(overrides)
        # Should have trained from the one valid row
        assert len(model.corrections) >= 1


# ── Integration Tests ────────────────────────────────────────────

class TestIntegration:
    def test_full_prediction_flow(self):
        """End-to-end: train → predict → verify."""
        model = ComfortModel(deadband_f=2.0)
        model.zone_offsets = {"upstairs": -1.0, "downstairs": 0.0}

        # Hot afternoon downstairs → baseline 78, no correction → 78F
        pred = model.predict(_inp(outdoor=88, hour=14, zone="downstairs"))
        assert pred.target_temp == 78
        assert pred.should_change is True

        # Hot afternoon upstairs → baseline 78 - 1 offset → 77F
        pred = model.predict(_inp(outdoor=88, hour=14, zone="upstairs"))
        assert pred.target_temp == 77

    def test_precool_window_exact_boundaries(self):
        """Verify exact pre-cool window boundaries."""
        model = ComfortModel()

        # 9:29 PM → normal (not pre-cool)
        pred = model.predict(_inp(hour=21, minute=29, outdoor=85))
        assert pred.baseline_temp == 78

        # 9:30 PM → pre-cool starts
        pred = model.predict(_inp(hour=21, minute=30, outdoor=85))
        assert pred.baseline_temp == 75

        # 10:29 PM → still pre-cool
        pred = model.predict(_inp(hour=22, minute=29, outdoor=85))
        assert pred.baseline_temp == 75

        # 10:30 PM → sleep target
        pred = model.predict(_inp(hour=22, minute=30, outdoor=85))
        assert pred.baseline_temp == 78

    def test_heating_full_range(self):
        """Verify heating targets across full outdoor range."""
        model = ComfortModel()

        # Extreme cold
        pred = model.predict(_inp(outdoor=10, mode="heating"))
        assert pred.baseline_temp == 72

        # Very cold
        pred = model.predict(_inp(outdoor=20, mode="heating"))
        assert pred.baseline_temp == 72

        # Moderate cold
        pred = model.predict(_inp(outdoor=42, mode="heating"))
        assert 69 <= pred.baseline_temp <= 71

        # Approaching mild
        pred = model.predict(_inp(outdoor=60, mode="heating"))
        assert 68 <= pred.baseline_temp <= 69

        # Mild
        pred = model.predict(_inp(outdoor=65, mode="heating"))
        assert pred.baseline_temp == 68
