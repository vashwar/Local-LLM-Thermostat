"""
Tests for agent.py core logic — directive builder, guardrails, validation.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass


# We need to set up sys.modules before importing agent to avoid
# circular import issues with telegram_bot
import sys
sys.modules["telegram"] = MagicMock()
sys.modules["telegram.error"] = MagicMock()
sys.modules["telegram.ext"] = MagicMock()

import agent
from nest_api import ThermostatState


@pytest.fixture(autouse=True)
def setup_config():
    """Load a minimal config for all tests."""
    agent._config = {
        "llm": {
            "model": "test",
            "endpoint": "http://localhost:8080/v1/chat/completions",
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 500,
            "timeout_seconds": 10,
            "server_exe": "test.exe",
            "model_path": "test.gguf",
        },
        "comfort": {
            "summer_range": [75, 80],
            "winter_range": [68, 72],
            "user_request_hours": 2,
            "sleep_cool_temp": 75,
            "sleep_cool_override_temp": 80,
            "pre_heat_minutes": 30,
        },
        "schedule": {
            "sleep_time": "23:00",
            "wake_time": "07:00",
        },
        "agent": {
            "loop_interval_minutes": 20,
            "db_path": ":memory:",
            "log_level": "INFO",
        },
        "weather": {
            "api_key": "test",
            "latitude": 33.0,
            "longitude": -96.0,
        },
        "nest": {
            "tokens_path": "test_tokens.json",
        },
        "telegram": {
            "bot_token": "test",
            "whitelisted_chat_ids": [],
        },
    }


def _make_thermo_state(name="Test Zone", mode="cooling", indoor_temp=72.0,
                        target_temp=72.0, humidity=45, hvac_running=False):
    return ThermostatState(
        name=name,
        device_id="test-device-id",
        indoor_temp=indoor_temp,
        humidity=humidity,
        mode=mode,
        target_temp=target_temp,
        hvac_running=hvac_running,
    )


# ── Regression: ISSUE-001 — mode string mismatch ──────────────────
# Found by /qa on 2026-03-31
# Report: .gstack/qa-reports/qa-report-2026-03-31.md

class TestModeStringRegression:
    """Regression tests for ISSUE-001: nest_api returns lowercase mode strings
    ('cooling', 'heating') but _build_directive used uppercase ('COOL', 'HEAT').
    This caused winter range to never be used and sleep logic to be inverted."""

    def test_heating_mode_uses_winter_range(self):
        """In heating mode, directive should reference winter comfort range (68-72F)."""
        state = _make_thermo_state(mode="heating", indoor_temp=65.0)
        now = datetime(2025, 12, 15, 14, 0)  # 2 PM, a home period
        comfort = agent._config["comfort"]
        sched = agent._config["schedule"]

        directive = agent._build_directive(state, MagicMock(current_temp=35.0),
                                           now, comfort, sched, [])

        assert "68-72F" in directive, \
            f"Heating mode should use winter range (68-72F), got: {directive}"
        assert "75-80F" not in directive, \
            f"Heating mode should NOT use summer range, got: {directive}"

    def test_cooling_mode_uses_summer_range(self):
        """In cooling mode, directive should reference summer comfort range (75-80F)."""
        state = _make_thermo_state(mode="cooling", indoor_temp=78.0)
        now = datetime(2025, 7, 15, 14, 0)
        comfort = agent._config["comfort"]
        sched = agent._config["schedule"]

        directive = agent._build_directive(state, MagicMock(current_temp=95.0),
                                           now, comfort, sched, [])

        assert "75-80F" in directive
        assert "68-72F" not in directive

    def test_sleep_cooling_mode_uses_cool_branch(self):
        """During sleep in cooling mode, directive should mention summer/cooling logic."""
        state = _make_thermo_state(mode="cooling", indoor_temp=74.0)
        now = datetime(2025, 7, 15, 23, 30)  # 11:30 PM — sleep time
        comfort = agent._config["comfort"]
        sched = agent._config["schedule"]

        with patch("agent.weather") as mock_weather:
            mock_weather.get_forecast_analysis.return_value = None
            directive = agent._build_directive(state, MagicMock(current_temp=78.0),
                                               now, comfort, sched, [])

        assert "summer" in directive.lower() or "cool" in directive.lower(), \
            f"Sleep+cooling should use summer/cool logic, got: {directive}"

    def test_sleep_heating_mode_uses_heat_branch(self):
        """During sleep in heating mode, directive should mention winter/heat logic."""
        state = _make_thermo_state(mode="heating", indoor_temp=66.0)
        now = datetime(2025, 12, 15, 23, 30)  # 11:30 PM — sleep time
        comfort = agent._config["comfort"]
        sched = agent._config["schedule"]

        with patch("agent.weather") as mock_weather:
            mock_weather.get_forecast_analysis.return_value = None
            directive = agent._build_directive(state, MagicMock(current_temp=30.0),
                                               now, comfort, sched, [])

        assert "winter" in directive.lower() or "heat" in directive.lower() or "cold" in directive.lower(), \
            f"Sleep+heating should use heat logic, got: {directive}"


# ── Regression: ISSUE-002 — message age timezone mismatch ─────────
# Found by /qa on 2026-03-31

class TestMessageAgeTimezone:
    """Regression tests for ISSUE-002: _get_message_age_minutes compared UTC
    timestamps from SQLite with local time from datetime.now(), causing the
    age to be off by the local timezone offset."""

    def test_message_age_utc_timestamp(self):
        """A message stored 10 minutes ago in UTC should report ~10 minutes age."""
        now = datetime.now()
        # Simulate a SQLite UTC timestamp from 10 minutes ago
        utc_now = datetime.now(timezone.utc)
        ten_min_ago_utc = utc_now - timedelta(minutes=10)
        ts = ten_min_ago_utc.strftime("%Y-%m-%d %H:%M:%S")

        msg = {"timestamp": ts, "text": "test"}
        age = agent._get_message_age_minutes(msg, now)

        assert age is not None
        assert 8 <= age <= 12, \
            f"Message from 10 min ago should report ~10 min age, got {age:.1f}"

    def test_message_age_returns_none_for_missing_timestamp(self):
        """Messages without timestamps should return None."""
        msg = {"text": "test"}
        assert agent._get_message_age_minutes(msg, datetime.now()) is None

    def test_message_age_returns_none_for_empty_timestamp(self):
        msg = {"timestamp": "", "text": "test"}
        assert agent._get_message_age_minutes(msg, datetime.now()) is None


# ── Validate Response Tests ───────────────────────────────────────

class TestValidateResponse:
    def test_valid_set_temperature(self):
        resp = json.dumps({
            "action": "set_temperature",
            "temperature": 72,
            "reasoning": "It's cold"
        })
        valid, error = agent.validate_response(resp)
        assert valid is True
        assert error is None

    def test_valid_no_change(self):
        resp = json.dumps({
            "action": "no_change",
            "temperature": None,
            "reasoning": "Comfortable"
        })
        valid, error = agent.validate_response(resp)
        assert valid is True

    def test_invalid_action(self):
        resp = json.dumps({"action": "turn_off", "reasoning": "test"})
        valid, error = agent.validate_response(resp)
        assert valid is False
        assert "Invalid action" in error

    def test_missing_action(self):
        resp = json.dumps({"temperature": 72, "reasoning": "test"})
        valid, error = agent.validate_response(resp)
        assert valid is False
        assert "Missing 'action'" in error

    def test_missing_reasoning(self):
        resp = json.dumps({"action": "no_change"})
        valid, error = agent.validate_response(resp)
        assert valid is False
        assert "Missing 'reasoning'" in error

    def test_temperature_below_min(self):
        resp = json.dumps({
            "action": "set_temperature",
            "temperature": 50,
            "reasoning": "test"
        })
        valid, error = agent.validate_response(resp)
        assert valid is False
        assert "out of range" in error

    def test_temperature_above_max(self):
        resp = json.dumps({
            "action": "set_temperature",
            "temperature": 90,
            "reasoning": "test"
        })
        valid, error = agent.validate_response(resp)
        assert valid is False
        assert "out of range" in error

    def test_set_temperature_missing_temp(self):
        resp = json.dumps({
            "action": "set_temperature",
            "temperature": None,
            "reasoning": "test"
        })
        valid, error = agent.validate_response(resp)
        assert valid is False

    def test_no_change_with_temperature(self):
        resp = json.dumps({
            "action": "no_change",
            "temperature": 72,
            "reasoning": "test"
        })
        valid, error = agent.validate_response(resp)
        assert valid is False

    def test_invalid_json(self):
        valid, error = agent.validate_response("not json at all")
        assert valid is False
        assert "Invalid JSON" in error

    def test_boundary_temp_min(self):
        resp = json.dumps({
            "action": "set_temperature",
            "temperature": 65,
            "reasoning": "at minimum"
        })
        valid, error = agent.validate_response(resp)
        assert valid is True

    def test_boundary_temp_max(self):
        resp = json.dumps({
            "action": "set_temperature",
            "temperature": 80,
            "reasoning": "at maximum"
        })
        valid, error = agent.validate_response(resp)
        assert valid is True


# ── Guardrails Tests ──────────────────────────────────────────────

class TestGuardrails:
    def test_no_change_always_allowed(self):
        decision = {"action": "no_change", "reasoning": "fine"}
        allowed, reason = agent.check_guardrails(decision)
        assert allowed is True

    def test_temperature_in_range(self):
        with patch("agent.database") as mock_db:
            mock_db.count_temp_changes_last_hour.return_value = 0
            mock_db.get_last_manual_override_time.return_value = None
            decision = {"action": "set_temperature", "temperature": 72, "reasoning": "test"}
            allowed, reason = agent.check_guardrails(decision)
            assert allowed is True

    def test_temperature_below_min_blocked(self):
        decision = {"action": "set_temperature", "temperature": 50, "reasoning": "test"}
        allowed, reason = agent.check_guardrails(decision)
        assert allowed is False
        assert "outside bounds" in reason

    def test_temperature_above_max_blocked(self):
        decision = {"action": "set_temperature", "temperature": 90, "reasoning": "test"}
        allowed, reason = agent.check_guardrails(decision)
        assert allowed is False
        assert "outside bounds" in reason

    def test_rate_limit_blocks(self):
        with patch("agent.database") as mock_db:
            mock_db.count_temp_changes_last_hour.return_value = 6
            mock_db.get_last_manual_override_time.return_value = None
            decision = {"action": "set_temperature", "temperature": 72, "reasoning": "test"}
            allowed, reason = agent.check_guardrails(decision)
            assert allowed is False
            assert "Rate limit" in reason

    def test_user_triggered_bypasses_rate_limit(self):
        with patch("agent.database") as mock_db:
            mock_db.count_temp_changes_last_hour.return_value = 10
            mock_db.get_last_manual_override_time.return_value = None
            decision = {"action": "set_temperature", "temperature": 72, "reasoning": "test"}
            allowed, reason = agent.check_guardrails(decision, user_triggered=True)
            assert allowed is True


# ── Time Period Tests ─────────────────────────────────────────────

class TestTimePeriod:
    def test_sleep_time_late_night(self):
        now = datetime(2025, 7, 15, 23, 30)  # 11:30 PM
        period = agent._get_time_period(now, agent._config["schedule"])
        assert period == "sleep"

    def test_sleep_time_early_morning(self):
        now = datetime(2025, 7, 15, 3, 0)  # 3 AM
        period = agent._get_time_period(now, agent._config["schedule"])
        assert period == "sleep"

    def test_pre_wake(self):
        now = datetime(2025, 7, 15, 6, 45)  # 6:45 AM, 15 min before 7 AM wake
        period = agent._get_time_period(now, agent._config["schedule"])
        assert period == "pre_wake"

    def test_waking_up(self):
        now = datetime(2025, 7, 15, 7, 30)  # 7:30 AM
        period = agent._get_time_period(now, agent._config["schedule"])
        assert period == "waking_up"

    def test_work_hours_weekday(self):
        now = datetime(2025, 7, 14, 10, 0)  # Monday 10 AM
        period = agent._get_time_period(now, agent._config["schedule"])
        assert period == "work"

    def test_home_afternoon(self):
        now = datetime(2025, 7, 15, 18, 0)  # 6 PM Tuesday
        period = agent._get_time_period(now, agent._config["schedule"])
        assert period == "home"

    def test_winding_down(self):
        now = datetime(2025, 7, 15, 22, 15)  # 10:15 PM
        period = agent._get_time_period(now, agent._config["schedule"])
        assert period == "winding_down"


# ── User Request Detection ────────────────────────────────────────

class TestFindUserRequest:
    def test_finds_temperature_request(self):
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)
        ts = (utc_now - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
        messages = [{"text": "set upstairs to 78", "timestamp": ts}]
        result = agent._find_recent_user_request(messages, now, max_hours=2)
        assert result == "set upstairs to 78"

    def test_ignores_old_request(self):
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)
        ts = (utc_now - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
        messages = [{"text": "set to 78", "timestamp": ts}]
        result = agent._find_recent_user_request(messages, now, max_hours=2)
        assert result is None

    def test_ignores_non_temperature_message(self):
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)
        ts = (utc_now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        messages = [{"text": "what's the weather?", "timestamp": ts}]
        result = agent._find_recent_user_request(messages, now, max_hours=2)
        assert result is None
