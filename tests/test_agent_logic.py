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
    """Load a minimal config and reset module state for all tests."""
    agent._evaluation_counter = 0
    agent._user_message_eval_counter = None
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

    def test_temperature_out_of_range_passes_validation(self):
        """validate_response no longer checks range — guardrails clamp instead."""
        for temp in [50, 90]:
            resp = json.dumps({
                "action": "set_temperature",
                "temperature": temp,
                "reasoning": "test"
            })
            valid, error = agent.validate_response(resp)
            assert valid is True, f"temp {temp} should pass validation (guardrails clamp later)"

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

    def test_guardrail_clamps_low(self):
        """Temperature 45 → clamped to 65, allowed=True."""
        with patch("agent.database") as mock_db:
            mock_db.count_temp_changes_last_hour.return_value = 0
            mock_db.get_last_manual_override_time.return_value = None
            decision = {"action": "set_temperature", "temperature": 45, "reasoning": "user wants cold"}
            allowed, reason = agent.check_guardrails(decision)
            assert allowed is True
            assert decision["temperature"] == 65
            assert "[Clamped 45F->65F]" in decision["reasoning"]

    def test_guardrail_clamps_high(self):
        """Temperature 90 → clamped to 80, allowed=True."""
        with patch("agent.database") as mock_db:
            mock_db.count_temp_changes_last_hour.return_value = 0
            mock_db.get_last_manual_override_time.return_value = None
            decision = {"action": "set_temperature", "temperature": 90, "reasoning": "user wants warm"}
            allowed, reason = agent.check_guardrails(decision)
            assert allowed is True
            assert decision["temperature"] == 80
            assert "[Clamped 90F->80F]" in decision["reasoning"]

    def test_guardrail_boundary_not_clamped(self):
        """Boundary values (65 and 80) are not clamped."""
        with patch("agent.database") as mock_db:
            mock_db.count_temp_changes_last_hour.return_value = 0
            mock_db.get_last_manual_override_time.return_value = None
            for temp in [65, 80]:
                decision = {"action": "set_temperature", "temperature": temp, "reasoning": "test"}
                allowed, reason = agent.check_guardrails(decision)
                assert allowed is True
                assert decision["temperature"] == temp
                assert "Clamped" not in decision["reasoning"]

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

    def test_awake_morning(self):
        now = datetime(2025, 7, 15, 7, 30)  # 7:30 AM
        period = agent._get_time_period(now, agent._config["schedule"])
        assert period == "awake"

    def test_awake_afternoon(self):
        now = datetime(2025, 7, 15, 14, 0)  # 2 PM
        period = agent._get_time_period(now, agent._config["schedule"])
        assert period == "awake"

    def test_awake_weekday_work_hours(self):
        """Former 'work' period is now just 'awake'."""
        now = datetime(2025, 7, 14, 10, 0)  # Monday 10 AM
        period = agent._get_time_period(now, agent._config["schedule"])
        assert period == "awake"

    def test_winding_down(self):
        now = datetime(2025, 7, 15, 22, 15)  # 10:15 PM
        period = agent._get_time_period(now, agent._config["schedule"])
        assert period == "winding_down"


# ── User Message Detection (no regex — LLM parses intent) ────────

class TestFindUserMessage:
    def test_finds_temperature_request(self):
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)
        ts = (utc_now - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
        messages = [{"text": "set upstairs to 78", "timestamp": ts}]
        result = agent._find_recent_user_message(messages, now, max_hours=2)
        assert result == "set upstairs to 78"

    def test_ignores_old_message(self):
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)
        ts = (utc_now - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
        messages = [{"text": "set to 78", "timestamp": ts}]
        result = agent._find_recent_user_message(messages, now, max_hours=2)
        assert result is None

    def test_returns_any_recent_message(self):
        """No regex — ANY recent message is returned for LLM to parse."""
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)
        ts = (utc_now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        messages = [{"text": "I'm hot", "timestamp": ts}]
        result = agent._find_recent_user_message(messages, now, max_hours=2)
        assert result == "I'm hot"

    def test_returns_greeting(self):
        """Even a greeting is returned — LLM decides it's not a temp request."""
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)
        ts = (utc_now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        messages = [{"text": "hello", "timestamp": ts}]
        result = agent._find_recent_user_message(messages, now, max_hours=2)
        assert result == "hello"

    def test_returns_question(self):
        """Questions are returned too — LLM handles them."""
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)
        ts = (utc_now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        messages = [{"text": "what's the weather?", "timestamp": ts}]
        result = agent._find_recent_user_message(messages, now, max_hours=2)
        assert result == "what's the weather?"

    def test_returns_most_recent(self):
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)
        ts_old = (utc_now - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
        ts_new = (utc_now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        messages = [
            {"text": "set to 72", "timestamp": ts_old},
            {"text": "actually set to 75", "timestamp": ts_new},
        ]
        result = agent._find_recent_user_message(messages, now, max_hours=2)
        assert result == "actually set to 75"


# ── Directive Tests ──────────────────────────────────────────────

class TestDirective:
    def _ts_minutes_ago(self, minutes):
        utc_now = datetime.now(timezone.utc)
        return (utc_now - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")

    def _activate_user_message(self, cycles_ago=0):
        """Simulate a user message that arrived cycles_ago eval cycles back."""
        agent._evaluation_counter = 10
        agent._user_message_eval_counter = 10 - cycles_ago

    def test_user_message_has_zone_routing(self):
        """When user message exists, directive includes zone identification."""
        self._activate_user_message(cycles_ago=0)
        state = _make_thermo_state(name="Upstairs", mode="cooling", indoor_temp=76.0)
        now = datetime(2025, 7, 15, 14, 0)
        comfort = agent._config["comfort"]
        sched = agent._config["schedule"]
        messages = [{"text": "set bedroom to 75", "timestamp": self._ts_minutes_ago(5)}]

        with patch("agent.weather") as mock_weather:
            mock_weather.get_forecast_analysis.return_value = None
            directive = agent._build_directive(state, MagicMock(current_temp=95.0),
                                               now, comfort, sched, messages)

        assert "YOUR ZONE: Upstairs" in directive
        assert "OTHER ZONE:" in directive

    def test_user_message_has_both_all_handling(self):
        """Directive includes guidance for 'both'/'all' keywords."""
        self._activate_user_message(cycles_ago=0)
        state = _make_thermo_state(name="Downstairs", mode="cooling", indoor_temp=76.0)
        now = datetime(2025, 7, 15, 14, 0)
        comfort = agent._config["comfort"]
        sched = agent._config["schedule"]
        messages = [{"text": "set both to 72", "timestamp": self._ts_minutes_ago(5)}]

        with patch("agent.weather") as mock_weather:
            mock_weather.get_forecast_analysis.return_value = None
            directive = agent._build_directive(state, MagicMock(current_temp=95.0),
                                               now, comfort, sched, messages)

        assert "'both'" in directive and "'all'" in directive

    def test_user_message_active_on_second_cycle(self):
        """User message is still honored on the cycle after it arrived."""
        self._activate_user_message(cycles_ago=1)
        state = _make_thermo_state(name="Upstairs", mode="cooling", indoor_temp=76.0)
        now = datetime(2025, 7, 15, 14, 0)
        comfort = agent._config["comfort"]
        sched = agent._config["schedule"]
        messages = [{"text": "set to 75", "timestamp": self._ts_minutes_ago(5)}]

        with patch("agent.weather") as mock_weather:
            mock_weather.get_forecast_analysis.return_value = None
            directive = agent._build_directive(state, MagicMock(current_temp=95.0),
                                               now, comfort, sched, messages)

        assert "RULE 1:" in directive
        assert "set to 75" in directive

    def test_user_message_expired_after_two_cycles(self):
        """User message is disregarded after 2 evaluation cycles."""
        self._activate_user_message(cycles_ago=2)
        state = _make_thermo_state(mode="cooling", indoor_temp=76.0)
        now = datetime(2025, 7, 15, 14, 0)
        comfort = agent._config["comfort"]
        sched = agent._config["schedule"]
        messages = [{"text": "set to 75", "timestamp": self._ts_minutes_ago(5)}]

        with patch("agent.weather") as mock_weather:
            mock_weather.get_forecast_analysis.return_value = None
            directive = agent._build_directive(state, MagicMock(current_temp=95.0),
                                               now, comfort, sched, messages)

        assert "PRIORITY" not in directive
        assert "Prefer no_change to save energy" in directive

    def test_no_message_has_energy_saving(self):
        """When no user message, directive includes energy-saving preference."""
        state = _make_thermo_state(mode="cooling", indoor_temp=76.0)
        now = datetime(2025, 7, 15, 14, 0)  # Awake period, no user message
        comfort = agent._config["comfort"]
        sched = agent._config["schedule"]

        with patch("agent.weather") as mock_weather:
            mock_weather.get_forecast_analysis.return_value = None
            directive = agent._build_directive(state, MagicMock(current_temp=85.0),
                                               now, comfort, sched, [])

        assert "Prefer no_change to save energy" in directive

    def test_comfort_range_is_guide(self):
        """Comfort range says 'guide only' so LLM doesn't treat it as hard limit."""
        state = _make_thermo_state(mode="cooling", indoor_temp=76.0)
        now = datetime(2025, 7, 15, 14, 0)
        comfort = agent._config["comfort"]
        sched = agent._config["schedule"]

        with patch("agent.weather") as mock_weather:
            mock_weather.get_forecast_analysis.return_value = None
            directive = agent._build_directive(state, MagicMock(current_temp=85.0),
                                               now, comfort, sched, [])

        assert "guide only" in directive

    def test_winding_down_precool_hot_day(self):
        """Hot outdoor + warm indoor → pre-cool directive."""
        state = _make_thermo_state(mode="cooling", indoor_temp=79.5)
        now = datetime(2025, 7, 15, 22, 15)  # Winding down
        comfort = agent._config["comfort"]
        sched = agent._config["schedule"]

        with patch("agent.weather") as mock_weather:
            mock_weather.get_forecast_analysis.return_value = None
            directive = agent._build_directive(state, MagicMock(current_temp=90.0),
                                               now, comfort, sched, [])

        assert "Pre-cool" in directive

    def test_winding_down_no_precool_mild(self):
        """Mild outdoor → prefer no_change, don't pre-cool."""
        state = _make_thermo_state(mode="cooling", indoor_temp=76.0)
        now = datetime(2025, 7, 15, 22, 15)  # Winding down
        comfort = agent._config["comfort"]
        sched = agent._config["schedule"]

        with patch("agent.weather") as mock_weather:
            mock_weather.get_forecast_analysis.return_value = None
            directive = agent._build_directive(state, MagicMock(current_temp=72.0),
                                               now, comfort, sched, [])

        assert "Prefer no_change" in directive
        assert "Pre-cool" not in directive

    def test_no_work_period(self):
        """Verify 'work hours' text never appears in directives."""
        state = _make_thermo_state(mode="cooling", indoor_temp=76.0)
        comfort = agent._config["comfort"]
        sched = agent._config["schedule"]

        # Test across many different times — none should produce "work hours"
        for hour in range(7, 23):
            now = datetime(2025, 7, 14, hour, 0)  # Monday
            with patch("agent.weather") as mock_weather:
                mock_weather.get_forecast_analysis.return_value = None
                directive = agent._build_directive(state, MagicMock(current_temp=85.0),
                                                   now, comfort, sched, [])
            assert "Work hours" not in directive, \
                f"'Work hours' found at {hour}:00 — {directive}"
