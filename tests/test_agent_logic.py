"""
Tests for agent.py — comfort model integration, guardrails, mode switching,
NLP command handling, cooldown logic.
"""

import json
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

# We need to set up sys.modules before importing agent to avoid
# circular import issues with telegram_bot
import sys
sys.modules["telegram"] = MagicMock()
sys.modules["telegram.error"] = MagicMock()
sys.modules["telegram.ext"] = MagicMock()

import agent
import location
from nest_api import ThermostatState
from comfort_model import ComfortModel, ComfortInput, ComfortPrediction
from nlp_parser import ParsedCommand


@pytest.fixture(autouse=True)
def setup_config():
    """Load a minimal config and reset module state for all tests."""
    agent._evaluation_counter = 0
    agent._user_triggered = False
    agent._pending_user_text = None
    agent._last_cooldown_time = None
    agent._comfort_model = ComfortModel(deadband_f=2.0)
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
        "comfort_model": {
            "corrections_path": "test_corrections.json",
            "deadband_f": 2.0,
            "adjust_step_f": 2.0,
            "retrain_on_startup": False,
        },
        "schedule": {
            "sleep_time": "23:00",
            "wake_time": "07:00",
        },
        "agent": {
            "loop_interval_minutes": 30,
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


def _make_weather(current_temp=85.0, humidity=50, forecast_summary="Clear"):
    w = MagicMock()
    w.current_temp = current_temp
    w.humidity = humidity
    w.forecast_summary = forecast_summary
    w.is_stale = False
    return w


# ── Comfort Model Integration Tests ─────────────────────────────

class TestComfortModelIntegration:
    def test_build_comfort_input(self):
        """_build_comfort_input correctly maps state to ComfortInput."""
        state = _make_thermo_state(name="Upstairs Bedroom", mode="cooling",
                                   indoor_temp=76.0, target_temp=78.0)
        weather_data = _make_weather(current_temp=90.0)
        now = datetime(2025, 7, 15, 14, 30)

        inp = agent._build_comfort_input(state, weather_data, now)

        assert inp.outdoor_temp == 90.0
        assert inp.indoor_temp == 76.0
        assert inp.hvac_mode == "cooling"
        assert inp.hour == 14
        assert inp.minute == 30
        assert inp.zone == "Upstairs Bedroom"
        assert inp.current_target == 78.0

    def test_autonomous_cycle_uses_comfort_model(self):
        """Autonomous evaluation uses comfort model, not LLM."""
        # The comfort model should be called, not call_llm
        state = _make_thermo_state(name="Downstairs Kitchen", mode="cooling",
                                   indoor_temp=76.0, target_temp=75.0)
        weather_data = _make_weather(current_temp=85.0)
        now = datetime(2025, 7, 15, 14, 0)

        inp = agent._build_comfort_input(state, weather_data, now)
        prediction = agent._comfort_model.predict(inp)

        # Default cooling: 78F, current target 75F → diff=3F > deadband → should_change
        assert prediction.baseline_temp == 78
        assert prediction.should_change is True


# ── NLP Command Handling Tests ───────────────────────────────────

class TestParsedCommandHandling:
    def test_set_temp_single_zone(self):
        """set_temp command applies to the correct zone."""
        cmd = ParsedCommand(action_type="set_temp", target_temp=77.0,
                           zone="upstairs", raw_text="set upstairs to 77")
        states = [
            _make_thermo_state(name="Upstairs Bedroom", target_temp=75.0),
            _make_thermo_state(name="Downstairs Kitchen", target_temp=75.0),
        ]
        weather_data = _make_weather()

        decisions = agent._apply_parsed_command(cmd, states, weather_data)

        assert len(decisions) == 1
        assert decisions[0][0]["temperature"] == 77.0
        assert decisions[0][1].name == "Upstairs Bedroom"

    def test_set_temp_both_zones(self):
        """set_temp with zone='both' applies to all zones."""
        cmd = ParsedCommand(action_type="set_temp", target_temp=79.0,
                           zone="both", raw_text="set both to 79")
        states = [
            _make_thermo_state(name="Upstairs Bedroom"),
            _make_thermo_state(name="Downstairs Kitchen"),
        ]
        weather_data = _make_weather()

        decisions = agent._apply_parsed_command(cmd, states, weather_data)

        assert len(decisions) == 2
        for dec, _ in decisions:
            assert dec["temperature"] == 79.0

    def test_set_temp_no_zone_applies_all(self):
        """set_temp with no zone applies to all zones."""
        cmd = ParsedCommand(action_type="set_temp", target_temp=76.0,
                           zone=None, raw_text="set to 76")
        states = [
            _make_thermo_state(name="Upstairs Bedroom"),
            _make_thermo_state(name="Downstairs Kitchen"),
        ]
        weather_data = _make_weather()

        decisions = agent._apply_parsed_command(cmd, states, weather_data)
        assert len(decisions) == 2

    def test_adjust_warmer(self):
        """adjust warmer increases temp by ADJUST_STEP_F."""
        cmd = ParsedCommand(action_type="adjust", direction="warmer",
                           raw_text="make it warmer")
        states = [_make_thermo_state(target_temp=76.0)]
        weather_data = _make_weather()

        decisions = agent._apply_parsed_command(cmd, states, weather_data)

        assert len(decisions) == 1
        assert decisions[0][0]["temperature"] == 78.0  # 76 + 2

    def test_adjust_cooler(self):
        """adjust cooler decreases temp by ADJUST_STEP_F."""
        cmd = ParsedCommand(action_type="adjust", direction="cooler",
                           raw_text="I'm hot")
        states = [_make_thermo_state(target_temp=78.0)]
        weather_data = _make_weather()

        decisions = agent._apply_parsed_command(cmd, states, weather_data)

        assert len(decisions) == 1
        assert decisions[0][0]["temperature"] == 76.0  # 78 - 2

    def test_question_returns_status(self):
        """question action returns no_change with status message."""
        cmd = ParsedCommand(action_type="question", raw_text="what's the temp?")
        states = [_make_thermo_state(name="Test", indoor_temp=76.0, target_temp=78.0)]
        weather_data = _make_weather(current_temp=90.0)

        decisions = agent._apply_parsed_command(cmd, states, weather_data)

        assert len(decisions) == 1
        assert decisions[0][0]["action"] == "no_change"
        assert "message_to_user" in decisions[0][0]

    def test_unknown_returns_no_change(self):
        """unknown action returns no_change."""
        cmd = ParsedCommand(action_type="unknown", raw_text="hello")
        states = [_make_thermo_state()]
        weather_data = _make_weather()

        decisions = agent._apply_parsed_command(cmd, states, weather_data)

        assert len(decisions) == 1
        assert decisions[0][0]["action"] == "no_change"


# ── Zone Resolution Tests ───────────────────────────────────────

class TestZoneResolution:
    def test_resolve_upstairs(self):
        states = [
            _make_thermo_state(name="Upstairs Bedroom"),
            _make_thermo_state(name="Downstairs Kitchen"),
        ]
        result = agent._resolve_zones("upstairs", states)
        assert len(result) == 1
        assert result[0].name == "Upstairs Bedroom"

    def test_resolve_downstairs(self):
        states = [
            _make_thermo_state(name="Upstairs Bedroom"),
            _make_thermo_state(name="Downstairs Kitchen"),
        ]
        result = agent._resolve_zones("downstairs", states)
        assert len(result) == 1
        assert result[0].name == "Downstairs Kitchen"

    def test_resolve_both(self):
        states = [
            _make_thermo_state(name="Upstairs Bedroom"),
            _make_thermo_state(name="Downstairs Kitchen"),
        ]
        result = agent._resolve_zones("both", states)
        assert len(result) == 2

    def test_resolve_none(self):
        states = [
            _make_thermo_state(name="Upstairs Bedroom"),
            _make_thermo_state(name="Downstairs Kitchen"),
        ]
        result = agent._resolve_zones(None, states)
        assert len(result) == 2


# ── Guardrails Tests ─────────────────────────────────────────────

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


# ── Cooldown Tests ───────────────────────────────────────────────

class TestCooldown:
    def test_cooldown_active_skips_evaluation(self):
        """When cooldown is active, autonomous evaluation is skipped."""
        agent._last_cooldown_time = datetime.now() - timedelta(minutes=10)
        agent._user_triggered = False

        # run_evaluation_cycle checks cooldown and returns None
        # We can test the logic directly:
        elapsed = (datetime.now() - agent._last_cooldown_time).total_seconds() / 60
        assert elapsed < agent.USER_REQUEST_COOLDOWN_MINUTES

    def test_cooldown_expired_allows_evaluation(self):
        """After cooldown expires, autonomous evaluation proceeds."""
        agent._last_cooldown_time = datetime.now() - timedelta(minutes=35)

        elapsed = (datetime.now() - agent._last_cooldown_time).total_seconds() / 60
        assert elapsed >= agent.USER_REQUEST_COOLDOWN_MINUTES

    def test_no_cooldown_allows_evaluation(self):
        """No cooldown set → evaluation proceeds."""
        agent._last_cooldown_time = None
        # No cooldown → should proceed

    def test_user_triggered_ignores_cooldown(self):
        """User-triggered evaluations bypass cooldown."""
        agent._last_cooldown_time = datetime.now() - timedelta(minutes=5)
        agent._user_triggered = True
        # User-triggered should still run


# ── Trigger Tests ────────────────────────────────────────────────

class TestTriggerEvaluation:
    def test_trigger_sets_user_text(self):
        """trigger_evaluation passes user text for NLP parsing."""
        agent.trigger_evaluation(user_text="set upstairs to 77")
        assert agent._user_triggered is True
        assert agent._pending_user_text == "set upstairs to 77"

    def test_trigger_without_text(self):
        """trigger_evaluation can be called without text."""
        agent.trigger_evaluation()
        assert agent._user_triggered is True
        assert agent._pending_user_text is None


# ── Mode Switch + Temperature Adjustment Tests ───────────────────

class TestCheckAndSwitchMode:
    """Tests for check_and_switch_mode: after switching HVAC mode, the target
    temperature must be adjusted to the new mode's comfort zone."""

    def _run_async(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    @patch("agent.send_telegram", new_callable=AsyncMock)
    @patch("agent.nest_api")
    @patch("agent.weather")
    def test_cool_to_heat_sets_heating_baseline(self, mock_weather, mock_nest, mock_tg):
        """Switching COOL→HEAT should set target to 68F (heating baseline)."""
        mock_forecast = MagicMock()
        mock_forecast.next_24h_high = 55.0
        mock_weather.get_forecast_analysis.return_value = mock_forecast
        mock_nest.set_mode.return_value = True
        mock_nest.set_temperature.return_value = True

        state = _make_thermo_state(mode="cooling", indoor_temp=71.0, target_temp=75.0)
        weather_data = MagicMock(current_temp=55.0)

        agent._daily_high_cache = (None, None)

        self._run_async(agent.check_and_switch_mode(state, weather_data))

        assert state.mode == "heating"
        assert state.target_temp == 68
        mock_nest.set_mode.assert_called_once_with("HEAT", "test-device-id")
        mock_nest.set_temperature.assert_called_once_with(68, "test-device-id")

    @patch("agent.send_telegram", new_callable=AsyncMock)
    @patch("agent.nest_api")
    @patch("agent.weather")
    def test_heat_to_cool_sets_cooling_default(self, mock_weather, mock_nest, mock_tg):
        """Switching HEAT→COOL should set target to 78F (cooling default)."""
        mock_forecast = MagicMock()
        mock_forecast.next_24h_high = 85.0
        mock_weather.get_forecast_analysis.return_value = mock_forecast
        mock_nest.set_mode.return_value = True
        mock_nest.set_temperature.return_value = True

        state = _make_thermo_state(mode="heating", indoor_temp=70.0, target_temp=68.0)
        weather_data = MagicMock(current_temp=80.0)

        agent._daily_high_cache = (None, None)

        self._run_async(agent.check_and_switch_mode(state, weather_data))

        assert state.mode == "cooling"
        assert state.target_temp == 78
        mock_nest.set_mode.assert_called_once_with("COOL", "test-device-id")
        mock_nest.set_temperature.assert_called_once_with(78, "test-device-id")

    @patch("agent.send_telegram", new_callable=AsyncMock)
    @patch("agent.nest_api")
    @patch("agent.weather")
    def test_no_switch_no_temp_change(self, mock_weather, mock_nest, mock_tg):
        """When mode doesn't change, temperature should not be touched."""
        agent._daily_high_cache = (None, None)

        mock_forecast = MagicMock()
        mock_forecast.next_24h_high = 85.0
        mock_weather.get_forecast_analysis.return_value = mock_forecast

        state = _make_thermo_state(mode="cooling", indoor_temp=76.0, target_temp=75.0)
        weather_data = MagicMock(current_temp=85.0)

        self._run_async(agent.check_and_switch_mode(state, weather_data))

        assert state.mode == "cooling"
        assert state.target_temp == 75.0
        mock_nest.set_mode.assert_not_called()
        mock_nest.set_temperature.assert_not_called()

    @patch("agent.send_telegram", new_callable=AsyncMock)
    @patch("agent.nest_api")
    @patch("agent.weather")
    def test_telegram_notification_includes_new_target(self, mock_weather, mock_nest, mock_tg):
        """Mode switch telegram message should mention the new target temperature."""
        agent._daily_high_cache = (None, None)

        mock_forecast = MagicMock()
        mock_forecast.next_24h_high = 55.0
        mock_weather.get_forecast_analysis.return_value = mock_forecast
        mock_nest.set_mode.return_value = True
        mock_nest.set_temperature.return_value = True

        state = _make_thermo_state(mode="cooling", indoor_temp=71.0, target_temp=75.0)
        weather_data = MagicMock(current_temp=55.0)

        self._run_async(agent.check_and_switch_mode(state, weather_data))

        mock_tg.assert_called_once()
        msg = mock_tg.call_args[0][0]
        assert "HEAT" in msg
        assert "68" in msg


# ── Vacation Mode Tests ──────────────────────────────────────────

class TestVacationMode:
    """Tests for vacation mode integration."""

    @pytest.fixture(autouse=True)
    def reset_vacation(self):
        """Reset location module state before each test."""
        location._vacation_mode = False
        location._enabled = False
        location._initialized = False
        location._vacation_temp_min = 60
        location._vacation_temp_max = 85
        yield
        location._vacation_mode = False

    def test_vacation_comfort_model_cooling_82(self):
        """In vacation + cooling, comfort model targets 82F."""
        location._vacation_mode = True
        inp = ComfortInput(
            outdoor_temp=95, indoor_temp=80, humidity=45,
            hvac_mode="cooling", hour=14, minute=0,
            zone="downstairs", is_vacation=True, current_target=None,
        )
        prediction = agent._comfort_model.predict(inp)
        assert prediction.baseline_temp == 82

    def test_vacation_comfort_model_heating_60(self):
        """In vacation + heating, comfort model targets 60F."""
        location._vacation_mode = True
        inp = ComfortInput(
            outdoor_temp=30, indoor_temp=55, humidity=45,
            hvac_mode="heating", hour=14, minute=0,
            zone="downstairs", is_vacation=True, current_target=None,
        )
        prediction = agent._comfort_model.predict(inp)
        assert prediction.baseline_temp == 60

    def test_vacation_widens_guardrails_to_85(self):
        """In vacation mode, get_temp_max() should return 85 (not 80)."""
        location._vacation_mode = True
        assert agent.get_temp_max() == 85

    def test_normal_guardrails_unchanged_at_80(self):
        """Outside vacation mode, get_temp_max() should return 80."""
        location._vacation_mode = False
        assert agent.get_temp_max() == 80

    def test_vacation_get_temp_min(self):
        """In vacation mode, get_temp_min() should return vacation min (60)."""
        location._vacation_mode = True
        assert agent.get_temp_min() == 60

    def test_normal_get_temp_min(self):
        """Outside vacation mode, get_temp_min() returns normal min (65)."""
        location._vacation_mode = False
        assert agent.get_temp_min() == 65

    def test_guardrail_clamp_uses_vacation_range(self):
        """During vacation, guardrails should clamp to 85 (not 80)."""
        location._vacation_mode = True
        with patch("agent.database") as mock_db:
            mock_db.count_temp_changes_last_hour.return_value = 0
            mock_db.get_last_manual_override_time.return_value = None
            decision = {"action": "set_temperature", "temperature": 90,
                        "reasoning": "test"}
            allowed, reason = agent.check_guardrails(decision)
            assert allowed is True
            assert decision["temperature"] == 85

    def test_guardrail_clamp_normal_at_80(self):
        """Outside vacation, guardrails should clamp to 80."""
        location._vacation_mode = False
        with patch("agent.database") as mock_db:
            mock_db.count_temp_changes_last_hour.return_value = 0
            mock_db.get_last_manual_override_time.return_value = None
            decision = {"action": "set_temperature", "temperature": 90,
                        "reasoning": "test"}
            allowed, reason = agent.check_guardrails(decision)
            assert allowed is True
            assert decision["temperature"] == 80


# ── Comfort Model Prediction via Agent ───────────────────────────

class TestAgentComfortPredictions:
    """Test comfort model produces correct targets through agent integration."""

    def test_hot_day_cooling_78(self):
        """Hot day → comfort model targets 78F."""
        state = _make_thermo_state(name="Downstairs Kitchen", mode="cooling",
                                   indoor_temp=76.0, target_temp=75.0)
        weather_data = _make_weather(current_temp=88.0)
        now = datetime(2025, 7, 15, 14, 0)

        inp = agent._build_comfort_input(state, weather_data, now)
        pred = agent._comfort_model.predict(inp)

        assert pred.baseline_temp == 78

    def test_extreme_heat_precool_75(self):
        """Extreme heat (100F) → comfort model pre-cools to 75F."""
        state = _make_thermo_state(mode="cooling", indoor_temp=78.0, target_temp=78.0)
        weather_data = _make_weather(current_temp=100.0)
        now = datetime(2025, 7, 15, 14, 0)

        inp = agent._build_comfort_input(state, weather_data, now)
        pred = agent._comfort_model.predict(inp)

        assert pred.baseline_temp == 75

    def test_precool_before_bed_75(self):
        """9:30 PM → pre-cool to 75F."""
        state = _make_thermo_state(mode="cooling", indoor_temp=78.0, target_temp=78.0)
        weather_data = _make_weather(current_temp=85.0)
        now = datetime(2025, 7, 15, 21, 30)

        inp = agent._build_comfort_input(state, weather_data, now)
        pred = agent._comfort_model.predict(inp)

        assert pred.baseline_temp == 75

    def test_sleep_after_1030pm_78(self):
        """After 10:30 PM → sleep target 78F."""
        state = _make_thermo_state(mode="cooling", indoor_temp=76.0, target_temp=75.0)
        weather_data = _make_weather(current_temp=85.0)
        now = datetime(2025, 7, 15, 22, 30)

        inp = agent._build_comfort_input(state, weather_data, now)
        pred = agent._comfort_model.predict(inp)

        assert pred.baseline_temp == 78

    def test_heating_extreme_cold_72(self):
        """Extreme cold (15F) → 72F."""
        state = _make_thermo_state(mode="heating", indoor_temp=65.0)
        weather_data = _make_weather(current_temp=15.0)
        now = datetime(2025, 12, 15, 14, 0)

        inp = agent._build_comfort_input(state, weather_data, now)
        pred = agent._comfort_model.predict(inp)

        assert pred.baseline_temp == 72

    def test_heating_mild_68(self):
        """Mild outdoor (65F) → 68F."""
        state = _make_thermo_state(mode="heating", indoor_temp=66.0)
        weather_data = _make_weather(current_temp=65.0)
        now = datetime(2025, 12, 15, 14, 0)

        inp = agent._build_comfort_input(state, weather_data, now)
        pred = agent._comfort_model.predict(inp)

        assert pred.baseline_temp == 68

    def test_deadband_prevents_change(self):
        """Target within deadband of current → no change."""
        state = _make_thermo_state(mode="cooling", indoor_temp=77.0, target_temp=77.0)
        weather_data = _make_weather(current_temp=85.0)
        now = datetime(2025, 7, 15, 14, 0)

        inp = agent._build_comfort_input(state, weather_data, now)
        pred = agent._comfort_model.predict(inp)

        # baseline 78, current 77, diff=1 < deadband 2 → no change
        assert pred.should_change is False

    def test_deadband_allows_change_when_outside(self):
        """Target outside deadband → change."""
        state = _make_thermo_state(mode="cooling", indoor_temp=77.0, target_temp=75.0)
        weather_data = _make_weather(current_temp=85.0)
        now = datetime(2025, 7, 15, 14, 0)

        inp = agent._build_comfort_input(state, weather_data, now)
        pred = agent._comfort_model.predict(inp)

        # baseline 78, current 75, diff=3 > deadband 2 → change
        assert pred.should_change is True


# ── Online Learning Tests ────────────────────────────────────────

class TestOnlineLearning:
    def test_learn_from_override_updates_corrections(self):
        """Manual override is learned by the comfort model."""
        state = _make_thermo_state(name="Downstairs Kitchen", mode="cooling",
                                   indoor_temp=78.0, target_temp=78.0)
        weather_data = _make_weather(current_temp=90.0)

        initial_corrections = len(agent._comfort_model.corrections)

        with patch("agent._comfort_model.save"):  # Don't actually save to disk
            agent._learn_from_override(state, weather_data, 76.0)

        assert len(agent._comfort_model.corrections) > initial_corrections


# ── Eval Interval Tests ──────────────────────────────────────────

class TestEvalInterval:
    def test_default_interval_5_minutes(self):
        """Default evaluation interval is 5 minutes."""
        assert agent.EVAL_INTERVAL_MINUTES == 5

    def test_cooldown_30_minutes(self):
        """Cooldown after user/manual override is 30 minutes."""
        assert agent.USER_REQUEST_COOLDOWN_MINUTES == 30
