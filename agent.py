#!/usr/bin/env python3
"""
AI Thermostat Agent — The Brain
Hybrid architecture: comfort model for autonomous decisions, LLM for NLP only.
"""

import asyncio
import json
import logging
import sys
import time
import requests
import yaml
from datetime import datetime, timedelta
from typing import Optional, Tuple

# Fix __main__ double-import: when run as "python agent.py", this module
# is loaded as "__main__". But telegram_bot.py does "import agent" which
# creates a SECOND copy. This ensures both names point to the same object,
# so set_telegram_send_fn() updates the variable agent_loop() reads.
if __name__ == "__main__":
    sys.modules["agent"] = sys.modules[__name__]

import database
import location
import weather
import nest_api
import llm_server
from comfort_model import ComfortModel, ComfortInput
import nlp_parser

logger = logging.getLogger(__name__)

# ── Hard-coded guardrails (not configurable) ──────────────────────
TEMP_MIN_NORMAL = 65
TEMP_MAX_NORMAL = 80
MAX_CHANGES_PER_HOUR = 6
MANUAL_OVERRIDE_BACKOFF_MINUTES = 30   # Cooldown after manual override
USER_REQUEST_COOLDOWN_MINUTES = 30     # Cooldown after user message
EVAL_INTERVAL_MINUTES = 5             # Autonomous evaluation interval
ADJUST_STEP_F = 2.0                   # Temperature step for "warmer"/"cooler"


def get_temp_min() -> int:
    """Return the current minimum guardrail temperature (wider in vacation mode)."""
    if location.is_vacation_mode():
        return location.get_vacation_temp_min()
    return TEMP_MIN_NORMAL


def get_temp_max() -> int:
    """Return the current maximum guardrail temperature (wider in vacation mode)."""
    if location.is_vacation_mode():
        return location.get_vacation_temp_max()
    return TEMP_MAX_NORMAL


# ── Module-level state ────────────────────────────────────────────
_config: dict = {}
_comfort_model: Optional[ComfortModel] = None
_last_evaluation_result: Optional[dict] = None
_evaluation_counter: int = 0
_user_triggered: bool = False
_pending_user_text: Optional[str] = None  # User message waiting to be parsed
_telegram_send_fn = None
_last_cooldown_time: Optional[datetime] = None  # Cooldown after user/manual override
_daily_high_cache: tuple = (None, None)


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    global _config
    with open(path) as f:
        _config = yaml.safe_load(f)
    return _config


def get_config() -> dict:
    return _config


def set_telegram_send_fn(fn):
    """Register the Telegram send function (called by telegram_bot)."""
    global _telegram_send_fn
    _telegram_send_fn = fn


async def send_telegram(message: str):
    """Send a message via Telegram if the send function is registered."""
    if _telegram_send_fn:
        try:
            logger.info("Sending Telegram: %s", message[:80])
            await _telegram_send_fn(message)
        except Exception as e:
            logger.error("Failed to send Telegram message: %s", e)
    else:
        logger.warning("Telegram send function not registered — message dropped: %s", message[:80])


def call_llm(system_prompt: str) -> Tuple[Optional[str], bool]:
    """
    Call the local LLM via llama.cpp /v1/chat/completions.
    Returns (response_text, is_valid_json).
    Retries once on JSON failure.
    """
    llm_cfg = _config["llm"]
    payload = {
        "model": llm_cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Parse the message above."}
        ],
        "temperature": llm_cfg["temperature"],
        "top_p": llm_cfg["top_p"],
        "max_tokens": llm_cfg["max_tokens"]
    }

    for attempt in range(2):
        try:
            resp = requests.post(
                llm_cfg["endpoint"],
                json=payload,
                timeout=llm_cfg["timeout_seconds"]
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            # Strip markdown code fences (e.g. ```json ... ```)
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                content = "\n".join(lines).strip()

            # Try to parse as JSON
            try:
                json.loads(content)
                return content, True
            except json.JSONDecodeError:
                if attempt == 0:
                    logger.warning("LLM returned invalid JSON (attempt 1), retrying...")
                    continue
                return content, False

        except requests.exceptions.Timeout:
            logger.error("LLM request timed out (attempt %d)", attempt + 1)
            if attempt == 0:
                continue
            return "TIMEOUT", False
        except requests.exceptions.ConnectionError:
            logger.error("LLM connection error — is llama-server running on %s?",
                         llm_cfg["endpoint"])
            return "CONNECTION_ERROR", False
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return f"ERROR: {e}", False

    return None, False


# ── Comfort Model Integration ────────────────────────────────────

def init_comfort_model(config: dict):
    """Initialize and train the comfort model on startup."""
    global _comfort_model
    cm_cfg = config.get("comfort_model", {})
    comfort = config.get("comfort", {})
    schedule = config.get("schedule", {})

    _comfort_model = ComfortModel(
        deadband_f=cm_cfg.get("deadband_f", 2.0),
        summer_range=comfort.get("summer_range", [75, 80]),
        winter_range=comfort.get("winter_range", [68, 72]),
        sleep_time=schedule.get("sleep_time", "23:00"),
        wake_time=schedule.get("wake_time", "07:00"),
        precool_time=schedule.get("precool_time"),
        sleep_cool_temp=comfort.get("sleep_cool_temp"),
    )

    # Load saved corrections
    corrections_path = cm_cfg.get("corrections_path", "comfort_corrections.json")
    _comfort_model.load(corrections_path)

    # Train from historical data if configured
    if cm_cfg.get("retrain_on_startup", True):
        _run_initial_training()

    logger.info("Comfort model initialized (deadband=%.1fF, corrections=%d)",
                _comfort_model.deadband_f, len(_comfort_model.corrections))


def _run_initial_training():
    """Train comfort model from historical database data."""
    try:
        # Train zone offsets from climate data
        climate_data = database.get_all_climate_data()
        if climate_data:
            _comfort_model.train_zone_offsets(climate_data)
            logger.info("Zone offsets trained from %d climate records", len(climate_data))

        # Train corrections from manual overrides
        overrides = database.get_manual_overrides_with_context()
        if overrides:
            _comfort_model.train_from_overrides(overrides)
            logger.info("Corrections trained from %d overrides", len(overrides))

        # Save trained model
        cm_cfg = _config.get("comfort_model", {})
        _comfort_model.save(cm_cfg.get("corrections_path", "comfort_corrections.json"))

    except Exception as e:
        logger.warning("Initial training failed (will use baseline only): %s", e)


def _build_comfort_input(thermo_state, weather_data, now: datetime) -> ComfortInput:
    """Build a ComfortInput from current state."""
    return ComfortInput(
        outdoor_temp=weather_data.current_temp,
        indoor_temp=thermo_state.indoor_temp,
        humidity=thermo_state.humidity,
        hvac_mode=thermo_state.mode,
        hour=now.hour,
        minute=now.minute,
        zone=thermo_state.name,
        is_vacation=location.is_vacation_mode(),
        current_target=thermo_state.target_temp,
    )


def _learn_from_override(thermo_state, weather_data, user_target: float):
    """Feed a manual override or user request to the comfort model for online learning."""
    if _comfort_model is None:
        return
    try:
        now = datetime.now()
        inp = _build_comfort_input(thermo_state, weather_data, now)
        _comfort_model.learn_override(inp, user_target)

        # Save updated corrections
        cm_cfg = _config.get("comfort_model", {})
        _comfort_model.save(cm_cfg.get("corrections_path", "comfort_corrections.json"))
    except Exception as e:
        logger.warning("Failed to learn from override: %s", e)


# ── Parsed command handling ──────────────────────────────────────

def _resolve_zones(zone_str: Optional[str], all_states: list) -> list:
    """Resolve a zone string to a list of thermostat states."""
    if zone_str is None or zone_str == "both":
        return all_states

    for state in all_states:
        name_lower = state.name.lower()
        if zone_str == "upstairs" and "upstairs" in name_lower:
            return [state]
        if zone_str == "downstairs" and "downstairs" in name_lower:
            return [state]

    # No match — apply to all zones
    return all_states


def _apply_parsed_command(command, all_states: list, weather_data) -> list:
    """
    Apply a parsed NLP command to thermostat states.
    Returns list of (decision_dict, thermo_state) tuples.
    """
    decisions = []

    if command.action_type == "set_temp" and command.target_temp is not None:
        targets = _resolve_zones(command.zone, all_states)
        for state in targets:
            decisions.append(({
                "action": "set_temperature",
                "temperature": command.target_temp,
                "reasoning": f"User requested {command.target_temp:.0f}F"
                             + (f" for {command.zone}" if command.zone else ""),
            }, state))

    elif command.action_type == "adjust":
        targets = _resolve_zones(command.zone, all_states)
        for state in targets:
            current = state.target_temp or 75
            if command.direction == "warmer":
                new_temp = current + ADJUST_STEP_F
            elif command.direction == "cooler":
                new_temp = current - ADJUST_STEP_F
            else:
                continue
            decisions.append(({
                "action": "set_temperature",
                "temperature": new_temp,
                "reasoning": f"User wants it {command.direction} ({current:.0f}F → {new_temp:.0f}F)",
            }, state))

    elif command.action_type in ("question", "status_query"):
        # No temperature change — return status info
        decisions.append(({
            "action": "no_change",
            "reasoning": "User asked a question",
            "message_to_user": _build_status_message(all_states, weather_data),
        }, all_states[0] if all_states else None))

    else:
        # Unknown — no change
        if all_states:
            decisions.append(({
                "action": "no_change",
                "reasoning": f"Could not parse: {command.raw_text}",
                "message_to_user": f"I didn't understand that. Current status:\n"
                                   + _build_status_message(all_states, weather_data),
            }, all_states[0]))

    return decisions


def _build_status_message(all_states: list, weather_data) -> str:
    """Build a status summary for user questions."""
    lines = []
    for s in all_states:
        lines.append(f"{s.name}: {s.indoor_temp:.0f}F, target {s.target_temp or '?'}F, "
                     f"{'running' if s.hvac_running else 'idle'}")
    lines.append(f"Outdoor: {weather_data.current_temp:.0f}F")
    return "\n".join(lines)


# ── Guardrails ───────────────────────────────────────────────────

def check_guardrails(decision: dict, user_triggered: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Enforce safety rules before executing a decision.
    Returns (allowed, reason_if_blocked).
    User-triggered evaluations skip the rate limit (user explicitly asked).
    """
    if decision["action"] != "set_temperature":
        return True, None

    temp = decision.get("temperature")

    # Temperature bounds — clamp instead of block
    if temp is not None and not (get_temp_min() <= temp <= get_temp_max()):
        clamped = max(get_temp_min(), min(get_temp_max(), temp))
        logger.warning("Guardrail clamp: requested %sF, clamped to %sF", temp, clamped)
        decision["temperature"] = clamped
        decision["reasoning"] = f"[Clamped {temp}F->{clamped}F] " + decision.get("reasoning", "")

    # Rate limit (skipped for user-triggered evaluations)
    if not user_triggered:
        changes = database.count_temp_changes_last_hour()
        if changes >= MAX_CHANGES_PER_HOUR:
            return False, f"Rate limit: {changes} changes in last hour (max {MAX_CHANGES_PER_HOUR})"

    # Manual override backoff (skipped for user-triggered evaluations)
    if not user_triggered:
        last_override = database.get_last_manual_override_time()
        if last_override:
            elapsed = (datetime.utcnow() - last_override).total_seconds() / 60
            if elapsed < MANUAL_OVERRIDE_BACKOFF_MINUTES:
                remaining = int(MANUAL_OVERRIDE_BACKOFF_MINUTES - elapsed)
                return False, f"Manual override backoff: {remaining} minutes remaining"

    return True, None


# ── Decision execution ───────────────────────────────────────────

async def execute_decision(decision: dict, thermo_state, weather_data, raw_response: str,
                           user_triggered: bool = False):
    """Execute the agent's decision — call Nest API if needed, log to DB."""
    reasoning = decision.get("reasoning", "")
    temperature = decision.get("temperature")
    zone = thermo_state.name

    # Skip redundant temperature changes BEFORE logging
    if decision["action"] == "set_temperature" and temperature is not None:
        current_target = thermo_state.target_temp
        if current_target is not None and abs(temperature - current_target) < 0.5:
            logger.info("[%s] Skipping — target already at %.0fF", zone, current_target)
            decision["action"] = "no_change"
            decision["reasoning"] = f"Already at {current_target:.0f}F (requested {temperature:.0f}F)"

    action = decision["action"]

    # Log the decision (now reflects the actual action taken)
    database.log_decision(
        indoor_temp=thermo_state.indoor_temp,
        outdoor_temp=weather_data.current_temp,
        action=action,
        temperature=temperature if action == "set_temperature" else None,
        reasoning=reasoning,
        raw_response=raw_response,
        zone=zone
    )

    # Log climate snapshot
    database.log_climate(
        indoor_temp=thermo_state.indoor_temp,
        indoor_humidity=thermo_state.humidity,
        outdoor_temp=weather_data.current_temp,
        outdoor_humidity=weather_data.humidity,
        forecast=weather_data.forecast_summary,
        hvac_mode=thermo_state.mode,
        hvac_running=thermo_state.hvac_running,
        target_temp=thermo_state.target_temp or 0,
        action=action,
        reasoning=reasoning,
        zone=zone
    )

    # Execute temperature change
    if action == "set_temperature" and temperature is not None:
        # Edge case: user wants to heat when system is in cool mode.
        if (user_triggered
                and thermo_state.mode == "cooling"
                and temperature > thermo_state.indoor_temp
                and 65 <= thermo_state.indoor_temp <= 70):
            logger.info("[%s] User wants %.0fF but indoor is %.0fF in cool mode — switching to HEAT",
                        zone, temperature, thermo_state.indoor_temp)
            await asyncio.to_thread(nest_api.set_mode, "HEAT", thermo_state.device_id)
            await asyncio.sleep(10)

        success = await asyncio.to_thread(
            nest_api.set_temperature, temperature, thermo_state.device_id
        )
        if success:
            logger.info("[%s] Temperature set to %.0fF (was %.0fF) — %s",
                        zone, temperature, thermo_state.target_temp or 0, reasoning)
        else:
            logger.error("[%s] Failed to set temperature to %.0fF", zone, temperature)
            database.log_error("nest_api", "set_temperature_failed",
                               f"[{zone}] Tried to set {temperature}F")

    return action


# ── Mode switching ───────────────────────────────────────────────

async def check_and_switch_mode(thermo_state, weather_data) -> str:
    """
    Check if HVAC mode needs to be switched based on forecast daily high.
    Returns the current/new mode as lowercase string ("heating", "cooling", etc).

    The daily high is locked on the first evaluation of each day to prevent
    mode flip-flopping as the rolling 24h forecast window shifts.
    """
    global _daily_high_cache
    zone = thermo_state.name
    today = datetime.now().strftime("%Y-%m-%d")

    if _daily_high_cache[0] == today:
        daily_high = _daily_high_cache[1]
    else:
        forecast = weather.get_forecast_analysis()
        daily_high = forecast.next_24h_high if forecast else weather_data.current_temp
        _daily_high_cache = (today, daily_high)
        logger.info("Daily high locked for %s: %.0fF", today, daily_high)

    desired_mode = "COOL" if daily_high >= 65 else "HEAT"
    current_mode = thermo_state.mode
    mode_map = {"COOL": "cooling", "HEAT": "heating"}

    if current_mode != mode_map.get(desired_mode):
        logger.info("[%s] Switching HVAC mode to %s (daily high %.0fF)",
                    zone, desired_mode, daily_high)
        await asyncio.to_thread(nest_api.set_mode, desired_mode, thermo_state.device_id)
        thermo_state.mode = mode_map[desired_mode]

        # Set temperature to the new mode's comfort zone
        if desired_mode == "HEAT":
            new_target = 68  # Heating baseline low
        else:
            new_target = 78  # Cooling default
        await asyncio.sleep(5)
        await asyncio.to_thread(nest_api.set_temperature, new_target, thermo_state.device_id)
        thermo_state.target_temp = new_target
        logger.info("[%s] Set target to %.0fF for %s mode comfort zone",
                    zone, new_target, desired_mode)

        await send_telegram(
            f"[{zone}] HVAC mode switched to {desired_mode} → target set to {new_target:.0f}F "
            f"(forecast daily high: {daily_high:.0f}F)"
        )

    return thermo_state.mode


# ── Weekly report ────────────────────────────────────────────────

def generate_weekly_report() -> str:
    """Generate a Sunday summary from climate_log."""
    since = datetime.utcnow() - timedelta(days=7)
    entries = database.get_climate_log_since(since)

    if not entries:
        return "No climate data recorded this week."

    temps = [e["indoor_temp"] for e in entries if e["indoor_temp"]]
    outdoor = [e["outdoor_temp"] for e in entries if e["outdoor_temp"]]
    changes = [e for e in entries if e["action"] == "set_temperature"]
    running = [e for e in entries if e["hvac_running"]]

    report = "Weekly Climate Report\n"
    report += "=" * 30 + "\n"
    if temps:
        report += f"Indoor: avg {sum(temps)/len(temps):.1f}F, "
        report += f"range {min(temps):.0f}-{max(temps):.0f}F\n"
    if outdoor:
        report += f"Outdoor: avg {sum(outdoor)/len(outdoor):.1f}F, "
        report += f"range {min(outdoor):.0f}-{max(outdoor):.0f}F\n"
    report += f"Temperature changes: {len(changes)}\n"
    report += f"HVAC active snapshots: {len(running)} of {len(entries)}\n"
    report += f"Total data points: {len(entries)}\n"

    return report


# ── Evaluation cycle ─────────────────────────────────────────────

async def run_evaluation_cycle() -> Optional[dict]:
    """
    Run one complete evaluation cycle for ALL thermostats.

    Two paths:
    - Autonomous (no user message): comfort model decides, no LLM needed
    - User-triggered: LLM parses message, command applied, 30-min cooldown
    """
    global _user_triggered, _last_cooldown_time, _pending_user_text
    user_triggered = _user_triggered
    user_text = _pending_user_text
    _user_triggered = False
    _pending_user_text = None
    logger.info("Starting evaluation cycle (user_triggered=%s)", user_triggered)

    # Cooldown check — skip autonomous evaluation if within cooldown period
    if not user_triggered and _last_cooldown_time is not None:
        elapsed = (datetime.now() - _last_cooldown_time).total_seconds() / 60
        if elapsed < USER_REQUEST_COOLDOWN_MINUTES:
            remaining = int(USER_REQUEST_COOLDOWN_MINUTES - elapsed)
            logger.info("Cooldown active: %d minutes remaining, skipping evaluation", remaining)
            return None

    last_decision = None
    _cycle_decisions = []

    try:
        # 0. ARP presence check (updates vacation mode before thermostat logic)
        location.check_arp_presence()

        # 1. Get ALL thermostat states
        all_states = await asyncio.to_thread(nest_api.get_all_thermostat_states)
        for s in all_states:
            logger.info("[%s] %.1fF, %d%% humidity, mode=%s, target=%s",
                         s.name, s.indoor_temp, s.humidity, s.mode, s.target_temp)

        # 2. Check for manual overrides — triggers cooldown + learning
        for s in all_states:
            if nest_api.detect_manual_override(s.device_id, s.target_temp):
                database.log_manual_override(
                    s.target_temp,
                    nest_api._last_known_targets.get(s.device_id, 0),
                    zone=s.name
                )
                logger.info("[%s] Manual override detected → cooldown + learning", s.name)

                # Get weather for learning
                weather_data = await asyncio.to_thread(weather.get_weather)
                _learn_from_override(s, weather_data, s.target_temp)

                _last_cooldown_time = datetime.now()
                await send_telegram(
                    f"[{s.name}] Manual override detected "
                    f"(target changed to {s.target_temp}F). "
                    f"Backing off for {MANUAL_OVERRIDE_BACKOFF_MINUTES} minutes."
                )
                return None  # Skip rest of evaluation

        # 3. Get weather (shared across zones)
        weather_data = await asyncio.to_thread(weather.get_weather)
        logger.info("Weather: %.1fF, %d%% humidity, stale=%s",
                     weather_data.current_temp, weather_data.humidity, weather_data.is_stale)

        # Check weather alerts (once)
        alerts = weather.check_weather_alerts(weather_data)
        for alert in alerts:
            logger.warning("Weather alert: %s", alert)
            await send_telegram(f"Weather Alert: {alert}")

        # ── Path A: User-triggered — LLM parses message ──
        if user_triggered and user_text:
            logger.info("User message: '%s' — starting LLM for NLP parse", user_text)

            # Start LLM, parse, stop LLM
            server_started = await asyncio.to_thread(llm_server.start)
            if server_started:
                command = await asyncio.to_thread(
                    nlp_parser.parse, user_text, call_llm
                )
                await asyncio.to_thread(llm_server.stop)
            else:
                # LLM failed — use regex fallback
                logger.warning("LLM server failed to start — using regex fallback")
                command = nlp_parser.parse(user_text)

            logger.info("Parsed command: %s (temp=%s, zone=%s, dir=%s)",
                        command.action_type, command.target_temp,
                        command.zone, command.direction)

            # Apply command
            decisions = _apply_parsed_command(command, all_states, weather_data)

            for decision, thermo_state in decisions:
                if thermo_state is None:
                    continue

                # Check and switch mode first
                await check_and_switch_mode(thermo_state, weather_data)

                # Guardrails
                allowed, block_reason = check_guardrails(decision, user_triggered=True)
                if not allowed:
                    logger.warning("[%s] Guardrail blocked: %s", thermo_state.name, block_reason)
                    continue

                # Execute
                raw = json.dumps({"parsed": command.__dict__, "decision": decision})
                await execute_decision(decision, thermo_state, weather_data, raw,
                                       user_triggered=True)
                decision["zone"] = thermo_state.name
                _cycle_decisions.append(decision)
                last_decision = decision

                # Learn from user request + log to manual_overrides for startup replay
                if decision["action"] == "set_temperature" and decision.get("temperature"):
                    _learn_from_override(thermo_state, weather_data, decision["temperature"])
                    database.log_manual_override(
                        decision["temperature"],
                        thermo_state.target_temp,
                        zone=thermo_state.name
                    )

            # Set cooldown
            _last_cooldown_time = datetime.now()

        # ── Path B: Autonomous — comfort model decides, NO LLM ──
        else:
            now = datetime.now()

            for thermo_state in all_states:
                zone = thermo_state.name
                logger.info("[%s] Running comfort model evaluation...", zone)

                # Check and switch HVAC mode
                await check_and_switch_mode(thermo_state, weather_data)

                # Get comfort model prediction
                inp = _build_comfort_input(thermo_state, weather_data, now)
                prediction = _comfort_model.predict(inp)

                logger.info("[%s] Comfort model: target=%.0fF (baseline=%.0fF, corr=%.1f), "
                            "should_change=%s",
                            zone, prediction.target_temp, prediction.baseline_temp,
                            prediction.correction, prediction.should_change)

                if not prediction.should_change:
                    decision = {
                        "action": "no_change",
                        "reasoning": prediction.reasoning,
                    }
                else:
                    decision = {
                        "action": "set_temperature",
                        "temperature": prediction.target_temp,
                        "reasoning": prediction.reasoning,
                    }

                # Guardrails
                allowed, block_reason = check_guardrails(decision)
                if not allowed:
                    logger.warning("[%s] Guardrail blocked: %s", zone, block_reason)
                    database.log_decision(
                        indoor_temp=thermo_state.indoor_temp,
                        outdoor_temp=weather_data.current_temp,
                        action=f"BLOCKED:{decision['action']}",
                        temperature=decision.get("temperature"),
                        reasoning=f"BLOCKED: {block_reason} | {decision.get('reasoning', '')}",
                        raw_response=json.dumps(prediction.__dict__),
                        zone=zone
                    )
                    continue

                # Execute
                raw = json.dumps(prediction.__dict__)
                await execute_decision(decision, thermo_state, weather_data, raw)
                decision["zone"] = zone
                _cycle_decisions.append(decision)
                last_decision = decision

                logger.info("[%s] Cycle complete: action=%s, temp=%s",
                             zone, decision["action"], decision.get("temperature"))

        # Update heartbeat
        database.update_heartbeat()

        # Telegram notifications for temperature changes
        temp_changes = [d for d in _cycle_decisions if d["action"] == "set_temperature"]
        if temp_changes and not user_triggered:
            # Only send autonomous change notifications (user-triggered handled by telegram_bot)
            parts = []
            for d in temp_changes:
                reason = d.get("reasoning", "")[:200]
                parts.append(f"{d['zone']}: set to {d['temperature']:.0f}F — {reason}")
            await send_telegram("\n".join(parts))

        return last_decision

    except Exception as e:
        logger.error("Evaluation cycle failed: %s", e, exc_info=True)
        database.log_error("agent", type(e).__name__, str(e))
        try:
            await asyncio.to_thread(llm_server.stop)
        except Exception:
            pass
        return None


def trigger_evaluation(user_text: Optional[str] = None):
    """Called by Telegram bot to wake the agent loop for an immediate evaluation."""
    global _user_triggered, _pending_user_text
    _user_triggered = True
    _pending_user_text = user_text
    logger.info("Evaluation triggered by user message: %s", (user_text or "")[:50])


def get_last_evaluation_result() -> Optional[dict]:
    """Get the result of the last evaluation cycle (for /status command)."""
    return _last_evaluation_result


def get_evaluation_counter() -> int:
    """Get the evaluation counter (for detecting new cycles)."""
    return _evaluation_counter


# ── Agent loop ───────────────────────────────────────────────────

async def agent_loop():
    """
    Main async loop — runs evaluation every 30 minutes or when triggered.
    """
    global _last_evaluation_result, _evaluation_counter, _user_triggered

    normal_interval = EVAL_INTERVAL_MINUTES * 60

    logger.info("Agent loop starting (interval: %d minutes)", normal_interval // 60)

    # Wait for Telegram bot to register its send function
    logger.info("Waiting for Telegram bot to initialize...")
    for i in range(15):
        if _telegram_send_fn is not None:
            logger.info("Telegram bot ready (took %ds)", (i + 1) * 2)
            break
        await asyncio.sleep(2)
    else:
        logger.warning("Telegram bot did not register in time — summary messages will be skipped")

    # Weekly report day check
    last_report_day = None
    last_cleanup_day = None

    while True:
        try:
            result = await asyncio.wait_for(
                run_evaluation_cycle(),
                timeout=300
            )
            _last_evaluation_result = result
            _evaluation_counter += 1

            # Weekly report on Sundays
            now = datetime.now()
            if now.weekday() == 6 and last_report_day != now.date():
                report = generate_weekly_report()
                await send_telegram(report)
                last_report_day = now.date()

            # Daily cleanup
            if last_cleanup_day != now.date():
                database.cleanup_old_records()
                last_cleanup_day = now.date()

        except asyncio.TimeoutError:
            logger.error("Evaluation cycle timed out after 5 minutes")
            database.log_error("agent", "timeout", "Evaluation cycle exceeded 5 minutes")
        except Exception as e:
            logger.error("Agent loop error: %s", e, exc_info=True)
            database.log_error("agent", type(e).__name__, str(e))

        # If a trigger came in during the cycle, run again immediately
        if _user_triggered:
            logger.info("Trigger received during cycle — running again immediately")
            continue

        # Compute interval: longer during vacation mode
        if location.is_vacation_mode():
            interval_seconds = location.get_vacation_eval_interval_minutes() * 60
        else:
            interval_seconds = normal_interval
        logger.debug("Next evaluation in %d minutes", interval_seconds // 60)

        # Wait for next interval, checking for triggers every 2 seconds
        waited = 0
        while waited < interval_seconds:
            await asyncio.sleep(2)
            waited += 2
            if _user_triggered:
                logger.info("Agent loop woken by user trigger (waited %ds)", waited)
                break


async def _on_vacation_mode_change(is_vacation: bool):
    """Callback fired by location module when vacation mode transitions."""
    if is_vacation:
        vmin = location.get_vacation_temp_min()
        vmax = location.get_vacation_temp_max()
        msg = (f"Vacation mode ACTIVATED — all phones are away.\n"
               f"Widening comfort range to {vmin}-{vmax}F, "
               f"eval interval → {location.get_vacation_eval_interval_minutes()} min.")
    else:
        msg = (f"Vacation mode DEACTIVATED — welcome home!\n"
               f"Restoring normal comfort range ({TEMP_MIN_NORMAL}-{TEMP_MAX_NORMAL}F), "
               f"eval interval → {EVAL_INTERVAL_MINUTES} min.")
    logger.info(msg)
    await send_telegram(msg)


async def main():
    """Entry point — init all modules, run agent loop + Telegram bot."""
    config = load_config()

    # Setup logging
    log_level = config["agent"].get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger.info("AI Thermostat Agent starting...")

    # Init database
    database.init_db(config["agent"]["db_path"])

    # Init weather
    w = config["weather"]
    weather.init_weather(w["api_key"], w["latitude"], w["longitude"],
                         w.get("cache_minutes", 20), w.get("stale_hours", 6))

    # Init Nest API
    nest_api.init_nest(config["nest"]["tokens_path"],
                       devices=config["nest"].get("devices"))

    # Init LLM server manager (for NLP parsing only — not started until needed)
    llm_cfg = config["llm"]
    llm_port = int(llm_cfg["endpoint"].split(":")[-1].split("/")[0])
    llm_server.init(
        server_exe=llm_cfg["server_exe"],
        model_path=llm_cfg["model_path"],
        port=llm_port
    )

    # Init comfort model (replaces LLM for autonomous decisions)
    init_comfort_model(config)

    # Init location tracking (OwnTracks)
    location.init_location(config)
    location.set_vacation_change_callback(_on_vacation_mode_change)

    # Init Telegram bot (imported here to avoid circular imports)
    import telegram_bot
    telegram_bot.init_bot(config)

    logger.info("All modules initialized. Starting agent loop + Telegram bot.")

    await asyncio.gather(
        agent_loop(),
        telegram_bot.start_bot(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user.")
        sys.exit(0)
